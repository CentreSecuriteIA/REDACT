"""Drive generators that yield :class:`LLMRequest`: one with a blocking call,
many round by round, or a pool of them topped up from a lazy source.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator, Hashable, Iterable

from redact.llms.client import ModelClient

from .request import LLMRequest


def drive_sync(gen: Generator, call: Callable[[LLMRequest], str]):
    """Drive one generator to completion with a blocking call function.

    The generator yields :class:`LLMRequest`s and is resumed with each reply.
    :func:`drive_generators` drives many generators at once.

    Args:
        gen: Generator yielding :class:`LLMRequest`s.
        call: Blocking function that turns one request into a reply string;
            it is handed the whole request, limit included.

    Returns:
        The generator's return value.
    """
    try:
        request = next(gen)
        while True:
            reply = call(request)
            request = gen.send(reply)
    except StopIteration as stop:
        return stop.value


def drive_generators(
    gens: dict[Hashable, Generator],
    *,
    resolve: Callable[[str], object] | None = None,
    finalize: Callable[[Hashable, object], object],
    on_error: Callable[[Hashable, Exception], object] | None = None,
    progress: str | None = None,
) -> dict[Hashable, object]:
    """Advance many generators round by round, batching LLM calls per model.

    Each generator yields :class:`LLMRequest`s and is resumed with the reply.
    Every round, the pending requests are grouped by ``request.client`` (by
    ``request.model`` when none is attached) and ``request.max_tokens``, and
    sent as one ``generate()`` call per group.

    Args:
        gens: ``{key: generator}``. Keys are returned unchanged.
        resolve: ``model_name -> ModelClient`` for requests without a
            client. Defaults to :meth:`ModelClient.create`.
        finalize: ``(key, return_value) -> result``, called when a generator
            finishes. An exception from it propagates.
        on_error: ``(key, exc) -> result``, called when a generator raises
            or yields something other than an :class:`LLMRequest`. ``None``
            lets the exception propagate. A failed batch dispatch always
            propagates and never goes through ``on_error``.
        progress: Label prefix for progress logging. ``None`` logs nothing.

    Returns:
        ``{key: result}`` for every input key.

    Raises:
        RuntimeError: A model returned a different number of replies than
            it was sent requests.
    """
    if resolve is None:
        resolve = ModelClient.create

    results: dict[Hashable, object] = {}
    pending: dict[Hashable, LLMRequest] = {}

    def _fail(key: Hashable, exc: Exception):
        if on_error is None:
            raise exc
        results[key] = on_error(key, exc)

    def advance(key: Hashable, response):
        try:
            request = gens[key].send(response)
            if not isinstance(request, LLMRequest):
                raise TypeError(
                    f"generator yielded {type(request).__name__}, not an LLMRequest"
                )
            pending[key] = request
            return
        except StopIteration as stop:
            value = stop.value
        except Exception as exc:  # noqa: BLE001 — isolate one unit
            _fail(key, exc)
            return
        results[key] = finalize(key, value)

    # Prime every generator to its first request (or completion): sending
    # None starts a generator.
    for key in list(gens):
        advance(key, None)

    round_idx = 0
    while pending:
        round_idx += 1
        groups: dict[Hashable, list] = defaultdict(list)
        for key, req in pending.items():
            groups[(req.client if req.client is not None else req.model,
                    req.max_tokens)].append(key)

        round_requests = dict(pending)
        pending.clear()

        for keys in groups.values():
            first = round_requests[keys[0]]
            model = first.model
            client = first.client if first.client is not None else resolve(model)
            messages_list = [round_requests[k].messages for k in keys]
            kw = {"progress": _label(progress, round_idx, first)} if progress else {}
            if first.max_tokens is not None:
                kw["max_tokens"] = first.max_tokens
            # Pass internals_ids only when a request in this batch wants
            # capture. A client whose backend cannot capture rejects any
            # list, even one that is all None.
            batch_internals_ids = [round_requests[k].internals_id for k in keys]
            if any(i is not None for i in batch_internals_ids):
                kw["internals_ids"] = batch_internals_ids
            # A dispatch failure propagates and does not go through on_error.
            # It is the transport failing, not these units, and turning it
            # into results would let the caller record work that never ran.
            responses = client.generate(messages_list, **kw)
            if len(responses) != len(keys):
                raise RuntimeError(
                    f"Model {model!r} returned {len(responses)} replies for "
                    f"{len(keys)} requests."
                )
            for k, resp in zip(keys, responses):
                advance(k, resp)

    return results


def _label(progress: str, round_idx: int, first: LLMRequest) -> str:
    """Progress label of one call; names the limit when the group has one."""
    limit = f", max_tokens={first.max_tokens}" if first.max_tokens is not None else ""
    return f"{progress} round {round_idx} ({first.model}{limit})"


def _group_key(request: LLMRequest) -> Hashable:
    """The requests that may share one ``generate()`` call.

    A call carries one set of settings, so requests share one only when they
    agree on the reply limit (``None`` meaning the model's default); beyond
    that the same registry name on the same setup is enough, whichever client
    object made the request. A request without a client is keyed by name and
    resolved at dispatch.
    """
    client = request.client
    if client is None:
        return request.model, request.max_tokens
    return client.model, client.backend.backend_name, request.max_tokens


def drive_units(
    source: Iterable[tuple[Hashable, Generator]],
    *,
    on_done: Callable[[Hashable, object], object],
    on_error: Callable[[Hashable, Exception], object] | None = None,
    pool_size: int,
    cap: int | None = None,
    resolve: Callable[[str], ModelClient] | None = None,
    progress: str | None = None,
) -> int:
    """Drive units from a lazy source, keeping ``pool_size`` of them live.

    A unit is ``(key, generator)``; the generator yields :class:`LLMRequest`s
    and is resumed with each reply. Each round admits units from ``source``
    into free slots, groups every live unit's pending request by model, setup
    and reply limit (by ``request.model`` and limit alone when no client is
    attached), sends every group that fills a call — ``cap`` requests, oldest
    first, one ``generate()`` each — or the largest group when none does, and
    hands the replies back. A unit that finishes is reported at once, so the
    caller can save it without waiting for the others. Each distinct limit on
    one model is another group: at ``pool_size = 2 * cap`` up to three keep
    the calls full, more leave them part-full.

    Args:
        source: ``(key, generator)`` pairs, pulled only as slots free up.
            Keys must be unique among live units.
        on_done: ``(key, return_value)``, called when a unit finishes. An
            exception from it propagates.
        on_error: ``(key, exc)``, called when a unit raises or yields
            something other than an :class:`LLMRequest`. ``None`` lets the
            exception propagate. A failed ``generate()`` always propagates
            and never goes through ``on_error``.
        pool_size: Units live at once, at least 1. About twice ``cap``, so
            one group can fill while another drains.
        cap: Requests per ``generate()`` call, at least 1. Defaults to
            ``pool_size``. Requests above it, and a group below it while
            another is full, wait for a later round.
        resolve: ``model_name -> ModelClient`` for requests without a
            client. Defaults to :meth:`ModelClient.create`.
        progress: Label prefix for progress logging. ``None`` logs nothing.

    Returns:
        The number of units that finished or failed.

    Raises:
        RuntimeError: A model returned a different number of replies than
            it was sent requests.
        ValueError: ``pool_size`` or ``cap`` below 1, or a key already live.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be at least 1, got {pool_size}.")
    if cap is None:
        cap = pool_size
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}.")
    if resolve is None:
        resolve = ModelClient.create

    units = iter(source)
    exhausted = False
    live: dict[Hashable, Generator] = {}     # admitted, not finished
    pending: dict[Hashable, LLMRequest] = {}  # awaiting dispatch, oldest first
    finished = 0

    def advance(key, reply):
        nonlocal finished
        try:
            request = live[key].send(reply)
            if not isinstance(request, LLMRequest):
                raise TypeError(
                    f"generator yielded {type(request).__name__}, not an LLMRequest")
            pending[key] = request
            return
        except StopIteration as stop:
            value = stop.value
        except Exception as exc:  # noqa: BLE001 — isolate one unit
            del live[key]
            finished += 1
            if on_error is None:
                raise
            on_error(key, exc)
            return
        del live[key]
        finished += 1
        on_done(key, value)          # outside the try: its exception propagates unchained

    def admit():
        nonlocal exhausted
        while not exhausted and len(live) < pool_size:
            try:
                key, gen = next(units)
            except StopIteration:
                exhausted = True
                return
            if key in live:
                raise ValueError(f"Unit {key!r} is already live.")
            live[key] = gen
            advance(key, None)       # sending None starts it; finishing here frees the slot at once

    round_idx = 0
    while True:
        admit()
        if not pending:
            return finished
        round_idx += 1
        groups: dict[Hashable, list] = defaultdict(list)
        for key, req in pending.items():
            groups[_group_key(req)].append(key)
        # Full calls first; a smaller group waits for the others to fill it.
        # When none is full the largest goes (the oldest on a tie), so a
        # trickle of requests still drains and the run ends.
        full = [keys for keys in groups.values() if len(keys) >= cap]
        for keys in full or [max(groups.values(), key=len)]:
            keys = keys[:cap]                      # the rest stay pending, ahead of later yields
            requests = [pending.pop(k) for k in keys]
            first = requests[0]
            client = first.client if first.client is not None else resolve(first.model)
            kw = {"progress": _label(progress, round_idx, first)} if progress else {}
            if first.max_tokens is not None:
                kw["max_tokens"] = first.max_tokens
            ids = [r.internals_id for r in requests]
            if any(i is not None for i in ids):    # a non-capturing client rejects any list
                kw["internals_ids"] = ids
            responses = client.generate([r.messages for r in requests], **kw)
            if len(responses) != len(keys):
                raise RuntimeError(
                    f"Model {first.model!r} returned {len(responses)} replies for "
                    f"{len(keys)} requests.")
            for k, resp in zip(keys, responses):
                advance(k, resp)
