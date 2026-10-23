"""Drive generators that yield :class:`LLMRequest`: one with a blocking call,
many round by round, or a pool of them topped up from a lazy source.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from collections import defaultdict
from collections.abc import Callable, Generator, Hashable, Iterable
from concurrent.futures import Future

from redact.exceptions import PromptTooLongError
from redact.llms import aio
from redact.llms.client import ModelClient

from .request import LLMRequest

logger = logging.getLogger(__name__)


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
    ``request.model`` when none is attached), ``request.max_tokens`` and
    ``request.temperature``, and sent as one ``generate()`` call per group.

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
                    req.max_tokens, req.temperature)].append(key)

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
            if first.temperature is not None:
                kw["temperature"] = first.temperature
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
    """Progress label of one call; names the limit and the temperature when
    the group has them."""
    limit = f", max_tokens={first.max_tokens}" if first.max_tokens is not None else ""
    if first.temperature is not None:
        limit += f", temperature={first.temperature}"
    return f"{progress} round {round_idx} ({first.model}{limit})"


def _group_key(request: LLMRequest) -> Hashable:
    """The requests that may share one ``generate()`` call.

    A call carries one set of settings, so requests share one only when they
    agree on the reply limit and the temperature (``None`` meaning the
    model's default); beyond that the same registry name on the same setup
    is enough, whichever client object made the request. A request without a
    client is keyed by name and resolved at dispatch.
    """
    client = request.client
    if client is None:
        return request.model, request.max_tokens, request.temperature
    return (client.model, client.backend.backend_name, request.max_tokens,
            request.temperature)


def _has_async_calls(client) -> bool:
    """Whether ``client``'s requests go on the async lane.

    A client that declares no flags makes synchronous calls.
    """
    config = getattr(client, "compute_config", None)
    return getattr(config, "supports_async_calls", False)


async def _cancel(tasks: set[asyncio.Task]) -> None:
    """Cancel the requests still on the loop and wait until each has stopped."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def pool_size(batch_size: int, *clients: ModelClient | None) -> int:
    """Units a stage keeps live: two calls' worth, or its widest async lane.

    A local engine runs hundreds of requests at once, far more than a
    synchronous call holds, and each live unit has one request in flight.
    A hosted model's lane is narrow and changes nothing.
    """
    lanes = [client.backend.max_workers for client in clients
             if client is not None and _has_async_calls(client)]
    return max([2 * batch_size, *lanes])


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
    into free slots, groups every live unit's pending request by model, setup,
    reply limit and temperature (by ``request.model``, limit and temperature
    alone when no client is attached), sends every group that fills a call —
    ``cap`` requests, oldest first, one ``generate()`` each — or the largest
    group when none does, and
    hands the replies back. A unit that finishes is reported at once, so the
    caller can save it without waiting for the others. Each distinct limit on
    one model is another group: at ``pool_size = 2 * cap`` two keep the
    calls full, more leave them part-full.

    A request whose client has async calls (``supports_async_calls``) is not
    grouped. It is sent on the shared event loop the moment its unit yields
    it, and its reply is handed back the moment it arrives, so its slot never
    waits for a longer reply. One model has at most ``backend.max_workers``
    such requests in flight. A reply that has arrived is handed back before a
    round's calls go out. While any is in flight only full calls go out: a
    smaller group waits for the next reply, which may fill it, and the
    largest group goes once none is in flight.

    An async request that fails with
    :class:`~redact.exceptions.PromptTooLongError` is raised inside its unit,
    at the ``yield`` that made it: the unit may catch it and go on, and the
    other requests stay in flight. A ``generate()`` that fails with it fails
    the run, as with any error: which of its items was too long is not known.

    Args:
        source: ``(key, generator)`` pairs, pulled only as slots free up.
            Keys must be unique among live units.
        on_done: ``(key, return_value)``, called when a unit finishes. An
            exception from it propagates.
        on_error: ``(key, exc)``, called when a unit raises or yields
            something other than an :class:`LLMRequest`. ``None`` lets the
            exception propagate. A failed ``generate()`` or async request
            always propagates and never goes through ``on_error``, unless it
            was raised inside its unit.
        pool_size: Units live at once, at least 1. About twice ``cap``, so
            one group can fill while another drains.
        cap: Requests per ``generate()`` call, at least 1. Defaults to
            ``pool_size``. Requests above it, and a group below it while
            another is full, wait for a later round. Async requests are not
            counted against it.
        resolve: ``model_name -> ModelClient`` for requests without a
            client. Defaults to :meth:`ModelClient.create`.
        progress: Label prefix for progress logging. ``None`` logs nothing.
            Async requests log one line when a model's first one goes out,
            and one every ``cap`` finished units.

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

    flying: dict[Hashable, Future] = {}       # async requests in flight
    arrived: queue.Queue = queue.Queue()      # their keys, as each one ends
    gates: dict[Hashable, asyncio.Semaphore] = {}
    tasks: set[asyncio.Task] = set()          # touched on the event loop only
    by_name: dict[str, ModelClient] = {}
    reported = 0

    async def fly(key, client, request, gate) -> str:
        """Make one request on the event loop, under its model's ``gate``."""
        task = asyncio.current_task()
        tasks.add(task)
        try:
            async with gate:
                return await client.agenerate(
                    request.messages,
                    max_tokens=request.max_tokens, temperature=request.temperature,
                    internals_id=request.internals_id,
                )
        finally:
            tasks.discard(task)
            # From the loop, as the last step: the driver then sees the ends
            # in the order they happened.
            arrived.put(key)

    def launch(key, request) -> bool:
        """Send ``request`` on the async lane, if its client has one."""
        client = request.client
        if client is None:
            if request.model not in by_name:
                by_name[request.model] = resolve(request.model)
            client = by_name[request.model]
        if not _has_async_calls(client):
            return False
        gate_key = client.model, client.backend.backend_name
        if gate_key not in gates:
            gates[gate_key] = asyncio.Semaphore(client.backend.max_workers)
            if progress:
                logger.info("%s: %s on the async lane, at most %d in flight",
                            progress, client.model, client.backend.max_workers)
        flying[key] = aio.submit(fly(key, client, request, gates[gate_key]))
        return True

    def advance(key, reply, error=None):
        """Resume a unit with ``reply``, or with ``error`` raised at its ``yield``."""
        nonlocal finished
        try:
            unit = live[key]
            request = unit.send(reply) if error is None else unit.throw(error)
            if not isinstance(request, LLMRequest):
                raise TypeError(
                    f"generator yielded {type(request).__name__}, not an LLMRequest")
        except StopIteration as stop:
            value = stop.value
        except Exception as exc:  # noqa: BLE001 — isolate one unit
            del live[key]
            finished += 1
            if on_error is None:
                raise
            on_error(key, exc)
            return
        else:
            # Outside the try: failing to send a request is not the unit's error.
            if not launch(key, request):
                pending[key] = request
            return
        del live[key]
        finished += 1
        on_done(key, value)          # outside the try: its exception propagates unchained

    def land(wait: bool) -> bool:
        """Hand back the async reply that arrived first, if one has.

        Args:
            wait: Block until one arrives.

        Returns:
            Whether a unit was advanced.
        """
        nonlocal reported
        if not flying:
            return False
        while True:
            try:
                # Wake every second: on Windows a blocked get() does not see
                # Ctrl-C.
                key = arrived.get(block=wait, timeout=1.0)
                break
            except queue.Empty:
                if not wait:
                    return False
        # The result follows its key within a turn of the loop. A failed
        # request raises here: it is the transport failing, as a failed
        # generate() is, so it propagates. A prompt too long for the model
        # fails that one request and nothing else, so it goes to its unit.
        try:
            reply, error = flying.pop(key).result(), None
        except PromptTooLongError as exc:
            reply, error = None, exc
        advance(key, reply, error)
        if progress and finished - reported >= cap:
            reported = finished
            logger.info("%s: %d units done, %d in flight",
                        progress, finished, len(flying))
        return True

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
    try:
        while True:
            admit()
            if land(wait=not pending):
                continue                           # a slot may be free: admit first
            if not pending:
                return finished
            groups: dict[Hashable, list] = defaultdict(list)
            for key, req in pending.items():
                groups[_group_key(req)].append(key)
            # Full calls first; a smaller group waits for the others to fill it.
            # When none is full the largest goes (the oldest on a tie), so a
            # trickle of requests still drains and the run ends.
            full = [keys for keys in groups.values() if len(keys) >= cap]
            if not full and land(wait=True):       # a reply in flight may fill one
                continue
            round_idx += 1
            for keys in full or [max(groups.values(), key=len)]:
                keys = keys[:cap]                      # the rest stay pending, ahead of later yields
                requests = [pending.pop(k) for k in keys]
                first = requests[0]
                client = first.client if first.client is not None else resolve(first.model)
                kw = {"progress": _label(progress, round_idx, first)} if progress else {}
                if first.max_tokens is not None:
                    kw["max_tokens"] = first.max_tokens
                if first.temperature is not None:
                    kw["temperature"] = first.temperature
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
    finally:
        # On any exit with requests in flight (a failed request, an exception
        # from on_done or the source, Ctrl-C): none may outlive the call.
        if flying:
            aio.run(_cancel(tasks))
