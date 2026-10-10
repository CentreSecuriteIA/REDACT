"""Drive generators that yield :class:`LLMRequest`: one with a blocking call,
or many round by round with one batch per model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator, Hashable

from redact.llms.client import ModelClient

from .request import LLMRequest


def drive_sync(gen: Generator, call: Callable[[LLMRequest], str]):
    """Drive one generator to completion with a blocking call function.

    The generator yields :class:`LLMRequest`s and is resumed with each reply.
    :func:`drive_generators` drives many generators at once.

    Args:
        gen: Generator yielding :class:`LLMRequest`s.
        call: Blocking function that turns one request into a reply string.

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
    ``request.model`` when none is attached) and sent as one ``generate()``
    call per group.

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
            groups[req.client if req.client is not None else req.model].append(key)

        round_requests = dict(pending)
        pending.clear()

        for keys in groups.values():
            first = round_requests[keys[0]]
            model = first.model
            client = first.client if first.client is not None else resolve(model)
            messages_list = [round_requests[k].messages for k in keys]
            kw = (
                {"progress": f"{progress} round {round_idx} ({model})"}
                if progress else {}
            )
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
