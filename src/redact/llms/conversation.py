"""Conversation primitives shared by ``jailbreak/`` and ``multi_turn/``.

- :class:`LLMRequest`: a pending model call yielded by a generator.
- :class:`Step` / :class:`Transcript`: a typed step log for multi-turn
  conversations.
- :func:`drive_sync`: drive one generator to completion with a blocking call.
- :func:`drive_generators`: drive many generators round by round, batching
  their requests per model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator, Hashable
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LLMRequest:
    """A pending LLM call yielded by a generator.

    Attributes:
        model: Model name. Pending requests are grouped by it and sent one
            batch per model.
        messages: Chat messages for the call.
        internals_id: Capture this call's internals under this id. ``None``
            captures nothing.
    """

    model: str
    messages: list[dict]
    internals_id: str | None = None


# Step kinds. Only "message" and "reply" steps are sent to models.
StepType = Literal["strategy", "message", "reply", "analysis", "evaluation"]
_VISIBLE = ("message", "reply")


@dataclass
class Step:
    """One entry in a conversation's step log.

    ``message`` and ``reply`` steps carry a chat ``role`` and are rendered
    into the messages sent to a model. ``strategy``, ``analysis`` and
    ``evaluation`` steps record the actor's ideas, analyses and scores, and
    are never sent.
    """

    type: StepType
    actor: str
    content: str
    role: str | None = None          # ChatMessage role for visible steps
    model: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Transcript:
    """Ordered step log of a conversation."""

    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def message(self, actor: str, content: str, role: str = "user",
                model: str | None = None, meta: dict | None = None) -> Step:
        return self.add(Step("message", actor, content, role=role, model=model, meta=meta or {}))

    def reply(self, actor: str, content: str, role: str = "assistant",
              model: str | None = None, meta: dict | None = None) -> Step:
        return self.add(Step("reply", actor, content, role=role, model=model, meta=meta or {}))

    def note(self, type: StepType, actor: str, content: str,
             model: str | None = None, meta: dict | None = None) -> Step:
        """Log a provenance event (strategy / analysis / evaluation)."""
        return self.add(Step(type, actor, content, model=model, meta=meta or {}))

    def as_messages(self, system: str | None = None) -> list[dict]:
        """Render the visible turns into a chat ``messages`` list."""
        msgs: list[dict] = [{"role": "system", "content": system}] if system else []
        for s in self.steps:
            if s.type in _VISIBLE and s.role:
                msgs.append({"role": s.role, "content": s.content})
        return msgs

    def to_records(self) -> list[dict]:
        """The full step log as a list of dicts, for JSON storage."""
        return [vars(s) for s in self.steps]


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
    verbose: bool = True,
    progress: str | None = None,
) -> dict[Hashable, object]:
    """Advance many generators round by round, batching LLM calls per model.

    Each generator yields :class:`LLMRequest`s and is resumed with the reply.
    Every round, the pending requests are grouped by ``request.model`` and
    sent as one ``generate()`` call per model. Used by the jailbreak engine
    and the multi-turn pipeline.

    Args:
        gens: ``{key: generator}``. Keys are returned unchanged.
        resolve: ``model_name -> ModelClient``. Defaults to
            :meth:`ModelClient.create`.
        finalize: ``(key, return_value) -> result``, called when a generator
            finishes.
        on_error: ``(key, exc) -> result``, called when a generator raises.
            ``None`` lets the exception propagate. A failed batch dispatch
            always propagates and never goes through ``on_error``.
        verbose: Pass a progress label to ``generate`` when ``progress`` is
            also set.
        progress: Label prefix for progress logging.

    Returns:
        ``{key: result}`` for every input key.
    """
    if resolve is None:
        from .client import ModelClient  # local: avoids an import cycle
        resolve = ModelClient.create

    results: dict[Hashable, object] = {}
    pending: dict[Hashable, LLMRequest] = {}

    def _fail(key: Hashable, exc: Exception):
        if on_error is None:
            raise exc
        results[key] = on_error(key, exc)

    def advance(key: Hashable, response):
        try:
            pending[key] = gens[key].send(response)
        except StopIteration as stop:
            results[key] = finalize(key, stop.value)
        except Exception as exc:  # noqa: BLE001 — isolate one unit
            _fail(key, exc)

    # Prime every generator to its first request (or completion).
    for key in list(gens):
        try:
            pending[key] = next(gens[key])
        except StopIteration as stop:
            results[key] = finalize(key, stop.value)
        except Exception as exc:  # noqa: BLE001
            _fail(key, exc)

    round_idx = 0
    while pending:
        round_idx += 1
        groups: dict[str, list] = defaultdict(list)
        for key, req in pending.items():
            groups[req.model].append(key)

        round_requests = dict(pending)
        pending.clear()

        for model, keys in groups.items():
            messages_list = [round_requests[k].messages for k in keys]
            kw = (
                {"progress": f"{progress} round {round_idx} ({model})"}
                if (verbose and progress) else {}
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
            responses = resolve(model).generate(messages_list, **kw)
            for k, resp in zip(keys, responses):
                advance(k, resp)

    return results
