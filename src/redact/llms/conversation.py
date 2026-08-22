"""Conversation primitives shared across the library (model-layer).

These are use-case-agnostic and live in ``llms/`` so both ``jailbreak/`` (technique
generators) and ``multi_turn/`` (conversation actors) reuse them without a
cross-dependency (``multi_turn/`` and the jailbreak layer depend on ``llms/``,
never the reverse).

Contents:
- :class:`LLMRequest` — a pending model call (``model`` + ``messages``) yielded by
  any generator; ``model`` is the routing key. (Historically defined in
  ``jailbreak/protocol.py``; it is the model layer's type and is now defined here,
  re-exported there for backward compatibility.)
- :class:`Step` / :class:`Transcript` — a **typed step log** for multi-turn
  conversations: visible turns (``message``/``reply`` — Inspect ``ChatMessage``-like,
  rendered into a backend ``messages`` list) plus provenance events
  (``strategy``/``analysis``/``evaluation``) that are logged but never sent to models.
- :func:`drive_sync` — drive one generator to completion with a blocking call fn
  (the single-conversation / single-sample path; the batched engine drives many).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator, Hashable
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LLMRequest:
    """A single pending LLM call yielded by a generator.

    ``model`` is the resolved model name and the engine's routing key (pending
    requests are grouped by ``model`` and dispatched one batch per model).
    ``internals_id`` is optional — set by a caller that wants this specific
    call's internals captured (see ``LLMBackend.supports_internals``); left
    ``None`` by default, which is a no-op all the way down.
    """

    model: str
    messages: list[dict]
    internals_id: str | None = None


# Step kinds: visible turns vs provenance events (Inspect-transcript-event-like).
StepType = Literal["strategy", "message", "reply", "analysis", "evaluation"]
_VISIBLE = ("message", "reply")


@dataclass
class Step:
    """One entry in a conversation's typed step log.

    ``message``/``reply`` steps carry a ChatMessage ``role`` and render into the
    ``messages`` sent to a backend; ``strategy``/``analysis``/``evaluation`` steps
    are provenance (the actor's *ideas*/analyses/scores) and are not sent to models.
    """

    type: StepType
    actor: str
    content: str
    role: str | None = None          # ChatMessage role for visible steps
    model: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Transcript:
    """Ordered step log of a conversation (the trace)."""

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
        """Render the visible turns into a backend ``messages`` list."""
        msgs: list[dict] = [{"role": "system", "content": system}] if system else []
        for s in self.steps:
            if s.type in _VISIBLE and s.role:
                msgs.append({"role": s.role, "content": s.content})
        return msgs

    def to_records(self) -> list[dict]:
        """Serializable form of the full step log (for JSON storage)."""
        return [vars(s) for s in self.steps]


def drive_sync(gen: Generator, call: Callable[[LLMRequest], str]):
    """Drive one generator to completion with a blocking ``call(LLMRequest) -> str``.

    The generator yields :class:`LLMRequest`s and is resumed with the reply string;
    returns whatever the generator returns (e.g. a ``Trajectory`` or ``(text, info)``).
    :func:`drive_generators` drives many such generators at once instead.
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
    router,
    finalize: Callable[[Hashable, object], object],
    on_error: Callable[[Hashable, Exception], object] | None = None,
    verbose: bool = True,
    progress: str | None = None,
) -> dict[Hashable, object]:
    """Advance many generators **round-by-round, batching LLM calls by model**.

    The generic engine behind both the jailbreak combination engine and the
    multi-turn conversation runner. Each generator in ``gens`` yields
    :class:`LLMRequest`s and is resumed with the reply string. Every round, all
    live generators' pending requests are pooled by ``request.model`` and
    dispatched in one ``router.batch_generate(model, messages_list, ...)`` per
    model; replies are fed back via ``.send``.

    Args:
        gens: ``{key: generator}``. Keys are arbitrary hashables (returned as-is).
        router: object with ``batch_generate(model, messages_list, **kw) -> list[str]``.
        finalize: ``(key, return_value) -> result`` — called when a generator
            completes (``StopIteration``).
        on_error: ``(key, exc) -> result`` — called if a generator (or a whole
            batch) raises, isolating that unit. If ``None``, the exception
            propagates.
        verbose / progress: when both set, a progress label is passed to
            ``batch_generate`` (per model per round); otherwise no ``progress``
            kwarg is passed (preserving the minimal ``batch_generate`` contract).

    Returns:
        ``{key: result}`` for every input key.
    """
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
            # Only pass internals_ids when at least one request in this
            # model's batch actually wants capture — a list of all-None would
            # still trip BatchCaller's "backend doesn't support this" guard on
            # a non-introspection model, since it only treats a bare `None`
            # (the kwarg omitted) as "capture not requested".
            batch_internals_ids = [round_requests[k].internals_id for k in keys]
            if any(i is not None for i in batch_internals_ids):
                kw["internals_ids"] = batch_internals_ids
            try:
                responses = router.batch_generate(model, messages_list, **kw)
            except Exception as exc:  # noqa: BLE001 — whole batch failed
                for k in keys:
                    _fail(k, exc)
                continue
            for k, resp in zip(keys, responses):
                advance(k, resp)

    return results
