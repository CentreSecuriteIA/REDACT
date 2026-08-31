"""The telemetry emit hook — and deliberately nothing else.

``llms/`` is the layer everything else depends on, and it imports nothing
upward. So it cannot own a collector, a sink, or any vendor SDK: it can only
*emit facts* and let something above decide what they are worth.

That is this module. :func:`record` hands an event to whatever emitter is
registered, defaulting to a no-op — so a fresh ``import redact.llms`` with no
telemetry configured costs one dict lookup per call and pulls in no
dependencies. ``redact.telemetry`` installs the real emitter via
:func:`set_emitter`; see its docstring for sinks and configuration.

A module-level function rather than a method on :class:`LLMBackend`, because
``vllm._engine()`` is a module function (the engine is cached per checkpoint,
not per backend instance) and has to emit its load/lifetime events too.

**Granularity contract:** a ``call`` event is one *real transport call*, not
one batch item. A native-batching pass over 32 prompts is a single event with
``n_items=32``. This is why telemetry cannot hang off ``on_complete``, which
fires per item and would count a shared engine pass 32 times.
"""

import threading
from collections.abc import Callable

# The installed sink. None = nobody is listening; record() returns immediately.
_emitter: Callable[[dict], None] | None = None

# The stage a call belongs to. A plain module global, not a ContextVar: stages
# run strictly sequentially in run_pipeline, and ThreadPoolExecutor does *not*
# propagate context to its workers — a ContextVar would simply read empty for
# every call BatchCaller fans out.
_stage: str | None = None
_lock = threading.Lock()


def set_emitter(fn: Callable[[dict], None] | None) -> None:
    """Install (or clear, with ``None``) the process-wide event sink.

    Called by ``redact.telemetry``; there is no reason for anything else to.
    """
    global _emitter
    _emitter = fn


def set_stage(stage: str | None) -> None:
    """Label the pipeline stage subsequent events belong to.

    Set by ``redact.telemetry.stage()``. Safe to leave unset — events then
    carry ``stage: None``, which is the normal case for direct API callers who
    never went through ``run_pipeline``.
    """
    global _stage
    with _lock:
        _stage = stage


def current_stage() -> str | None:
    """The stage label attached to events right now."""
    return _stage


def record(event: dict) -> None:
    """Emit one telemetry event.

    A no-op when no emitter is installed — the default. Never raises: a
    misbehaving sink must not take down a generation run that is otherwise
    working, so exceptions from the emitter are swallowed deliberately.

    Args:
        event: Must carry an ``"ev"`` key naming the event type
            (``call`` / ``engine`` / ``stage`` / ``run`` / ``residency`` /
            ``preload_failed``). ``stage`` is filled in here if absent.
    """
    emitter = _emitter
    if emitter is None:
        return
    event.setdefault("stage", _stage)
    try:
        emitter(event)
    except Exception:  # noqa: BLE001 — telemetry must never break a run
        pass
