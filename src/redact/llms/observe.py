"""Telemetry emit hook for the LLM layer.

:func:`record` passes an event to the installed emitter and does nothing when
none is installed. ``redact.telemetry`` installs the emitter and owns the
sinks, so nothing here imports from above ``llms/``.

A ``call`` event is one transport call, not one batch item: a native vLLM pass
over 32 prompts is a single event with ``n_items=32``.
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# The installed sink. None means record() returns immediately.
_emitter: Callable[[dict], None] | None = None

# The stage current events belong to. A module global because thread pools do
# not propagate context variables to their workers, so a ContextVar would read
# empty in every call BatchCaller fans out.
_stage: str | None = None


def set_emitter(fn: Callable[[dict], None] | None) -> None:
    """Install the process-wide event sink, or clear it with ``None``.

    Called by ``redact.telemetry``.
    """
    global _emitter
    _emitter = fn


def set_stage(stage: str | None) -> None:
    """Set the pipeline stage label attached to later events.

    Called by ``redact.telemetry.stage()``. Events carry ``stage: None`` when
    no stage is set.
    """
    global _stage
    _stage = stage


def current_stage() -> str | None:
    """The stage label currently attached to events."""
    return _stage


def record(event: dict) -> None:
    """Emit one telemetry event.

    Does nothing when no emitter is installed. Never raises: an exception from
    the emitter is logged at DEBUG, so a broken sink cannot stop a run.

    Args:
        event: Must carry an ``"ev"`` key naming the event type (e.g.
            ``call`` or ``local``). ``stage`` is filled in here if absent.
    """
    emitter = _emitter
    if emitter is None:
        return
    event.setdefault("stage", _stage)
    try:
        emitter(event)
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a run
        logger.debug("Telemetry emitter failed (%s: %s)", type(exc).__name__, exc)
