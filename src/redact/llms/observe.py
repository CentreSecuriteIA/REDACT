"""Telemetry emit hook for the LLM layer.

:func:`record` passes an event to the installed emitter and does nothing when
none is installed. ``redact.telemetry`` installs the emitter and owns the
sinks, so nothing here imports from above ``llms/``.

A ``call`` event is one transport call, with ``n_items`` the items it carried.
The vLLM engine takes each prompt as its own request, so a batch of 32 is 32
events there as well.
"""

import logging
import threading
from collections.abc import Callable, Hashable

logger = logging.getLogger(__name__)

# The installed sink. None means record() returns immediately.
_emitter: Callable[[dict], None] | None = None

# The stage current events belong to. A module global because thread pools do
# not propagate context variables to their workers, so a ContextVar would read
# empty in every call BatchCaller fans out.
_stage: str | None = None

# What first_in_stage() has seen since the stage label was last set.
_seen: set[Hashable] = set()
_seen_lock = threading.Lock()


def set_emitter(fn: Callable[[dict], None] | None) -> None:
    """Install the process-wide event sink, or clear it with ``None``.

    Called by ``redact.telemetry``.
    """
    global _emitter
    _emitter = fn


def set_stage(stage: str | None) -> None:
    """Set the pipeline stage label attached to later events.

    Called by ``redact.telemetry.stage()``, on entering a stage and on
    leaving it. Events carry ``stage: None`` when no stage is set. What
    :func:`first_in_stage` has seen is forgotten, so a stage run again (a
    second run in a notebook) warns again.
    """
    global _stage
    with _seen_lock:
        _stage = stage
        _seen.clear()


def first_in_stage(key: Hashable) -> bool:
    """Whether ``key`` is new since the stage label was last set: for a
    warning given once per stage run, such as a model's first cut reply."""
    with _seen_lock:
        if key in _seen:
            return False
        _seen.add(key)
        return True


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
