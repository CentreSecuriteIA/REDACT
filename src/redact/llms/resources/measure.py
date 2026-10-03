"""Measure what a local model load took on the GPU, for telemetry.

Planning does not use these numbers. It uses
:mod:`redact.llms.resources.estimate`, because:

- vLLM preallocates ``gpu_memory_utilization x card``, so ``claimed_gib``
  describes that setting and not the model.
- ``weights_gib`` comes from torch's allocator in this process. vLLM runs its
  engine in a separate process, so the figure is usually unavailable there.
  It is meaningful for ``introspection.py``, which loads in-process.

All sizes are GiB (bytes / 1024**3).
"""


import logging

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024 ** 3


def _torch():
    """Return ``torch`` if it is importable and CUDA is usable, else ``None``."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:  # noqa: BLE001 — a broken CUDA install must not raise here
        return None
    return torch


def free_total_gib() -> tuple[float, float] | None:
    """``(free_gib, total_gib)`` for the current device, or ``None`` without CUDA."""
    torch = _torch()
    if torch is None:
        return None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        logger.debug("mem_get_info failed (%s: %s)", type(exc).__name__, exc)
        return None
    return free / _BYTES_PER_GIB, total / _BYTES_PER_GIB


class Measurement:
    """Context manager that measures a local model load.

    Wrap the load in it and read the results afterwards. Every result stays
    ``None`` without CUDA::

        with Measurement() as m:
            engine = LLM(...)
        print(m.claimed_gib, m.weights_gib)

    Attributes:
        claimed_gib: Drop in free device memory over the load
            (``mem_get_info()``). Everything the runtime took, including a
            preallocated pool.
        weights_gib: Peak memory torch's allocator added in this process
            during the load. A lower bound: a model cannot run in its
            weights alone (:func:`estimate.planning_gib` adds KV and
            headroom).
        total_gib: Total memory of the device.
    """

    def __init__(self):
        self._torch = _torch()
        self.before_free_gib: float | None = None
        self.before_allocated_gib: float | None = None
        self.claimed_gib: float | None = None
        self.weights_gib: float | None = None
        self.total_gib: float | None = None

    def __enter__(self) -> "Measurement":
        ft = free_total_gib()
        if ft is not None:
            self.before_free_gib, self.total_gib = ft
            with _suppress():
                # reset_peak_memory_stats() resets the peak to what is
                # currently allocated, not to zero. Record that baseline so
                # the delta in __exit__ counts only what this load added,
                # even with another model already loaded.
                allocated = self._torch.cuda.memory_allocated()
                self.before_allocated_gib = allocated / _BYTES_PER_GIB
                self._torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> bool:
        ft = free_total_gib()
        if ft is not None and self.before_free_gib is not None:
            free_after, _ = ft
            self.claimed_gib = round(max(0.0, self.before_free_gib - free_after), 2)
            with _suppress():
                # torch's allocator counts only tensors allocated by torch in
                # this process.
                peak = self._torch.cuda.max_memory_allocated() / _BYTES_PER_GIB
                delta = peak - (self.before_allocated_gib or 0.0)
                self.weights_gib = round(delta, 2) if delta > 0 else None
        return False  # never swallow a load failure



class _suppress:
    """Swallow and log exceptions from a best-effort CUDA probe.

    ``KeyboardInterrupt`` and ``SystemExit`` pass through.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc_type is None or issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            return False
        logger.debug("CUDA probe failed (%s: %s)", exc_type.__name__, exc)
        return True
