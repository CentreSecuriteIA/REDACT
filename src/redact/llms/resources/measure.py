"""What a local load *actually* took on the GPU — a diagnostic, not a planner.

Planning runs on :mod:`redact.llms.resources.estimate`, which computes a
footprint from the checkpoint's config. This module is the other question:
after a real load, how much of the card went away? That is the right question
for telemetry ("what did this engine hold") and for a human checking an
estimate against reality — and the wrong one for planning, because:

- **vLLM preallocates** ``gpu_memory_utilization x card`` for weights *and* KV,
  so ``claimed_gib`` describes the *setting*, not the model. Plan from it and
  every model looks like it needs most of the card.
- **``weights_gib`` is usually unavailable for vLLM.** It comes from torch's
  own allocator, and vLLM V1 runs its engine core in a separate process, so
  this process never sees those allocations. It is meaningful for
  ``introspection.py``, which loads in-process via ``transformers``.

Nothing here is cached or fed back into planning. That loop existed, and was
removed: it measured a number that is either unobtainable (vLLM weights) or
unusable (claimed), while the one number it could have provided is computable
from the config without loading anything.

Units are **GiB** (bytes / 1024**3) throughout, matching ``estimate.py``,
``nvidia-smi``, and what GPU spec sheets mean by "GB".
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
    """``(free_gb, total_gib)`` for the current device, or ``None`` without CUDA."""
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
    """Brackets a local model load and reports what it took.

    Use as a context manager around the load. Everything degrades to ``None``
    without CUDA, so the calling backend needs no guard of its own::

        with Measurement() as m:
            engine = LLM(...)
        m.record(path, key, settings, backend="vllm")

    Two meters, deliberately, because they answer different questions:

    - ``claimed_gib`` — ``mem_get_info()``, i.e. the whole card at driver
      level. What the *runtime* took, KV pool and all. For vLLM that is a fact
      about ``gpu_memory_utilization``, not about the model.
    - ``weights_gib`` — torch's own allocator, which never sees vLLM's KV pool.
      The parameters this load added, and nothing else.

    ``weights_gib`` is therefore a **reference floor, not a runnable size**: no
    model fits in its own weights. Activations, KV cache and the CUDA context
    all sit on top — see :func:`planning_gb`, which adds the KV estimate and
    carries the open question of how much further headroom belongs there.
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
                # Baseline for the delta below. reset_peak_memory_stats() does
                # NOT zero the peak — it resets it to *currently allocated* —
                # so with another checkpoint already resident the raw peak
                # after this load is (resident + mine). Recording that as this
                # model's floor would overstate it by whatever else happened to
                # be on the card, and residency.preload() loads local models
                # sequentially in one process, so that is the normal case, not
                # an edge one. Subtracting this baseline is what makes the
                # number mean "what *this* load added".
                allocated = self._torch.cuda.memory_allocated()
                self.before_allocated_gib = allocated / _BYTES_PER_GIB
                self._torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> bool:
        ft = free_total_gib()
        if ft is not None and self.before_free_gib is not None:
            free_after, _ = ft
            # Everything the runtime took off the card, however it took it.
            self.claimed_gib = round(max(0.0, self.before_free_gib - free_after), 2)
            with _suppress():
                # torch's own allocator only sees tensors torch allocated —
                # i.e. the weights. vLLM's KV pool is outside it, which is
                # precisely what makes this the floor rather than the total.
                peak = self._torch.cuda.max_memory_allocated() / _BYTES_PER_GIB
                delta = peak - (self.before_allocated_gib or 0.0)
                self.weights_gib = round(delta, 2) if delta > 0 else None
        return False  # never swallow a load failure



class _suppress:
    """Swallow anything a best-effort CUDA probe throws.

    Deliberately NOT a blanket catch: ``KeyboardInterrupt`` and ``SystemExit``
    pass through. They are not probe failures, and swallowing them makes a
    Ctrl-C during a multi-minute model load do nothing visible.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc_type is None or issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            return False
        logger.debug("CUDA probe failed (%s: %s)", exc_type.__name__, exc)
        return True
