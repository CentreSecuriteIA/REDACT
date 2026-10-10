"""Probe the machine's GPUs, and measure what an in-process load took on them.

:func:`detect_gpus` and :func:`detect_gpu_memory_gib` ask ``nvidia-smi`` for
the cards, for the planner and the cost roll-up. :func:`free_total_gib` is
the planner's fallback for the card size.

Only the transformers backend measures its loads (:class:`Measurement`), for
telemetry. vLLM loads are not measured, and planning uses
:mod:`redact.llms.resources.estimate`.

All sizes are GiB (bytes / 1024**3).
"""

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024 ** 3

#TODO: Implementation of a separate measure capability which loads only one model and then saves the value, nothing more. Discuss first if even useful.

def _torch():
    """Return ``torch`` if it is importable and CUDA is usable, else ``None``."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — a broken install raises OSError, not ImportError
        if not isinstance(exc, ImportError):
            logger.debug("torch failed to import (%s: %s)", type(exc).__name__, exc)
        return None
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:  # noqa: BLE001 — a broken CUDA install must not raise here
        return None
    return torch


def free_total_gib() -> tuple[float, float] | None:
    """``(free_gib, total_gib)`` of the current device, or ``None`` if unreadable."""
    torch = _torch()
    if torch is None:
        return None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        logger.debug("mem_get_info failed (%s: %s)", type(exc).__name__, exc)
        return None
    return free / _BYTES_PER_GIB, total / _BYTES_PER_GIB


def detect_gpus() -> tuple[str | None, int]:
    """Which GPU this machine has, and how many, via ``nvidia-smi``.

    Returns:
        ``(name, count)`` — one line per device from
        ``nvidia-smi --query-gpu=name --format=csv,noheader``, so the name
        matches the pricing table and the line count is the device count.
        ``(None, 0)`` when ``nvidia-smi`` is absent or fails: a machine without
        it reports no GPUs rather than raising.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, 0
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi failed (%s: %s); reporting no GPUs", type(exc).__name__, exc)
        return None, 0
    names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return (names[0] if names else None), len(names)


def detect_gpu_memory_gib() -> float | None:
    """Total VRAM per device, via ``nvidia-smi``.

    The residency planner's other capacity source, ``torch.cuda.mem_get_info``,
    needs torch installed *and* CUDA usable — neither is true on a machine that
    merely has a card. ``nvidia-smi`` answers without either, so capacity is
    known wherever a GPU is, not only where the full stack is.

    Returns:
        Per-device total in **GiB** (the first device; mixed-card machines
        are not modelled), or ``None`` when ``nvidia-smi`` is absent or
        unparseable.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi memory query failed (%s)", exc)
        return None
    for line in out.splitlines():
        line = line.strip()
        if line:
            try:
                # MiB -> GiB. 1024-based, matching resources/estimate.py:
                # mixing this with a decimal-GB divisor there is what made
                # the same card read 48.0 or 44.7 depending on the path.
                return round(int(line) / 1024, 1)
            except ValueError:
                return None
    return None


def _devices(cuda) -> list:
    """Indices of the visible devices, or ``[None]`` (the current device)
    when they cannot be counted."""
    try:
        return list(range(cuda.device_count())) or [None]
    except Exception:  # noqa: BLE001
        return [None]


def _on(probe, device):
    """Call a ``torch.cuda`` probe for one device; ``None`` is the current one."""
    return probe() if device is None else probe(device)


class Measurement:
    """Context manager that measures an in-process (transformers) model load.

    Wrap the load in it and read the results afterwards. Figures are summed
    over all visible devices, so a ``device_map="auto"`` load is counted
    whole. Every result stays ``None`` without CUDA::

        with Measurement() as m:
            model = AutoModelForCausalLM.from_pretrained(...)
        print(m.claimed_gib, m.weights_gib)

    Probing a device creates this process's CUDA context on it.

    Attributes:
        claimed_gib: Drop in free device memory over the load
            (``mem_get_info()``). Everything any process took in that time.
        weights_gib: Peak memory torch's allocator held during the load,
            above what it held before. Covers this process only: the weights
            plus any load-time temporaries, and ``None`` when the load
            allocated nothing here.
        before_free_gib: Free device memory when the load started.
        before_allocated_gib: What torch's allocator held when the load
            started.
    """

    def __init__(self):
        self._torch = _torch()
        self._devices: list = []
        self.before_free_gib: float | None = None
        self.before_allocated_gib: float | None = None
        self.claimed_gib: float | None = None
        self.weights_gib: float | None = None

    def __enter__(self) -> "Measurement":
        if self._torch is None:
            return self
        with _suppress():
            cuda = self._torch.cuda
            self._devices = _devices(cuda)
            info = [_on(cuda.mem_get_info, d) for d in self._devices]
            self.before_free_gib = sum(free for free, _ in info) / _BYTES_PER_GIB
            # reset_peak_memory_stats() resets the peak to what is currently
            # allocated, not to zero, so record that as the baseline.
            allocated = sum(_on(cuda.memory_allocated, d) for d in self._devices)
            self.before_allocated_gib = allocated / _BYTES_PER_GIB
            for d in self._devices:
                _on(cuda.reset_peak_memory_stats, d)
        return self

    def __exit__(self, *exc) -> bool:
        if self.before_free_gib is not None:
            with _suppress():
                cuda = self._torch.cuda
                free = sum(_on(cuda.mem_get_info, d)[0] for d in self._devices)
                self.claimed_gib = round(
                    max(0.0, self.before_free_gib - free / _BYTES_PER_GIB), 2)
                if self.before_allocated_gib is not None:
                    peak = sum(_on(cuda.max_memory_allocated, d) for d in self._devices)
                    delta = peak / _BYTES_PER_GIB - self.before_allocated_gib
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
