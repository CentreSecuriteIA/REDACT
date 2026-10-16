"""Probe the machine's GPUs, and measure what an in-process load took on them.

:func:`gpu_cards` asks ``nvidia-smi`` for each card, for the planner and the
loader, :func:`detect_gpus` for the cards' name and count, for the cost
roll-up, and :func:`card_use` for what each card holds, for the run's check
before it loads a model.
:func:`free_total_gib` is the planner's fallback for the card size.

A vLLM build narrows ``CUDA_VISIBLE_DEVICES`` for the whole process
(:func:`visible_only`); :data:`cuda_lock` keeps what starts CUDA in this
process from starting it meanwhile.

Only the transformers backend measures its loads (:class:`Measurement`), for
telemetry. vLLM loads are not measured, and planning uses
:mod:`redact.llms.resources.estimate`.

All sizes are GiB (bytes / 1024**3).
"""

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024 ** 3

#: Held while a build has narrowed ``CUDA_VISIBLE_DEVICES``, and while a load
#: starts CUDA in this process: CUDA reads the variable once, when it starts.
#: Taken after a backend's own load lock, never before it.
cuda_lock = threading.Lock()
# Guards _outer and the variable while visible_only() changes them, so that
# visible_gpu_ids() reads the two together.
_env_lock = threading.Lock()
# CUDA_VISIBLE_DEVICES as it was before visible_only() narrowed it, as a
# one-tuple; None when it is not narrowed.
_outer: tuple[str | None] | None = None


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
    """``(free_gib, total_gib)`` of the current device, or ``None`` if unreadable.

    Also ``None`` while an engine is being built on narrowed cards: asking
    torch then would start CUDA in this process on those cards only.
    """
    if not cuda_lock.acquire(blocking=False):
        logger.info("[residency] an engine is being built, so torch is not "
                    "asked for the card size now.")
        return None
    try:
        torch = _torch()
        if torch is None:
            return None
        try:
            free, total = torch.cuda.mem_get_info()
        except Exception as exc:  # noqa: BLE001
            logger.debug("mem_get_info failed (%s: %s)", type(exc).__name__, exc)
            return None
        return free / _BYTES_PER_GIB, total / _BYTES_PER_GIB
    finally:
        cuda_lock.release()


def visible_gpu_ids() -> list[str] | None:
    """The cards ``CUDA_VISIBLE_DEVICES`` leaves this process, as it names them.

    While a build has narrowed the variable (:func:`visible_only`), the
    value from before.

    Returns:
        The variable's entries in its order (indices or UUIDs), ``[]`` when
        it is set and empty, or ``None`` when it is unset.
    """
    with _env_lock:
        value = os.environ.get("CUDA_VISIBLE_DEVICES") if _outer is None else _outer[0]
    if value is None:
        return None
    return [entry.strip() for entry in value.split(",") if entry.strip()]


@contextlib.contextmanager
def visible_only(ids: list[str]):
    """Make ``ids`` the only cards a process started inside can see.

    An engine's processes read ``CUDA_VISIBLE_DEVICES`` as they start, so
    the variable is set for the build and put back after it, also when the
    build fails. The variable is the whole process's: it is changed under
    :data:`cuda_lock`, and :func:`visible_gpu_ids` reads the value from
    before meanwhile. Without ``ids`` nothing is changed and no lock taken.
    """
    global _outer
    if not ids:
        yield
        return
    with cuda_lock:
        with _env_lock:
            before = os.environ.get("CUDA_VISIBLE_DEVICES")
            _outer = (before,)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        try:
            yield
        finally:
            with _env_lock:
                if before is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = before
                _outer = None


def bus_order() -> None:
    """Make CUDA count the cards in PCI bus order, as ``nvidia-smi`` does.

    CUDA's own order puts the fastest card first, so on a machine of mixed
    cards an index the plan takes from ``nvidia-smi`` could name another card
    in a ``device_map`` or in ``CUDA_VISIBLE_DEVICES``. Sets
    ``CUDA_DEVICE_ORDER`` unless it is set, for this process and the engine
    processes it starts. CUDA reads it once, when it starts: a warning says
    so when it has started in this process already.
    """
    if "CUDA_DEVICE_ORDER" in os.environ:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Looked up, not imported: importing torch here would cost an API-only run.
    torch = sys.modules.get("torch")
    try:
        started = torch is not None and torch.cuda.is_initialized()
    except Exception:  # noqa: BLE001 — a broken CUDA install must not raise here
        started = False
    if started:
        logger.warning(
            "[residency] CUDA started in this process before CUDA_DEVICE_ORDER "
            "was set, so it counts the cards fastest first, not as nvidia-smi "
            "and the plan do: a transformers model's device_map may name another "
            "card than the plan's. Set CUDA_DEVICE_ORDER=PCI_BUS_ID before CUDA "
            "starts.")


def detect_gpus(visible: bool = False) -> tuple[str | None, int]:
    """Which GPU this machine has, and how many, via ``nvidia-smi``.

    Args:
        visible: Count the cards ``CUDA_VISIBLE_DEVICES`` leaves this
            process, when it is set, instead of the machine's. The planner
            counts those; the bill counts the machine's, since the pod is
            rented whole.

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
    ids = visible_gpu_ids() if visible else None
    return (names[0] if names else None), len(names if ids is None else ids)


def detect_gpu_memory_gib() -> float | None:
    """The size of the smallest card this process can see, in GiB.

    ``nvidia-smi`` answers without torch or a usable CUDA, so the size is
    known wherever a card is, not only where the full stack is.

    Returns:
        The smallest total over the cards :func:`gpu_cards` finds visible:
        a plan has one card size. ``None`` when ``nvidia-smi`` is absent or
        unparseable, or no card is visible.
    """
    cards = gpu_cards(visible=True)
    return min(card.total_gib for card in cards) if cards else None


@dataclass(frozen=True)
class Card:
    """One card, as ``nvidia-smi`` reports it."""

    #: The card's index as ``nvidia-smi`` numbers it, in PCI bus order.
    index: int
    uuid: str
    used_gib: float
    total_gib: float
    name: str = ""


@dataclass(frozen=True)
class CardUse(Card):
    """What one card holds, as ``nvidia-smi`` reports it."""

    #: ``(pid, name, GiB)`` of each compute process on the card. The GiB is
    #: ``None`` where ``nvidia-smi`` cannot say.
    processes: tuple[tuple[int, str, float | None], ...] = ()


def _query(query: str) -> list[str] | None:
    """The lines of ``nvidia-smi`` answering ``query``, in CSV without header
    or units, or ``None`` when it is absent or fails."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi %s failed (%s: %s)", query, type(exc).__name__, exc)
        return None
    return [line for line in out.splitlines() if line.strip()]


def _gib(mib: str) -> float | None:
    """A MiB figure of ``nvidia-smi`` in GiB, ``None`` for ``[N/A]``."""
    return None if mib.strip().startswith("[") else round(int(mib) / 1024, 1)


def gpu_cards(visible: bool = False) -> list[Card] | None:
    """Each card's index, UUID, name and memory, from one ``nvidia-smi`` query.

    Args:
        visible: Only the cards ``CUDA_VISIBLE_DEVICES`` leaves this process,
            in the variable's order, so that the k-th is the card the
            planner and the loader number k. An entry names a card by its
            index or by its UUID, or a prefix of it; as CUDA does, an entry
            that names no card ends the list.

    Returns:
        One entry per card, in ``nvidia-smi``'s order unless ``visible``, or
        ``None`` when ``nvidia-smi`` is absent, fails, lists no card or
        answers in a form not read here.
    """
    lines = _query("--query-gpu=index,uuid,memory.used,memory.total,name")
    if not lines:
        return None
    cards = []
    try:
        for line in lines:
            index, uuid, used, total, name = (
                part.strip() for part in line.split(",", 4))
            used_gib, total_gib = _gib(used), _gib(total)
            if used_gib is None or total_gib is None:
                return None
            cards.append(Card(int(index), uuid, used_gib, total_gib, name))
    except ValueError:
        return None
    ids = visible_gpu_ids() if visible else None
    if ids is None:
        return cards
    shown = []
    for entry in ids:
        by_uuid = entry.startswith("GPU-")
        card = next((card for card in cards if entry == str(card.index)
                     or (by_uuid and card.uuid.startswith(entry))), None)
        if card is None:
            break
        shown.append(card)
    return shown


def card_use() -> list[CardUse] | None:
    """Each card's memory in use and the compute processes holding it.

    Two ``nvidia-smi`` queries: the cards (:func:`gpu_cards`), and the
    compute processes.

    Returns:
        One entry per card, in ``nvidia-smi``'s order, or ``None`` when
        ``nvidia-smi`` is absent, fails, lists no card or answers in a form
        not read here.
    """
    cards = gpu_cards()
    apps = _query("--query-compute-apps=pid,used_memory,gpu_uuid,process_name")
    if not cards or apps is None:
        return None
    held: dict[str, list] = {}
    try:
        for line in apps:
            pid, used, uuid, name = (part.strip() for part in line.split(",", 3))
            held.setdefault(uuid, []).append((int(pid), name, _gib(used)))
    except ValueError:
        return None
    return [CardUse(card.index, card.uuid, card.used_gib, card.total_gib, card.name,
                    tuple(held.get(card.uuid, ()))) for card in cards]


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
