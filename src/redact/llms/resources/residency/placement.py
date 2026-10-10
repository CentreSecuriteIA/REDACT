"""Place a run's local models on the machine's cards."""

import logging
import math
import re

from redact.llms.resources import measure

from .footprint import _CARD_CEILING, _SPANNING_DEVICE_MAPS, ModelFootprint, footprint
from .lifecycle import loaded_engines
from .plan import ResidencyPlan

logger = logging.getLogger(__name__)

#: Size used in the placement arithmetic for a transformers model whose
#: footprint is neither declared nor estimable. The plan reports it as unknown.
_UNKNOWN_GB = 0.0

#: Slack for comparing GiB sums.
_EPS = 1e-6


class _Cards:
    """Memory per card while one group of models is placed."""

    def __init__(self, count: int, per_gpu_gb: float | None):
        self.per = per_gpu_gb
        #: GiB per card not held by transformers models, whose claims are
        #: fixed. ``None`` when the card size is unknown.
        self.free = [per_gpu_gb] * count if per_gpu_gb else None
        #: The vLLM engines on each card.
        self.shared: list[list[ModelFootprint]] = [[] for _ in self.free or ()]
        #: Cards a tensor-parallel engine holds whole.
        self.whole: set[int] = set()

    def _room(self, card: int, ceiling: bool = False) -> float:
        """GiB a new claim can take on a card."""
        if card in self.whole:
            # Negative, so a claim of 0 GiB (unknown size) cannot join either.
            return -math.inf
        room = self.free[card] - sum(fp.padded_gb or 0.0 for fp in self.shared[card])
        if ceiling or self.shared[card]:
            room -= (1 - _CARD_CEILING) * self.per
        return room

    def place(self, fp: ModelFootprint, force: bool = False) -> bool:
        """Claim the model's cards and record them on the footprint.

        Returns ``False``, claiming nothing, when the model does not fit
        beside what is already placed. ``force`` places it regardless, for a
        model that opens a group.
        """
        devices, reserve = self._claim(fp)
        vllm = fp.setup == "vllm"
        fits = self.free is None or all(
            d < len(self.free)
            # vLLM grants sharing a card stay under the ceiling together.
            and self._room(d, vllm) + _EPS >= reserve[d]
            # A tensor-parallel engine takes untouched cards only.
            and not (vllm and fp.min_gpus > 1
                     and (self.shared[d] or self.free[d] < self.per))
            for d in devices
        )
        if not (fits or force):
            return False
        fp.devices = devices
        if self.free is not None and devices:
            if not vllm:
                fp.reserved_gb = max(reserve.values())
            for d in devices:
                if d >= len(self.free):
                    continue
                if vllm:
                    self.shared[d].append(fp)
                else:
                    self.free[d] -= reserve[d]
                if vllm and fp.min_gpus > 1:
                    self.whole.add(d)
        return True

    def settle(self) -> None:
        """Grant each vLLM engine its padded need plus an equal share of
        the spare.

        Per card: ``pool = 0.9 x card - fixed claims`` and
        ``grant = padded need + (pool - sum of padded needs) / engines``,
        never below 0. A grant below its padded need makes the model oversized.
        """
        for card, engines in enumerate(self.shared):
            if not engines:
                continue
            pool = max(0.0, self.free[card] - (1 - _CARD_CEILING) * self.per)
            spare = pool - sum(fp.padded_gb or 0.0 for fp in engines)
            for fp in engines:
                padded = fp.padded_gb or 0.0
                fp.reserved_gb = max(0.0, padded + spare / len(engines))
                # Rounded down, to keep a card's fractions under the ceiling.
                fraction = math.floor(fp.reserved_gb / self.per * 1e4 + 1e-6) / 1e4
                if fp.reserved_gb + _EPS >= padded:
                    # Rounded up to cover the need, even where that takes
                    # the card's fractions a rounding step over the ceiling.
                    fraction = max(
                        fraction, math.ceil(padded / self.per * 1e4 - 1e-6) / 1e4)
                # vLLM rejects a fraction that is not positive.
                fp.planned_utilization = fraction if fraction > 0 else None

    def _claim(self, fp: ModelFootprint) -> tuple[tuple[int, ...], dict[int, float]]:
        """The cards a model would take and the GiB it claims on each."""
        if fp.setup == "vllm":
            # No device is chosen at load time, so vLLM takes the first cards.
            devices = tuple(range(fp.min_gpus))
            return devices, dict.fromkeys(devices, fp.padded_gb or 0.0)

        need = fp.gb or _UNKNOWN_GB
        device_map = str(fp.parts.get("device_map", "auto"))
        if device_map not in _SPANNING_DEVICE_MAPS:
            if device_map == "cpu":
                return (), {}
            index = re.search(r"(\d+)$", device_map)
            card = int(index.group(1)) if index else 0
            return (card,), {card: need}
        if self.free is None:
            return tuple(range(fp.min_gpus)), {}
        # A spanning device_map may split the model, so it is counted
        # against the cards' combined free memory.
        reserve: dict[int, float] = {}
        left = need
        for card in range(len(self.free)):
            take = min(max(self._room(card), 0.0), left)
            if take > _EPS:
                reserve[card] = take
                left -= take
        if left > _EPS or not reserve:
            # More than the cards hold: charge the excess to the last one.
            last = max(reserve, default=0)
            reserve[last] = reserve.get(last, 0.0) + left
        return tuple(reserve), reserve


def plan_residency(
    models: list[str],
    backend_types: dict | None = None,
    *,
    gpus: int | None = None,
    gib_per_gpu: float | None = None,
) -> ResidencyPlan:
    """Place a run's local models on the machine's cards.

    Models are placed in the order given, each on the first cards its
    runtime would use. One that does not fit beside those already placed
    starts a new group, and more than one group means the run does not fit
    (:attr:`ResidencyPlan.fits`).

    Args:
        models: Model names. Unregistered and non-local ones are ignored.
        backend_types: Optional per-model setup override, as passed to
            ``ModelClient.create``.
        gpus: Plan for this many cards instead of the detected ones.
        gib_per_gpu: Plan for cards of this size instead of the detected
            one. Give both to plan for another machine, e.g. a rented pod.

    Returns:
        A :class:`ResidencyPlan`. A plan that does not fit is returned, not
        raised.
    """
    backend_types = backend_types or {}
    footprints = []
    for m in dict.fromkeys(models):  # de-dup, keep order
        try:
            fp = footprint(m, backend_types.get(m))
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping %s in residency plan (%s)", m, exc)
            continue
        if fp is not None:
            footprints.append(fp)

    given = gpus is not None or gib_per_gpu is not None
    gpu_name, gpus_available = (
        (None, gpus) if gpus is not None else measure.detect_gpus())
    plan = ResidencyPlan(
        gpus_available=gpus_available, gpu_name=gpu_name, machine_given=given,
        per_gpu_gb=gib_per_gpu,
    )
    # Nothing local: return before probing, so an API-only run never
    # imports torch.
    if not footprints:
        return plan

    planned = {fp.engine for fp in footprints}
    resident = sorted(
        f"{key[0]}[{setup}]" for setup, keys in loaded_engines().items()
        for key in keys if (setup, *key) not in planned)
    if resident and not given:
        logger.warning(
            "[residency] already loaded and not in this plan, so the plan "
            "hands their memory out again: %s", ", ".join(resident))

    if gib_per_gpu is None:
        # nvidia-smi first: asking torch starts CUDA in this process, before
        # vLLM launches its engine subprocess.
        plan.per_gpu_gb = measure.detect_gpu_memory_gib()
        if plan.per_gpu_gb is None:
            ft = measure.free_total_gib()
            plan.per_gpu_gb = ft[1] if ft else None
    per_gpu_gb = plan.per_gpu_gb

    groups: list[list[ModelFootprint]] = []
    current: list[ModelFootprint] = []
    # Per engine: the model that placed it, and the group it is in.
    placed: dict[tuple, tuple[ModelFootprint, list[ModelFootprint]]] = {}
    twins: list[tuple[ModelFootprint, ModelFootprint]] = []
    # CUDA without nvidia-smi reports a size and no count: that is one card.
    count = gpus_available if gpus is not None else max(gpus_available, 1)
    if per_gpu_gb and gpus is None:
        plan.gpus_available = count
    cards = _Cards(count, per_gpu_gb)
    for fp in footprints:
        # Two models on the same engine share one load, in the engine's group.
        if fp.engine in placed:
            twin, group = placed[fp.engine]
            twins.append((fp, twin))
            group.append(fp)
            continue
        if not cards.place(fp):
            if current:
                cards.settle()
                groups.append(current)
                current = []
                cards = _Cards(count, per_gpu_gb)
            cards.place(fp, force=True)
        current.append(fp)
        placed[fp.engine] = (fp, current)
    cards.settle()
    groups.append(current)
    for fp, twin in twins:
        fp.devices, fp.reserved_gb = twin.devices, twin.reserved_gb
        fp.planned_utilization = twin.planned_utilization

    plan.groups = groups
    plan.gpus_required = max(
        (max(fp.devices) + 1 for fp in footprints if fp.devices), default=0
    )
    if per_gpu_gb:
        plan.oversized = [
            fp for fp in footprints
            if _exceeds_its_cards(fp, per_gpu_gb, gpus_available)
        ]
    return plan


def _exceeds_its_cards(fp: ModelFootprint, per_gpu_gb: float, gpus: int) -> bool:
    """Whether a model needs more than its grant, or than its cards hold."""
    if fp.setup == "vllm":
        if fp.reserved_gb is None:
            return False
        return fp.reserved_gb <= 0 or (
            fp.known and fp.padded_gb > fp.reserved_gb + _EPS)
    # No devices: the model runs on the CPU and takes no card.
    if not fp.known or not fp.devices:
        return False
    spans = (fp.setup == "introspect"
             and str(fp.parts.get("device_map", "auto")) in _SPANNING_DEVICE_MAPS)
    return fp.gb > per_gpu_gb * (max(gpus, 1) if spans else 1)
