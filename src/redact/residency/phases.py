"""Residency per stage: which local models are loaded while each stage runs."""

import logging
from dataclasses import dataclass

from .footprint import _engine_id, _local_setup
from .placement import _gpus_required, plan_residency
from .plan import ResidencyPlan

logger = logging.getLogger(__name__)


@dataclass
class Phase:
    """The local models resident while one stage runs, and their plan."""

    stage: str
    #: The stage's own models and those an earlier stage loaded that a later
    #: one still calls, in the order they are first needed.
    resident: list[str]
    plan: ResidencyPlan

    @property
    def loadable(self) -> list[str]:
        """The resident models that fit together: the plan's first group."""
        return [fp.model for fp in self.plan.groups[0]] if self.plan.groups else []


def plan_phases(
    stage_models: dict[str, list[str]],
    backend_types: dict | None = None,
    *,
    gpus: int | None = None,
    gib_per_gpu: float | None = None,
    copies: dict[str, int] | None = None,
) -> list[Phase]:
    """Plan residency stage by stage, in the order of ``stage_models``.

    An engine is resident from the first stage that calls it to the last, so
    it loads once. Each resident set is planned on its own, and a vLLM engine
    is granted the smallest share any of its stages plans for it, so the
    share it loads with holds for as long as it is resident. Likewise it
    holds only the cards every one of its stages plans for it, so it gives
    up the copies a more crowded stage has no card for.

    Args:
        stage_models: The models each stage calls. Unregistered and non-local
            ones are ignored.
        backend_types: Optional per-model setup override.
        gpus: Plan for this many cards instead of the detected ones.
        gib_per_gpu: Plan for cards of this size instead of the detected one.
        copies: Copies per model name, as for :func:`plan_residency`.

    Returns:
        One :class:`Phase` per stage. Stages with the same resident models
        share one plan object.
    """
    backend_types = backend_types or {}
    stages = list(stage_models)

    # Each engine's first and last stage.
    engine_of: dict[str, tuple] = {}
    span: dict[tuple, list[int]] = {}
    for i, stage in enumerate(stages):
        for name in stage_models[stage]:
            if name not in engine_of:
                try:
                    found = _local_setup(name, backend_types.get(name))
                except (KeyError, ValueError):
                    continue
                if found is None:
                    continue
                config, setup = found
                engine_of[name] = _engine_id(getattr(config, setup), setup)
            span.setdefault(engine_of[name], [i, i])[1] = i

    plans: dict[tuple, ResidencyPlan] = {}
    phases = []
    for i, stage in enumerate(stages):
        resident = [name for name, engine in engine_of.items()
                    if span[engine][0] <= i <= span[engine][1]]
        key = tuple(resident)
        if key not in plans:
            plans[key] = plan_residency(
                resident, backend_types, gpus=gpus, gib_per_gpu=gib_per_gpu,
                copies=copies)
        phases.append(Phase(stage, resident, plans[key]))

    # One share per engine for its whole lifetime: the smallest planned.
    smallest: dict[tuple, tuple[float, float | None]] = {}
    footprints = [fp for plan in plans.values() for fp in plan.footprints]
    for fp in footprints:
        if fp.planned_utilization is None:
            continue
        share = (fp.planned_utilization, fp.reserved_gb)
        if fp.engine not in smallest or share[0] < smallest[fp.engine][0]:
            smallest[fp.engine] = share
    lowered = set()
    for fp in footprints:
        if fp.planned_utilization is None:
            continue
        share, reserved = smallest[fp.engine]
        if share < fp.planned_utilization and fp.engine not in lowered:
            lowered.add(fp.engine)
            logger.info(
                "[residency] %s loads once, with gpu_memory_utilization=%.2f: "
                "its share in the stage where its card is most crowded.",
                fp.model, share)
        fp.planned_utilization, fp.reserved_gb = share, reserved

    # One set of cards per engine too: those it holds in every stage, so a
    # copy stays only where no stage plans another model.
    common: dict[tuple, tuple[int, ...]] = {}
    for fp in footprints:
        if fp.setup == "vllm":
            held = common.get(fp.engine, fp.devices)
            common[fp.engine] = tuple(d for d in held if d in fp.devices)
    told = set()
    for fp in footprints:
        cards = common.get(fp.engine, fp.devices)
        if len(cards) == len(fp.devices):
            continue
        # Too few shared cards for one copy: the stages disagree on where
        # the engine is, and no choice of cards here settles that.
        moved = not cards or len(cards) % fp.min_gpus != 0
        if fp.engine not in told:
            told.add(fp.engine)
            if moved:
                logger.warning(
                    "[residency] %s is planned on different cards in "
                    "different stages. It loads once and stays on the cards "
                    "it loaded on, so a later stage's plan may put another "
                    "model there.", fp.model)
            else:
                n = len(cards) // fp.min_gpus
                logger.info(
                    "[residency] %s loads once, with %d %s: on the cards it "
                    "holds in every stage.",
                    fp.model, n, "copy" if n == 1 else "copies")
        if not moved:
            fp.devices, fp.copies = cards, len(cards) // fp.min_gpus
    for plan in plans.values():
        plan.gpus_required = _gpus_required(plan.footprints)
    return phases
