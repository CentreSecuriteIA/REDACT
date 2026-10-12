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
    share it loads with holds for as long as it is resident. It also stays
    on the cards it loaded on: a stage places the engines it inherits first,
    on their cards, and its other models around them. Of its copies it
    keeps those on cards every one of its stages leaves it, so it gives up
    the ones a more crowded stage has no card for.

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
    # The cards each resident vLLM engine loaded on, kept until its release.
    held: dict[tuple, tuple[int, ...]] = {}
    counted = {engine_of[name] for name in copies or () if name in engine_of}
    phases = []
    for i, stage in enumerate(stages):
        resident = [name for name, engine in engine_of.items()
                    if span[engine][0] <= i <= span[engine][1]]
        engines = {engine_of[name] for name in resident}
        held = {engine: cards for engine, cards in held.items() if engine in engines}
        # The same models again are the same engines, resident since: no
        # lifetime has a gap, so the plan made for them still holds.
        key = tuple(resident)
        if key not in plans:
            plans[key] = plan_residency(
                resident, backend_types, gpus=gpus, gib_per_gpu=gib_per_gpu,
                copies=copies, pinned=held)
        for fp in plans[key].footprints:
            if fp.setup == "vllm" and fp.devices:
                # A copy on an idle card is decided stage by stage, so only
                # the cards the engine was placed on are kept for the next.
                placed = (fp.devices if fp.engine in counted
                          else fp.devices[:fp.min_gpus])
                held.setdefault(fp.engine, placed)
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
    # copy stays only where no stage plans another model. The cards an
    # engine was placed on are in every stage's set, being kept.
    common: dict[tuple, tuple[int, ...]] = {}
    for fp in footprints:
        if fp.setup == "vllm":
            so_far = common.get(fp.engine, fp.devices)
            common[fp.engine] = tuple(d for d in so_far if d in fp.devices)
    told = set()
    for fp in footprints:
        cards = common.get(fp.engine, fp.devices)
        if len(cards) == len(fp.devices):
            continue
        fp.devices, fp.copies = cards, len(cards) // fp.min_gpus
        if fp.engine not in told:
            told.add(fp.engine)
            logger.info(
                "[residency] %s loads once, with %d %s: on the cards it "
                "holds in every stage.",
                fp.model, fp.copies, "copy" if fp.copies == 1 else "copies")

    # A transformers model is not placed by the plan: its device_map placed
    # it, when it loaded. A later stage only guesses again.
    guessed: dict[tuple, tuple[int, ...]] = {}
    for fp in footprints:
        if fp.setup == "vllm":
            continue
        first = guessed.setdefault(fp.engine, fp.devices)
        if fp.devices != first and fp.engine not in told:
            told.add(fp.engine)
            logger.warning(
                "[residency] %s is planned on other cards in a later stage. A "
                "transformers model stays where its device_map put it when "
                "it loaded, so that stage's plan may hand out memory it holds.",
                fp.model)
    for plan in plans.values():
        plan.gpus_required = _gpus_required(plan.footprints)
    return phases
