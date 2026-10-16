"""Residency per stage: which local models are loaded while each stage runs."""

import logging
from dataclasses import dataclass

from .footprint import _engine_id, _local_setup
from .placement import _arrange, _exceeds_its_cards, _gpus_required, _Stage, _survey
from .plan import ResidencyPlan

logger = logging.getLogger(__name__)


@dataclass
class Phase:
    """The local models resident while one stage runs, and their plan."""

    stage: str
    #: The stage's own models, and those an earlier stage loaded that a
    #: later one still calls unless they are unloaded for this one, in the
    #: order they are first needed.
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

    An engine is resident from the first stage that calls it to the last,
    which loads it once, unless a stage in between has no room for it
    (below). Each resident set is planned on its own, and a vLLM engine
    is granted the smallest share any of its stages plans for it, so the
    share it loads with holds for as long as it is resident. It also stays
    on the cards it loaded on: a stage places the engines it inherits first,
    on their cards, and its other models around them. When a stage does not
    fit like that, other cards are looked for as :func:`plan_residency`
    does, for the run as a whole: an engine has the same cards in all its
    stages, and they are taken only when every stage fits on them. Of its
    copies an engine keeps those on cards every one of its stages has room
    for, so it gives up the ones a more crowded stage has no room for.

    A stage that still does not fit may hold engines it does not call. Each
    is unloaded for the stages between the two that call it, and loaded
    again for the second, if every stage of the run then fits. Otherwise
    none is. Of those releases, each one the run fits without is given back,
    the largest engine's first, so none is left that could be. Every time
    an engine is loaded is a lifetime of its own: what is said above of an
    engine's stages, its share, its cards and its copies holds within one.

    Args:
        stage_models: The models each stage calls. Unregistered and non-local
            ones are ignored.
        backend_types: Optional per-model setup override.
        gpus: Plan for this many cards instead of the detected ones.
        gib_per_gpu: Plan for cards of this size instead of the detected one.
        copies: Copies per model name, as for :func:`plan_residency`.

    Returns:
        One :class:`Phase` per stage. Stages with the same resident models
        in the same lifetimes share one plan object.
    """
    backend_types = backend_types or {}
    stages = list(stage_models)

    # The stages that call each engine.
    engine_of: dict[str, tuple] = {}
    calls: dict[tuple, list[int]] = {}
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
            calls.setdefault(engine_of[name], []).append(i)

    def place(released, survey, quiet=False):
        """Plan every stage, with each engine unloaded over its gaps in
        ``released``.

        Args:
            released: Gaps, each as an engine, the stage that calls it
                before the gap and the one that calls it after.
            survey: Makes a stage's :class:`_Stage` from the stage's index,
                its resident models and the lifetimes of their engines.

        Returns:
            The phases, and the :class:`_Stage` of each.
        """
        spans = {
            engine: _lifetimes(at, [gap[1:] for gap in released if gap[0] == engine])
            for engine, at in calls.items()}
        surveyed: dict[tuple, _Stage] = {}
        phases, staged = [], []
        for i, stage in enumerate(stages):
            resident, lifetimes = [], {}
            for name, engine in engine_of.items():
                first = next((a for a, b in spans[engine] if a <= i <= b), None)
                if first is None:
                    continue
                resident.append(name)
                if len(spans[engine]) > 1:
                    # Told apart by the stage it is loaded for.
                    lifetimes[engine] = (*engine, stages[first])
            # The same models again in the same lifetimes are resident
            # since, so one plan holds for every stage they share.
            key = (tuple(resident), tuple(lifetimes.values()))
            if key not in surveyed:
                surveyed[key] = survey(i, resident, lifetimes)
            phases.append(Phase(stage, resident, surveyed[key].plan))
            staged.append(surveyed[key])
        _arrange([s for s in surveyed.values() if s.footprints], quiet=quiet)
        return phases, staged

    phases, staged = place((), lambda i, resident, lifetimes: _survey(
        resident, backend_types, gpus, gib_per_gpu, copies))

    # A stage that does not fit may hold engines it does not call, kept for
    # a later stage. Each is unloaded over that gap if every stage then
    # fits. Of those gaps, the ones the run fits without are given back,
    # the largest engine's first: its load is the longest to repeat.
    unfit = [i for i, phase in enumerate(phases) if not phase.plan.fits]
    gaps = [(engine, before, after)
            for engine, at in calls.items() for before, after in zip(at, at[1:])
            if any(before < i < after for i in unfit)]
    once = staged       # the stages as surveyed, every engine loaded once

    def without(released, quiet=True):
        """The run planned with ``released`` unloaded, from the models as
        sized already: ``None`` unless every stage then fits."""
        found = place(released, lambda i, resident, lifetimes: once[i].only(
            resident, lifetimes), quiet)
        return found if all(phase.plan.fits for phase in found[0]) else None

    if gaps and without(gaps):
        size: dict[tuple, float] = {}
        for stage in once:
            for fp in stage.footprints:
                size.setdefault(fp.engine, fp.gb or 0.0)
        released = list(gaps)
        for gap in sorted(gaps, key=lambda g: -size[g[0]]):
            rest = [other for other in released if other != gap]
            if without(rest):
                released = rest
        # Planned once more, to log the cards if a search chose them.
        phases, staged = without(released, quiet=False)
        for engine, before, after in sorted(released, key=lambda g: g[1]):
            logger.info(
                "[residency] %s is unloaded for stage %s and loaded again for "
                "stage %s, so that %s fits. This costs one more load.",
                next(name for name, e in engine_of.items() if e == engine),
                stages[before + 1], stages[after],
                stages[next(i for i in unfit if before < i < after)])

    placed = list({id(stage): stage for stage in staged}.values())
    plans = [stage.plan for stage in placed]
    # Each footprint under what its engine has one share and one set of
    # cards for: the engine, or a lifetime of it.
    footprints = [(stage.lifetime(fp), fp)
                  for stage in placed for fp in stage.plan.footprints]

    def loads(key, fp):
        return "loads once" if key == fp.engine else f"loads for stage {key[-1]}"

    # One share per engine for a whole lifetime: the smallest planned.
    smallest: dict[tuple, tuple[float, float | None]] = {}
    for key, fp in footprints:
        if fp.planned_utilization is None:
            continue
        share = (fp.planned_utilization, fp.reserved_gb)
        if key not in smallest or share[0] < smallest[key][0]:
            smallest[key] = share
    lowered = set()
    for key, fp in footprints:
        if fp.planned_utilization is None:
            continue
        share, reserved = smallest[key]
        if share < fp.planned_utilization and key not in lowered:
            lowered.add(key)
            logger.info(
                "[residency] %s %s, with gpu_memory_utilization=%.2f: "
                "its share in the stage where its card is most crowded.",
                fp.model, loads(key, fp), share)
        fp.planned_utilization, fp.reserved_gb = share, reserved
    # A share a more crowded stage lowered may not cover the need on an
    # engine's smallest card.
    for stage in placed:
        stage.plan.oversized += [
            fp for fp in stage.plan.footprints
            if fp.setup == "vllm" and fp not in stage.plan.oversized
            and _exceeds_its_cards(fp, stage.plan.gib_per_card or [])]

    # One set of cards per engine too: placement gave its copies only the
    # cards every stage of the lifetime has room for.
    cut = set().union(*(stage.cut for stage in placed))
    told = set()
    for key, fp in footprints:
        if key in cut and key not in told:
            told.add(key)
            logger.info(
                "[residency] %s %s, with %d %s: on the cards it "
                "holds in every stage.",
                fp.model, loads(key, fp), fp.copies,
                "copy" if fp.copies == 1 else "copies")

    # A transformers model is not placed by the plan: its device_map placed
    # it, when it loaded. A later stage only guesses again.
    guessed: dict[tuple, tuple[int, ...]] = {}
    for key, fp in footprints:
        if fp.setup == "vllm":
            continue
        first = guessed.setdefault(key, fp.devices)
        if fp.devices != first and key not in told:
            told.add(key)
            logger.warning(
                "[residency] %s is planned on other cards in a later stage. A "
                "transformers model stays where its device_map put it when "
                "it loaded, so that stage's plan may hand out memory it holds.",
                fp.model)
    for plan in plans:
        plan.gpus_required = _gpus_required(plan.footprints)
    return phases


def _lifetimes(
    calls: list[int], released: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """The first and last stage of each time an engine is loaded.

    Args:
        calls: The stages that call the engine, in order.
        released: The gaps it is unloaded over, each as the call before the
            gap and the call after.
    """
    spans, first = [], calls[0]
    for before, after in zip(calls, calls[1:]):
        if (before, after) in released:
            spans.append((first, before))
            first = after
    return [*spans, (first, calls[-1])]
