"""One run of a recipe: serving, residency, telemetry, the stages, a record each.

Each stage writes ``{stage}.run.json`` under ``{data_dir}/Datasets/``: the
resolved params, models, how the stage's models were served, row counts,
spec version, timestamp and source files.

Usage::

    from redact import run_pipeline
    summary = run_pipeline("configs/runs/eval_example.json")
"""

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from redact import paths, telemetry
from redact.dataset import count_rows, discover_categories, iter_rows
from redact.llms.prompting import report_prompt_sources
from redact.recipe import (
    _DEFAULT_MODELS,
    _role_names,
    load_params,
    load_recipe,
    serving_of,
)

logger = logging.getLogger(__name__)


def record_path(stage: str, data_dir: str | Path | None = None) -> Path:
    """Path of a stage's run record: ``{data_dir}/Datasets/{stage}.run.json``."""
    return paths.datasets(data_dir) / f"{stage}.run.json"


def write_record(
    stage: str,
    *,
    data_dir: str | Path | None = None,
    params: dict | None = None,
    models: dict | None = None,
    counts: dict | None = None,
    sources: list | None = None,
    spec_version: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a stage's run record and return its path."""
    record = {
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params or {},
        "models": models or {},
        "counts": counts or {},
        "sources": sources or [],
    }
    if spec_version:
        record["spec_version"] = spec_version
    if extra:
        record.update(extra)

    path = record_path(stage, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
    return path


def read_record(stage: str, data_dir: str | Path | None = None) -> dict | None:
    """Read a stage's run record, or None if it doesn't exist."""
    path = record_path(stage, data_dir)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# Former names of the three record functions, kept for one release.
manifest_path = record_path
write_manifest = write_record
read_manifest = read_record


def _counts(df) -> dict:
    """Row counts for a stage's returned DataFrame."""
    if df is None:
        return {}
    out = {"rows": int(len(df))}
    if "accepted" in getattr(df, "columns", []):
        out["accepted"] = int(df["accepted"].sum())
    return out


def file_counts(
    files: Iterable[str | Path], keep: Mapping[str, Any] | None = None,
) -> dict:
    """Row counts of a stage's CSVs, read in chunks.

    What :func:`_counts` gives for the rows of ``files`` in one frame, without
    the frame. A file that does not exist has no rows.

    Args:
        keep: Count only the rows that hold these values in these columns.
            A file without one of the columns is not limited by it.
    """
    keep = keep or {}
    rows = accepted = 0
    verdicts = False        # whether any file has an ``accepted`` column
    for path in map(Path, files):
        if not path.exists():
            continue
        header = set(pd.read_csv(path, nrows=0).columns)
        columns = [c for c in dict.fromkeys(("accepted", *keep)) if c in header]
        verdicts |= "accepted" in header
        if not columns:
            rows += count_rows(path)
            continue
        for row in iter_rows(path, columns=columns):
            if all(row[c] == value for c, value in keep.items() if c in row):
                rows += 1
                accepted += row.get("accepted") is True
    return {"rows": rows, "accepted": accepted} if verdicts else {"rows": rows}


def _stage_files(
    stage: str, data_dir: str | Path | None, dataset_type: str, params: dict,
) -> tuple[list[str | Path], dict]:
    """The CSVs that hold a stage's rows, and what limits the rows to count.

    For :func:`file_counts`: together they are the rows the stage returns
    when it reads them back. ``params`` are the recipe's for the stage; a
    path, a paraphrase target or a style left out is the stage's default.
    """
    if stage == "inputs":
        seeded = dataset_type == "training"
        tree = (paths.constitution_inputs_dir if seeded else paths.datasets)(data_dir)
        files = [paths.category_csv(c, root=tree) for c in discover_categories(tree)]
        keep = {"accepted": True}
        # Seeded inputs of another style may share the tree.
        if seeded and params.get("style", "long"):
            keep["template_style"] = params.get("style", "long")
        return files, keep
    if stage == "paraphrase":
        target = params.get("target", "both")
        default = {"inputs": paths.paraphrases_inputs_csv,
                   "outputs": paths.paraphrases_outputs_csv}
        return [params.get(f"{name}_path") or default[name](data_dir)
                for name in default if target in (name, "both")], {}
    default = {
        "outputs": paths.output_responses_csv,
        "jailbreaks": paths.jailbreaks_csv,
        "build": paths.complete_dataset_csv,
    }[stage]
    return [params.get("output_path") or default(data_dir)], {}


def _plan_phases(
    models: dict,
    stages: list[str],
    verbose: bool = True,
    augmentations: dict | None = None,
    copies: dict | None = None,
) -> list:
    """Plan each stage's local models, report the plans and apply the grants.

    Runs before the stage loop, so a stage that does not fit says so before
    any stage spends its budget. Nothing here aborts the run: a plan that
    does not fit and a planning error are only logged. ``copies`` is the
    recipe's: a number of copies per model name.

    Returns:
        One :class:`~redact.residency.Phase` per stage, or ``[]`` when no
        stage calls a local model or planning failed.
    """
    from . import residency

    try:
        aug = augmentations or {}
        phases = residency.plan_phases(
            {stage: _stage_models(models, [stage], aug) for stage in stages},
            copies=copies or {})
        if not any(phase.resident for phase in phases):
            return []
        # One report per plan, naming the stages it covers.
        stages_of: dict[int, list[str]] = {}
        for phase in phases:
            if phase.resident:
                stages_of.setdefault(id(phase.plan), []).append(phase.stage)
        reported = set()
        for i, phase in enumerate(phases):
            if phase.resident and id(phase.plan) not in reported:
                reported.add(id(phase.plan))
                residency.report(
                    phase.plan, verbose=verbose, stages=stages_of[id(phase.plan)])
            # The first replaces what an earlier run left. An engine loaded
            # a second time has other values in a later plan, which it gets
            # when that stage is entered (_enter_phase).
            residency.apply_plan(phase.plan, replace=(i == 0))
    except Exception as exc:  # noqa: BLE001 — a planning bug must not abort a run
        logger.error("[residency] planning failed (%s: %s); the run continues "
                     "without a plan.", type(exc).__name__, exc)
        return []
    return phases


def _enter_phase(phases: list, i: int, warm: set[str], verbose: bool = True) -> None:
    """Make stage ``i``'s local models the resident ones.

    Unloads the engines the stage's plan leaves out, hands the loader the
    plan's shares and cards, then starts loading this stage's models in the
    background, and the next stage's too when they fit beside them without
    an unload. Nothing here aborts the run.

    Args:
        phases: The run's phases, as :func:`_plan_phases` returns them.
        i: Index of the stage about to run.
        warm: Models handed to the preloader and still resident; updated
            in place. A running preload skips a model that leaves it.
    """
    if not phases:
        return
    from . import residency

    try:
        phase = phases[i]
        warm &= set(phase.resident)
        dropped = set(phases[i - 1].resident) - set(phase.resident) if i else ()
        if dropped:
            if verbose:
                logger.info("[residency] stage %s: unloading %s",
                            phase.stage, ", ".join(sorted(dropped)))
            residency.unload_local(keep=phase.resident)
        # What loads from here on loads under this stage's plan.
        residency.apply_plan(phase.plan)
        ahead = list(phase.loadable)
        if i + 1 < len(phases):
            following = phases[i + 1]
            if (following.plan.co_resident
                    and set(phase.resident) <= set(following.resident)):
                ahead += following.loadable
                # An engine both stages hold has the same share and cards
                # in both plans.
                residency.apply_plan(following.plan, replace=False)
        todo = [m for m in dict.fromkeys(ahead) if m not in warm]
        if todo:
            warm |= set(todo)
            residency.preload(todo, verbose=verbose, wanted=warm)
    except Exception as exc:  # noqa: BLE001 — residency must not abort a run
        logger.error("[residency] stage %d: could not switch models (%s: %s); "
                     "the run continues.", i + 1, type(exc).__name__, exc)


def _stage_models(models: dict, stages: list[str], augmentations: dict) -> list[str]:
    """The models the stages will call, in the order the stages first need them.

    Each role resolves the way its stage resolves it, so the plan covers the
    models that will actually load.
    """
    wanted: list[str] = []
    for stage in stages:
        for role in STAGE_ROLES.get(stage, ()):
            try:
                wanted.extend(_resolve_role(models, role, augmentations))
            except (KeyError, ValueError):
                # No model registered for the role; the stage reports it.
                continue
    return list(dict.fromkeys(wanted))


def _resolve_role(models: dict, role: str, augmentations: dict) -> list[str]:
    """The models a recipe role stands for, resolved as its stage resolves it.

    Raises:
        KeyError, ValueError: No model is registered for the role.
    """
    from .content_moderation.paraphrase import paraphrase_pool
    from .llms.model_config import default_model_for_role

    name = models.get(role)
    if role == "paraphraser":
        # One model, or every paraphraser for the "distribution" pool.
        return paraphrase_pool(name)
    if role == "check":
        # The stages check with the generation model unless told otherwise.
        return [name] if name else _resolve_role(models, "gen", augmentations)
    if role == "translation" and not _uses_translation(augmentations):
        return []
    return [name or default_model_for_role(_ROLE_TO_REGISTRY_ROLE.get(role, role))]


def _uses_translation(augmentations: dict) -> bool:
    """Whether the jailbreak stage can sample a translation technique."""
    return (
        augmentations.get("include_translation", True)
        and augmentations.get("include_obfuscation", True)
        and not augmentations.get("pure_only", False)
    )


#: The model roles each pipeline stage calls, for planning and preloading
#: only the models a run will use.
STAGE_ROLES = {
    "constitution": ("constitution",),
    "inputs": ("gen", "check"),
    "outputs": ("gen", "check"),
    "paraphrase": ("paraphraser", "paraphrase_check"),
    "jailbreaks": ("gen", "translation"),
    "build": (),
}

#: Recipe role name -> registry role name, where they differ. The recipe speaks
#: in stage terms ("gen", "check"); the registry speaks in capability terms.
_ROLE_TO_REGISTRY_ROLE = {
    "gen": "uncensored_gen",
    "constitution": "constitution_gen",
    "paraphrase_check": "uncensored_gen",
}


def _served(
    models: dict, stages: list[str], augmentations: dict, serving: dict, phases: list,
) -> dict:
    """How the run serves the models its stages call, per role.

    For the summary and the run records. Called once the run's serving is
    set, so each setup is the one the run's clients get.

    Args:
        serving: The recipe's, as :func:`~redact.recipe.serving_of` returns it.
        phases: The run's plan, as :func:`_plan_phases` returns it.

    Returns:
        Per role, ``{"name", "backend_type"}`` with ``copies`` for a model
        served by vLLM (the plan's count, else the recipe's, else ``None``)
        or ``workers`` for a hosted one. A pool of several models is a list
        of these, and a role the run calls no model for is ``None``.
    """
    from .llms.backends import resolve_setup
    from .llms.model_config import get_model_config

    planned = {fp.model: fp.copies for phase in phases for fp in phase.plan.footprints}
    served: dict = {}
    roles = dict.fromkeys(
        role for stage in stages for role in STAGE_ROLES.get(stage, ()))
    for role in roles:
        try:
            names = _resolve_role(models, role, augmentations)
        except (KeyError, ValueError):
            names = []      # no model registered for the role; the stage reports it
        entries = []
        for name in names:
            entry = {"name": name}
            entries.append(entry)
            try:
                config = get_model_config(name)
                setup = resolve_setup(config)
            except (KeyError, TypeError, ValueError):
                continue    # not a registered model; the stage reports it
            entry["backend_type"] = setup
            given = serving.get(name, {})
            if setup == "vllm":
                entry["copies"] = planned.get(name, given.get("copies"))
            elif setup == "api":
                entry["workers"] = given.get(
                    "workers", config.api.recommended_max_workers)
        served[role] = entries[0] if len(entries) == 1 else entries or None
    return served


def run_pipeline(
    recipe: str | Path | dict,
    params: str | Path | dict | None = None,
    verbose: bool = True,
) -> dict:
    """Execute the stages named in a recipe, writing a run record per stage.

    ``eval`` recipes run the standalone path; ``training`` recipes run the
    constitution stage first and thread its DataFrame into ``generate_inputs``.
    No other stage reads its rows back: their counts are taken from the files.
    Returns a summary dict ``{"stages": {stage: {counts, manifest}}}``, with
    how each role's model was served under ``"serving"`` (:func:`_served`).
    Local models the run planned are released when it returns or raises, and
    the way the recipe serves its models holds only until then.
    """
    from redact import (
        build_dataset,
        generate_constitution,
        generate_inputs,
        generate_jailbreaks,
        generate_outputs,
        generate_paraphrases,
    )
    from redact.jailbreak import load_spec
    from redact.llms.backends import capabilities

    recipe = load_recipe(recipe)
    base_dir = recipe.get("_base_dir")
    if params is None:
        params = load_params(recipe.get("params_file"), base_dir=base_dir)
    else:
        params = load_params(params, base_dir=base_dir)

    data_dir = recipe.get("data_dir")
    dataset_type = recipe["dataset_type"]
    models = {**_DEFAULT_MODELS, **_role_names(recipe.get("models") or {})}
    serving = serving_of(recipe.get("models") or {})
    taxonomy = recipe.get("taxonomy", "content_moderation_categories")
    taxonomy_dir = recipe.get("taxonomy_dir")
    prompt_dir = recipe.get("prompt_dir")
    aug = dict(recipe.get("augmentations") or {})
    resume = recipe.get("resume", True)
    stages = recipe["stages"]

    def P(stage: str) -> dict:
        return dict(params.get(stage) or {})

    if verbose:
        logger.info("Run pipeline (%s) — stages: %s", dataset_type, stages)

    # Telemetry first, so the plan and preload below are themselves recorded.
    telemetry.install(data_dir=data_dir)
    if verbose:
        # Alongside the residency plan: one "here's your setup" block, before
        # anything runs. With prompts resolving per file against an overlay,
        # "which prompt ran?" stops being obvious, and an override that
        # silently matches nothing is invisible without this.
        report_prompt_sources(prompt_dir)

    summary: dict = {
        "dataset_type": dataset_type, "data_dir": str(data_dir), "stages": {}}
    constitution_df = None
    phases: list = []
    warm: set[str] = set()
    try:
        # Before the plan, which reads it as every client the run builds does.
        capabilities.set_run_serving(
            {name: {key: how[key] for key in ("backend_type", "workers") if key in how}
             for name, how in serving.items()})
        phases = _plan_phases(
            models, stages, verbose=verbose, augmentations=aug,
            copies={name: how["copies"] for name, how in serving.items()
                    if "copies" in how})
        if phases:
            summary["residency"] = {
                "fits": all(phase.plan.fits for phase in phases),
                "phases": [{"stage": phase.stage, "resident": phase.resident,
                            **phase.plan.as_dict()} for phase in phases],
            }
        served = summary["serving"] = _served(models, stages, aug, serving, phases)

        for i, stage in enumerate(stages):
            _enter_phase(phases, i, warm, verbose=verbose)
            with telemetry.stage(stage):
                if stage == "constitution":
                    constitution_df = generate_constitution(
                        taxonomy=taxonomy, taxonomy_dir=taxonomy_dir,
                        model=models["constitution"], data_dir=data_dir,
                        resume=resume, verbose=verbose, **P("constitution"),
                    )

                elif stage == "inputs":
                    kw = dict(
                        data_dir=data_dir, taxonomy=taxonomy, taxonomy_dir=taxonomy_dir,
                        prompt_dir=prompt_dir, model=models["gen"],
                        check_model=models["check"],
                        resume=resume, verbose=verbose, **P("inputs"),
                    )
                    if dataset_type == "training":
                        if constitution_df is None:
                            raise ValueError(
                                "training run: the 'constitution' stage must precede "
                                "'inputs' (so its entries can seed input generation)."
                            )
                        kw["constitution_df"] = constitution_df
                    generate_inputs(read_back=False, **kw)

                elif stage == "outputs":
                    generate_outputs(
                        data_dir=data_dir, model=models["gen"],
                        check_model=models["check"],
                        prompt_dir=prompt_dir, resume=resume, verbose=verbose,
                        read_back=False, **P("outputs"),
                    )

                elif stage == "paraphrase":
                    generate_paraphrases(
                        data_dir=data_dir, paraphraser=models["paraphraser"],
                        check_model=models["paraphrase_check"], prompt_dir=prompt_dir,
                        resume=resume, verbose=verbose, read_back=False,
                        **P("paraphrase"),
                    )

                elif stage == "jailbreaks":
                    generate_jailbreaks(
                        data_dir=data_dir, model=models["gen"],
                        translation_model=models["translation"],
                        resume=resume, verbose=verbose, read_back=False,
                        **aug, **P("jailbreaks"),
                    )

                elif stage == "build":
                    build_dataset(
                        data_dir=data_dir, mode=dataset_type, verbose=verbose,
                        read_back=False, **P("build"),
                    )

                else:  # pragma: no cover — load_recipe already validated
                    raise ValueError(f"Unknown stage {stage!r}")

                # The constitution is in memory, for the inputs stage. Every
                # other stage's rows are counted in its files.
                if stage == "constitution":
                    counts = _counts(constitution_df)
                else:
                    counts = file_counts(
                        *_stage_files(stage, data_dir, dataset_type, P(stage)))
                stage_params = {**(aug if stage == "jailbreaks" else {}), **P(stage)}
                spec = load_spec().get("version", "") if stage == "jailbreaks" else None
                mpath = write_record(
                    stage, data_dir=data_dir, params=stage_params, models=models,
                    counts=counts, spec_version=spec,
                    extra={"serving": {role: served[role]
                                       for role in STAGE_ROLES.get(stage, ())}},
                )
                summary["stages"][stage] = {"counts": counts, "manifest": str(mpath)}
                if verbose:
                    logger.info("[%s] %s -> manifest %s", stage, counts, mpath.name)
    finally:
        # The serving was this run's: a call made after it must not get it.
        capabilities.set_run_serving({})
        # The run's end is a transition too: release what the last stage held
        # and roll up, also after a failed stage. Nothing here may mask the
        # stage's own error.
        try:
            if phases:
                from . import residency
                from .llms.backends import vllm

                warm.clear()        # emptied first: the preloader skips the rest
                residency.unload_local()
                vllm.set_planned_utilization({})      # the shares were this run's
                vllm.set_planned_devices({})          # and so were the cards
            # Roll up tokens, engine time and cost. Also logged at INFO, so
            # run.py's existing --quiet/--debug control whether it is shown.
            summary["telemetry"] = telemetry.summary()
        except Exception as exc:  # noqa: BLE001 — must not mask the stage's error
            logger.error("[residency] end of run failed (%s: %s)",
                         type(exc).__name__, exc)
    trace = telemetry.collector()
    if trace is not None and trace.trace_path is not None:
        summary["trace"] = str(trace.trace_path)
    return summary
