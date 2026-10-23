"""One run of a recipe: serving, residency, telemetry, the stages, a record each.

Each stage writes ``{stage}.run.json`` under ``{data_dir}/Datasets/``: the
resolved params, models, how the stage's models were served, row counts,
spec version, timestamp, the files the stage read its inputs from
(``sources``), and the dataset type with its file's hash; with a
``config_dir``, the folder and the config files read from it
(``config_files``); the jailbreak stage's also holds the resolved
augmentation section.

Usage::

    from redact import run_pipeline
    summary = run_pipeline("configs/runs/eval_example.json")
"""

import contextlib
import json
import logging
import os
import threading
import time
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from redact import paths, telemetry
from redact.config_overlay import report_config_sources
from redact.dataset import count_rows, iter_rows
from redact.llms.prompting import report_prompt_sources
from redact.llms.resources import measure
from redact.recipe import (
    _DEFAULT_MODELS,
    _role_names,
    augmentations_of,
    check_dimensions,
    dataset_type_hash,
    load_params,
    load_recipe,
    resolve_role,
    serving_of,
    stage_params,
)
from redact.stages import STAGES, RunContext

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
        # UTC, with its offset, as the trace's timestamps are.
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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


#: Former names of the three record functions, read for one release with a
#: DeprecationWarning (a run record is not a ``*.manifest.jsonl``).
FORMER_RECORD_NAMES = {"manifest_path": "record_path",
                       "write_manifest": "write_record",
                       "read_manifest": "read_record"}


def former_record_name(module: str, name: str, stacklevel: int = 3):
    """The record function ``name``, a former name, stands for, with a
    ``DeprecationWarning`` naming ``module`` and the new name."""
    new = FORMER_RECORD_NAMES[name]
    warnings.warn(
        f"{module}.{name} is now {new}; the former name is read for this release.",
        DeprecationWarning, stacklevel=stacklevel)
    return globals()[new]


def __getattr__(name: str):
    if name in FORMER_RECORD_NAMES:
        return former_record_name(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def file_counts(
    files: Iterable[str | Path], keep: Mapping[str, Any] | None = None,
) -> dict:
    """Row counts of a stage's CSVs, read in chunks.

    ``rows``, and ``accepted`` when a file has an ``accepted`` column, as the
    rows of ``files`` read into one frame would give them. A file that does
    not exist, or has 0 bytes, has no rows.

    Args:
        keep: Count only the rows that hold these values in these columns.
            A file without one of the columns is not limited by it.
    """
    keep = keep or {}
    rows = accepted = 0
    verdicts = False        # whether any file has an ``accepted`` column
    for path in map(Path, files):
        # 0 bytes: a save killed before its first byte.
        if not path.exists() or not path.stat().st_size:
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
    does not fit and a planning error are only logged, at ERROR, and so is a
    plan with warnings, at WARNING, with ``verbose`` off too; a plan that
    fits is logged at INFO with ``verbose`` only. ``copies`` is the
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
                # A plan that does not fit, or warns, is logged whatever
                # verbose says: --quiet runs at WARNING and still shows it.
                residency.report(
                    phase.plan,
                    verbose=verbose or not phase.plan.fits or bool(phase.plan.warnings),
                    stages=stages_of[id(phase.plan)])
            # The first replaces what an earlier run left. An engine loaded
            # a second time has other values in a later plan, which it gets
            # when that stage is entered (_enter_phase).
            residency.apply_plan(phase.plan, replace=(i == 0))
    except Exception as exc:  # noqa: BLE001 — a planning bug must not abort a run
        logger.error(
            "[residency] planning failed (%s: %s); the run continues without a "
            "plan, so every local engine loads unplanned, at 0.9 of a whole "
            "card: a second engine on the same card will not start.",
            type(exc).__name__, exc, exc_info=exc)
        return []
    return phases


#: Memory an idle card may show in use: the driver's own and a display's.
_IDLE_GIB = 1.0


def _own_pids() -> set[int]:
    """This process and its children, such as the engine processes of a
    vLLM model it loaded. The children are found with psutil, which vLLM
    installs; without it only this process counts."""
    pids = {os.getpid()}
    try:
        import psutil
    except ImportError:
        return pids
    with contextlib.suppress(psutil.Error):
        pids.update(child.pid for child in psutil.Process().children(recursive=True))
    return pids


def _engine_parents(pids: list[int], own: set[int]) -> list[int]:
    """The vLLM engine processes above ``pids``, other than this process's.

    ``nvidia-smi`` lists the workers that hold the memory; after a
    ``kill -9`` their engine processes live on above them, orphaned. An
    ancestor counts when its name or command line starts as vLLM titles its
    processes (``VLLM::``, or ``$VLLM_PROCESS_NAME_PREFIX::``). Found with
    psutil; without it, none.
    """
    try:
        import psutil
    except ImportError:
        return []
    title = os.environ.get("VLLM_PROCESS_NAME_PREFIX", "VLLM") + "::"
    found: dict[int, None] = {}
    for pid in pids:
        with contextlib.suppress(psutil.Error):
            for parent in psutil.Process(pid).parents():
                if parent.pid == 1 or parent.pid in own or parent.pid in pids:
                    continue
                with contextlib.suppress(psutil.Error):
                    if any(text.startswith(title)
                           for text in (parent.name(), *parent.cmdline()[:1])):
                        found.setdefault(parent.pid)
    return list(found)


def _planned_card(
    cards: list[measure.CardUse], device: int,
) -> measure.CardUse | None:
    """The card the plan numbers ``device``, counting over the cards
    ``CUDA_VISIBLE_DEVICES`` leaves visible, or ``None``."""
    visible = measure.visible_gpu_ids()
    if visible is None:
        return next((card for card in cards if card.index == device), None)
    if device >= len(visible):
        return None
    # The variable names a card by its index or by its UUID, or a prefix of it.
    name = visible[device]
    return next((card for card in cards if name == str(card.index)
                 or (name.startswith("GPU-") and card.uuid.startswith(name))), None)


def _check_cards(phases: list) -> None:
    """Stop the run when another process holds memory on a card the plan uses.

    Reads ``nvidia-smi`` once, before any model loads. A process on a card
    the plan leaves free, and memory in use on a planned card with no other
    process listed, are warnings. Without ``nvidia-smi`` it does nothing.

    Raises:
        RuntimeError: Naming each planned card held and its processes, and
            how to free them.
    """
    cards = measure.card_use()
    if not cards:
        return
    devices = {d for phase in phases
               for fp in phase.plan.footprints for d in fp.devices}
    planned = {card.index for card in (_planned_card(cards, d) for d in devices)
               if card is not None}
    own = _own_pids()
    held, pids = [], []
    for card in cards:
        others = [proc for proc in card.processes if proc[0] not in own]
        names = ", ".join(
            f"{pid} ({name}, {gib} GiB)" if gib is not None else f"{pid} ({name})"
            for pid, name, gib in others)
        if card.index not in planned:
            if others:
                logger.warning("[residency] GPU %d is in use by another process, "
                               "%s; the run does not plan a model on it.",
                               card.index, names)
        elif others:
            held.append(f"  GPU {card.index}: {names}")
            pids += [pid for pid, _, _ in others]
        elif not card.processes and card.used_gib >= _IDLE_GIB:
            logger.warning(
                "[residency] GPU %d has %.1f of %.1f GiB in use, by no process "
                "nvidia-smi lists here; the plan counts it empty, so a model "
                "planned on it may not load.",
                card.index, card.used_gib, card.total_gib)
    if held:
        parents = _engine_parents(pids, own)
        lines = "\n".join(held)
        them = "them and their parent engine processes" if parents else "them"
        raise RuntimeError(
            f"The run plans a model on a card another process holds memory on, "
            f"so the model would fail to load there:\n{lines}\n"
            f"An earlier run killed with kill -9 leaves its vLLM engine "
            f"processes behind. If these are such, end {them} with "
            f"`kill -9 {' '.join(map(str, pids + parents))}` and check that "
            f"nvidia-smi shows the card empty; otherwise run on other cards "
            f"(CUDA_VISIBLE_DEVICES). Nothing has run yet.")


def _enter_phase(
    phases: list, i: int, warm: set[str], verbose: bool = True,
    kept: dict | None = None,
) -> tuple[threading.Thread, list[str]] | None:
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
        kept: Engines no unload releases, as
            :func:`~redact.residency.loaded_engines` returns them.

    Returns:
        The preload thread it started and the models handed to it, if any.
    """
    if not phases:
        return None
    from . import residency

    try:
        phase = phases[i]
        warm &= set(phase.resident)
        dropped = set(phases[i - 1].resident) - set(phase.resident) if i else ()
        if dropped:
            if verbose:
                logger.info("[residency] stage %s: unloading %s",
                            phase.stage, ", ".join(sorted(dropped)))
            residency.unload_local(keep=phase.resident, engines=kept)
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
            thread = residency.preload(todo, verbose=verbose, wanted=warm)
            return (thread, todo) if thread is not None else None
    except Exception as exc:  # noqa: BLE001 — residency must not abort a run
        logger.error("[residency] stage %d: could not switch models (%s: %s); "
                     "the run continues.", i + 1, type(exc).__name__, exc)
    return None


def _stage_models(models: dict, stages: list[str], augmentations: dict) -> list[str]:
    """The models the stages will call, in the order the stages first need them.

    Each role resolves the way its stage resolves it, so the plan covers the
    models that will actually load.
    """
    wanted: list[str] = []
    for stage in stages:
        for role, _ in STAGES[stage].roles:
            try:
                wanted.extend(resolve_role(models, role, augmentations))
            except (KeyError, ValueError):
                # No model registered for the role; the stage reports it.
                continue
    return list(dict.fromkeys(wanted))


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
        ``copies`` is what was planned, read before any stage runs: when a
        build with the plan's copies fails and one copy loads instead, the
        trace's ``load`` event says so (``fallback``, ``planned_copies``).
    """
    from .llms.backends import resolve_setup
    from .llms.model_config import get_model_config

    planned = {fp.model: fp.copies for phase in phases for fp in phase.plan.footprints}
    served: dict = {}
    roles = dict.fromkeys(
        role for stage in stages for role, _ in STAGES[stage].roles)
    for role in roles:
        try:
            names = resolve_role(models, role, augmentations)
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
    data_dir: str | Path | None = None,
) -> dict:
    """Run the stages a recipe names, in order, writing a run record per stage.

    ``params`` replaces the recipe's ``params_file``: a dict, or a file's
    path, which when relative is read from the current directory, else from
    the recipe's folder (:func:`redact.recipe.params_file_of`); with
    ``verbose`` the file read is logged. ``data_dir`` replaces the recipe's
    ``data_dir``. A params key that is not a parameter of its stage's front,
    or that the run passes the front itself, fails before any stage runs, as
    do an augmentation section the params contradict and a generation axis
    that varies the dimension of a family the section draws
    (:func:`redact.recipe.check_dimensions`); the jailbreak stage
    runs and records the section with the params' flat keywords applied
    over its adversarial kind. No stage reads its rows back: the counts
    are taken from each stage's files, and the inputs are seeded from the
    constitution saved under ``data_dir``. The local engines the run loaded
    are released when it returns or raises; those loaded before it, such as
    a notebook's, stay loaded unless the run calls one and unloads it
    between stages. The way the recipe serves its models holds only until then.
    Before any of them loads, the run stops if another process holds memory
    on a card the plan puts one on (:func:`_check_cards`). At the end a
    preload still loading is waited for at most ``EXIT_WAIT_S`` (60 s), then
    its load is ended, and the run's telemetry is rolled up and uninstalled:
    a front called after the run records into neither its trace nor its
    roll-up.

    Returns:
        ``dataset_type``, ``data_dir`` (the root the run wrote under),
        ``stages`` (per stage, its ``counts`` and the path of its run record
        under ``manifest``, a key kept by that name),
        ``serving`` (:func:`_served`) and ``telemetry``; ``residency`` when
        a stage calls a local model, and ``trace`` when one is written.
    """
    from redact.jailbreak import load_spec
    from redact.llms.backends import capabilities

    from . import residency

    recipe = load_recipe(recipe, params=params)
    params_file = recipe["_params_file"]
    params = load_params(params if isinstance(params, dict) else params_file,
                         config_dir=recipe.get("config_dir"))
    serving = serving_of(recipe.get("models") or {})
    stages = recipe["stages"]
    ctx = RunContext(
        data_dir=recipe.get("data_dir") if data_dir is None else data_dir,
        dataset_type=recipe["dataset_type"],
        models={**_DEFAULT_MODELS, **_role_names(recipe.get("models") or {})},
        prompt_dir=recipe.get("prompt_dir"),
        # null, as the examples write taxonomy_dir and prompt_dir, is the default.
        taxonomy=("harmful" if recipe.get("taxonomy") is None
                  else recipe["taxonomy"]),
        taxonomy_dir=recipe.get("taxonomy_dir"),
        benign_taxonomy=recipe["benign_taxonomy"],
        params=params,
        # The type file's flat keywords too, which load_params warns are read.
        augmentations=augmentations_of(recipe, stage_params(
            "jailbreaks", recipe["dataset_type"], params, recipe.get("config_dir"))
            if "jailbreaks" in stages else None),
        resume=recipe.get("resume", True),
        verbose=verbose,
        config_dir=recipe.get("config_dir"),
    )
    for name in stages:
        STAGES[name].check(ctx)     # a bad params key fails before any stage runs
    check_dimensions(ctx.params, ctx.augmentations, dataset_type=ctx.dataset_type,
                     config_dir=ctx.config_dir)
    noted = {"dataset_type": ctx.dataset_type,
             "dataset_type_hash": dataset_type_hash(ctx.dataset_type, ctx.config_dir)}

    if verbose:
        logger.info("Run pipeline (%s) — stages: %s", ctx.dataset_type, stages)
        if params_file is not None:
            logger.info("Params file: %s", params_file)

    # Telemetry first, so the plan and preload below are themselves recorded.
    telemetry.install(data_dir=ctx.data_dir)
    if verbose:
        # Alongside the residency plan: one "here's your setup" block, before
        # anything runs. With prompts resolving per file against an overlay,
        # "which prompt ran?" stops being obvious, and an override that
        # silently matches nothing is invisible without this.
        report_prompt_sources(ctx.prompt_dir)
    configs = report_config_sources(ctx.config_dir, verbose=verbose)
    if ctx.config_dir is not None:
        noted.update(config_dir=str(ctx.config_dir),
                     config_files=sorted(configs["overridden"] + configs["added"]))

    summary: dict = {
        "dataset_type": ctx.dataset_type,
        # The root the run wrote under, get_output_dir() when it names none.
        "data_dir": str(paths.datasets(ctx.data_dir).parent), "stages": {}}
    phases: list = []
    warm: set[str] = set()
    preloads: list[tuple[threading.Thread, list[str]]] = []
    before: dict = {}
    try:
        before = residency.loaded_engines()
        # Before the plan, which reads it as every client the run builds does.
        capabilities.set_run_serving(
            {name: {key: how[key] for key in ("backend_type", "workers") if key in how}
             for name, how in serving.items()})
        phases = _plan_phases(
            ctx.models, stages, verbose=verbose, augmentations=ctx.augmentations,
            copies={name: how["copies"] for name, how in serving.items()
                    if "copies" in how})
        if phases:
            _check_cards(phases)    # before any model loads
            summary["residency"] = {
                "fits": all(phase.plan.fits for phase in phases),
                "phases": [{"stage": phase.stage, "resident": phase.resident,
                            **phase.plan.as_dict()} for phase in phases],
            }
        served = summary["serving"] = _served(
            ctx.models, stages, ctx.augmentations, serving, phases)
        # Of the engines loaded before the run, those no stage calls stay
        # through every stage.
        unplanned = residency.loaded_engines(
            exclude=[name for phase in phases for name in phase.resident])

        for i, name in enumerate(stages):
            stage = STAGES[name]
            preload = _enter_phase(phases, i, warm, verbose=verbose, kept=unplanned)
            if preload is not None:
                preloads.append(preload)
            with telemetry.stage(name):
                stage.run(read_back=False, **stage.kwargs(ctx))
                counts = file_counts(*stage.files(ctx))
                if stage.extra_counts is not None:
                    counts.update(stage.extra_counts(ctx))
                jailbreaks = name == "jailbreaks"
                path = write_record(
                    name, data_dir=ctx.data_dir, params=stage.params(ctx),
                    models=ctx.models, counts=counts,
                    sources=[str(p) for p in stage.sources(ctx) if p.exists()],
                    spec_version=load_spec().get("version", "") if jailbreaks else None,
                    extra={"serving": {role: served[role] for role, _ in stage.roles},
                           **noted,
                           **({"augmentation": ctx.augmentations} if jailbreaks
                              else {})},
                )
                summary["stages"][name] = {"counts": counts, "manifest": str(path)}
                if verbose:
                    logger.info("[%s] %s -> record %s", name, counts, path.name)
    finally:
        # The serving was this run's: a call made after it must not get it.
        capabilities.set_run_serving({})
        # The run's end is a transition too: release what the last stage held
        # and roll up, also after a failed stage. Nothing here may mask the
        # stage's own error.
        try:
            if phases:
                from .llms.backends import vllm

                warm.clear()        # emptied first: the preloader skips the rest
                # A model still loading is released below, once it is loaded.
                # Not waited for longer than an exit waits: a load that does
                # not return by then is ended by the release.
                deadline = time.monotonic() + vllm.EXIT_WAIT_S
                for thread, names in preloads:
                    if thread.is_alive() and verbose:
                        logger.info("[residency] waiting for a model being "
                                    "preloaded, to release it")
                    thread.join(max(0.0, deadline - time.monotonic()))
                    if thread.is_alive():
                        logger.error(
                            "[residency] the preload of %s has not finished "
                            "%g s after the run's end; ending its load to "
                            "release the cards.", ", ".join(names), vllm.EXIT_WAIT_S)
                residency.unload_local(engines=before, wait=vllm._GRACE_S)
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
        # The trace and the roll-up were this run's: a front called after it,
        # as from a notebook, records into neither.
        telemetry.uninstall()
    return summary
