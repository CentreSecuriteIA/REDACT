"""Config-driven runs: recipe + input-params JSON, per-stage run manifests.

Two authored files + one generated per stage:

- **Recipe** (structural, *what to run*): ``dataset_type`` (``"eval"`` = no
  constitution, ``"training"`` = constitution-seeded), ``data_dir``, ``models``
  per role (a registry name, or an object that also says how the model is
  served in this run: ``{"name", "backend_type", "copies", "workers"}``, see
  :func:`_serving`), ``taxonomy``/``taxonomy_dir``/``prompt_dir``, an ordered
  ``stages`` list, ``augmentations`` (the jailbreak ``include_*`` /
  ``pure_only`` toggles), a ``params_file`` pointer, and ``resume``.
- **Input-params** (tunables, *how much / how hard*): per-stage numeric knobs
  (``inputs`` / ``outputs`` / ``jailbreaks`` / ``constitution``).
- **Run-manifest** (generated): each stage writes ``{stage}.run.json`` under
  ``{data_dir}/Datasets/`` recording the resolved params, models, row counts,
  spec version, timestamp, and source file(s) — so the next stage or a human can
  see how to continue.

Usage::

    from redact import run_pipeline
    summary = run_pipeline("configs/runs/eval_example.json")
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd

from redact import paths, telemetry
from redact.llms.prompting import report_prompt_sources

logger = logging.getLogger(__name__)

# Ordered canonical stage names. A recipe's ``stages`` is any subset, in order.
STAGES = ("constitution", "inputs", "outputs", "paraphrase", "jailbreaks", "build")

_DEFAULT_MODELS = {
    "gen": None, "check": None, "translation": None, "constitution": None,
    "paraphraser": None, "paraphrase_check": None,
}

# The keys of a role's object under ``models`` that say how its model is served.
_SERVING_KEYS = ("backend_type", "copies", "workers")


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------

def manifest_path(stage: str, data_dir: str | Path | None = None) -> Path:
    """Path of a stage's run-manifest: ``{data_dir}/Datasets/{stage}.run.json``."""
    return paths.datasets(data_dir) / f"{stage}.run.json"


def write_manifest(
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
    """Write a stage's run-manifest and return its path."""
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

    path = manifest_path(stage, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
    return path


def read_manifest(stage: str, data_dir: str | Path | None = None) -> dict | None:
    """Read a stage's run-manifest, or None if it doesn't exist."""
    path = manifest_path(stage, data_dir)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_recipe(recipe: str | Path | dict) -> dict:
    """Load + validate a recipe (JSON path or already-parsed dict)."""
    if isinstance(recipe, dict):
        data = dict(recipe)
        data.setdefault("_base_dir", None)
    else:
        path = Path(recipe)
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        data["_base_dir"] = str(path.parent)

    dtype = data.get("dataset_type", "eval")
    if dtype not in ("eval", "training"):
        raise ValueError(f"dataset_type must be 'eval' or 'training', got {dtype!r}")

    # `is None`, not `or`: an explicit `stages: []` means "run nothing", and
    # `or` turned that into the full default pipeline — the opposite of what
    # was asked for, with no error to notice.
    stages = data.get("stages")
    if stages is None:
        stages = ["inputs", "outputs", "jailbreaks", "build"]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; valid: {list(STAGES)}")
    if len(set(stages)) != len(stages):
        raise ValueError(f"A stage is listed more than once in {stages}.")
    if dtype == "eval" and "constitution" in stages:
        raise ValueError("dataset_type 'eval' cannot include the 'constitution' stage.")
    if data.get("copies") is not None:
        raise ValueError(
            'The top-level "copies" key is no longer read; write each count in '
            'its role\'s entry under "models" instead, e.g. '
            '"gen": {"name": "llama-3.2-3b-debug", "copies": 2}.')
    # For its errors. The entries stay as written, and a run reads them again.
    _serving(data.get("models") or {})
    data["stages"] = stages
    data["dataset_type"] = dtype
    return data


def load_params(params: str | Path | dict | None, base_dir: str | Path | None = None) -> dict:
    """Load an input-params file (JSON path or dict). ``base_dir`` resolves a
    relative ``params_file`` from the recipe's directory."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    p = Path(params)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def _role_names(models: dict) -> dict:
    """A recipe's ``models`` with each object reduced to its model name."""
    return {role: entry.get("name") if isinstance(entry, dict) else entry
            for role, entry in models.items()}


def _serving(models: dict) -> dict[str, dict]:
    """How a recipe's ``models`` say its models are served, per model name.

    A role's entry may be an object in place of a name:

    - ``name``: the model's registry name. Left out, the role's default
      model, as for ``null``.
    - ``backend_type``: the setup of the registry entry that serves it:
      ``"api"``, ``"vllm"`` or ``"introspect"``. Left out, the entry's default.
    - ``copies``: copies of a local vLLM model, one per card. Left out, a
      copy on each idle card; ``1`` turns that off.
    - ``workers``: requests a hosted model gets at once. Left out, the
      entry's ``recommended_max_workers``.

    One model is served one way in a run. A role that gives none of the last
    three keys takes what another role gives its model, so a ``check`` left
    out is served as the generation model is.

    Returns:
        The keys given, per model name. A model no role gives a key for is
        left out.

    Raises:
        ValueError: An object breaks a rule. The message names the role and
            says what to write instead.
    """
    from .llms.backends import SETUP_TYPES, resolve_setup, validate_concurrency
    from .llms.model_config import get_model_config

    names = {**_DEFAULT_MODELS, **_role_names(models)}
    serving: dict[str, dict] = {}
    said_by: dict[str, str] = {}
    for role, entry in models.items():
        if not isinstance(entry, dict):
            continue
        where = f"models[{role!r}]"
        if role not in _DEFAULT_MODELS:
            raise ValueError(
                f"{where} is not a model role; write one of {list(_DEFAULT_MODELS)}.")
        unknown = [key for key in entry if key not in ("name", *_SERVING_KEYS)]
        if unknown:
            raise ValueError(
                f"{where} has unknown key(s) {unknown}; write only "
                f"{['name', *_SERVING_KEYS]}.")
        if role == "paraphraser" and entry.get("name") == "distribution":
            raise ValueError(
                f"{where} names the pool of all paraphrasers, which takes no "
                f'object; write "distribution", or the name of one paraphraser.')
        # A null says what leaving the key out says.
        given = {key: entry[key] for key in _SERVING_KEYS
                 if entry.get(key) is not None}
        for key in ("copies", "workers"):
            if key in given and (type(given[key]) is not int or given[key] < 1):
                raise ValueError(
                    f"{where}: {key} must be a whole number of at least 1, got "
                    f"{given[key]!r}; write one, or leave {key} out.")
        if not given:
            continue

        # The rules below need the registry entry of the role's model.
        try:
            (name,) = _resolve_role(names, role, {})
            config = get_model_config(name)
        except (KeyError, TypeError, ValueError):
            asked = repr(names[role]) if names[role] else "the role's default model"
            raise ValueError(
                f"{where}: {asked} is not a registered model, so how it is "
                f'served cannot be set; write a registered "name".') from None
        present = [s for s in sorted(SETUP_TYPES) if getattr(config, s) is not None]
        try:
            setup = resolve_setup(config, given.get("backend_type"))
        except (TypeError, ValueError):
            raise ValueError(
                f"{where}: backend_type {given.get('backend_type')!r} is not a "
                f"setup of {name!r}; write one of {present}.") from None
        if "copies" in given and setup != "vllm":
            raise ValueError(
                f"{where}: copies are for a model served by vLLM, and {name!r} is "
                f"served by its {setup} setup; leave copies out"
                + (', or write "backend_type": "vllm".' if "vllm" in present else "."))
        if "workers" in given and setup != "api":
            raise ValueError(
                f"{where}: workers are for a hosted model, and {name!r} is served "
                f"by its {setup} setup, where requests at once are automatic; "
                f"leave workers out"
                + (', or write "backend_type": "api".' if "api" in present else "."))
        if "workers" in given:
            try:
                validate_concurrency(name, config.api.backend_type, given["workers"])
            except ValueError:
                raise ValueError(
                    f"{where}: {name!r} takes one request at a time (its "
                    f"{config.api.backend_type} backend makes no parallel calls); "
                    f'write "workers": 1, or leave workers out.') from None
        if serving.setdefault(name, given) != given:
            raise ValueError(
                f"models[{said_by[name]!r}] and {where} both resolve to {name!r} "
                f"but serve it differently ({serving[name]} and {given}); one "
                f"model is served one way in a run, so write the same keys in "
                f"both, or in one of them only.")
        said_by.setdefault(name, role)
    return serving


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _counts(df) -> dict:
    """Row counts for a stage's returned DataFrame."""
    if df is None:
        return {}
    out = {"rows": int(len(df))}
    if "accepted" in getattr(df, "columns", []):
        out["accepted"] = int(df["accepted"].sum())
    return out


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
    recipe's: a number of copies per model role.

    Returns:
        One :class:`~redact.residency.Phase` per stage, or ``[]`` when no
        stage calls a local model or planning failed.
    """
    from . import residency

    try:
        aug = augmentations or {}
        # The planner counts copies per model. A model two roles resolve to
        # takes the count of the role the recipe names first.
        per_model: dict[str, int] = {}
        for role, count in (copies or {}).items():
            try:
                names = _resolve_role(models, role, aug)
            except (KeyError, ValueError):
                # No model registered for the role, so none to count for.
                continue
            for name in names:
                per_model.setdefault(name, count)
        phases = residency.plan_phases(
            {stage: _stage_models(models, [stage], aug) for stage in stages},
            copies=per_model)
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
            # Every plan carries the same share per engine; the first replaces
            # what an earlier run left.
            residency.apply_plan(phase.plan, replace=(i == 0))
    except Exception as exc:  # noqa: BLE001 — a planning bug must not abort a run
        logger.error("[residency] planning failed (%s: %s); the run continues "
                     "without a plan.", type(exc).__name__, exc)
        return []
    return phases


def _enter_phase(phases: list, i: int, warm: set[str], verbose: bool = True) -> None:
    """Make stage ``i``'s local models the resident ones.

    Unloads the engines no stage from ``i`` on calls, then starts loading
    this stage's models in the background, and the next stage's too when
    they fit beside them without an unload. Nothing here aborts the run.

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
        ahead = list(phase.loadable)
        if i + 1 < len(phases):
            following = phases[i + 1]
            if (following.plan.co_resident
                    and set(phase.resident) <= set(following.resident)):
                ahead += following.loadable
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


def run_pipeline(
    recipe: str | Path | dict,
    params: str | Path | dict | None = None,
    verbose: bool = True,
) -> dict:
    """Execute the stages named in a recipe, writing a manifest per stage.

    ``eval`` recipes run the standalone path; ``training`` recipes run the
    constitution stage first and thread its DataFrame into ``generate_inputs``.
    Returns a summary dict ``{"stages": {stage: {counts, manifest}}}``.
    Local models the run planned are released when it returns or raises.
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

    recipe = load_recipe(recipe)
    base_dir = recipe.get("_base_dir")
    if params is None:
        params = load_params(recipe.get("params_file"), base_dir=base_dir)
    else:
        params = load_params(params, base_dir=base_dir)

    data_dir = recipe.get("data_dir")
    dataset_type = recipe["dataset_type"]
    models = {**_DEFAULT_MODELS, **_role_names(recipe.get("models") or {})}
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
    phases = _plan_phases(models, stages, verbose=verbose, augmentations=aug)

    summary: dict = {"dataset_type": dataset_type, "data_dir": str(data_dir), "stages": {}}
    if phases:
        summary["residency"] = {
            "fits": all(phase.plan.fits for phase in phases),
            "phases": [{"stage": phase.stage, "resident": phase.resident,
                        **phase.plan.as_dict()} for phase in phases],
        }
    constitution_df = None

    warm: set[str] = set()
    try:
        for i, stage in enumerate(stages):
            _enter_phase(phases, i, warm, verbose=verbose)
            with telemetry.stage(stage):
                if stage == "constitution":
                    constitution_df = generate_constitution(
                        taxonomy=taxonomy, taxonomy_dir=taxonomy_dir,
                        model=models["constitution"], data_dir=data_dir,
                        resume=resume, verbose=verbose, **P("constitution"),
                    )
                    df = constitution_df

                elif stage == "inputs":
                    kw = dict(
                        data_dir=data_dir, taxonomy=taxonomy, taxonomy_dir=taxonomy_dir,
                        prompt_dir=prompt_dir, model=models["gen"], check_model=models["check"],
                        resume=resume, verbose=verbose, **P("inputs"),
                    )
                    if dataset_type == "training":
                        if constitution_df is None:
                            raise ValueError(
                                "training run: the 'constitution' stage must precede 'inputs' "
                                "(so its entries can seed input generation)."
                            )
                        kw["constitution_df"] = constitution_df
                    df = generate_inputs(**kw)

                elif stage == "outputs":
                    df = generate_outputs(
                        data_dir=data_dir, model=models["gen"], check_model=models["check"],
                        prompt_dir=prompt_dir, resume=resume, verbose=verbose, **P("outputs"),
                    )

                elif stage == "paraphrase":
                    res = generate_paraphrases(
                        data_dir=data_dir, paraphraser=models["paraphraser"],
                        check_model=models["paraphrase_check"], prompt_dir=prompt_dir,
                        resume=resume, verbose=verbose, **P("paraphrase"),
                    )
                    frames = [v for v in res.values() if v is not None and not v.empty]
                    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

                elif stage == "jailbreaks":
                    df = generate_jailbreaks(
                        data_dir=data_dir, model=models["gen"],
                        translation_model=models["translation"],
                        resume=resume, verbose=verbose, **aug, **P("jailbreaks"),
                    )

                elif stage == "build":
                    df = build_dataset(
                        data_dir=data_dir, mode=dataset_type, verbose=verbose, **P("build"),
                    )

                else:  # pragma: no cover — load_recipe already validated
                    raise ValueError(f"Unknown stage {stage!r}")

                stage_params = {**(aug if stage == "jailbreaks" else {}), **P(stage)}
                spec = load_spec().get("version", "") if stage == "jailbreaks" else None
                mpath = write_manifest(
                    stage, data_dir=data_dir, params=stage_params, models=models,
                    counts=_counts(df), spec_version=spec,
                )
                summary["stages"][stage] = {"counts": _counts(df), "manifest": str(mpath)}
                if verbose:
                    logger.info("[%s] %s -> manifest %s", stage, _counts(df), mpath.name)
    finally:
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
