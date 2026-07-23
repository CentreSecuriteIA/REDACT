"""Config-driven runs: recipe + input-params JSON, per-stage run manifests.

Two authored files + one generated per stage:

- **Recipe** (structural, *what to run*): ``dataset_type`` (``"eval"`` = no
  constitution, ``"training"`` = constitution-seeded), ``data_dir``, ``models``
  per role, ``taxonomy``/``taxonomy_dir``/``prompt_dir``, an ordered ``stages``
  list, ``augmentations`` (the jailbreak ``include_*`` / ``pure_only`` toggles),
  a ``params_file`` pointer, and ``resume``.
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
import time
from pathlib import Path

from redact import paths

# Ordered canonical stage names. A recipe's ``stages`` is any subset, in order.
STAGES = ("constitution", "inputs", "outputs", "jailbreaks", "build")

_DEFAULT_MODELS = {"gen": None, "check": None, "translation": None, "constitution": None}


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

    stages = data.get("stages") or ["inputs", "outputs", "jailbreaks", "build"]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; valid: {list(STAGES)}")
    if dtype == "eval" and "constitution" in stages:
        raise ValueError("dataset_type 'eval' cannot include the 'constitution' stage.")
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


def run_pipeline(
    recipe: str | Path | dict,
    params: str | Path | dict | None = None,
    verbose: bool = True,
) -> dict:
    """Execute the stages named in a recipe, writing a manifest per stage.

    ``eval`` recipes run the standalone path; ``training`` recipes run the
    constitution stage first and thread its DataFrame into ``generate_inputs``.
    Returns a summary dict ``{"stages": {stage: {counts, manifest}}}``.
    """
    from redact import (
        generate_constitution, generate_inputs, generate_outputs,
        generate_jailbreaks, build_dataset,
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
    models = {**_DEFAULT_MODELS, **(recipe.get("models") or {})}
    taxonomy = recipe.get("taxonomy", "content_moderation_categories")
    taxonomy_dir = recipe.get("taxonomy_dir")
    prompt_dir = recipe.get("prompt_dir")
    aug = dict(recipe.get("augmentations") or {})
    resume = recipe.get("resume", True)
    stages = recipe["stages"]

    def P(stage: str) -> dict:
        return dict(params.get(stage) or {})

    if verbose:
        print(f"\n{'='*60}\nRun pipeline ({dataset_type}) — stages: {stages}\n{'='*60}")

    summary: dict = {"dataset_type": dataset_type, "data_dir": str(data_dir), "stages": {}}
    constitution_df = None

    for stage in stages:
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
                resume=resume, verbose=verbose, **P("outputs"),
            )

        elif stage == "jailbreaks":
            df = generate_jailbreaks(
                data_dir=data_dir, model=models["gen"],
                translation_model=models["translation"],
                resume=resume, verbose=verbose, **aug, **P("jailbreaks"),
            )

        elif stage == "build":
            df = build_dataset(data_dir=data_dir, verbose=verbose)

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
            print(f"  [{stage}] {_counts(df)} -> manifest {mpath.name}")

    return summary
