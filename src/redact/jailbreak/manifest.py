"""Planning pass + JSONL ledger for jailbreak runs.

A jailbreak run is **planned in full before any generation**:
:func:`plan_run` walks every input sample × every iteration, assigns a
technique combination (deterministically, with per-sample dedup so repeated
iterations don't draw the same combination), and writes one JSON line per unit
to a manifest. Generation then streams the manifest in chunks.

Identity is the **prompt-content MD5** (the same ``id`` ``dataset/io.py`` writes
for input samples), so the manifest, the assignment seed, and every output row
are all keyed to the exact prompt — stable across re-runs and content-based.

Resume is driven by the **output CSV as source of truth**: any
``(sample_id, iteration)`` already present there is skipped on a re-run. The
manifest itself is immutable after planning, so a chunk crash never corrupts the
plan.

Manifest row schema (one JSON object per line)::

    {"sample_id": "<md5>", "iteration": 0,
     "combination": ["to_noble_goal", "to_rot13"],
     "settings": {"max_complexity": 6, ..., "seed": 42},
     "category": "Cyber", "entry_type": "harmful", "status": "planned"}

The ``iteration`` + ``settings`` fields exist so a future multi-round /
increasing-complexity pipeline can lay out N iterations per sample up front;
this module only plans, it does not run the rounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from redact.dataset.io import _hash_text

from .utils import assign_combination


def compute_sample_id(prompt: str) -> str:
    """Prompt-content id — matches ``dataset.io._hash_text`` (MD5, 16 hex)."""
    return _hash_text(prompt)


def default_manifest_path(output_path) -> Path:
    """Manifest path alongside the jailbreak output CSV (``*.manifest.jsonl``)."""
    out = Path(output_path)
    return out.with_name(out.stem + ".manifest.jsonl")


def plan_run(
    inputs: pd.DataFrame,
    pool: list,
    *,
    manifest_path,
    seed: int = 42,
    iterations: int = 1,
    settings_per_iteration: list[dict] | None = None,
    sample_kwargs: dict | None = None,
    text_col: str = "prompt",
    id_col: str = "id",
    verbose: bool = True,
) -> Path:
    """Write the full run plan to ``manifest_path`` (one JSON line per unit).

    Args:
        inputs: Input samples. Needs ``text_col`` (and optionally ``id_col``,
            ``category``, ``entry_type``); ``sample_id`` is taken from ``id_col``
            when present, else computed from the prompt text.
        pool: Tagged technique functions to sample combinations from.
        manifest_path: Where to write the JSONL plan (overwritten each call —
            planning is idempotent).
        seed: Global run seed (combined with sample_id + iteration).
        iterations: Units to plan per sample (1 for a single-round run).
        settings_per_iteration: Optional per-iteration ``sample_kwargs`` (for the
            future increasing-complexity pipeline). Falls back to ``sample_kwargs``.
        sample_kwargs: Base kwargs forwarded to ``sample_combination`` via
            ``assign_combination`` (``max_complexity``, ``include_*``,
            ``sampling_probs``, ...).
        text_col / id_col: Column names in ``inputs``.

    Returns:
        The manifest path.
    """
    sample_kwargs = sample_kwargs or {}
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    has_id = id_col in inputs.columns
    n_units = 0
    with manifest_path.open("w", encoding="utf-8") as fh:
        for _, row in inputs.iterrows():
            if has_id and pd.notna(row.get(id_col)) and str(row.get(id_col)):
                sid = str(row[id_col])
            else:
                sid = compute_sample_id(str(row[text_col]))

            used: list[tuple] = []
            for it in range(iterations):
                settings = (
                    settings_per_iteration[it]
                    if settings_per_iteration is not None
                    else sample_kwargs
                )
                names = assign_combination(
                    sid, pool, seed=seed, iteration=it, used=used, **settings
                )
                used.append(tuple(names))
                rec = {
                    "sample_id": sid,
                    "iteration": it,
                    "combination": names,
                    "settings": {**settings, "seed": seed},
                    "category": str(row.get("category", "")),
                    "entry_type": str(row.get("entry_type", "")),
                    "status": "planned",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_units += 1

    if verbose:
        print(f"  Planned {n_units} units ({len(inputs)} samples x {iterations}) -> {manifest_path}")
    return manifest_path


def load_plan(manifest_path) -> list[dict]:
    """Read all plan rows from a manifest JSONL file."""
    rows: list[dict] = []
    with Path(manifest_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plan_index(rows: list[dict]) -> dict[str, list[dict]]:
    """Group plan rows by ``sample_id`` (preserves iteration order per sample)."""
    index: dict[str, list[dict]] = {}
    for r in rows:
        index.setdefault(str(r["sample_id"]), []).append(r)
    return index


def completed_from_output(output_path) -> set[tuple[str, int]]:
    """Return ``(sample_id, iteration)`` units already written to the output CSV.

    The output CSV is the resume source of truth — anything present here is
    skipped on a re-run, so partial chunks never duplicate rows.
    """
    p = Path(output_path)
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    if df.empty or "input_id" not in df.columns:
        return set()
    iterations = (
        df["iteration"] if "iteration" in df.columns
        else pd.Series([0] * len(df))
    )
    return set(zip(df["input_id"].astype(str), iterations.astype(int)))
