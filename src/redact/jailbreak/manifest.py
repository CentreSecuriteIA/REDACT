"""Planning pass for jailbreak runs (built on the shared ``Manifest``).

A jailbreak run is **planned in full before any generation**:
:func:`plan_run` walks every input sample × every iteration, assigns a
technique combination (deterministically, with per-sample dedup so repeated
iterations don't draw the same combination), and writes one JSON line per unit
via :class:`redact.dataset.Manifest`. Generation then streams the manifest in
chunks.

Identity is the **prompt-content MD5** (the same ``sample_id`` the input
samples carry), so the manifest, the assignment seed, and every output
row are all keyed to the exact prompt — stable across re-runs and content-based.

Resume is driven by the sidecar ``jailbreaks.state.jsonl`` **ledger**
(``redact.dataset.Ledger``, keyed ``(input_id, iteration)``, acked per chunk
after the CSV append) alone. The output CSV is read only once, to take over a
run written before the ledger recorded sizes — see
:func:`redact.pipelines.generate_jailbreaks`. The manifest itself
is immutable after planning,
so a chunk crash never corrupts the plan.

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

import logging
from pathlib import Path

import pandas as pd

from redact.dataset import Manifest, row_id

from .assignment import assign_combination

logger = logging.getLogger(__name__)


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
    id_col: str = "sample_id",
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
    planned = 0

    # Yielded to the manifest one at a time: the plan is never held whole.
    def units():
        nonlocal planned
        for _, row in inputs.iterrows():
            sid = row_id(row, text_col, id_col)

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
                planned += 1
                yield {
                    "sample_id": sid,
                    "iteration": it,
                    "combination": names,
                    "settings": {**settings, "seed": seed},
                    "category": str(row.get("category", "")),
                    "entry_type": str(row.get("entry_type", "")),
                    "status": "planned",
                }

    Manifest(manifest_path).write(units())

    if verbose:
        logger.info("Planned %d units (%d samples x %d) -> %s",
                    planned, len(inputs), iterations, manifest_path)
    return manifest_path


def plan_index(rows: list[dict]) -> dict[str, list[dict]]:
    """Group plan rows by ``sample_id`` (preserves iteration order per sample)."""
    index: dict[str, list[dict]] = {}
    for r in rows:
        index.setdefault(str(r["sample_id"]), []).append(r)
    return index
