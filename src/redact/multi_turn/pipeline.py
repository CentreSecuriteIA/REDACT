"""Batched generation of multi-turn conversation datasets.

Plays a :class:`~redact.multi_turn.core.Setting` on each seed (× ``iterations``)
to a :class:`~redact.multi_turn.core.Trajectory`, driven at scale by
:func:`redact.llms.conversation.drive_generators` (one batch per model per round).
Resumable via a sidecar ``*.state.jsonl`` **ledger** + a full-plan ``*.manifest.jsonl``,
mirroring the paraphrase / output pipelines. The unit key ``(input_id, iteration)``
and the ``source`` provenance label match the rest of the library, so conversation
datasets join / dedup / resume exactly like inputs, jailbreaks, and paraphrases.
Writes ``conversations.csv`` (flat metadata + the JSON step log per row).
"""

from __future__ import annotations

import copy
import json
import logging
from math import ceil
from pathlib import Path

import pandas as pd

from redact import paths
from redact.dataset.io import _hash_text
from redact.dataset.ledger import Ledger
from redact.dataset.manifest import Manifest
from redact.llms import get_router
from redact.llms.conversation import drive_generators

from .core import Setting, conversation_gen

logger = logging.getLogger(__name__)


def _seed_text_col(df: pd.DataFrame) -> str:
    for c in ("seed", "sample", "prompt", "text"):
        if c in df.columns:
            return c
    raise ValueError("seeds must have a 'seed' / 'sample' / 'prompt' / 'text' column.")


def _ledger(out: Path) -> Ledger:
    """Shared resume-ledger for conversation units, keyed ``(input_id, iteration)``."""
    return Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                          casters={"input_id": str, "iteration": int})


def generate_conversations(
    seeds: pd.DataFrame,
    setting: Setting | callable,
    data_dir: str | Path | None = None,
    iterations: int = 1,
    resume: bool = True,
    batch_size: int = 256,
    output_path: str | Path | None = None,
    verbose: bool = True,
    router=None,
) -> pd.DataFrame:
    """Generate conversation trajectories from seeds under a Setting.

    Args:
        seeds: DataFrame of opening prompts/goals; the seed text comes from a
            ``seed``/``sample``/``prompt``/``text`` column, the id from ``sample_id``
            (else a content hash), plus optional ``category``/``entry_type`` metadata.
        setting: a :class:`Setting` **template** (deep-copied per conversation so
            stateful actors don't leak) **or a zero-arg factory** returning a fresh
            Setting per conversation (preferred for stateful actors).
        data_dir: working root; artifact defaults to ``{data_dir}/Datasets/conversations.csv``.
        iterations: conversations generated per seed — the ``iteration`` axis (0-based),
            named to match the jailbreak / paraphrase pipelines. These are independent
            stochastic variants (not escalation rounds).
        resume: skip ``(input_id, iteration)`` units already in the sidecar ledger.
        batch_size: conversations per chunk (each chunk driven together, pooled by model).
        output_path: artifact override.
        router: router override (defaults to the process-wide one).

    Returns:
        DataFrame of all rows in ``conversations.csv``: ``sample_id``, ``input_id``,
        ``iteration``, ``setting``, ``turns_used``, ``stop_reason``, ``transcript``
        (JSON step log), ``category``, ``entry_type``, ``source``.
    """
    router = router or get_router()
    if seeds is None or len(seeds) == 0:
        raise ValueError("generate_conversations needs a non-empty `seeds` DataFrame.")
    seeds = seeds.reset_index(drop=True)
    tcol = _seed_text_col(seeds)

    out = Path(output_path) if output_path else paths.conversations_csv(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    ledger, manifest = _ledger(out), Manifest.sidecar(out)

    make_setting = setting if callable(setting) else (lambda: copy.deepcopy(setting))

    def _input_id(row) -> str:
        existing = str(row.get("sample_id", "")).strip()
        return existing if existing and existing.lower() != "nan" else _hash_text(str(row[tcol]))

    if not resume:
        ledger.reset()
        manifest.reset()
        if out.exists():
            out.unlink()
    completed = ledger.completed() if resume else set()

    # Plan the full run → manifest (every (input_id, iteration) unit).
    units: list[tuple] = []  # (input_id, iteration, seed_text, meta)
    for _, row in seeds.iterrows():
        iid, stext = _input_id(row), str(row[tcol])
        meta = {"category": str(row.get("category", "")), "entry_type": str(row.get("entry_type", ""))}
        for it in range(iterations):
            units.append((iid, it, stext, meta))
    manifest.write({"input_id": iid, "iteration": it, **meta} for iid, it, _t, meta in units)

    pending = [u for u in units if (u[0], u[1]) not in completed]
    if verbose:
        logger.info("Generate Conversations")
        logger.info("Seeds: %d | iterations: %d | units: %d | pending: %d",
                    len(seeds), iterations, len(units), len(pending))
    if not pending:
        return pd.read_csv(out) if out.exists() else pd.DataFrame()

    n = len(pending)
    n_chunks = ceil(n / batch_size) if batch_size else 1

    def _row(key, *, setting_name, turns, stop, transcript_json, meta):
        return {
            "sample_id": _hash_text(f"{key[0]}:{key[1]}"),
            "input_id": key[0], "iteration": key[1], "setting": setting_name,
            "turns_used": turns, "stop_reason": stop, "transcript": transcript_json,
            "category": meta["category"], "entry_type": meta["entry_type"],
            "source": "conversation",
        }

    for start in range(0, n, batch_size):
        chunk = pending[start : start + batch_size]
        gens, keymeta = {}, {}
        for iid, it, stext, meta in chunk:
            key = (iid, it)
            gens[key] = conversation_gen(make_setting(), stext, input_id=iid)
            keymeta[key] = meta

        def finalize(key, traj):
            return _row(key, setting_name=traj.setting, turns=traj.turns_used,
                        stop=traj.stop_reason,
                        transcript_json=json.dumps(traj.transcript.to_records(), ensure_ascii=False),
                        meta=keymeta[key])

        def on_error(key, exc):
            return _row(key, setting_name="", turns=0, stop=f"ERROR: {exc}",
                        transcript_json="[]", meta=keymeta[key])

        results = drive_generators(
            gens, router=router, finalize=finalize, on_error=on_error,
            verbose=verbose,
            progress=f"conversations {start // batch_size + 1}/{n_chunks}" if verbose else None,
        )
        rows = [results[(iid, it)] for iid, it, _t, _m in chunk]
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        # Ack only after the CSV append (crash-safe), carrying stop_reason for context.
        ledger.record([
            {"input_id": r["input_id"], "iteration": r["iteration"], "stop_reason": r["stop_reason"]}
            for r in rows
        ])
        if verbose:
            logger.info("[%d/%d] wrote %d conversations", start + len(rows), n, len(rows))

    return pd.read_csv(out) if out.exists() else pd.DataFrame()
