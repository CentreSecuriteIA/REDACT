"""The conversation stage: plan, drive the units in chunks, save.

A unit is one ``(input_id, iteration)``: the Setting played once on one seed.
Its row in ``conversations.csv`` holds the conversation's step log as JSON.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from itertools import islice, repeat
from math import ceil
from pathlib import Path

import pandas as pd

from redact import paths
from redact.dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    content_id,
    read_rows,
    release_lock,
    resume_state,
    row_ids,
)
from redact.llm_pipeline import drive_generators

from .conversation import Setting, conversation_gen

logger = logging.getLogger(__name__)

#TODO(pool): move onto llm_pipeline.drive_units, like outputs and jailbreaks.
# This stage still drives fixed chunks.


def _seed_text_col(df: pd.DataFrame) -> str:
    for c in ("seed", "sample", "prompt", "text"):
        if c in df.columns:
            return c
    raise ValueError("seeds must have a 'seed' / 'sample' / 'prompt' / 'text' column.")


def _ledger(out: Path) -> Ledger:
    """The ledger of conversation units, keyed ``(input_id, iteration)``."""
    return Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                          casters={"input_id": str, "iteration": int})


def generate_conversations(
    seeds: pd.DataFrame,
    setting: Setting | Callable[[], Setting],
    data_dir: str | Path | None = None,
    iterations: int = 1,
    resume: bool = True,
    batch_size: int = 256,
    output_path: str | Path | None = None,
    verbose: bool = True,
    resolve=None,
) -> pd.DataFrame:
    """Play ``setting`` on every seed ``iterations`` times and save the conversations.

    Args:
        seeds: The opening messages, from a ``seed``, ``sample``, ``prompt`` or
            ``text`` column. The id is ``sample_id`` when present, else the
            content id of the text; ``category`` and ``entry_type`` are copied
            to the rows.
        setting: A Setting, deep-copied for each conversation, or a factory
            that returns a new one; prefer the factory when an actor holds state.
        iterations: Conversations per seed, numbered from 0.
        batch_size: Conversations driven together per chunk.
        resolve: ``model name -> ModelClient``; defaults to ``ModelClient.create``.

    Returns:
        Every row of ``conversations.csv``.
    """
    if seeds is None or len(seeds) == 0:
        raise ValueError("generate_conversations needs a non-empty `seeds` DataFrame.")
    seeds = seeds.reset_index(drop=True)
    tcol = _seed_text_col(seeds)

    out = Path(output_path) if output_path else paths.conversations_csv(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    ledger, manifest = _ledger(out), Manifest.sidecar(out)

    make_setting = setting if callable(setting) else (lambda: copy.deepcopy(setting))

    # Before the manifest is written: the resume takes the stage's lock, and
    # cuts off the rows of a save whose acks did not land.
    completed = resume_state(ledger, resume=resume, artifact=out, manifest=manifest)

    ids = row_ids(seeds, tcol)

    def _units():
        """The run's units in plan order, ``(input_id, iteration, seed_text, meta)``.

        Built anew on every call, so no list of them is held.
        """
        blank = repeat("")      # for seeds without the column
        categories = seeds.get("category", blank)
        entry_types = seeds.get("entry_type", blank)
        for iid, stext, cat, et in zip(ids, seeds[tcol], categories, entry_types):
            meta = {"category": str(cat), "entry_type": str(et)}
            for it in range(iterations):
                yield iid, it, str(stext), meta

    manifest.write(
        {"input_id": iid, "iteration": it, **meta} for iid, it, _t, meta in _units())

    planned = n = 0     # units of the run, and those not finished
    for iid, it, *_ in _units():
        planned += 1
        n += (iid, it) not in completed
    if verbose:
        logger.info("Generate Conversations")
        logger.info("Seeds: %d | iterations: %d | units: %d | pending: %d",
                    len(seeds), iterations, planned, n)
    if not n:
        release_lock(ledger)    # no CommitBuffer will
        return read_rows(out)

    pending = (u for u in _units() if u[:2] not in completed)
    n_chunks = ceil(n / batch_size) if batch_size else 1

    def _row(key, *, setting_name, turns, stop, transcript_json, meta):
        return {
            "sample_id": content_id(f"{key[0]}:{key[1]}"),
            "input_id": key[0], "iteration": key[1], "setting": setting_name,
            "turns_used": turns, "stop_reason": stop, "transcript": transcript_json,
            "category": meta["category"], "entry_type": meta["entry_type"],
            "source": "conversation",
        }

    # One save per chunk, at the flush below: ``every`` is not reached before it.
    with CommitBuffer(
        append=lambda rows: append_rows(out, rows), ledger=ledger, every=n
    ) as buf:
        for start in range(0, n, batch_size):
            chunk = list(islice(pending, batch_size))
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
                gens, resolve=resolve, finalize=finalize, on_error=on_error,
                progress=f"conversations {start // batch_size + 1}/{n_chunks}" if verbose else None,
            )
            rows = [results[(iid, it)] for iid, it, _t, _m in chunk]
            # The flush appends the chunk's rows, then acks them.
            for r in rows:
                buf.add([r], {"input_id": r["input_id"], "iteration": r["iteration"],
                              "stop_reason": r["stop_reason"]})
            buf.flush()
            if verbose:
                logger.info("[%d/%d] wrote %d conversations", start + len(rows), n, len(rows))

    return read_rows(out)
