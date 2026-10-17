"""The conversation stage: plan, drive the units on the pool, save.

A unit is one ``(input_id, iteration)``: the Setting played once on one seed.
Its row in ``conversations.csv`` holds the conversation's step log as JSON.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from itertools import repeat
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
    resume_state,
    row_ids,
)
from redact.llm_pipeline import drive_units, pool_size
from redact.llms import ModelClient

from .actors import ModelActor
from .conversation import Setting, conversation_gen

logger = logging.getLogger(__name__)


def _seed_text_col(df: pd.DataFrame) -> str:
    for c in ("seed", "sample", "prompt", "text"):
        if c in df.columns:
            return c
    raise ValueError("seeds must have a 'seed' / 'sample' / 'prompt' / 'text' column.")


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
    read_back: bool = True,
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
        batch_size: Requests per model call.
        resolve: ``model name -> ModelClient``; defaults to ``ModelClient.create``.

    Returns:
        Every row of ``conversations.csv``, or an empty frame when
        ``read_back`` is False.
    """
    if seeds is None or len(seeds) == 0:
        raise ValueError("generate_conversations needs a non-empty `seeds` DataFrame.")
    seeds = seeds.reset_index(drop=True)
    tcol = _seed_text_col(seeds)

    out = Path(output_path) if output_path else paths.conversations_csv(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                            casters={"input_id": str, "iteration": int})
    manifest = Manifest.sidecar(out)

    make_setting = setting if callable(setting) else (lambda: copy.deepcopy(setting))
    # The clients of the models that reply, which size the pool. A vLLM client
    # loads its engine here, before the stage knows whether a unit is pending.
    resolve = resolve or ModelClient.create
    clients = [resolve(actor.model) for actor in make_setting().participants
               if isinstance(actor, ModelActor)]

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

    planned = n = written = 0     # units of the run, those not finished, rows saved
    if verbose:     # one pass over the plan, for the numbers in the log lines
        for iid, it, *_ in _units():
            planned += 1
            n += (iid, it) not in completed
        logger.info("Generate Conversations")
        logger.info("Seeds: %d | iterations: %d | units: %d | pending: %d",
                    len(seeds), iterations, planned, n)

    # (input_id, iteration, meta) of each live unit, by its place among the
    # pending units, so two seeds with one id stay two units.
    live: dict[int, tuple[str, int, dict]] = {}

    def _pending_units():
        pending = (u for u in _units() if u[:2] not in completed)
        for i, (iid, it, stext, meta) in enumerate(pending):
            live[i] = (iid, it, meta)
            yield i, conversation_gen(make_setting(), stext, input_id=iid)

    def _add(i, *, setting_name, turns, stop, transcript_json):
        iid, it, meta = live.pop(i)
        row = {
            "sample_id": content_id(f"{iid}:{it}"),
            "input_id": iid, "iteration": it, "setting": setting_name,
            "turns_used": turns, "stop_reason": stop, "transcript": transcript_json,
            "category": meta["category"], "entry_type": meta["entry_type"],
            "source": "conversation",
        }
        buf.add([row], {"input_id": iid, "iteration": it, "stop_reason": stop})

    def _on_done(i, traj):
        records = json.dumps(traj.transcript.to_records(), ensure_ascii=False)
        _add(i, setting_name=traj.setting, turns=traj.turns_used, stop=traj.stop_reason,
             transcript_json=records)

    # A unit that raises gets an error row and is acked.
    def _on_error(i, exc):
        _add(i, setting_name="", turns=0, stop=f"ERROR: {exc}", transcript_json="[]")

    def _append(rows: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(out, rows)
        written += len(rows)
        if verbose:
            logger.info("[%d/%d] wrote %d conversations", written, n, len(rows))
        return sizes

    # The buffer appends finished rows, then acks them, every 32 units and on
    # any exit.
    with CommitBuffer(append=_append, ledger=ledger) as buf:
        drive_units(
            _pending_units(),
            on_done=_on_done,
            on_error=_on_error,
            pool_size=pool_size(batch_size, *clients),
            cap=batch_size,
            resolve=resolve,
            progress="conversations" if verbose else None,
        )

    return read_rows(out) if read_back else pd.DataFrame()
