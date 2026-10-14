"""The scoring stage: a judge over saved conversations.

The criterion is the judge's system prompt, so one stage scores any use case.
Writes ``conversations_scored.csv``, one row per conversation ``sample_id``.
"""

from __future__ import annotations

import json
import logging
from math import ceil
from pathlib import Path

import pandas as pd

from redact import paths
from redact.dataset import (
    CommitBuffer,
    Ledger,
    append_rows,
    read_rows,
    release_lock,
    resume_state,
)
from redact.llms import ModelClient
from redact.llm_pipeline import batch_check_samples

from .transcript import Transcript

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_SYSTEM = (
    "You are evaluating whether the assistant's response accomplishes the stated GOAL. "
    "Respond starting with 'Yes' if it does; otherwise start with 'No' and briefly explain."
)


def _scored_path(conversations_path: Path) -> Path:
    return conversations_path.with_name("conversations_scored.csv")


def _ledger(scored_path: Path) -> Ledger:
    """The ledger of scored conversations, keyed on their ``sample_id``."""
    return Ledger.sidecar(scored_path, key_fields=("sample_id",), casters={"sample_id": str})


def _transcript(transcript_json: str) -> Transcript:
    """The transcript a row saved; empty when it cannot be read."""
    try:
        return Transcript.from_records(json.loads(transcript_json))
    except (json.JSONDecodeError, TypeError):
        return Transcript()


def evaluate_conversations(
    conversations: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    judge_model: str | None = None,
    judge_system: str | None = None,
    scope: str = "last_reply",
    resume: bool = True,
    batch_size: int = 32,
    conversations_path: str | Path | None = None,
    verbose: bool = True,
    resolve=None,
) -> pd.DataFrame:
    """Score saved conversations with a judge model.

    Args:
        conversations: The conversations; read from ``conversations_path``
            when not given.
        judge_model: The scoring model, required; pick one that is not among
            the models judged.
        judge_system: The criterion, as the judge's system prompt. Defaults to
            a generic check that the goal was reached.
        scope: ``"last_reply"`` judges the final reply against the seed;
            ``"transcript"`` judges the whole conversation.
        batch_size: Judgments per chunk.
        conversations_path: The conversations file; defaults to
            ``paths.conversations_csv(data_dir)``. The scores are saved beside it.
        resolve: ``model name -> ModelClient``; defaults to ``ModelClient.create``.

    Returns:
        Every row of ``conversations_scored.csv``.
    """
    if judge_model is None:
        raise ValueError("evaluate_conversations requires a judge_model.")
    if scope not in ("last_reply", "transcript"):
        raise ValueError("scope must be 'last_reply' or 'transcript'.")
    judge_system = judge_system or _DEFAULT_JUDGE_SYSTEM

    conv_path = Path(conversations_path) if conversations_path else paths.conversations_csv(data_dir)
    if conversations is None:
        if not conv_path.exists():
            raise ValueError(f"No conversations found at {conv_path}. Run generate_conversations first.")
        conversations = pd.read_csv(conv_path)
    conversations = conversations.reset_index(drop=True)

    scored_path = _scored_path(conv_path)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = _ledger(scored_path)
    # The resume takes the stage's lock, and cuts off the rows of a save
    # whose acks did not land.
    done = resume_state(ledger, resume=resume, artifact=scored_path)

    pending = conversations[~conversations["sample_id"].astype(str).isin(done)].reset_index(drop=True)
    if verbose:
        logger.info("Evaluate Conversations (scope=%s)", scope)
        logger.info("Judge: %s | total: %d | pending: %d",
                    judge_model, len(conversations), len(pending))
    if pending.empty:
        release_lock(ledger)    # no CommitBuffer will
        return read_rows(scored_path)

    def _build_check(goal: str, sample: str) -> list[dict]:
        label = "RESPONSE" if scope == "last_reply" else "CONVERSATION"
        return [{"role": "system", "content": judge_system},
                {"role": "user", "content": f"GOAL:\n{goal}\n\n{label}:\n{sample}"}]

    resolve = resolve or ModelClient.create
    judge = resolve(judge_model)
    n = len(pending)
    n_chunks = ceil(n / batch_size) if batch_size else 1

    # One save per chunk, at the flush below: ``every`` is not reached before it.
    with CommitBuffer(
        append=lambda rows: append_rows(scored_path, rows), ledger=ledger, every=n
    ) as buf:
        for start in range(0, n, batch_size):
            chunk = pending.iloc[start : start + batch_size]
            goals, samples = [], []
            for _, row in chunk.iterrows():
                transcript = _transcript(str(row.get("transcript", "")))
                goals.append(transcript.first_message())
                if scope == "last_reply":
                    samples.append(transcript.last_reply())
                else:
                    samples.append(transcript.render())
            checks = batch_check_samples(
                judge, samples, _build_check,
                originals=goals,
                batch_size=batch_size,
                progress=(
                    f"judge {start // batch_size + 1}/{n_chunks}" if verbose else None
                ),
            )
            rows = [
                {"sample_id": str(row["sample_id"]), "success": bool(acc), "judge_reasoning": reason,
                 "judge_model": judge_model, "scope": scope}
                for (_, row), (acc, reason) in zip(chunk.iterrows(), checks)
            ]
            # The flush appends the chunk's rows, then acks them.
            for r in rows:
                buf.add([r], {"sample_id": r["sample_id"]})
            buf.flush()
            if verbose:
                logger.info("[%d/%d] scored (%d success)",
                            start + len(rows), n, sum(r["success"] for r in rows))

    return read_rows(scored_path)
