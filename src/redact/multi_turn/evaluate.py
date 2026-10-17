"""The scoring stage: a judge over saved conversations.

The criterion is the judge's system prompt, so one stage scores any use case.
Writes ``conversations_scored.csv``, one row per conversation ``sample_id``.
"""

from __future__ import annotations

import json
import logging
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
from redact.exceptions import PromptTooLongError
from redact.llm_pipeline import LLMRequest, drive_units, is_accepted, pool_size
from redact.llms import ModelClient
from redact.llms.prompting import PromptTemplate

from .transcript import Transcript

logger = logging.getLogger(__name__)


def judge_prompt(system: str | None = None, prompt_dir=None) -> PromptTemplate:
    """The judge's prompt, called with ``goal``, ``label`` and ``sample``.

    ``label`` names the sample: ``"RESPONSE"`` for a reply, ``"CONVERSATION"``
    for a whole conversation. ``system`` replaces the criterion, as written;
    the prompt is read through the overlay ``prompt_dir``.
    """
    prompt = PromptTemplate.load("conversation", "judge", prompt_dir=prompt_dir)
    if system:
        prompt.system_prompt = system
    return prompt


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
    read_back: bool = True,
    prompt_dir: str | Path | None = None,
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
        batch_size: Judgments per model call.
        conversations_path: The conversations file; defaults to
            ``paths.conversations_csv(data_dir)``. The scores are saved beside it.
        resolve: ``model name -> ModelClient``; defaults to ``ModelClient.create``.
        prompt_dir: The prompt overlay the judge's prompt is read through.

    Returns:
        Every row of ``conversations_scored.csv``, or an empty frame when
        ``read_back`` is False.
    """
    if judge_model is None:
        raise ValueError("evaluate_conversations requires a judge_model.")
    if scope not in ("last_reply", "transcript"):
        raise ValueError("scope must be 'last_reply' or 'transcript'.")

    if conversations_path:
        conv_path = Path(conversations_path)
        scored_path = conv_path.with_name(paths.CONVERSATIONS_SCORED_FILENAME)
    else:
        conv_path = paths.conversations_csv(data_dir)
        scored_path = paths.conversations_scored_csv(data_dir)
    if conversations is None:
        if not conv_path.exists():
            raise ValueError(f"No conversations found at {conv_path}. Run generate_conversations first.")
        conversations = read_rows(conv_path)
    conversations = conversations.reset_index(drop=True)

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.sidecar(scored_path, key_fields=("sample_id",),
                            casters={"sample_id": str})
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
        return read_rows(scored_path) if read_back else pd.DataFrame()

    judge_messages = judge_prompt(judge_system, prompt_dir)
    label = "RESPONSE" if scope == "last_reply" else "CONVERSATION"
    resolve = resolve or ModelClient.create
    judge = resolve(judge_model)
    n, written = len(pending), 0

    def _judgment(messages: list[dict]):
        """One judge call: ``(True, "")`` on a yes, else ``(False, reply)``."""
        reply = yield LLMRequest.for_client(judge, messages)
        return (True, "") if is_accepted(reply) else (False, reply)

    # The sample_id of each live unit, by its row in ``pending``.
    live: dict[int, str] = {}

    def _units():
        for i, (_, row) in enumerate(pending.iterrows()):
            transcript = _transcript(str(row.get("transcript", "")))
            sample = (transcript.last_reply() if scope == "last_reply"
                      else transcript.render())
            live[i] = str(row["sample_id"])
            messages = judge_messages(
                goal=transcript.first_message(), label=label, sample=sample)
            yield i, _judgment(messages)

    def _on_done(i, verdict):
        success, reasoning = verdict
        sample_id = live.pop(i)
        row = {"sample_id": sample_id, "success": success, "judge_reasoning": reasoning,
               "judge_model": judge_model, "scope": scope}
        buf.add([row], {"sample_id": sample_id})

    # A transcript too long for the judge is scored failed and acked: sent
    # again it would be too long again. Any other unit error stops the stage.
    def _on_error(i, exc):
        if not isinstance(exc, PromptTooLongError):
            raise exc
        logger.warning("Conversation %s not scored, too long for the judge: %s",
                       live[i], exc)
        _on_done(i, (False, f"prompt too long: {exc}"))

    def _append(rows: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(scored_path, rows)
        written += len(rows)
        if verbose:
            logger.info("[%d/%d] scored (%d success)",
                        written, n, sum(r["success"] for r in rows))
        return sizes

    # The buffer appends finished rows, then acks them, every 32 units and on
    # any exit.
    with CommitBuffer(append=_append, ledger=ledger) as buf:
        drive_units(
            _units(),
            on_done=_on_done,
            on_error=_on_error,
            pool_size=pool_size(batch_size, judge),
            cap=batch_size,
            progress="judge" if verbose else None,
        )

    return read_rows(scored_path) if read_back else pd.DataFrame()
