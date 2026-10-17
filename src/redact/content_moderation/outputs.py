"""The outputs stage: one checked model response per input row."""

import functools
import logging
from pathlib import Path

import pandas as pd

from ..dataset import Ledger, Manifest, append_rows, content_id, read_rows, row_ids
from ..dataset.resume import resume_state
from ..dataset.save import CommitBuffer
from ..exceptions import PromptTooLongError
from ..llm_pipeline import checked, drive_units, pool_size
from ..llms.client import ModelClient
from ..llms.prompting import build_messages, load_prompt
from .checker import build_output_quality_checker
from .feedback import with_feedback

logger = logging.getLogger(__name__)


def run_output_generation(
    inputs: pd.DataFrame,
    client: ModelClient,
    check_outputs: bool,
    check: ModelClient | None,
    batch_size: int,
    out_path: Path,
    prompt_dir: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    max_attempts: int = 1,
    check_max_tokens: int | None = None,
    read_back: bool = True,
) -> pd.DataFrame:
    """Generate one checked response per input row, appended to ``out_path``.

    Each row is a ``llm_pipeline.checked`` unit with its own entry-type-aware
    output checker: a refusal of a harmful input is rejected, one of a benign
    input is judged as any reply. With ``max_attempts > 1`` a rejected
    response is regenerated with the verdict as feedback, and the row holds
    the last attempt. An input whose prompt is too long for the model gets a
    rejected row with an empty response. A call carries up to ``batch_size``
    requests; ``check_max_tokens`` limits a verdict's reply.

    A row's ``input_id`` is its input's id (``row_ids``); its ``sample_id``
    is ``content_id`` of the response. The ledger beside ``out_path`` is
    keyed on ``input_id``; ``resume=False`` starts the stage over.

    Returns:
        The rows of ``out_path``, from this run and earlier ones, or an empty
        frame when ``read_back`` is False.
    """
    # Internals capture (opt-in): only true for an introspection-capable
    # backend (e.g. TransformersIntrospectionBackend); a no-op for the default
    # Venice/vLLM path — no internals_ids kwarg is passed below. Captures land
    # under {log_dir}/{input_id}/output/ and .../val_out/ — a pure side channel,
    # never a CSV column (see .claude/introspection_backend_plan.md).
    capture_internals_gen = client.compute_config.supports_internals
    prompt_config = load_prompt("output", "generation", prompt_dir=prompt_dir)
    capture_internals_check = (
        check is not None and check.compute_config.supports_internals
    )

    text_col = "sample" if "sample" in inputs.columns else "prompt"
    inputs = inputs.reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.sidecar(out_path, key_fields=("input_id",), casters={"input_id": str})

    # Stable per-input id (content-hash) for resume — matches the jailbreak id.
    # The reset above gave this stage its own frame, so the column stays here.
    inputs["_state_id"] = row_ids(inputs, text_col)

    # Before the manifest is written: the resume takes the stage's lock. A
    # non-resume run wipes prior output, ledger and plan; a resume skips the ids the
    # ledger records, and cuts off the rows of a save whose acks did not land:
    # their inputs run again, and no input ends with two rows.
    manifest = Manifest.sidecar(out_path)
    completed = resume_state(
        ledger, resume=resume, artifact=out_path, manifest=manifest)

    # Plan → manifest: one row per input_id, written before the batch loop so the
    # run's full intended scope is inspectable up front (keyed like the ledger).
    manifest.write(
        {"input_id": sid, "status": "planned"} for sid in inputs["_state_id"]
    )

    if completed:
        before = len(inputs)
        inputs = inputs[~inputs["_state_id"].isin(completed)].reset_index(drop=True)
        if verbose and before != len(inputs):
            logger.info("Resume: skipping %d already-completed inputs", before - len(inputs))

    if verbose:
        logger.info("Generate Output Responses")
        logger.info("Samples: %d | Model: %s | Checker: %s | Batch: %d",
                    len(inputs), client.model,
                    f"enabled ({check.model})" if check_outputs and check else "disabled",
                    batch_size)

    @functools.cache
    def _checker_for(category: str, entry_type: str):
        return build_output_quality_checker(
            category=category, entry_type=entry_type, prompt_dir=prompt_dir
        )

    written = 0

    def _append_rows(rows: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(out_path, rows)
        written += len(rows)
        if verbose:
            accepted_count = sum(1 for r in rows if r["accepted"])
            logger.info("[%d/%d] %d/%d accepted -> appended to %s",
                        written, len(inputs), accepted_count, len(rows), out_path.name)
        return sizes

    use_checker = check_outputs and check is not None
    live_rows: dict[int, dict] = {}   # rows of live units, by position in inputs

    # One generate-check unit per row, keyed by position, so two rows with one
    # input_id stay two units.
    def _unit(i, row):
        input_text = row[text_col]
        category = str(row.get("category", "unknown"))
        entry_type = str(row.get("entry_type", "harmful"))
        subcategory = str(row.get("subcategory", ""))
        input_id = str(row["_state_id"])

        base = build_messages(
            prompt_config, input_prompt=input_text, Category=category
        )
        live_rows[i] = {
            "input_id": input_id,
            "input_prompt": input_text,
            "category": category,
            "subcategory": subcategory,
            "entry_type": entry_type,
            "model": client.model,
            "source": str(row.get("source", "")),
        }
        return i, checked(
            client,
            lambda feedback: with_feedback(base, feedback),
            check if use_checker else None,
            _checker_for(category, entry_type) if use_checker else None,
            original=input_text,
            internals_ids=(
                f"{input_id}/output" if capture_internals_gen else None,
                f"{input_id}/val_out" if capture_internals_check else None,
            ),
            max_tokens=(None, check_max_tokens),
            max_attempts=max_attempts,
        )

    # A finished unit's row. sample_id is this row's own content-hash identity
    # (distinct from input_id, which points back to the origin sample) — only
    # knowable once the response text exists.
    def _on_done(i, value):
        resp, accepted, reasoning = value
        r = live_rows.pop(i)
        r["sample_id"] = content_id(resp)
        r["output_response"] = resp
        r["accepted"] = accepted
        r["rejection_reason"] = reasoning
        buf.add([r], {"input_id": r["input_id"]})

    # The row of a prompt that is too long is acked like any other: sent
    # again, the prompt would be too long again. Any other unit error stops
    # the stage.
    def _on_error(i, exc):
        if not isinstance(exc, PromptTooLongError):
            raise exc
        logger.warning("Input %s rejected, its prompt is too long: %s",
                       live_rows[i]["input_id"], exc)
        _on_done(i, ("", False, f"prompt too long: {exc}"))

    # Generation and check calls, topped up from the source as units finish.
    # The buffer appends finished rows, then acks them (see dataset/save.py),
    # every 32 units and on any exit.
    with CommitBuffer(append=_append_rows, ledger=ledger) as buf:
        drive_units(
            (_unit(i, row) for i, (_, row) in enumerate(inputs.iterrows())),
            on_done=_on_done,
            on_error=_on_error,
            pool_size=pool_size(batch_size, client, check),
            cap=batch_size,
            progress="outputs" if verbose else None,
        )

    if not read_back:
        return pd.DataFrame()

    # Return the full dataset on disk (includes rows from prior resumed runs),
    # not just this run's newly-written delta.
    output_df = read_rows(out_path)
    if verbose:
        accepted_total = (
            int(output_df["accepted"].sum())
            if not output_df.empty and "accepted" in output_df.columns
            else 0
        )
        logger.info("Output CSV now holds %d responses (%d accepted) at %s",
                    len(output_df), accepted_total, out_path)

    return output_df
