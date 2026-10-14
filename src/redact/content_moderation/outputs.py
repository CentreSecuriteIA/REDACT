"""The outputs stage: one checked model response per input row."""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ..dataset import Ledger, Manifest, append_rows, content_id, read_rows, row_ids
from ..dataset.resume import resume_state
from ..dataset.save import CommitBuffer
from ..exceptions import PromptTooLongError
from ..llm_pipeline import checked, drive_units
from ..llms.client import ModelClient
from ..llms.prompting import build_messages, load_prompt
from .checker import build_output_quality_checker
from .feedback import with_feedback
from .inputs import _pool_size

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
    """Generate model responses for input samples, batched and quality-checked.

    Pipeline:
      1. One generate-check unit per row (``llm_pipeline.checked``), with the
         row's own entry-type-aware output checker. Refusals on harmful
         inputs are rejected; refusals on benign inputs are evaluated normally.
      2. ``drive_units`` keeps ``2 * batch_size`` units live, or as many as
         the widest async lane of the two models holds when that is more,
         and sends calls of up to ``batch_size``. With ``max_attempts > 1``
         rejected rows go round again, regenerated with the checker's
         verdict as feedback, in the same calls as new rows.
      3. Finished rows are appended to the output CSV every 32 units, in the
         order they finish — crash-resilient.

    An input whose prompt is longer than the model's context (a
    ``PromptTooLongError`` from a request on the async lane) gets a rejected
    row with an empty response and the ``rejection_reason``
    ``"prompt too long: <message>"``.

    Resume: each input gets a stable content-hash id (its ``id``/``sample_id``
    column when present, else ``content_id(prompt)``). Completed ids are
    recorded in a sidecar ``*.state.jsonl`` ledger beside ``out_path``. On a
    re-run (``resume=True``) inputs already in the ledger are skipped, so a
    crash loses only the live units. ``resume=False`` clears both the CSV and
    the ledger first.

    Args:
        inputs: Input samples DataFrame — already loaded/capped by the caller.
        client: Generation model, bound to its transport.
        check_outputs: Run the entry-type-aware output quality checker.
            Set False to skip checking (accept everything).
        check: Checker model, or None when ``check_outputs`` is False.
        batch_size: Inputs per engine pass.
        out_path: Resolved output CSV path.
        prompt_dir: Root prompt directory for the generation **and**
            output-check prompts. ``None`` falls back to the bundled ``prompts/``.
        resume: When True (default), skip inputs already recorded in the sidecar
            state ledger. When False, clear the output CSV and ledger first.
        verbose: Print per-batch progress.
        max_attempts: Generations per input. With more than 1, a rejected
            response is regenerated with the checker's verdict as feedback
            and the row holds the last attempt.
        check_max_tokens: Reply limit for the check calls; a verdict needs
            few tokens. ``None`` keeps the checker model's default.
        read_back: False returns an empty DataFrame and does not read the
            output CSV.

    Returns:
        DataFrame of all rows in the output CSV (the full dataset, including
        rows from prior resumed runs) with the unified output schema:
        ``input_id``, ``sample_id``, ``input_prompt``, ``category``,
        ``subcategory``, ``entry_type``, ``output_response``, ``accepted``,
        ``rejection_reason``, ``model``, ``source``. ``sample_id`` is this
        row's own content-hash identity (``content_id(output_response)``),
        distinct from ``input_id`` (the origin sample it responds to).
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
    # non-resume run wipes prior output + ledger; a resume skips the ids the
    # ledger records, and cuts off the rows of a save whose acks did not land:
    # their inputs run again, and no input ends with two rows.
    completed = resume_state(ledger, resume=resume, artifact=out_path)

    # Plan → manifest: one row per input_id, written before the batch loop so the
    # run's full intended scope is inspectable up front (keyed like the ledger).
    Manifest.sidecar(out_path).write(
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

    # Per-(category, entry_type) checker cache so we build each prompt once.
    checker_cache: dict[tuple[str, str], Callable[[str, str], list[dict]]] = {}

    def _checker_for(category: str, entry_type: str):
        key = (category, entry_type)
        if key not in checker_cache:
            checker_cache[key] = build_output_quality_checker(
                category=category, entry_type=entry_type, prompt_dir=prompt_dir
            )
        return checker_cache[key]

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
            lambda feedback: with_feedback(base, feedback, "output"),
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
            pool_size=_pool_size(batch_size, client, check),
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
