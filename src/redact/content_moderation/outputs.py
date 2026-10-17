"""The outputs stage: one checked model response per input row, each dealt
one spec of the response axes."""

import functools
import logging
from pathlib import Path

import pandas as pd

from ..dataset import (
    Ledger,
    Manifest,
    append_rows,
    check_columns,
    content_id,
    read_rows,
    row_ids,
)
from ..dataset.planfile import changed_plan, plan_path, plan_record, write_plan
from ..dataset.resume import release_lock, resume_state
from ..dataset.save import CommitBuffer
from ..exceptions import PromptTooLongError
from ..llm_pipeline import checked, drive_units, pool_size
from ..llms.client import ModelClient
from ..llms.prompting import build_messages, load_prompt
from .axes import (
    deal_specs,
    deal_weights,
    render_spec,
    spec_columns,
    warn_unshown,
    write_stage_coverage,
)
from .checker import build_output_quality_checker
from .feedback import with_feedback

logger = logging.getLogger(__name__)

# The prefix of a row's column for each response axis.
_PREFIX = "response_"


def _stop_on_an_earlier_file(out_path: Path, columns: list[str]) -> None:
    """Raise when ``out_path`` has a header without one of ``columns``: it was
    written before the response axes, or without them.

    Raises:
        ValueError: The column missing, and the remedies.
    """
    try:
        check_columns(out_path, [dict.fromkeys(columns, "")])
    except ValueError as error:
        raise ValueError(
            f"{error} Start the stage over with resume=False (a recipe's "
            '"resume": false), run it in a new data_dir, or run it without the '
            "response axes (params outputs.axes: false).") from None


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
    axes: dict | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate one checked response per input row, appended to ``out_path``.

    Each row is a ``llm_pipeline.checked`` unit with its own entry-type-aware
    output checker: a refusal of a harmful input is rejected, one of a benign
    input is judged as any reply. With ``max_attempts > 1`` a rejected
    response is regenerated with the verdict as feedback, and the row holds
    the last attempt. An input whose prompt is too long for the model gets a
    rejected row with an empty response. A call carries up to ``batch_size``
    requests; ``check_max_tokens`` limits a verdict's reply.

    ``axes`` (:func:`redact.content_moderation.axes.resolve`) deals each
    row one spec per ``(category, entry_type)`` from ``seed``; the
    template's ``{response_spec}`` shows it, the row records it in a
    ``response_<axis>`` column per axis, and ``<stem>.coverage.csv`` beside
    the manifest counts the specs planned and accepted. The checker is not
    shown the spec. ``None``: no specs, and ``{response_spec}`` is empty.

    A row's ``input_id`` is its input's id (``row_ids``); its ``sample_id``
    is ``content_id`` of the response, or of ``"{input_id}:prompt too long"``
    for a prompt too long, which has none. The ledger beside ``out_path`` is
    keyed on ``input_id``, an ack holding the verdict too; ``resume=False``
    starts the stage over. A resume deals the specs again, and
    ``<stem>.plan.json`` (``seed``, the axes' weights) stops one that would
    deal others; a file without the response columns stops it before any
    call.

    Returns:
        The rows of ``out_path``, from this run and earlier ones, or an empty
        frame when ``read_back`` is False.

    Raises:
        ValueError: A resume whose ``seed`` or axes' weights differ from those
            of the plan file, or onto a file without the response columns.
    """
    # Internals capture (opt-in): only true for an introspection-capable
    # backend (e.g. TransformersIntrospectionBackend); a no-op for the default
    # Venice/vLLM path — no internals_ids kwarg is passed below. Captures land
    # under {log_dir}/{input_id}/output/ and .../val_out/ — a pure side channel,
    # never a CSV column (see .claude/introspection_backend_plan.md).
    capture_internals_gen = client.compute_config.supports_internals
    prompt_config = load_prompt("output", "generation", prompt_dir=prompt_dir)
    if axes:
        warn_unshown(prompt_config, "response_spec", "of the outputs' generation")
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

    made_with = plan_record(seed=seed, axes=deal_weights(axes))
    changed = changed_plan(manifest.path, made_with) if resume else []
    try:
        if changed:
            raise ValueError(
                f"Outputs: this run's {changed} differ from those the responses "
                f"in {out_path.name} were dealt with (see "
                f"{plan_path(manifest.path).name}), and a resume deals them again "
                "as they were. Run with resume=False to start over, or with the "
                "settings the plan was made with.")
        if resume and out_path.exists():
            _stop_on_an_earlier_file(out_path, list(spec_columns(
                axes, {}, prefix=_PREFIX)))
    except ValueError:
        release_lock(ledger)
        raise
    if changed is None and completed:
        logger.warning(
            "Outputs: %s has finished rows and no %s, so whether this run's seed "
            "and axes are the ones they were dealt with is not checked.",
            out_path.name, plan_path(manifest.path).name)
    write_plan(manifest.path, made_with)

    groups = [(str(row.get("category", "unknown")),
               str(row.get("entry_type", "harmful")))
              for _, row in inputs.iterrows()]
    specs = deal_specs(
        axes, [(group, (sid, 0)) for group, sid in zip(groups, inputs["_state_id"])],
        seed=seed)
    # Plan → manifest: one row per input_id, written before the batch loop so the
    # run's full intended scope is inspectable up front (keyed like the ledger).
    manifest.write(
        {"input_id": sid, "category": category, "entry_type": entry_type,
         **({"spec": spec} if axes else {}), "status": "planned"}
        for sid, (category, entry_type), spec in zip(inputs["_state_id"], groups, specs)
    )

    if completed:
        before = len(inputs)
        pending = ~inputs["_state_id"].isin(completed)
        specs = [spec for spec, keep in zip(specs, pending) if keep]
        inputs = inputs[pending].reset_index(drop=True)
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
            prompt_config, input_prompt=input_text, Category=category,
            response_spec=render_spec(axes, specs[i], {"subcategory": subcategory}),
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
    def _on_done(i, value, sample_id=None):
        resp, accepted, reasoning = value
        r = live_rows.pop(i)
        r["sample_id"] = sample_id or content_id(resp)
        r["output_response"] = resp
        r["accepted"] = accepted
        r["rejection_reason"] = reasoning
        r.update(spec_columns(axes, specs[i], prefix=_PREFIX))
        # The verdict, what the coverage table counts.
        buf.add([r], {"input_id": r["input_id"], "accepted": bool(accepted)})

    # The row of a prompt that is too long is acked like any other: sent
    # again, the prompt would be too long again. Any other unit error stops
    # the stage.
    def _on_error(i, exc):
        if not isinstance(exc, PromptTooLongError):
            raise exc
        input_id = live_rows[i]["input_id"]
        logger.warning("Input %s rejected, its prompt is too long: %s",
                       input_id, exc)
        # No response to hash: the id comes from the input's, so two such
        # rows never share one.
        _on_done(i, ("", False, f"prompt too long: {exc}"),
                 sample_id=content_id(f"{input_id}:prompt too long"))

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
    write_stage_coverage(axes, manifest, ledger)

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
