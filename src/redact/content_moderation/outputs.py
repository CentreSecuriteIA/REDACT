"""The outputs stage: one checked model response per input row, each dealt an
answer mode its input's entry type allows and one spec of the response axes.

The mode (:mod:`redact.content_moderation.modes`) picks the generation prompt
(``prompts/output/generation/<mode>/``) and the checker
(``prompts/output/quality_check/<mode>/``), and gives the row its label."""

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
from .modes import MODES, OUTPUT_LABEL, deal_modes, resolve_modes, response_label

logger = logging.getLogger(__name__)

# The prefix of a row's column for each response axis.
_PREFIX = "response_"

# The columns of a row that say how it answers its input.
_MODE_COLUMNS = ("response_mode", "output_label", "response_label")

# The entry type of an input without one.
_DEFAULT_ENTRY_TYPE = "harmful"


def _entry_type(cell) -> str:
    """An input's entry type: :data:`_DEFAULT_ENTRY_TYPE` for a missing, NaN
    or empty cell."""
    if cell is None or (pd.api.types.is_scalar(cell) and pd.isna(cell)):
        return _DEFAULT_ENTRY_TYPE
    text = str(cell)
    return text if text.strip() else _DEFAULT_ENTRY_TYPE


def _stop_on_an_earlier_file(out_path: Path, axes: dict | None) -> None:
    """Raise when ``out_path`` has a header without the answer-mode columns
    (it was written before the modes) or without a column of ``axes`` (it
    was written before the response axes, or without them).

    Raises:
        ValueError: The column missing, and the remedies.
    """
    remedy = ('Start the stage over with resume=False (a recipe\'s "resume": '
              "false), or run it in a new data_dir")
    for columns, written, other in (
        (list(_MODE_COLUMNS), "before the answer modes", ""),
        (list(spec_columns(axes, {}, prefix=_PREFIX)),
         "before the response axes, or without them",
         ", or run it without the response axes (params outputs.axes: false)"),
    ):
        try:
            check_columns(out_path, [dict.fromkeys(columns, "")])
        except ValueError as error:
            raise ValueError(
                f"{error} It was written {written}. {remedy}{other}.") from None


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
    modes: dict | None = None,
) -> pd.DataFrame:
    """Generate one checked response per input row, appended to ``out_path``.

    Each row is dealt an answer mode, ``harmful`` or ``helpful``, from those
    ``modes`` (:func:`redact.content_moderation.modes.resolve_modes`; ``None``
    the default) allows its input's entry type, per ``(category,
    entry_type)`` from ``seed``. The mode picks the generation prompt
    ``output/generation/<mode>`` and the checker ``output/quality_check/<mode>``,
    and gives the row ``response_mode``, ``output_label`` (``harmful`` or
    ``benign``) and ``response_label``, ``<entry_type>_<output_label>``. An
    input without an entry type is ``harmful``.

    Each row is a ``llm_pipeline.checked`` unit. With ``max_attempts > 1`` a
    rejected response is regenerated with the verdict as feedback, and the
    row holds the last attempt. An input whose prompt is too long for the
    model gets a rejected row with an empty response. A call carries up to
    ``batch_size`` requests; ``check_max_tokens`` limits a verdict's reply.

    ``axes`` (:func:`redact.content_moderation.axes.resolve`) deals each
    row one spec per ``(category, entry_type)`` from ``seed``; the
    template's ``{response_spec}`` shows it, the row records it in a
    ``response_<axis>`` column per axis. The checker is not shown the spec.
    ``None``: no specs, and ``{response_spec}`` is empty. ``<stem>.coverage.csv``
    beside the manifest counts the modes and the specs planned and accepted.

    A row's ``input_id`` is its input's id (``row_ids``); its ``sample_id``
    is ``content_id`` of the response, or of ``"{input_id}:prompt too long"``
    for a prompt too long, which has none. The ledger beside ``out_path`` is
    keyed on ``input_id``, an ack holding the verdict too; ``resume=False``
    starts the stage over. A resume deals the modes and the specs again, and
    ``<stem>.plan.json`` (``seed``, ``modes``, the axes' weights) stops one
    that would deal others; a file without the answer-mode or response
    columns stops it before any call.

    Returns:
        The rows of ``out_path``, from this run and earlier ones, or an empty
        frame when ``read_back`` is False.

    Raises:
        ValueError: An input of an entry type ``modes`` does not map, raised
            before anything is cleared; a resume whose ``seed``, ``modes`` or
            axes' weights differ from those of the plan file, or onto a file
            without the answer-mode or response columns.
    """
    # Internals capture (opt-in): only true for an introspection-capable
    # backend (e.g. TransformersIntrospectionBackend); a no-op for the default
    # Venice/vLLM path — no internals_ids kwarg is passed below. Captures land
    # under {log_dir}/{input_id}/output/ and .../val_out/ — a pure side channel,
    # never a CSV column (see .claude/introspection_backend_plan.md).
    capture_internals_gen = client.compute_config.supports_internals
    capture_internals_check = (
        check is not None and check.compute_config.supports_internals
    )
    use_checker = check_outputs and check is not None
    modes = resolve_modes(modes)

    text_col = "sample" if "sample" in inputs.columns else "prompt"
    inputs = inputs.reset_index(drop=True)
    rows = [row for _, row in inputs.iterrows()]
    # Stable per-input id (content-hash) for resume — matches the jailbreak id.
    sids = [str(sid) for sid in row_ids(inputs, text_col)]
    groups = [(str(row.get("category", "unknown")), _entry_type(row.get("entry_type")))
              for row in rows]
    unknown = sorted({entry_type for _, entry_type in groups} - set(modes))
    if unknown:
        raise ValueError(
            f"Outputs: the inputs hold the entry type(s) {unknown}, which "
            f"outputs.modes does not map; it maps {list(modes)}.")

    # One unit per input row, keyed by its position in ``units``, so two rows
    # with one input_id stay two units. Every unit is dealt its mode and its
    # spec, done or not: the deal is a function of the whole plan.
    units = [(i, 0) for i in range(len(rows))]
    keys = [(groups[i], (sids[i], iteration)) for i, iteration in units]
    unit_modes = deal_modes(modes, keys, seed=seed)
    specs = deal_specs(axes, keys, seed=seed)

    # The prompts of the modes dealt, read before anything is cleared.
    prompts = {}
    for mode in (mode for mode in MODES if mode in unit_modes):
        prompts[mode] = load_prompt(
            "output", f"generation/{mode}", prompt_dir=prompt_dir)
        if axes:
            warn_unshown(prompts[mode], "response_spec", f"output/generation/{mode}")
        if use_checker:
            load_prompt("output", f"quality_check/{mode}", prompt_dir=prompt_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.sidecar(
        out_path, key_fields=("input_id",), casters={"input_id": str})

    # Before the manifest is written: the resume takes the stage's lock. A
    # non-resume run wipes prior output, ledger and plan; a resume skips the ids the
    # ledger records, and cuts off the rows of a save whose acks did not land:
    # their inputs run again, and no input ends with two rows.
    manifest = Manifest.sidecar(out_path)
    completed = resume_state(
        ledger, resume=resume, artifact=out_path, manifest=manifest)

    made_with = plan_record(seed=seed, modes=modes, axes=deal_weights(axes))
    changed: list[str] | None = []
    try:
        # A file of an earlier shape first: its plan file, if any, lacks keys
        # no setting of this run can match.
        if resume and out_path.exists():
            _stop_on_an_earlier_file(out_path, axes)
        changed = changed_plan(manifest.path, made_with) if resume else []
        if changed:
            raise ValueError(
                f"Outputs: this run's {changed} differ from those the responses "
                f"in {out_path.name} were dealt with (see "
                f"{plan_path(manifest.path).name}), and a resume deals them again "
                "as they were. Run with resume=False to start over, or with the "
                "settings the plan was made with.")
    except ValueError:
        release_lock(ledger)
        raise
    if changed is None and completed:
        logger.warning(
            "Outputs: %s has finished rows and no %s, so whether this run's seed, "
            "modes and axes are the ones they were dealt with is not checked.",
            out_path.name, plan_path(manifest.path).name)
    write_plan(manifest.path, made_with)

    # Plan → manifest: one row per unit, written before the batch loop so the
    # run's full intended scope is inspectable up front (keyed like the ledger).
    manifest.write(
        {"input_id": sids[i], "category": category, "entry_type": entry_type,
         "mode": mode, **({"spec": spec} if axes else {}), "status": "planned"}
        for (i, _), ((category, entry_type), _), mode, spec
        in zip(units, keys, unit_modes, specs, strict=True)
    )

    pending = [u for u, (i, _) in enumerate(units) if sids[i] not in completed]
    if verbose and len(pending) != len(units):
        logger.info("Resume: skipping %d already-completed units",
                    len(units) - len(pending))

    if verbose:
        logger.info("Generate Output Responses")
        logger.info("Samples: %d | Model: %s | Checker: %s | Batch: %d",
                    len(pending), client.model,
                    f"enabled ({check.model})" if use_checker else "disabled",
                    batch_size)

    @functools.cache
    def _checker_for(category: str, mode: str):
        return build_output_quality_checker(
            category=category, mode=mode, prompt_dir=prompt_dir)

    written = 0

    def _append_rows(appended: list[dict]) -> dict[Path, int]:
        nonlocal written
        sizes = append_rows(out_path, appended)
        written += len(appended)
        if verbose:
            accepted_count = sum(1 for r in appended if r["accepted"])
            logger.info("[%d/%d] %d/%d accepted -> appended to %s",
                        written, len(pending), accepted_count, len(appended),
                        out_path.name)
        return sizes

    live_rows: dict[int, dict] = {}   # rows of live units, by unit index

    # One generate-check unit per unit index.
    def _unit(u):
        i, _ = units[u]
        row = rows[i]
        input_text = row[text_col]
        category, entry_type = groups[i]
        subcategory = str(row.get("subcategory", ""))
        input_id = sids[i]
        mode = unit_modes[u]

        base = build_messages(
            prompts[mode], input_prompt=input_text, Category=category,
            response_spec=render_spec(axes, specs[u], {"subcategory": subcategory}),
        )
        live_rows[u] = {
            "input_id": input_id,
            "input_prompt": input_text,
            "category": category,
            "subcategory": subcategory,
            "entry_type": entry_type,
            "model": client.model,
            "source": str(row.get("source", "")),
        }
        return u, checked(
            client,
            lambda feedback: with_feedback(base, feedback),
            check if use_checker else None,
            _checker_for(category, mode) if use_checker else None,
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
    def _on_done(u, value, sample_id=None):
        resp, accepted, reasoning = value
        r = live_rows.pop(u)
        mode = unit_modes[u]
        r["sample_id"] = sample_id or content_id(resp)
        r["output_response"] = resp
        r["accepted"] = accepted
        r["rejection_reason"] = reasoning
        r["response_mode"] = mode
        r["output_label"] = OUTPUT_LABEL[mode]
        r["response_label"] = response_label(r["entry_type"], mode)
        r.update(spec_columns(axes, specs[u], prefix=_PREFIX))
        # The verdict, what the coverage table counts.
        buf.add([r], {"input_id": r["input_id"], "accepted": bool(accepted)})

    # The row of a prompt that is too long is acked like any other: sent
    # again, the prompt would be too long again. Any other unit error stops
    # the stage.
    def _on_error(u, exc):
        if not isinstance(exc, PromptTooLongError):
            raise exc
        input_id = live_rows[u]["input_id"]
        logger.warning("Input %s rejected, its prompt is too long: %s",
                       input_id, exc)
        # No response to hash: the id comes from the input's, so two such
        # rows never share one.
        _on_done(u, ("", False, f"prompt too long: {exc}"),
                 sample_id=content_id(f"{input_id}:prompt too long"))

    # Generation and check calls, topped up from the source as units finish.
    # The buffer appends finished rows, then acks them (see dataset/save.py),
    # every 32 units and on any exit.
    with CommitBuffer(append=_append_rows, ledger=ledger) as buf:
        drive_units(
            (_unit(u) for u in pending),
            on_done=_on_done,
            on_error=_on_error,
            pool_size=pool_size(batch_size, client, check),
            cap=batch_size,
            progress="outputs" if verbose else None,
        )
    write_stage_coverage(axes, manifest, ledger, extra=("mode",))

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
