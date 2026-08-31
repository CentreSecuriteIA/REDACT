"""Content moderation input sample generation pipeline.

Generates multi-sample batches via LLM, extracts individual samples via
regex, checks each sample individually, and saves all samples (accepted
and rejected) incrementally to per-category CSVs.

This module owns the *active* constitution-seeded path
(``InputPipeline.run_from_constitution``) plus the shared core every mode
needs (``__init__``, ``_build_generation_messages``). The *deprecated*
standalone (meta-prompt) path — ``run_category``, ``run_standalone``, and
the ``generate_batch``/``check_samples``/``run_turn`` primitives only it
uses — lives in ``standalone_generation.py`` as a mixin
(:class:`~redact.content_moderation.standalone_generation._StandaloneGenerationMixin`)
that ``InputPipeline`` inherits from, so the two previously-fused paths are
now separately readable while every existing ``pipeline.run_category(...)``
/ ``pipeline.run_turn(...)`` call site keeps working unchanged. Result
dataclasses (``SampleResult``, ``TurnResult``, ``CategoryResult``,
``ConstitutionInputResult``) live in ``results.py`` so neither this module
nor ``standalone_generation.py`` has to import the other.

Usage (the active constitution-seeded path — prefer this over
``run_category``/``run_standalone`` in ``standalone_generation.py``, which
are deprecated):
    from redact.llms import ModelClient, load_prompt
    from redact.content_moderation import InputPipeline

    prompt_config = load_prompt("input", "generation/from_constitution/long")

    pipeline = InputPipeline(
        gen=ModelClient.create("venice-uncensored"),
        check=ModelClient.create("venice-uncensored"),
        dataset_dir="./Datasets/constitution_inputs",
    )
    result = pipeline.run_from_constitution(
        constitution_df=constitution_df,  # from ConstitutionPipeline
        prompt_config=prompt_config,
        samples_per_entry=3, style="long",
    )

Most callers should use the higher-level ``redact.generate_inputs(data_dir=...,
constitution_df=...)`` wrapper in ``pipelines.py`` instead of constructing
``InputPipeline`` directly — it resolves the backend/model from the registry
role and the dataset path from ``data_dir`` for you.
"""

import logging
from collections.abc import Callable
from math import ceil
from pathlib import Path

import pandas as pd

from ..dataset.io import _default_dataset_dir, _hash_text, append_samples
from ..dataset.ledger import Ledger
from ..dataset.manifest import Manifest
from ..dataset.merge import merge_all
from ..llms.client import ModelClient
from ..llms.extraction import extract_and_clean
from ..llms.prompts import build_messages, load_prompt
from ..llms.router import batch_generate_samples, is_accepted
from ..llms.wrappers import assert_single_sample_per_call
from .checker import build_output_quality_checker, build_quality_checker
from .results import CategoryResult, ConstitutionInputResult, SampleResult, TurnResult
from .standalone_generation import _StandaloneGenerationMixin

# CategoryResult/TurnResult aren't used in this file's own code (only by the
# standalone mixin, in standalone_generation.py) — re-exported here anyway
# since existing code (e.g. tests/content_moderation/test_generation.py)
# imports them from `redact.content_moderation.generation`, not `.results`.
__all__ = [
    "InputPipeline",
    "SampleResult",
    "TurnResult",
    "CategoryResult",
    "ConstitutionInputResult",
    "run_output_generation",
]

logger = logging.getLogger(__name__)

# Cap on existing samples shown in the "don't repeat these" prompt section.
_PROHIBITED_PREVIEW_LIMIT = 20
# Truncation length for the sample_description preview in verbose progress lines.
_DESC_PREVIEW_LEN = 60


def _constitution_inputs_ledger(dataset_dir: str | Path) -> Ledger:
    """Shared resume-ledger for constitution-seeded input generation.

    One JSON object per completed constitution entry, keyed on
    ``(sample_description, entry_type, style)`` and stored at
    ``{dataset_dir}/constitution_inputs.state.jsonl`` beside the per-category
    CSVs.
    """
    return Ledger(
        Path(dataset_dir) / "constitution_inputs.state.jsonl",
        key_fields=("sample_description", "entry_type", "style"),
        casters={"sample_description": str, "entry_type": str, "style": str},
    )


def _constitution_inputs_manifest(dataset_dir: str | Path) -> Manifest:
    """Run-plan for constitution-seeded input generation, at
    ``{dataset_dir}/constitution_inputs.manifest.jsonl`` beside the ledger. One
    row per constitution entry, keyed like the ledger
    (``sample_description``/``entry_type``/``style``)."""
    return Manifest(Path(dataset_dir) / "constitution_inputs.manifest.jsonl")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class InputPipeline(_StandaloneGenerationMixin):
    """Multi-sample content moderation input generation pipeline.

    Workflow per turn (standalone mode, see ``standalone_generation.py``):
    1. Build messages from prompt config + seed kwargs + format instruction
    2. Optionally inject prohibited outputs (already-generated samples)
    3. Call generator LLM (single request producing N samples)
    4. Extract individual samples via regex (numbered list by default)
    5. Exact-dedup against existing samples (skip duplicates before checking)
    6. Check each sample individually via checker LLM
    7. Save all samples (accepted + rejected) to category CSV with metadata
    8. Collect rejection feedback for next turn

    Constitution-seeded mode (``run_from_constitution``, below) batches
    generation + checking across many entries per LLM engine pass instead.
    """

    def __init__(
        self,
        gen: ModelClient,
        check: ModelClient,
        extraction_style: str = "numbered",
        dataset_dir: str | Path | None = None,
    ):
        """Create an InputPipeline.

        Args:
            gen: Generation model, bound to its transport.
            check: Checker model (may be the same client as ``gen``).
        """
        self.gen = gen
        self.check = check
        self.extraction_style = extraction_style
        self.dataset_dir = dataset_dir if dataset_dir is not None else _default_dataset_dir()

    def _build_generation_messages(
        self,
        prompt_config: dict,
        samples_per_request: int,
        feedback: str = "",
        prohibited: set[str] | None = None,
        **seed_kwargs: str,
    ) -> list[dict]:
        """Build the generation message list with format instruction.

        Args:
            prompt_config: Loaded prompt JSON config.
            samples_per_request: Number of samples to request per LLM call.
            feedback: Rejection feedback from a previous turn (optional).
            prohibited: Set of existing sample texts to avoid repeating.
            **seed_kwargs: Template variables (Category, SeedPrompts, etc.).

        Returns:
            Chat messages list ready for generate_sample().
        """
        # Build the base message list — format_style appends the matching
        # output-format instruction to system_prompt and is exactly the
        # style extract_and_clean() is later called with (self.extraction_style),
        # so the two can't independently drift.
        messages = build_messages(
            prompt_config,
            format_style=self.extraction_style,
            num_samples=str(samples_per_request),
            **seed_kwargs,
        )

        # Append prohibited outputs so the LLM avoids repeating them
        if prohibited:
            # Show a sample of existing outputs (cap to avoid prompt bloat)
            sample_list = list(prohibited)[:_PROHIBITED_PREVIEW_LIMIT]
            prohibited_text = "\n".join(f"- {s}" for s in sample_list)
            prohibited_msg = (
                f"\n\nIMPORTANT: The following samples already exist in the "
                f"dataset. Do NOT generate anything similar to these:\n"
                f"{prohibited_text}"
            )
            if len(prohibited) > _PROHIBITED_PREVIEW_LIMIT:
                prohibited_msg += (
                    f"\n(... and {len(prohibited) - _PROHIBITED_PREVIEW_LIMIT} more. "
                    f"Ensure all your outputs are novel.)"
                )
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += prohibited_msg
            else:
                messages.append({"role": "user", "content": prohibited_msg})

        # Append rejection feedback from prior turn
        if feedback:
            feedback_msg = (
                "\n\n[FEEDBACK FROM PREVIOUS ATTEMPT]\n"
                "Some previously generated samples were rejected for these "
                f"reasons:\n{feedback}\n"
                "Please avoid similar issues in your new samples."
            )
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += feedback_msg
            else:
                messages.append({"role": "user", "content": feedback_msg})

        return messages

    # ------------------------------------------------------------------
    # Constitution-seeded mode
    # ------------------------------------------------------------------

    def run_from_constitution(
        self,
        constitution_df: pd.DataFrame,
        prompt_config: dict,
        samples_per_entry: int = 3,
        use_checker: bool = True,
        save: bool = True,
        verbose: bool = True,
        batch_size: int = 32,
        fresh: bool = False,
        style: str = "",
    ) -> ConstitutionInputResult:
        """Run constitution-seeded input generation, batched across entries.

        Each row in ``constitution_df`` becomes a generation request. The
        per-entry feedback loop is handled inside the LLM checker — failed
        samples are saved with their rejection reason but no regeneration
        loop runs here (this mode prioritises throughput over per-sample
        retries; the standalone/deprecated path in
        ``standalone_generation.py`` is the one place in this library that
        does per-sample feedback-driven regeneration, hand-rolled rather
        than built on a shared retry primitive).

        Each batch of ``batch_size`` entries triggers exactly one
        ``batch_generate`` call (one vLLM engine pass) followed by one
        checker batch_generate. CSV append happens per batch, so a crash
        mid-run loses at most ``batch_size`` entries' worth of work.

        Args:
            constitution_df: DataFrame with columns ``source_category``,
                ``sample_description``, ``constitution_subcategory``,
                ``entry_type`` (matches the output of
                ``ConstitutionPipeline.to_dataframe()``).
            prompt_config: Loaded prompt JSON config from
                ``prompts/input/generation/from_constitution/{style}/``.
            samples_per_entry: Prompts to generate per constitution entry.
            use_checker: Whether to run the entry-type-aware quality checker.
            save: Whether to append rows to CSV per batch.
            verbose: Log per-entry progress (INFO for stage milestones, DEBUG
                for per-entry detail).
            batch_size: Entries per LLM engine pass.
            fresh: When True, clears this style's rows from the per-category
                CSVs and the ledger before generating (style-aware — other
                styles' saved rows/ledger entries are left alone).
            style: Template style under ``from_constitution/{style}`` — also
                used as part of the ledger key and saved as a CSV column.

        Returns:
            ConstitutionInputResult with generation statistics.
        """
        if constitution_df.empty:
            logger.warning("run_from_constitution: empty constitution_df, nothing to do")
            return ConstitutionInputResult()

        # Internals capture (opt-in, non-interfering — no CSV column, see
        # .claude/introspection_backend_plan.md). Generation asks for
        # samples_per_entry outputs in one completion, so one forward pass can't
        # be attributed to any single resulting sample unless there's exactly
        # one — assert_single_sample_per_call is a no-op unless the gen client
        # actually supports internals, in which case it raises if not.
        assert_single_sample_per_call(self.gen, samples_per_entry)
        capture_input = self.gen.compute_config.supports_internals
        # The checker dispatches one call per already-extracted sample (a flat
        # batch), so it's always attributable regardless of samples_per_entry —
        # no gate needed here.
        capture_val_in = use_checker and self.check.compute_config.supports_internals

        result = ConstitutionInputResult()

        # Sidecar resume-ledger, keyed on one constitution entry
        # (sample_description, entry_type, style) — the same crash-safe pattern
        # as the constitution / output / paraphrase stages. A unit is acked only
        # after its rows are saved, so a crash loses at most one entry.
        ledger = (
            _constitution_inputs_ledger(self.dataset_dir)
            if (save and self.dataset_dir is not None)
            else None
        )

        # Plan → manifest: one row per constitution entry (full set for this
        # style), keyed like the ledger, written before the batch loop.
        if save and self.dataset_dir is not None:
            _constitution_inputs_manifest(self.dataset_dir).write(
                {
                    "sample_description": str(entry.get("sample_description", "")),
                    "entry_type": str(entry.get("entry_type", "harmful")),
                    "style": style,
                    "status": "planned",
                }
                for _, entry in constitution_df.iterrows()
            )

        if fresh and self.dataset_dir is not None:
            # Whole-dir fresh wipes every unit's acks; style-aware fresh keeps
            # other styles' acks (they aren't being regenerated).
            if ledger is not None and not style:
                ledger.reset()
            for csv_path in Path(self.dataset_dir).glob("*/samples.csv"):
                if style:
                    # Style-aware fresh: only remove rows for this style so
                    # data from other styles in the same dir is preserved.
                    try:
                        df_existing = pd.read_csv(csv_path)
                        if "template_style" in df_existing.columns:
                            df_existing = df_existing[
                                df_existing["template_style"] != style
                            ].reset_index(drop=True)
                            if df_existing.empty:
                                csv_path.unlink()
                            else:
                                df_existing.to_csv(csv_path, index=False)
                        else:
                            csv_path.unlink()
                    except Exception as exc:
                        logger.warning(
                            "Style-aware fresh could not read %s (%s: %s); "
                            "deleting it instead of filtering by style.",
                            csv_path, type(exc).__name__, exc,
                        )
                        csv_path.unlink()
                else:
                    csv_path.unlink()
                if verbose:
                    logger.info("Cleared %s", csv_path)
        elif not fresh and self.dataset_dir is not None:
            # Resume source of truth is the sidecar ledger; we also union the
            # existing-CSV set so runs created before the ledger existed still
            # resume (back-compat). Keys are stringified on both sides to match.
            processed: set[tuple[str, str, str]] = (
                set(ledger.completed()) if ledger is not None else set()
            )
            existing = merge_all(self.dataset_dir, accepted_only=False)
            # Handle both old column name (sample_description) and new
            # (source_sample_description) so resume works across both formats.
            desc_col = (
                "source_sample_description" if "source_sample_description" in existing.columns
                else "sample_description" if "sample_description" in existing.columns
                else None
            )
            if not existing.empty and desc_col:
                style_vals = (
                    existing["template_style"]
                    if "template_style" in existing.columns
                    else pd.Series([""] * len(existing), index=existing.index)
                )
                processed |= {
                    (str(d), str(t), str(s))
                    for d, t, s in zip(existing[desc_col], existing["entry_type"], style_vals)
                }
            if processed:
                original_count = len(constitution_df)
                constitution_df = constitution_df[
                    ~constitution_df.apply(
                        lambda r: (str(r["sample_description"]), str(r["entry_type"]), str(style))
                        in processed,
                        axis=1,
                    )
                ].reset_index(drop=True)
                skipped = original_count - len(constitution_df)
                if verbose and skipped > 0:
                    logger.info("Resuming: skipped %d already-processed entries, %d remaining",
                                skipped, len(constitution_df))
                if constitution_df.empty:
                    if verbose:
                        logger.info("All entries already processed — nothing to do.")
                    return result
        prohibited: set[str] = set()
        checker_cache: dict[tuple[str, str, str], Callable[[str, str], list[dict]]] = {}

        entries = list(constitution_df.iterrows())
        total = len(entries)
        n_chunks = (total + batch_size - 1) // batch_size if batch_size else 1

        if verbose:
            style_label = f" | style='{style}'" if style else ""
            logger.info("Constitution-seeded generation: %d entries, batch_size=%d, "
                        "samples_per_entry=%d%s", total, batch_size, samples_per_entry, style_label)

        for batch_start in range(0, total, batch_size):
            batch = entries[batch_start : batch_start + batch_size]
            batch_idx = batch_start // batch_size + 1

            # 1. Build per-entry generation messages
            messages_list = []
            entry_ids = []  # pre-call identity (composite of the ledger's own key
                             # fields) — samples don't exist yet, so this entry is
                             # the only thing capturable internals can be rooted at.
            for _, entry in batch:
                seed_kwargs = {
                    "Category": str(entry.get("source_category", "")),
                    "sample_description": str(entry.get("sample_description", "")),
                    "constitution_subcategory": str(
                        entry.get("constitution_subcategory", "")
                    ),
                    "entry_type": str(entry.get("entry_type", "harmful")),
                }
                messages_list.append(
                    self._build_generation_messages(
                        prompt_config,
                        samples_per_request=samples_per_entry,
                        prohibited=prohibited,
                        **seed_kwargs,
                    )
                )
                entry_ids.append(_hash_text(
                    f"{seed_kwargs['sample_description']}:{seed_kwargs['entry_type']}:{style}"
                ))

            # 2. Single batched generation for the whole chunk (rate-limited,
            #    capability-aware) — routed through the shared router entry
            #    point, not a hand-rolled BatchCaller.
            raw_outputs = batch_generate_samples(
                self.gen, messages_list,
                batch_size=len(messages_list) or 1,
                progress=f"gen batch {batch_idx}/{n_chunks}" if verbose else None,
                internals_ids=(
                    [f"{eid}/input_{style}" if style else f"{eid}/input" for eid in entry_ids]
                    if capture_input else None
                ),
            )

            # 3. Extract per entry
            per_entry_extracted: list[list[str]] = []
            for (_, entry), raw_output in zip(batch, raw_outputs):
                extracted = extract_and_clean(
                    raw_output, style=self.extraction_style
                )
                if prohibited:
                    extracted = [s for s in extracted if s not in prohibited]
                per_entry_extracted.append(extracted)

            # 4. Build flat checker list + single batch_generate for checks
            flat_samples: list[str] = []
            flat_check_msgs: list[list[dict]] = []
            # sample_id per extracted sample — its own eventual CSV identity
            # (append_samples() computes the same hash), known pre-checker-call
            # since the sample text already exists by this point.
            flat_internals_ids: list[str] = []
            entry_ranges: list[tuple[int, int]] = []
            for (_, entry), extracted in zip(batch, per_entry_extracted):
                start = len(flat_samples)
                flat_samples.extend(extracted)
                entry_ranges.append((start, len(flat_samples)))

                if use_checker and extracted:
                    category = str(entry.get("source_category", "unknown"))
                    entry_type = str(entry.get("entry_type", "harmful"))
                    subcategory = str(entry.get("constitution_subcategory", ""))
                    cache_key = (category, entry_type, subcategory)
                    if cache_key not in checker_cache:
                        checker_cache[cache_key] = build_quality_checker(
                            category=category,
                            entry_type=entry_type,
                            subcategory=subcategory,
                        )
                    checker = checker_cache[cache_key]
                    flat_check_msgs.extend(checker("", s) for s in extracted)
                    if capture_val_in:
                        suffix = f"val_in_{style}" if style else "val_in"
                        flat_internals_ids.extend(f"{_hash_text(s)}/{suffix}" for s in extracted)

            flat_check_results: list[tuple[bool, str]]
            if use_checker and flat_check_msgs:
                # Not batch_check_samples: each entry in this chunk can carry a
                # different (category, entry_type, subcategory) checker, so
                # there's no single build_check_messages describing the whole
                # flat batch — messages are already built per-entry above.
                responses = batch_generate_samples(
                    self.check, flat_check_msgs,
                    batch_size=len(flat_check_msgs),
                    progress=f"check batch {batch_idx}/{n_chunks}" if verbose else None,
                    internals_ids=flat_internals_ids if capture_val_in else None,
                )
                flat_check_results = []
                for response in responses:
                    accepted = is_accepted(response)
                    flat_check_results.append((accepted, "" if accepted else response))
            else:
                flat_check_results = [(True, "") for _ in flat_samples]

            # 5. Per-entry save (incremental) and stats
            for entry_idx, (_, entry) in enumerate(batch):
                global_idx = batch_start + entry_idx
                start, end = entry_ranges[entry_idx]
                extracted = per_entry_extracted[entry_idx]

                category = str(entry.get("source_category", "unknown"))
                entry_type = str(entry.get("entry_type", "harmful"))
                subcategory = str(entry.get("constitution_subcategory", ""))
                sample_desc = str(entry.get("sample_description", ""))

                if verbose:
                    desc_preview = sample_desc[:_DESC_PREVIEW_LEN]
                    if len(sample_desc) > _DESC_PREVIEW_LEN:
                        desc_preview += "..."
                    logger.debug("[%d/%d] %s | %s | %s",
                                 global_idx + 1, total, category, entry_type, desc_preview)

                if not extracted:
                    result.skipped_entries += 1
                    if verbose:
                        logger.debug("-> no samples extracted")
                    continue

                sample_results = [
                    SampleResult(
                        text=text,
                        accepted=accepted,
                        reasoning=reasoning,
                        turn=0,
                    )
                    for text, (accepted, reasoning) in zip(
                        extracted, flat_check_results[start:end]
                    )
                ]

                if save:
                    extra = [
                        {
                            "entry_type": entry_type,
                            "subcategory": subcategory,
                            "source_sample_description": sample_desc,
                            "constitution_category": str(
                                entry.get("constitution_category", "")
                            ),
                            "source_group_tag": str(
                                entry.get("source_group_tag", "")
                            ),
                            "template_style": style,
                            "rejection_reason": sr.reasoning,
                        }
                        for sr in sample_results
                    ]
                    append_samples(
                        [sr.text for sr in sample_results],
                        category=category,
                        turn=0,
                        accepted=[sr.accepted for sr in sample_results],
                        source="constitution",
                        extra_columns=extra,
                        dataset_dir=self.dataset_dir,
                    )
                    # Ack this entry only after its rows are on disk (crash-safe).
                    if ledger is not None:
                        ledger.record([{
                            "sample_description": sample_desc,
                            "entry_type": entry_type,
                            "style": style,
                        }])

                accepted_count = sum(1 for sr in sample_results if sr.accepted)
                rejected_count = len(sample_results) - accepted_count

                result.total_entries_processed += 1
                result.total_prompts_generated += len(sample_results)
                result.total_prompts_accepted += accepted_count
                result.total_prompts_rejected += rejected_count

                for sr in sample_results:
                    prohibited.add(sr.text)

                if verbose:
                    logger.debug("-> %d accepted, %d rejected", accepted_count, rejected_count)

        if verbose:
            style_label = f" | style='{style}'" if style else ""
            logger.info("Constitution-seeded run complete%s: %d/%d accepted (%.0f%%)",
                        style_label, result.total_prompts_accepted,
                        result.total_prompts_generated, result.acceptance_rate * 100)

        return result


# ---------------------------------------------------------------------------
# Output generation — moved here from pipelines.py so that module stays a
# thin path/model-resolution wrapper. pipelines.py's generate_outputs()
# resolves inputs/model/backend/out_path, then calls this directly.
# ---------------------------------------------------------------------------


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
) -> pd.DataFrame:
    """Generate model responses for input samples, batched and quality-checked.

    Pipeline per ``batch_size`` chunk:
      1. Build all messages upfront.
      2. ``batch_generate_samples(...)`` — single vLLM engine pass (or
         thread-pool / sequential per backend capability). One rate-limit
         slot per batch.
      3. Each (input, output) pair checked with its own row's entry-type-aware
         output checker (``build_output_quality_checker`` keyed by category/
         entry_type — not ``batch_check_samples``, since a chunk can mix
         several checkers) via another ``batch_generate_samples`` pass.
         Refusals on harmful inputs are rejected; refusals on benign inputs
         are evaluated normally.
      4. Incremental append to the output CSV per batch — crash-resilient.

    Resume: each input gets a stable content-hash id (its ``id``/``sample_id``
    column when present, else ``_hash_text(prompt)``). Completed ids are
    recorded in a sidecar ``*.state.jsonl`` ledger beside ``out_path``. On a
    re-run (``resume=True``) inputs already in the ledger are skipped, so a
    crash loses at most one chunk. ``resume=False`` clears both the CSV and
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

    Returns:
        DataFrame of all rows in the output CSV (the full dataset, including
        rows from prior resumed runs) with the unified output schema:
        ``input_id``, ``sample_id``, ``input_prompt``, ``category``,
        ``subcategory``, ``entry_type``, ``output_response``, ``accepted``,
        ``rejection_reason``, ``model``, ``source``. ``sample_id`` is this
        row's own content-hash identity (``_hash_text(output_response)``),
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
    def _input_id(row) -> str:
        existing = str(row.get("sample_id", "")).strip()
        if existing and existing.lower() != "nan":
            return existing
        return _hash_text(str(row[text_col]))

    inputs = inputs.copy()
    inputs["_state_id"] = [_input_id(row) for _, row in inputs.iterrows()]

    # Plan → manifest: one row per input_id, written before the batch loop so the
    # run's full intended scope is inspectable up front (keyed like the ledger).
    Manifest.sidecar(out_path).write(
        {"input_id": sid, "status": "planned"} for sid in inputs["_state_id"]
    )

    # Non-resume run wipes prior output + ledger; resume skips already-done ids.
    if not resume:
        if out_path.exists():
            out_path.unlink()
        ledger.reset()
    completed = ledger.completed() if resume else set()
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

    all_rows: list[dict] = []
    n_chunks = ceil(len(inputs) / batch_size) if batch_size else 1

    for batch_start in range(0, len(inputs), batch_size):
        chunk = inputs.iloc[batch_start : batch_start + batch_size]
        chunk_idx = batch_start // batch_size + 1

        # 1. Build all messages for this chunk
        messages_list = []
        chunk_rows = []
        for _, row in chunk.iterrows():
            input_text = row[text_col]
            category = str(row.get("category", "unknown"))
            entry_type = str(row.get("entry_type", "harmful"))
            subcategory = str(row.get("subcategory", ""))

            messages_list.append(
                build_messages(
                    prompt_config,
                    input_prompt=input_text,
                    Category=category,
                )
            )
            chunk_rows.append({
                "input_id": str(row["_state_id"]),
                "input_prompt": input_text,
                "category": category,
                "subcategory": subcategory,
                "entry_type": entry_type,
                "model": client.model,
                "source": str(row.get("source", "")),
            })

        # 2. Single batched generation
        responses = batch_generate_samples(
            client, messages_list,
            batch_size=len(messages_list) or 1,
            progress=f"gen chunk {chunk_idx}/{n_chunks}" if verbose else None,
            internals_ids=(
                [f'{r["input_id"]}/output' for r in chunk_rows]
                if capture_internals_gen else None
            ),
        )

        # 3. Batched output checking (per-row checker, flat batch). Not
        # batch_check_samples: each row's checker is keyed by its own
        # (category, entry_type), so there's no single build_check_messages
        # for the whole chunk — messages are built per-row below instead.
        if check_outputs and check is not None:
            check_msgs_list: list[list[dict]] = []
            for r, resp in zip(chunk_rows, responses):
                check_msgs_list.append(
                    _checker_for(r["category"], r["entry_type"])(r["input_prompt"], resp)
                )
            check_responses = batch_generate_samples(
                check, check_msgs_list,
                batch_size=len(check_msgs_list) or 1,
                progress=f"check chunk {chunk_idx}/{n_chunks}" if verbose else None,
                internals_ids=(
                    [f'{r["input_id"]}/val_out' for r in chunk_rows]
                    if capture_internals_check else None
                ),
            )
            check_results = []
            for cr in check_responses:
                accepted = is_accepted(cr)
                check_results.append((accepted, "" if accepted else cr))
        else:
            check_results = [(True, "")] * len(chunk_rows)

        # 4. Assemble + incremental append. sample_id is this row's own content-hash
        # identity (distinct from input_id, which points back to the origin sample) —
        # only knowable once the response text exists.
        for r, resp, (accepted, reasoning) in zip(chunk_rows, responses, check_results):
            r["sample_id"] = _hash_text(resp)
            r["output_response"] = resp
            r["accepted"] = accepted
            r["rejection_reason"] = reasoning
        all_rows.extend(chunk_rows)

        new_df = pd.DataFrame(chunk_rows)
        if out_path.exists():
            new_df.to_csv(out_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(out_path, index=False)
        # Record completion only after the CSV append succeeds, so a crash
        # mid-chunk leaves those ids un-acked and they re-run next time.
        ledger.record([{"input_id": r["input_id"]} for r in chunk_rows])

        if verbose:
            accepted_count = sum(1 for r in chunk_rows if r["accepted"])
            logger.info("[%d/%d] %d/%d accepted -> appended to %s",
                        batch_start + len(chunk_rows), len(inputs),
                        accepted_count, len(chunk_rows), out_path.name)

    # Return the full dataset on disk (includes rows from prior resumed runs),
    # not just this run's newly-written delta.
    output_df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(all_rows)
    if verbose:
        accepted_total = (
            int(output_df["accepted"].sum())
            if not output_df.empty and "accepted" in output_df.columns
            else 0
        )
        logger.info("Output CSV now holds %d responses (%d accepted) at %s",
                    len(output_df), accepted_total, out_path)

    return output_df
