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
from collections.abc import Callable, Collection
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path

import pandas as pd

from .. import paths
from ..dataset import (
    Ledger,
    Manifest,
    append_rows,
    check_columns,
    content_id,
    read_rows,
    row_ids,
    sample_row,
)
from ..dataset.resume import CommitBuffer, drop_units, release_lock, resume_state
from ..exceptions import PromptTooLongError
from ..llms.client import ModelClient
from ..llms.prompting import extract_and_clean
from ..llms.prompting import build_messages, load_prompt
from ..llm_pipeline import LLMRequest, checked, drive_units, is_accepted
from ..llm_pipeline import extracted as extracted_unit
from ..llms.router import assert_single_sample_per_call
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
# The category files of constitution-seeded inputs, under the stage's folder.
_CATEGORY_FILES = f"*/{paths.SAMPLES_FILENAME}"


_INPUT_FEEDBACK = (
    "\n\n[FEEDBACK FROM PREVIOUS ATTEMPT]\n"
    "Some previously generated samples were rejected for these "
    "reasons:\n{feedback}\n"
    "Please avoid similar issues in your new samples."
)
_OUTPUT_FEEDBACK = (
    "\n\n[FEEDBACK FROM PREVIOUS ATTEMPT]\n"
    "Your previous response was rejected for this reason:\n{feedback}\n"
    "Please avoid this issue in your new response."
)


def _append_feedback(
    messages: list[dict], feedback: str, template: str = _INPUT_FEEDBACK,
) -> list[dict]:
    """``messages`` with rejection feedback appended to the last user turn.

    Returns a new list; ``messages`` itself for an empty ``feedback``.
    """
    if not feedback:
        return messages
    feedback_msg = template.format(feedback=feedback)
    if messages and messages[-1]["role"] == "user":
        last = {**messages[-1], "content": messages[-1]["content"] + feedback_msg}
        return [*messages[:-1], last]
    return [*messages, {"role": "user", "content": feedback_msg}]


def _pool_size(batch_size: int, *clients: ModelClient | None) -> int:
    """Units a stage keeps live: two calls' worth, or its widest async lane.

    A local engine runs hundreds of requests at once, far more than a
    synchronous call holds, and each live unit has one request in flight.
    A hosted model's lane is narrow and changes nothing.
    """
    lanes = [client.backend.max_workers for client in clients
             if client is not None and client.compute_config.supports_async_calls]
    return max([2 * batch_size, *lanes])


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


def _entries_in(path: Path) -> dict[tuple[str, str, str], list[str]]:
    """Each entry with rows in a category file, by its ledger key, with their ids.

    Read once, when a folder written before the ledger recorded sizes and
    ids is taken over. A file of that time may name the description
    ``sample_description`` and have no style column: its rows are of the
    style ``""``. A file without a description, an entry type or an id
    column gives no entry.
    """
    entries: dict[tuple[str, str, str], list[str]] = {}
    # Every cell as the text it is: an id of digits is not a number.
    with pd.read_csv(
            path, dtype=str, keep_default_na=False, chunksize=50_000) as chunks:
        for chunk in chunks:
            columns = set(chunk.columns)
            described = next(
                (name for name in ("source_sample_description", "sample_description")
                 if name in columns), None)
            if described is None or not {"entry_type", "sample_id"} <= columns:
                return {}
            styles = chunk.get("template_style", repeat(""))
            for description, entry_type, style, sample_id in zip(
                    chunk[described], chunk["entry_type"], styles, chunk["sample_id"]):
                key = str(description), str(entry_type), str(style)
                entries.setdefault(key, []).append(str(sample_id))
    return entries


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
        self.dataset_dir = dataset_dir if dataset_dir is not None else paths.datasets()

    def _build_generation_messages(
        self,
        prompt_config: dict,
        samples_per_request: int,
        feedback: str = "",
        prohibited: Collection[str] | None = None,
        more: int = 0,
        **seed_kwargs: str,
    ) -> list[dict]:
        """Build the generation message list with format instruction.

        Args:
            prompt_config: Loaded prompt JSON config.
            samples_per_request: Number of samples to request per LLM call.
            feedback: Rejection feedback from a previous turn (optional).
            prohibited: Existing sample texts to avoid repeating. The first
                20 are shown, the rest counted.
            more: How many such texts exist beyond ``prohibited``.
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
            more += len(prohibited) - len(sample_list)
            if more:
                prohibited_msg += (
                    f"\n(... and {more} more. "
                    f"Ensure all your outputs are novel.)"
                )
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += prohibited_msg
            else:
                messages.append({"role": "user", "content": prohibited_msg})

        return _append_feedback(messages, feedback)

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
        max_attempts: int = 3,
        check_max_tokens: int | None = None,
    ) -> ConstitutionInputResult:
        """Run constitution-seeded input generation, batched across entries.

        Each row in ``constitution_df`` becomes a generation request.
        Samples the checker rejects are saved with their rejection reason
        and not regenerated.

        Each entry is one unit: a generation call (again for a reply that is
        rejected whole), then one check call per sample. ``drive_units``
        keeps ``2 * batch_size`` entries live, or as many as the widest async
        lane of the two models holds when that is more, and sends calls of up
        to ``batch_size`` requests, so a retry or a check shares its call with
        other entries' requests. Finished entries are appended every 32
        entries, so a crash mid-run loses only the live entries' work.

        An entry whose prompt is longer than the model's context (a
        ``PromptTooLongError`` from a request on the async lane) is acked as
        ``rejected`` at once, with no rows, and skipped on resume.

        A text is written once in the whole folder, whatever its category
        and style: one that a row already holds is dropped, and the entry
        that gave it is finished with the rows it has left. Which texts the
        rows hold is recorded in the ledger, so a resume reads no row.

        The folder is this stage's alone. On a resume, a category's
        ``samples.csv`` that the ledger does not record is moved aside, and
        rows another writer added to one it records are cut off: both read
        as what a save that did not finish left.

        Args:
            constitution_df: DataFrame with columns ``source_category``,
                ``sample_description``, ``constitution_subcategory``,
                ``entry_type`` (matches the output of
                ``ConstitutionPipeline.to_dataframe()``).
            prompt_config: Loaded prompt JSON config from
                ``prompts/input/generation/from_constitution/{style}/``.
            samples_per_entry: Prompts to generate per constitution entry.
            use_checker: Whether to run the entry-type-aware quality checker.
            save: Whether to append rows to CSV as entries finish. False
                leaves the folder alone: nothing in it is resumed from,
                cleared or written.
            verbose: Log per-entry progress (INFO for stage milestones, DEBUG
                for per-entry detail).
            batch_size: Requests per model call; twice as many entries are live.
            fresh: When True, starts this style over: its rows and its acks
                are removed before generating, and those of other styles
                stay. Without a style the whole folder is cleared.
            style: Template style under ``from_constitution/{style}`` — also
                used as part of the ledger key and saved as a CSV column.
            max_attempts: Generations per entry. A reply with the wrong
                number of samples is rejected whole and regenerated; an
                entry still rejected after the last attempt is acked as
                ``rejected`` and skipped on resume.
            check_max_tokens: Reply limit for the check calls; a verdict
                needs few tokens. ``None`` keeps the checker model's default.

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
        # The checker sends one request per already-extracted sample, so it's
        # always attributable regardless of samples_per_entry — no gate needed
        # here.
        capture_val_in = use_checker and self.check.compute_config.supports_internals

        result = ConstitutionInputResult()

        # The stage's ledger, keyed on one constitution entry
        # (sample_description, entry_type, style). A unit is acked only after
        # its rows are saved, so a crash loses only unsaved entries.
        saves = save and self.dataset_dir is not None
        folder = Path(self.dataset_dir) if saves else None
        ledger = _constitution_inputs_ledger(folder) if saves else None

        seen: set[str] = set()      # the content ids of the rows of every style
        preview: list[str] = []     # the first texts this run generated
        unshown = 0                 # how many it generated after those
        if ledger is not None:
            tree = {"artifact": folder, "pattern": _CATEGORY_FILES,
                    "units_in": _entries_in}
            manifest = _constitution_inputs_manifest(folder)
            if fresh and style:
                # The other styles keep their rows and their acks.
                drop_units(
                    ledger,
                    rows=lambda row: row.get("template_style") == style,
                    acks=lambda ack: ack.get("style") == style,
                    **tree,
                )
            # Before anything is written in the folder: the resume takes the
            # stage's lock. It cuts off what a save that did not finish left,
            # and a fresh run of every style clears the folder.
            processed = resume_state(
                ledger, resume=bool(style) or not fresh, manifest=manifest, **tree)
            seen = set(ledger.ids)

            # Plan → manifest: one row per constitution entry of this style,
            # keyed like the ledger.
            manifest.write(
                {
                    "sample_description": str(entry.get("sample_description", "")),
                    "entry_type": str(entry.get("entry_type", "harmful")),
                    "style": style,
                    "status": "planned",
                }
                for _, entry in constitution_df.iterrows()
            )

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
                    release_lock(ledger)    # no CommitBuffer will
                    return result
        checker_cache: dict[tuple[str, str, str], Callable[[str, str], list[dict]]] = {}

        total = len(constitution_df)

        if verbose:
            style_label = f" | style='{style}'" if style else ""
            logger.info("Constitution-seeded generation: %d entries, batch_size=%d, "
                        "samples_per_entry=%d%s", total, batch_size, samples_per_entry, style_label)

        rejected_entries = 0
        finished = 0   # entries finished so far

        def _append(rows: list[dict]) -> dict[Path, int]:
            """Append a save's rows, each to its category's file."""
            by_file: dict[Path, list[dict]] = {}
            for r in rows:
                by_file.setdefault(
                    paths.category_csv(r["category"], base=folder), []).append(r)
            # Before the first file is written: a save that stopped on a later
            # file would leave the earlier ones to be cut back.
            for path, its_rows in by_file.items():
                check_columns(path, its_rows)
            sizes: dict[Path, int] = {}
            for path, its_rows in by_file.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                sizes.update(append_rows(path, its_rows))
            if verbose:
                n_acc = sum(1 for r in rows if r["accepted"])
                logger.info("[%d/%d] %d/%d samples accepted -> appended",
                            finished, total, n_acc, len(rows))
            return sizes

        def _entry_unit(entry):
            """Generate an entry's samples, drop those already seen, check each."""
            seed_kwargs = {
                "Category": str(entry.get("source_category", "")),
                "sample_description": str(entry.get("sample_description", "")),
                "constitution_subcategory": str(
                    entry.get("constitution_subcategory", "")
                ),
                "entry_type": str(entry.get("entry_type", "harmful")),
            }
            # Pre-call identity (the ledger's own key fields): the samples
            # do not exist yet, so captured internals are rooted here.
            eid = content_id(
                f"{seed_kwargs['sample_description']}:{seed_kwargs['entry_type']}:{style}"
            )
            # A reply with any other number of samples is rejected whole and
            # regenerated, up to max_attempts.
            extracted, attempts = yield from extracted_unit(
                self.gen,
                lambda: self._build_generation_messages(
                    prompt_config,
                    samples_per_request=samples_per_entry,
                    prohibited=preview,
                    more=unshown,
                    **seed_kwargs,
                ),
                lambda raw: extract_and_clean(
                    raw, style=self.extraction_style,
                    expected_count=samples_per_entry,
                ),
                internals_id=(
                    (f"{eid}/input_{style}" if style else f"{eid}/input")
                    if capture_input else None
                ),
                max_attempts=max_attempts,
            )
            rejected = not extracted
            # Outside the extraction: a repeat is not a reason to regenerate.
            extracted = [s for s in extracted if content_id(s) not in seen]

            verdicts = [(True, "")] * len(extracted)
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
                suffix = f"val_in_{style}" if style else "val_in"
                verdicts = []
                for s in extracted:
                    # The sample's own content hash is its eventual sample_id
                    # (sample_row() computes the same hash).
                    response = yield LLMRequest.for_client(
                        self.check, checker("", s),
                        internals_id=(
                            f"{content_id(s)}/{suffix}" if capture_val_in else None
                        ),
                        max_tokens=check_max_tokens,
                    )
                    accepted = is_accepted(response)
                    verdicts.append((accepted, "" if accepted else response))
            return extracted, verdicts, attempts, rejected

        live: dict[int, pd.Series] = {}   # entries of live units, by position

        def _units():
            for j, (_, entry) in enumerate(constitution_df.iterrows()):
                live[j] = entry
                yield j, _entry_unit(entry)

        def _on_done(j, value):
            nonlocal rejected_entries, finished, unshown
            entry = live.pop(j)
            extracted, verdicts, attempts, rejected = value
            finished += 1

            category = str(entry.get("source_category", "unknown"))
            entry_type = str(entry.get("entry_type", "harmful"))
            subcategory = str(entry.get("constitution_subcategory", ""))
            sample_desc = str(entry.get("sample_description", ""))

            if verbose:
                desc_preview = sample_desc[:_DESC_PREVIEW_LEN]
                if len(sample_desc) > _DESC_PREVIEW_LEN:
                    desc_preview += "..."
                logger.debug("[%d/%d] %s | %s | %s",
                             finished, total, category, entry_type, desc_preview)

            if not extracted:
                result.skipped_entries += 1
                rejected_entries += rejected
                # Ack it anyway: un-acked, it would be generated, and paid
                # for, again on every resume.
                if buf is not None:
                    buf.add([], {
                        "sample_description": sample_desc,
                        "entry_type": entry_type,
                        "style": style,
                        "status": "rejected" if rejected else "no_samples",
                        "attempts": attempts,
                    })
                if verbose:
                    logger.debug("-> no samples extracted")
                return

            sample_results = [
                SampleResult(
                    text=text,
                    accepted=accepted,
                    reasoning=reasoning,
                    turn=0,
                )
                for text, (accepted, reasoning) in zip(extracted, verdicts)
            ]

            # Again here: an entry that was live with this one may have given
            # one of its texts, and both passed the check when generated.
            rows = []
            for sr in sample_results:
                row = sample_row(
                    sr.text, category, turn=0, accepted=sr.accepted,
                    source="constitution",
                    extra={
                        "entry_type": entry_type,
                        "subcategory": subcategory,
                        "source_sample_description": sample_desc,
                        "constitution_category": str(
                            entry.get("constitution_category", "")
                        ),
                        "source_group_tag": str(entry.get("source_group_tag", "")),
                        "template_style": style,
                        "rejection_reason": sr.reasoning,
                    },
                )
                if row["sample_id"] in seen:
                    continue
                seen.add(row["sample_id"])
                rows.append(row)
                if len(preview) < _PROHIBITED_PREVIEW_LIMIT:
                    preview.append(sr.text)
                else:
                    unshown += 1
            if buf is not None:
                # The ids of the rows it wrote: what a resume knows them by.
                buf.add(rows, {
                    "sample_description": sample_desc,
                    "entry_type": entry_type,
                    "style": style,
                    "ids": [row["sample_id"] for row in rows],
                })

            accepted_count = sum(1 for sr in sample_results if sr.accepted)
            rejected_count = len(sample_results) - accepted_count

            result.total_entries_processed += 1
            result.total_prompts_generated += len(sample_results)
            result.total_prompts_accepted += accepted_count
            result.total_prompts_rejected += rejected_count

            if verbose:
                logger.debug("-> %d accepted, %d rejected", accepted_count, rejected_count)

        # An entry whose prompt is too long ends as one whose every reply was
        # rejected: no rows, acked as rejected. How many generations it made
        # is not known here; the ack says 1. Any other unit error stops the
        # stage.
        def _on_error(j, exc):
            if not isinstance(exc, PromptTooLongError):
                raise exc
            logger.warning(
                "Entry %r (%s) rejected, its prompt is too long: %s",
                str(live[j].get("sample_description", "")),
                str(live[j].get("entry_type", "harmful")), exc,
            )
            _on_done(j, ([], [], 1, True))

        # Generation and check calls, topped up from the entries as units
        # finish. The buffer appends finished entries' rows, then acks them
        # (see dataset/resume.py), every 32 entries and on any exit.
        with (
            CommitBuffer(append=_append, ledger=ledger) if ledger is not None
            else nullcontext()
        ) as buf:
            drive_units(
                _units(),
                on_done=_on_done,
                on_error=_on_error,
                pool_size=_pool_size(batch_size, self.gen, self.check),
                cap=batch_size,
                progress="inputs" if verbose else None,
            )

        if rejected_entries:
            logger.warning(
                "%d entries rejected after %d attempt(s) each. They are acked "
                "and skipped on resume; run with fresh=True, or delete their "
                "'rejected' lines from the ledger, to try them again.",
                rejected_entries, max_attempts,
            )
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
            lambda feedback: _append_feedback(base, feedback, _OUTPUT_FEEDBACK),
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
    # The buffer appends finished rows, then acks them (see dataset/resume.py),
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
