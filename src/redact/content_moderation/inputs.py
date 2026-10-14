"""The input stages: ``InputPipeline``, which turns constitution entries into
checked samples in the per-category tree.

The deprecated standalone path's methods come from the mixin in
``standalone.py``.

Usage::

    from redact.llms import ModelClient, load_prompt
    from redact.content_moderation import InputPipeline

    pipeline = InputPipeline(
        gen=ModelClient.create("venice-uncensored"),
        check=ModelClient.create("venice-uncensored"),
        dataset_dir="./Datasets/constitution_inputs",
    )
    result = pipeline.run_from_constitution(
        constitution_df=constitution_df,
        prompt_config=load_prompt("input", "generation/from_constitution/long"),
        samples_per_entry=3, style="long",
    )
"""

import functools
import logging
from collections.abc import Collection
from itertools import repeat
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .. import paths
from ..dataset import (
    Ledger,
    Manifest,
    append_rows,
    check_columns,
    content_id,
    sample_row,
)
from ..dataset.drop import drop_units
from ..dataset.resume import release_lock, resume_state
from ..dataset.save import CommitBuffer
from ..exceptions import PromptTooLongError
from ..llm_pipeline import LLMRequest, drive_units, is_accepted, pool_size
from ..llm_pipeline import extracted as extracted_unit
from ..llms.client import ModelClient
from ..llms.prompting import build_messages, extract_and_clean
from ..llms.router import assert_single_sample_per_call
from .checker import build_quality_checker
from .feedback import (
    PROHIBITED_SHOWN,
    append_to_user,
    prohibited_note,
    with_feedback,
)
from .results import ConstitutionInputResult
from .standalone import _StandaloneGenerationMixin

logger = logging.getLogger(__name__)

# Truncation length for the sample_description preview in verbose progress lines.
_DESC_PREVIEW_LEN = 60
# The category files of constitution-seeded inputs, under the stage's folder.
_CATEGORY_FILES = f"*/{paths.SAMPLES_FILENAME}"


def _constitution_inputs_ledger(dataset_dir: str | Path) -> Ledger:
    """The stage's ledger: one ack per entry, keyed like the manifest."""
    return Ledger(
        Path(dataset_dir) / "constitution_inputs.state.jsonl",
        key_fields=("sample_description", "entry_type", "style"),
        casters={"sample_description": str, "entry_type": str, "style": str},
    )


def _constitution_inputs_manifest(dataset_dir: str | Path) -> Manifest:
    """The stage's plan: one row per entry, by description, entry type and style."""
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


class _Entry(NamedTuple):
    """The fields of a constitution entry that the stage reads."""

    description: str
    entry_type: str
    category: str
    subcategory: str
    constitution_category: str
    group_tag: str


def _entry_fields(entry: pd.Series) -> _Entry:
    """Read an entry's fields once, each missing one as its default."""
    return _Entry(
        description=str(entry.get("sample_description", "")),
        entry_type=str(entry.get("entry_type", "harmful")),
        category=str(entry.get("source_category", "unknown")),
        subcategory=str(entry.get("constitution_subcategory", "")),
        constitution_category=str(entry.get("constitution_category", "")),
        group_tag=str(entry.get("source_group_tag", "")),
    )


class InputPipeline(_StandaloneGenerationMixin):
    """What the input stages share: two clients, a reply format and a folder.

    ``run_from_constitution`` is the seeded stage; the deprecated standalone
    methods come from ``_StandaloneGenerationMixin``.
    """

    def __init__(
        self,
        gen: ModelClient,
        check: ModelClient,
        extraction_style: str = "numbered",
        dataset_dir: str | Path | None = None,
    ):
        """Bind the clients; ``dataset_dir`` defaults to ``paths.datasets()``."""
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
        """Build the generation messages, with the format instruction.

        Args:
            prompt_config: Loaded prompt JSON config.
            samples_per_request: Number of samples to request per LLM call.
            feedback: Rejection feedback from a previous turn (optional).
            prohibited: Existing sample texts to avoid repeating. The first
                20 are shown, the rest counted.
            more: How many such texts exist beyond ``prohibited``.
            **seed_kwargs: Template variables (Category, SeedPrompts, etc.).
        """
        # The format instruction matches the style the reply is extracted with.
        messages = build_messages(
            prompt_config,
            format_style=self.extraction_style,
            num_samples=str(samples_per_request),
            **seed_kwargs,
        )
        if prohibited:
            messages = append_to_user(messages, prohibited_note(prohibited, more))
        return with_feedback(messages, feedback)

    def run_from_constitution(
        self,
        constitution_df: pd.DataFrame,
        prompt_config: dict,
        samples_per_entry: int = 3,
        use_checker: bool = True,
        verbose: bool = True,
        batch_size: int = 32,
        resume: bool = True,
        style: str = "",
        max_attempts: int = 3,
        check_max_tokens: int | None = None,
    ) -> ConstitutionInputResult:
        """Generate ``samples_per_entry`` checked samples per constitution entry.

        Each entry is one unit: a generation call, made again for a reply with
        the wrong number of samples, up to ``max_attempts``; then one check
        call per sample. A sample the checker rejects is saved with its
        reason. An entry rejected on every attempt, or whose prompt is too
        long for the model, is acked ``rejected`` with no rows. A call carries
        up to ``batch_size`` requests; ``check_max_tokens`` limits a verdict.

        Rows go to ``{dataset_dir}/{category}/samples.csv``, and a text is
        written once in the folder, whatever its category and style. The
        ledger beside them is keyed on ``(sample_description, entry_type,
        style)``, and a resume reads it alone. ``resume=False`` starts this
        style over and keeps the rows and acks of the others; without a
        style it clears the folder.

        ``constitution_df`` has the constitution's columns
        (``source_category``, ``sample_description``,
        ``constitution_subcategory``, ``entry_type``, ...); ``prompt_config``
        is a prompt of ``input/generation/from_constitution/{style}``.
        """
        if constitution_df.empty:
            logger.warning("run_from_constitution: empty constitution_df, nothing to do")
            return ConstitutionInputResult()

        # One forward pass's internals can be told apart for one sample only.
        assert_single_sample_per_call(self.gen, samples_per_entry)
        capture_input = self.gen.compute_config.supports_internals
        capture_val_in = use_checker and self.check.compute_config.supports_internals

        folder = Path(self.dataset_dir)
        ledger = _constitution_inputs_ledger(folder)
        manifest = _constitution_inputs_manifest(folder)
        tree = {"artifact": folder, "pattern": _CATEGORY_FILES, "units_in": _entries_in}
        if not resume and style:
            # The other styles keep their rows and their acks.
            drop_units(
                ledger,
                name=style,
                rows=lambda row: row.get("template_style") == style,
                acks=lambda ack: ack.get("style") == style,
                **tree,
            )
        # Before anything is written in the folder: it takes the stage's lock.
        processed = resume_state(
            ledger, resume=resume or bool(style), manifest=manifest, **tree)
        seen = set(ledger.ids)      # the content ids of the rows of every style
        preview: list[str] = []     # the first texts this run generated
        unshown = 0                 # how many it generated after those

        entries = [_entry_fields(entry) for _, entry in constitution_df.iterrows()]
        manifest.write(
            {"sample_description": e.description, "entry_type": e.entry_type,
             "style": style, "status": "planned"}
            for e in entries
        )
        pending = [e for e in entries
                   if (e.description, e.entry_type, style) not in processed]
        result = ConstitutionInputResult()
        if verbose and len(pending) < len(entries):
            logger.info("Resuming: skipped %d already-processed entries, %d remaining",
                        len(entries) - len(pending), len(pending))
        if not pending:
            if verbose:
                logger.info("All entries already processed — nothing to do.")
            release_lock(ledger)    # no CommitBuffer will
            return result

        total = len(pending)
        style_label = f" | style='{style}'" if style else ""
        if verbose:
            logger.info("Constitution-seeded generation: %d entries, batch_size=%d, "
                        "samples_per_entry=%d%s", total, batch_size, samples_per_entry,
                        style_label)

        @functools.cache
        def _checker(category: str, entry_type: str, subcategory: str):
            return build_quality_checker(
                category=category, entry_type=entry_type, subcategory=subcategory)

        rejected_entries = 0
        finished = 0

        def _append(rows: list[dict]) -> dict[Path, int]:
            """Append a save's rows, each to its category's file."""
            by_file: dict[Path, list[dict]] = {}
            for r in rows:
                by_file.setdefault(
                    paths.category_csv(r["category"], root=folder), []).append(r)
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

        def _entry_unit(e: _Entry):
            """Generate an entry's samples, drop those already seen, check each."""
            # The samples do not exist yet: internals are captured under the entry.
            eid = content_id(f"{e.description}:{e.entry_type}:{style}")
            extracted, attempts = yield from extracted_unit(
                self.gen,
                lambda: self._build_generation_messages(
                    prompt_config,
                    samples_per_request=samples_per_entry,
                    prohibited=preview,
                    more=unshown,
                    Category=e.category,
                    sample_description=e.description,
                    constitution_subcategory=e.subcategory,
                    entry_type=e.entry_type,
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
                checker = _checker(e.category, e.entry_type, e.subcategory)
                suffix = f"val_in_{style}" if style else "val_in"
                verdicts = []
                for s in extracted:
                    # The sample's content id is its sample_id.
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

        live: dict[int, _Entry] = {}   # entries of live units, by position

        def _units():
            for j, e in enumerate(pending):
                live[j] = e
                yield j, _entry_unit(e)

        def _on_done(j, value):
            nonlocal rejected_entries, finished, unshown
            e = live.pop(j)
            extracted, verdicts, attempts, rejected = value
            finished += 1
            if verbose:
                desc_preview = e.description[:_DESC_PREVIEW_LEN]
                if len(e.description) > _DESC_PREVIEW_LEN:
                    desc_preview += "..."
                logger.debug("[%d/%d] %s | %s | %s",
                             finished, total, e.category, e.entry_type, desc_preview)

            key = {"sample_description": e.description, "entry_type": e.entry_type,
                   "style": style}
            if not extracted:
                result.skipped_entries += 1
                rejected_entries += rejected
                # Acked: un-acked, it would be generated, and paid for, again
                # on every resume.
                buf.add([], {**key, "status": "rejected" if rejected else "no_samples",
                             "attempts": attempts})
                if verbose:
                    logger.debug("-> no samples extracted")
                return

            # Again here: an entry that was live with this one may have given
            # one of its texts, and both passed the check when generated.
            rows = []
            for text, (accepted, reasoning) in zip(extracted, verdicts):
                row = sample_row(
                    text, e.category, turn=0, accepted=accepted,
                    source="constitution",
                    extra={
                        "entry_type": e.entry_type,
                        "subcategory": e.subcategory,
                        "source_sample_description": e.description,
                        "constitution_category": e.constitution_category,
                        "source_group_tag": e.group_tag,
                        "template_style": style,
                        "rejection_reason": reasoning,
                    },
                )
                if row["sample_id"] in seen:
                    continue
                seen.add(row["sample_id"])
                rows.append(row)
                if len(preview) < PROHIBITED_SHOWN:
                    preview.append(text)
                else:
                    unshown += 1
            # The ids of the rows it wrote: what a resume knows them by.
            buf.add(rows, {**key, "ids": [row["sample_id"] for row in rows]})

            accepted_count = sum(accepted for accepted, _ in verdicts)
            result.total_entries_processed += 1
            result.total_prompts_generated += len(extracted)
            result.total_prompts_accepted += accepted_count
            result.total_prompts_rejected += len(extracted) - accepted_count
            if verbose:
                logger.debug("-> %d accepted, %d rejected",
                             accepted_count, len(extracted) - accepted_count)

        # A prompt too long for the model ends the entry as one rejected on
        # every attempt; the ack says 1. Any other unit error stops the stage.
        def _on_error(j, exc):
            if not isinstance(exc, PromptTooLongError):
                raise exc
            logger.warning("Entry %r (%s) rejected, its prompt is too long: %s",
                           live[j].description, live[j].entry_type, exc)
            _on_done(j, ([], [], 1, True))

        with CommitBuffer(append=_append, ledger=ledger) as buf:
            drive_units(
                _units(),
                on_done=_on_done,
                on_error=_on_error,
                pool_size=pool_size(batch_size, self.gen, self.check),
                cap=batch_size,
                progress="inputs" if verbose else None,
            )

        if rejected_entries:
            logger.warning(
                "%d entries rejected after %d attempt(s) each. They are acked "
                "and skipped on resume; run with resume=False, or delete their "
                "'rejected' lines from the ledger, to try them again.",
                rejected_entries, max_attempts,
            )
        if verbose:
            logger.info("Constitution-seeded run complete%s: %d/%d accepted (%.0f%%)",
                        style_label, result.total_prompts_accepted,
                        result.total_prompts_generated, result.acceptance_rate * 100)

        return result
