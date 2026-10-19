"""The inputs stage: ``InputPipeline``, which turns constitution entries into
checked samples in the per-category tree.

Each entry is ``samples_per_entry`` units, one per iteration. A unit is dealt
one spec of the generation axes, generates one sample in one call and checks
it; every unit is its own generator on the ``drive_units`` pool.

Usage::

    from redact.llms import ModelClient, load_prompt
    from redact.content_moderation import InputPipeline, axes

    pipeline = InputPipeline(
        gen=ModelClient.create("venice-uncensored"),
        check=ModelClient.create("venice-uncensored"),
        dataset_dir="./Datasets/constitution_inputs",
    )
    result = pipeline.run_from_constitution(
        constitution_df=constitution_df,
        prompt_config=load_prompt("input", "generation/from_constitution/long"),
        samples_per_entry=3, style="long", axes=axes.resolve("inputs"),
    )
"""

import functools
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .. import paths
from ..constitution.parse import strip_end_marker
from ..dataset import (
    Ledger,
    Manifest,
    append_rows,
    check_columns,
    content_id,
    sample_row,
)
from ..dataset.drop import drop_units
from ..dataset.planfile import changed_plan, plan_path, plan_record, write_plan
from ..dataset.resume import release_lock, resume_state
from ..dataset.save import CommitBuffer
from ..exceptions import PromptTooLongError
from ..llm_pipeline import (
    CHECK_TEMPERATURE,
    LLMRequest,
    drive_units,
    is_accepted,
    pool_size,
)
from ..llm_pipeline import extracted as extracted_unit
from ..llms.client import ModelClient
from ..llms.prompting import PromptTemplate, extract_and_clean
from .axes import (
    deal_specs,
    deal_weights,
    render_spec,
    spec_columns,
    warn_unshown,
    write_stage_coverage,
)
from .checker import build_quality_checker
from .results import ConstitutionInputResult

logger = logging.getLogger(__name__)

# Truncation length for the sample_description preview in verbose progress lines.
_DESC_PREVIEW_LEN = 60
# The category files of constitution-seeded inputs, under the stage's folder.
_CATEGORY_FILES = f"*/{paths.SAMPLES_FILENAME}"
# A unit's key in the ledger and the manifest.
_KEY = ("sample_description", "entry_type", "style", "iteration")
# The columns of a category file that name a row's unit, in key order, and its id.
_UNIT_COLUMNS = ("source_sample_description", "entry_type", "template_style",
                 "iteration", "sample_id")


def _constitution_inputs_ledger(dataset_dir: str | Path) -> Ledger:
    """The stage's ledger: one ack per unit, keyed like the manifest."""
    return Ledger(
        Path(dataset_dir) / "constitution_inputs.state.jsonl",
        key_fields=_KEY,
        casters={"sample_description": str, "entry_type": str, "style": str,
                 "iteration": int},
    )


def _constitution_inputs_manifest(dataset_dir: str | Path) -> Manifest:
    """The stage's plan: one row per unit, by description, entry type, style
    and iteration."""
    return Manifest(Path(dataset_dir) / "constitution_inputs.manifest.jsonl")


def _entries_in(path: Path) -> dict[tuple[str, str, str, int], list[str]]:
    """Each unit with a row in a category file, by its ledger key, with its ids.

    Read once, when a folder whose ledger records no sizes, or has none, is
    taken over. A file without one of the columns gives no unit: one written
    before units had an iteration stops the stage before this.
    """
    entries: dict[tuple[str, str, str, int], list[str]] = {}
    # Every cell as the text it is: an id of digits is not a number.
    with pd.read_csv(
            path, dtype=str, keep_default_na=False, chunksize=50_000) as chunks:
        for chunk in chunks:
            if not set(_UNIT_COLUMNS) <= set(chunk.columns):
                return {}
            for description, entry_type, style, iteration, sample_id in zip(
                    *(chunk[column] for column in _UNIT_COLUMNS)):
                if iteration.isdigit():
                    key = description, entry_type, style, int(iteration)
                    entries.setdefault(key, []).append(sample_id)
    return entries


def _stop_on_earlier_files(
        ledger: Ledger, files: list[Path], columns: list[str]) -> None:
    """Raise when the folder was written before its units were these: an ack
    without ``iteration`` (an entry was one call then), or a category file
    whose header lacks one of ``columns``.

    Raises:
        ValueError: What was found, and the remedies.
    """
    remedy = (
        "Start this style over with resume=False (a recipe's \"resume\": false), "
        "which removes its rows and acks, or run the stage in a new data_dir; "
        "a folder without the axis columns also takes params inputs.axes: false.")
    earlier = Ledger(ledger.path, key_fields=_KEY[:-1])
    if any("iteration" not in ack for ack in earlier.records()):
        raise ValueError(
            f"{ledger.path} acks units without an iteration: it was written when "
            f"an entry was one call for all its samples. {remedy}")
    row = [dict.fromkeys(columns, "")]
    for path in files:
        try:
            check_columns(path, row)
        except ValueError as error:
            raise ValueError(f"{error} {remedy}") from None


class _Entry(NamedTuple):
    """The fields of a constitution entry that the stage reads."""

    description: str
    entry_type: str
    category: str
    subcategory: str
    constitution_category: str
    group_tag: str
    source_taxonomy: str | None     # None for a constitution without the column


def _entry_fields(entry: pd.Series) -> _Entry:
    """Read an entry's fields once, each missing one as its default."""
    taxonomy = entry.get("source_taxonomy")
    return _Entry(
        description=str(entry.get("sample_description", "")),
        entry_type=str(entry.get("entry_type", "harmful")),
        category=str(entry.get("source_category", "unknown")),
        subcategory=str(entry.get("constitution_subcategory", "")),
        constitution_category=str(entry.get("constitution_category", "")),
        group_tag=str(entry.get("source_group_tag", "")),
        source_taxonomy=None if taxonomy is None else str(taxonomy),
    )


class InputPipeline:
    """The inputs stage's two clients, reply format, folder and prompt
    overlay; ``run_from_constitution`` runs it."""

    def __init__(
        self,
        gen: ModelClient,
        check: ModelClient,
        extraction_style: str = "numbered",
        dataset_dir: str | Path | None = None,
        prompt_dir: str | Path | None = None,
    ):
        """Bind the clients; ``dataset_dir`` defaults to
        ``paths.constitution_inputs_dir()``.

        ``prompt_dir`` is the overlay the format instruction and the checks
        read their prompts through.
        """
        self.gen = gen
        self.check = check
        self.extraction_style = extraction_style
        self.dataset_dir = (dataset_dir if dataset_dir is not None
                            else paths.constitution_inputs_dir())
        self.prompt_dir = prompt_dir

    def _build_generation_messages(
        self,
        prompt_config: dict,
        **seed_kwargs: str,
    ) -> list[dict]:
        """Build the messages of one sample's generation, with the format
        instruction.

        Args:
            prompt_config: Loaded prompt JSON config.
            **seed_kwargs: Template variables (Category, sample_description,
                sample_spec, ...).
        """
        # The format instruction matches the style the reply is extracted with.
        kwargs = {"num_samples": "1", **seed_kwargs}
        return PromptTemplate(
            prompt_config, format_style=self.extraction_style,
            prompt_dir=self.prompt_dir, **kwargs,
        )(**kwargs)

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
        axes: dict | None = None,
        seed: int = 42,
    ) -> ConstitutionInputResult:
        """Generate ``samples_per_entry`` checked samples per constitution entry.

        Each entry is ``samples_per_entry`` units ``(entry, iteration)``. A
        unit makes one generation call for one sample, made again for a reply
        without its ``[END]`` marker or with another count, up to
        ``max_attempts``; then one check call. A sample the checker rejects is
        saved with its reason. A unit rejected on every attempt, or whose
        prompt is too long for the model, is acked ``rejected`` with no row;
        one whose sample the folder holds already is acked ``no_sample``. A
        call carries up to ``batch_size`` requests; ``check_max_tokens``
        limits a verdict.

        ``axes`` (:func:`redact.content_moderation.axes.resolve`) deals each
        unit one spec per ``(source_category, entry_type)``, an entry's units
        one after another, from ``seed``; the template's ``{sample_spec}``
        shows it, the row records it in a column per axis, and
        ``constitution_inputs.coverage.csv`` counts the specs planned and
        accepted. ``None``: no specs, and ``{sample_spec}`` is empty.

        Rows go to ``{dataset_dir}/{category}/samples.csv``, and a text is
        written once in the folder, whatever its category and style. The
        ledger beside them is keyed on ``(sample_description, entry_type,
        style, iteration)``, and a resume reads it alone; the deal is made
        again, and ``constitution_inputs.plan.json`` (``seed``,
        ``samples_per_entry``, the axes' weights) stops a resume that would
        deal another. ``resume=False`` starts this style over and keeps the
        rows and acks of the others; without a style it clears the folder. A
        folder written before its units had an iteration stops before any
        call.

        ``constitution_df`` has the constitution's columns
        (``source_category``, ``sample_description``,
        ``constitution_subcategory``, ``entry_type``, ...); its
        ``source_taxonomy``, when it has one, is copied onto the rows. A
        sample is checked against its ``source_category``: a harmful-taxonomy
        category, or a benign one for the general-benign part.
        ``prompt_config`` is a prompt of
        ``input/generation/from_constitution/{style}``.

        Raises:
            ValueError: A folder written before this layout, or a resume
                whose ``seed``, ``samples_per_entry`` or axes' weights differ
                from those of the plan file.
        """
        if constitution_df.empty:
            logger.warning("run_from_constitution: empty constitution_df, nothing to do")
            return ConstitutionInputResult()
        if axes:
            warn_unshown(prompt_config, "sample_spec", "of the inputs' generation")

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
        if resume or style:
            # Before anything is written: the files the run keeps must take
            # its rows.
            try:
                _stop_on_earlier_files(
                    ledger, sorted(folder.glob(_CATEGORY_FILES)),
                    ["iteration", *spec_columns(axes, {})])
            except ValueError:
                release_lock(ledger)    # drop_units keeps it for the resume
                raise
        # Before anything is written in the folder: it takes the stage's lock.
        processed = resume_state(
            ledger, resume=resume or bool(style), manifest=manifest, **tree)
        seen = set(ledger.ids)      # the content ids of the rows of every style

        made_with = plan_record(seed=seed, samples_per_entry=samples_per_entry,
                                axes=deal_weights(axes))
        changed = changed_plan(manifest.path, made_with) if resume else []
        if changed:
            release_lock(ledger)
            raise ValueError(
                f"Inputs: this run's {changed} differ from those the units in "
                f"{folder} were dealt with (see {plan_path(manifest.path).name}), "
                "and a resume deals them again as they were. Run with "
                "resume=False to start the style over, or with the settings the "
                "plan was made with.")
        if changed is None and processed:
            logger.warning(
                "Inputs: %s has finished units and no %s, so whether this run's "
                "seed, samples_per_entry and axes are the ones they were dealt "
                "with is not checked.", folder, plan_path(manifest.path).name)
        write_plan(manifest.path, made_with)

        entries = [_entry_fields(entry) for _, entry in constitution_df.iterrows()]
        units = [(e, iteration)
                 for e in entries for iteration in range(samples_per_entry)]
        specs = deal_specs(
            axes, [((e.category, e.entry_type), (e.description, iteration))
                   for e, iteration in units], seed=seed)
        manifest.write(
            {"sample_description": e.description, "entry_type": e.entry_type,
             "style": style, "iteration": iteration, "category": e.category,
             **({"spec": spec} if axes else {}), "status": "planned"}
            for (e, iteration), spec in zip(units, specs)
        )
        pending = [(e, iteration, spec) for (e, iteration), spec in zip(units, specs)
                   if (e.description, e.entry_type, style, iteration) not in processed]
        result = ConstitutionInputResult()
        if verbose and len(pending) < len(units):
            logger.info("Resuming: skipped %d already-processed units, %d remaining",
                        len(units) - len(pending), len(pending))
        if not pending:
            if verbose:
                logger.info("All units already processed — nothing to do.")
            release_lock(ledger)    # no CommitBuffer will
            write_stage_coverage(axes, manifest, ledger)
            return result

        total = len(pending)
        style_label = f" | style='{style}'" if style else ""
        if verbose:
            logger.info("Constitution-seeded generation: %d entries, %d samples per "
                        "entry, one call each: %d units, batch_size=%d%s",
                        len(entries), samples_per_entry, total, batch_size,
                        style_label)

        @functools.cache
        def _checker(category: str, entry_type: str, subcategory: str):
            return build_quality_checker(
                category=category, entry_type=entry_type, subcategory=subcategory,
                prompt_dir=self.prompt_dir)

        rejected_units = 0
        finished = 0
        unmarked = 0    # replies without their [END] marker

        def _one_sample(raw: str) -> list[str]:
            """The reply's one sample; ``[]`` for a reply without its ``[END]``
            marker (cut at the token limit) or with another count."""
            nonlocal unmarked
            text, ended = strip_end_marker(raw)
            if not ended:
                unmarked += 1
                return []
            return extract_and_clean(text, style=self.extraction_style,
                                     expected_count=1)

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

        input_label = f"input_{style}" if style else "input"
        check_label = f"val_in_{style}" if style else "val_in"

        def _unit(e: _Entry, iteration: int, spec: dict):
            """Generate one sample to ``spec``, then check it unless the folder
            holds it already."""
            # The sample does not exist yet: internals are captured under the entry.
            eid = content_id(f"{e.description}:{e.entry_type}:{style}")
            fields = {
                "Category": e.category,
                "sample_description": e.description,
                "constitution_subcategory": e.subcategory,
                "entry_type": e.entry_type,
                "sample_spec": render_spec(axes, spec, {"subcategory": e.subcategory}),
            }
            samples, attempts = yield from extracted_unit(
                self.gen,
                lambda: self._build_generation_messages(prompt_config, **fields),
                _one_sample,
                internals_id=(f"{eid}/{iteration}/{input_label}"
                              if capture_input else None),
                max_attempts=max_attempts,
            )
            if not samples:
                return None, None, attempts
            (text,) = samples
            # Outside the extraction: a repeat is not a reason to regenerate.
            if content_id(text) in seen:
                return text, None, attempts
            verdict = (True, "")
            if use_checker:
                # The sample's content id is its sample_id.
                checker = _checker(e.category, e.entry_type, e.subcategory)
                response = yield LLMRequest.for_client(
                    self.check, checker("", text),
                    internals_id=(f"{content_id(text)}/{check_label}"
                                  if capture_val_in else None),
                    max_tokens=check_max_tokens,
                    temperature=CHECK_TEMPERATURE,
                )
                accepted = is_accepted(response)
                verdict = (accepted, "" if accepted else response)
            return text, verdict, attempts

        live: dict[int, tuple[_Entry, int, dict]] = {}   # live units, by position

        def _units():
            for j, unit in enumerate(pending):
                live[j] = unit
                yield j, _unit(*unit)

        def _on_done(j, value):
            nonlocal rejected_units, finished
            e, iteration, spec = live.pop(j)
            text, verdict, attempts = value
            finished += 1
            if verbose:
                desc_preview = e.description[:_DESC_PREVIEW_LEN]
                if len(e.description) > _DESC_PREVIEW_LEN:
                    desc_preview += "..."
                logger.debug("[%d/%d] %s | %s | %s | %d",
                             finished, total, e.category, e.entry_type, desc_preview,
                             iteration)

            key = {"sample_description": e.description, "entry_type": e.entry_type,
                   "style": style, "iteration": iteration}
            row = None
            if verdict is not None:
                accepted, reasoning = verdict
                row = sample_row(
                    text, e.category, turn=0, accepted=accepted,
                    source="constitution",
                    extra={
                        "entry_type": e.entry_type,
                        "subcategory": e.subcategory,
                        "source_sample_description": e.description,
                        "constitution_category": e.constitution_category,
                        "source_group_tag": e.group_tag,
                        # Copied when the constitution has it.
                        **({} if e.source_taxonomy is None
                           else {"source_taxonomy": e.source_taxonomy}),
                        "template_style": style,
                        "rejection_reason": reasoning,
                        "iteration": iteration,
                        **spec_columns(axes, spec),
                    },
                )
            # Again here: a unit that was live with this one may have given
            # its text, and both passed the check when generated.
            if row is None or row["sample_id"] in seen:
                result.skipped_entries += 1
                rejected_units += text is None
                # Acked: un-acked, it would be generated, and paid for, again
                # on every resume.
                status = "rejected" if text is None else "no_sample"
                buf.add([], {**key, "status": status, "attempts": attempts})
                if verbose:
                    logger.debug("-> no sample")
                return

            seen.add(row["sample_id"])
            # The id of the row it wrote, what a resume knows it by, and the
            # verdict, what the coverage table counts.
            buf.add([row], {**key, "ids": [row["sample_id"]], "accepted": accepted})
            result.total_entries_processed += 1
            result.total_prompts_generated += 1
            result.total_prompts_accepted += accepted
            result.total_prompts_rejected += not accepted
            if verbose:
                logger.debug("-> %s", "accepted" if accepted else "rejected")

        # A prompt too long for the model ends the unit as one rejected on
        # every attempt; the ack says 1. Any other unit error stops the stage.
        def _on_error(j, exc):
            if not isinstance(exc, PromptTooLongError):
                raise exc
            e, iteration, _ = live[j]
            logger.warning("Entry %r (%s), iteration %d, rejected, its prompt is too "
                           "long: %s", e.description, e.entry_type, iteration, exc)
            _on_done(j, (None, None, 1))

        with CommitBuffer(append=_append, ledger=ledger) as buf:
            drive_units(
                _units(),
                on_done=_on_done,
                on_error=_on_error,
                pool_size=pool_size(batch_size, self.gen, self.check),
                cap=batch_size,
                progress="inputs" if verbose else None,
            )
        write_stage_coverage(axes, manifest, ledger)

        if unmarked:
            logger.warning(
                "%d replies had no [END] marker at their end, cut at the token limit "
                "or written without it, and were asked again.", unmarked)
        if rejected_units:
            logger.warning(
                "%d units rejected after %d attempt(s) each. They are acked "
                "and skipped on resume; run with resume=False, or delete their "
                "'rejected' lines from the ledger, to try them again.",
                rejected_units, max_attempts,
            )
        if verbose:
            logger.info("Constitution-seeded run complete%s: %d/%d accepted (%.0f%%)",
                        style_label, result.total_prompts_accepted,
                        result.total_prompts_generated, result.acceptance_rate * 100)

        return result
