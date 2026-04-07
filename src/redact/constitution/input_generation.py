"""Constitution-to-input sample generation pipeline.

# CONSTITUTION-TO-INPUT PIPELINE
# Converts constitution entry descriptions into full realistic input prompts.
# This bridges the constitution pipeline (short descriptions) to the content
# moderation input pipeline (full prompts for classifier training).
#
# Each constitution entry's sample_description is expanded into multiple
# full-length prompts using configurable template styles ("long", "short",
# or any custom style added to prompts/constitution/input_generation/).
#
# Future work: chain with generate_outputs() for output samples, and
# generate_jailbreaks() for augmented training data.

Reads constitution CSVs from Data_cache/constitution/, expands each entry's
sample_description into full prompts using the content moderation InputPipeline
as the generation engine, and saves results in standard content moderation
format with constitution metadata preserved.

Usage:
    from redact.llms import get_backend, RateLimiter
    from redact.constitution.input_generation import ConstitutionInputPipeline

    backend = get_backend("venice-uncensored")
    pipeline = ConstitutionInputPipeline(
        gen_backend=backend, gen_model="venice-uncensored",
        check_backend=backend, check_model="venice-uncensored",
    )
    result = pipeline.run(style="long", samples_per_entry=3)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..content_moderation.generation import InputPipeline, SampleResult
from ..llms.base import LLMBackend
from ..llms.calls import batch_check_samples
from ..llms.extraction import extract_and_clean
from ..llms.prompts import load_prompt
from ..llms.wrappers import RateLimiter
from ..dataset.io import append_samples, get_existing_samples

logger = logging.getLogger(__name__)

# Prompt pipeline path for load_prompt()
_PROMPT_PIPELINE = "constitution/input_generation"

# Default prompt directory (prompts/ inside the redact package)
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_PROMPT_DIR = _PACKAGE_DIR / "prompts"


# ---------------------------------------------------------------------------
# Constitution-specific checker
# ---------------------------------------------------------------------------


def _build_constitution_checker(
    category: str,
    entry_type: str,
    subcategory: str = "",
    prompt_dir: str | Path | None = None,
) -> Callable[[str], list[dict]]:
    """Build a quality checker aware of entry type and subcategory.

    Loads ``prompts/constitution/checker/template.json`` and injects
    category, subcategory, and entry_type into the system prompt so the
    evaluator knows what kind of sample it is validating (harmful vs
    benign vs dual-use).

    This is separate from ``content_moderation.checker.build_quality_checker``
    which only handles harmful content. See checker.py for the TODO on
    extending that function when benign content moderation is added.
    """
    if prompt_dir is None:
        prompt_dir = _DEFAULT_PROMPT_DIR
    prompt_config = load_prompt("constitution", "checker", prompt_dir=prompt_dir)
    system_prompt = prompt_config["system_prompt"].format(
        Category=category,
        entry_type=entry_type,
        subcategory=subcategory or category,
    )

    def _build(sample: str) -> list[dict]:
        template = prompt_config["template"].format(sample=sample)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template},
        ]

    return _build


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class ConstitutionInputResult:
    """Aggregate result of constitution-to-input generation.

    # CONSTITUTION-TO-INPUT: Tracks generation statistics across all entries.
    """

    total_entries_processed: int = 0
    total_prompts_generated: int = 0
    total_prompts_accepted: int = 0
    total_prompts_rejected: int = 0
    skipped_entries: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_prompts_generated == 0:
            return 0.0
        return self.total_prompts_accepted / self.total_prompts_generated


# ---------------------------------------------------------------------------
# Style discovery
# ---------------------------------------------------------------------------


def get_available_styles(prompt_dir: Path | None = None) -> list[str]:
    """Return list of available template style names.

    # CONSTITUTION-TO-INPUT: Auto-discovers styles from prompt directories.
    # To add a new style, create prompts/constitution/input_generation/{style}/template.json

    Args:
        prompt_dir: Root prompts directory. Defaults to package prompts/.

    Returns:
        Sorted list of style names (e.g. ["long", "short"]).
    """
    base = (prompt_dir or _DEFAULT_PROMPT_DIR) / "constitution" / "input_generation"
    if not base.is_dir():
        return []
    return sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and (d / "template.json").exists()
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConstitutionInputPipeline:
    """Expand constitution descriptions into full input prompts.

    # CONSTITUTION-TO-INPUT: Core pipeline class.
    # Composes with content moderation InputPipeline for generation/extraction/checking.
    # Does NOT extend InputPipeline — the iteration pattern is fundamentally
    # different (per-entry with unique sample_description vs per-category multi-turn).

    Each constitution entry has a short sample_description (e.g. "Instructions
    for making pipe bombs"). This pipeline uses the content moderation
    InputPipeline to expand each description into full realistic prompts via
    LLM generation, numbered-list extraction, and optional quality checking.

    Output is saved in standard content moderation format (compatible with
    generate_outputs() and generate_jailbreaks()) with constitution metadata
    preserved as extra columns.
    """

    def __init__(
        self,
        gen_backend: LLMBackend,
        gen_model: str,
        check_backend: LLMBackend | None = None,
        check_model: str | None = None,
        rate_limiter: RateLimiter | None = None,
        extraction_style: str = "numbered",
        constitution_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        """Create a ConstitutionInputPipeline.

        Args:
            gen_backend: LLM backend for generation.
            gen_model: Model identifier for generation.
            check_backend: LLM backend for checking. Defaults to gen_backend.
            check_model: Model identifier for checking. Defaults to gen_model.
            rate_limiter: Optional shared rate limiter.
            extraction_style: Extraction style for InputPipeline ("numbered",
                "structured_qa", or "delimiter").
            constitution_dir: Where to read constitution CSVs.
                Defaults to Data_cache/constitution/.
            output_dir: Where to save generated input samples.
                Defaults to Datasets/constitution_inputs/.
        """
        check_backend = check_backend or gen_backend
        check_model = check_model or gen_model

        # Compose with InputPipeline for generate+extract+check
        self.input_pipeline = InputPipeline(
            gen_backend=gen_backend,
            gen_model=gen_model,
            check_backend=check_backend,
            check_model=check_model,
            rate_limiter=rate_limiter,
            extraction_style=extraction_style,
        )

        # Resolve default directories
        if constitution_dir is None:
            from redact import get_output_dir
            constitution_dir = get_output_dir() / "Data_cache" / "constitution"
        self.constitution_dir = Path(constitution_dir)

        if output_dir is None:
            from redact import get_output_dir
            output_dir = get_output_dir() / "Datasets" / "constitution_inputs"
        self.output_dir = Path(output_dir)

    # -----------------------------------------------------------------
    # Constitution loading
    # -----------------------------------------------------------------

    def _load_constitution(
        self,
        entry_types: list[str] | None = None,
        source_categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load and filter constitution entries from CSVs.

        # CONSTITUTION-TO-INPUT: Reads from Data_cache/constitution/.

        Tries merged.csv first, then falls back to loading individual
        type CSVs and concatenating them.

        Args:
            entry_types: Filter to these entry types (e.g. ["harmful", "benign"]).
            source_categories: Filter to these taxonomy categories.

        Returns:
            Filtered DataFrame of constitution entries.
        """
        merged_path = self.constitution_dir / "merged.csv"
        if merged_path.exists():
            df = pd.read_csv(merged_path)
        else:
            # Fall back to loading individual type CSVs
            parts = []
            for csv_path in self.constitution_dir.glob("*.csv"):
                if csv_path.name == "merged.csv":
                    continue
                try:
                    parts.append(pd.read_csv(csv_path))
                except Exception:
                    logger.warning("Failed to load %s", csv_path)
            if not parts:
                raise FileNotFoundError(
                    f"No constitution CSVs found in {self.constitution_dir}. "
                    f"Run generate_constitution() first."
                )
            df = pd.concat(parts, ignore_index=True)

        # Apply filters
        if entry_types is not None:
            df = df[df["entry_type"].isin(entry_types)]
        if source_categories is not None:
            df = df[df["source_category"].isin(source_categories)]

        if df.empty:
            logger.warning(
                "No constitution entries match filters: "
                "entry_types=%s, source_categories=%s",
                entry_types,
                source_categories,
            )

        return df.reset_index(drop=True)

    # -----------------------------------------------------------------
    # Per-entry generation
    # -----------------------------------------------------------------

    def generate_for_entry(
        self,
        entry: pd.Series,
        prompt_config: dict,
        samples_per_entry: int = 3,
        build_check_messages: Callable[[str], list[dict]] | None = None,
        prohibited: set[str] | None = None,
    ) -> list[SampleResult]:
        """Generate full prompts from a single constitution entry.

        # CONSTITUTION-TO-INPUT: Per-entry generation via InputPipeline composition.
        # Uses InputPipeline.generate_batch() for LLM call + extraction,
        # then InputPipeline.check_samples() for quality validation.

        Args:
            entry: Single row from the constitution DataFrame.
            prompt_config: Loaded prompt template config.
            samples_per_entry: Number of prompts to generate per entry.
            build_check_messages: Quality checker callable. If None, skips checking.
            prohibited: Existing sample texts to avoid.

        Returns:
            List of SampleResult objects (accepted or rejected).
        """
        # Build seed kwargs from constitution entry metadata
        seed_kwargs = {
            "Category": str(entry.get("source_category", "")),
            "sample_description": str(entry.get("sample_description", "")),
            "constitution_subcategory": str(entry.get("constitution_subcategory", "")),
            "entry_type": str(entry.get("entry_type", "")),
        }

        # Generate via InputPipeline (handles prompt rendering, format
        # instruction injection, LLM call, and numbered-list extraction)
        raw_output, extracted = self.input_pipeline.generate_batch(
            prompt_config=prompt_config,
            samples_per_request=samples_per_entry,
            prohibited=prohibited,
            **seed_kwargs,
        )

        logger.info(
            "Entry '%s': extracted %d samples",
            entry.get("sample_description", "")[:50],
            len(extracted),
        )

        # Dedup against existing samples
        if prohibited:
            before = len(extracted)
            extracted = [s for s in extracted if s not in prohibited]
            removed = before - len(extracted)
            if removed > 0:
                logger.info("Removed %d duplicates", removed)

        if not extracted:
            return []

        # Quality check via InputPipeline (optional)
        if build_check_messages is not None:
            return self.input_pipeline.check_samples(
                extracted, build_check_messages, turn_index=0
            )

        # Accept all without checking
        return [
            SampleResult(text=s, accepted=True, reasoning="", turn=0)
            for s in extracted
        ]

    # -----------------------------------------------------------------
    # Saving
    # -----------------------------------------------------------------

    def _save_results(
        self,
        entry: pd.Series,
        results: list[SampleResult],
        style: str,
    ) -> None:
        """Save generated prompts in standard content moderation format.

        # CONSTITUTION-TO-INPUT: Saves with constitution metadata as extra columns.
        # Output is compatible with generate_outputs() and generate_jailbreaks().

        Args:
            entry: Constitution entry that seeded the generation.
            results: List of SampleResult objects to save.
            style: Template style used (e.g. "long", "short").
        """
        if not results:
            return

        category = str(entry.get("source_category", "unknown"))
        texts = [r.text for r in results]
        accepted = [r.accepted for r in results]

        # Constitution metadata saved as extra columns for traceability
        extra = [
            {
                "constitution_category": str(entry.get("constitution_category", "")),
                "constitution_subcategory": str(entry.get("constitution_subcategory", "")),
                "sample_description": str(entry.get("sample_description", "")),
                "entry_type": str(entry.get("entry_type", "")),
                "source_group_tag": str(entry.get("source_group_tag", "")),
                "template_style": style,
                "reasoning": r.reasoning,
            }
            for r in results
        ]

        append_samples(
            texts,
            category=category,
            turn=0,
            accepted=accepted,
            source="constitution_to_input",
            extra_columns=extra,
            dataset_dir=self.output_dir,
        )

    # -----------------------------------------------------------------
    # Main run
    # -----------------------------------------------------------------

    def run(
        self,
        style: str = "long",
        samples_per_entry: int = 3,
        entry_types: list[str] | None = None,
        source_categories: list[str] | None = None,
        use_checker: bool = True,
        save: bool = True,
        verbose: bool = True,
        batch_size: int = 32,
    ) -> ConstitutionInputResult:
        """Run constitution-to-input generation with batched vLLM inference.

        # CONSTITUTION-TO-INPUT: Main pipeline entry point.
        # Processes entries in batches of ``batch_size`` for efficiency:
        #   1. Batch-generate prompts for all entries in the chunk (one vLLM pass)
        #   2. Extract samples per entry
        #   3. Batch-check all extracted samples across the chunk (one vLLM pass)
        #   4. Redistribute results back to their entries, save, update stats

        Args:
            style: Template style ("long", "short", or any custom style
                found in prompts/constitution/input_generation/).
            samples_per_entry: Number of prompts to generate per constitution entry.
            entry_types: Filter to specific entry types
                (e.g. ["harmful", "benign", "dual_use_harmful", "dual_use_benign"]).
            source_categories: Filter to specific taxonomy categories.
            use_checker: Whether to quality-check generated prompts via
                _build_constitution_checker(). Set False for faster iteration.
            save: Whether to save results to CSV.
            verbose: Print progress.
            batch_size: Number of entries to process per vLLM engine pass (default 32).

        Returns:
            ConstitutionInputResult with generation statistics.
        """
        # Load prompt template for the chosen style
        prompt_config = load_prompt(_PROMPT_PIPELINE, style)

        # Load and filter constitution entries
        constitution_df = self._load_constitution(entry_types, source_categories)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Constitution-to-Input Generation")
            print(f"{'='*60}")
            print(f"Style: {style}")
            print(f"Entries: {len(constitution_df)}")
            print(f"Samples per entry: {samples_per_entry}")
            print(f"Batch size: {batch_size}")
            print(f"Quality checker: {'enabled' if use_checker else 'disabled'}")
            if entry_types:
                print(f"Entry types: {entry_types}")
            if source_categories:
                print(f"Source categories: {source_categories}")

        result = ConstitutionInputResult()
        prohibited: set[str] = set()

        # Cache checkers per source_category to avoid rebuilding
        checker_cache: dict[tuple[str, str], Callable[[str], list[dict]]] = {}

        entries = list(constitution_df.iterrows())
        total = len(entries)

        for batch_start in range(0, total, batch_size):
            batch = entries[batch_start : batch_start + batch_size]

            # -----------------------------------------------------------------
            # Step 1: Build one generation message list per entry in the batch
            # -----------------------------------------------------------------
            messages_list = []
            for _, entry in batch:
                seed_kwargs = {
                    "Category": str(entry.get("source_category", "")),
                    "sample_description": str(entry.get("sample_description", "")),
                    "constitution_subcategory": str(entry.get("constitution_subcategory", "")),
                    "entry_type": str(entry.get("entry_type", "")),
                }
                messages = self.input_pipeline._build_generation_messages(
                    prompt_config,
                    samples_per_request=samples_per_entry,
                    prohibited=prohibited,
                    **seed_kwargs,
                )
                messages_list.append(messages)

            # -----------------------------------------------------------------
            # Step 2: One batch_generate call for all entries in the chunk
            # -----------------------------------------------------------------
            raw_outputs = self.input_pipeline.gen_backend.batch_generate(
                messages_list, self.input_pipeline.gen_model
            )

            # -----------------------------------------------------------------
            # Step 3: Extract samples per entry; track flat index ranges
            # -----------------------------------------------------------------
            # all_samples[i] is the flat list of extracted texts across entries.
            # entry_ranges[i] = (start, end) slice into all_samples for entry i.
            all_samples: list[str] = []
            entry_ranges: list[tuple[int, int]] = []
            per_entry_extracted: list[list[str]] = []

            for (_, entry), raw_output in zip(batch, raw_outputs):
                extracted = extract_and_clean(
                    raw_output,
                    style=self.input_pipeline.extraction_style,
                )

                logger.info(
                    "Entry '%s': extracted %d samples",
                    entry.get("sample_description", "")[:50],
                    len(extracted),
                )

                # Dedup against prohibited set
                if prohibited:
                    before = len(extracted)
                    extracted = [s for s in extracted if s not in prohibited]
                    removed = before - len(extracted)
                    if removed > 0:
                        logger.info("Removed %d duplicates", removed)

                start = len(all_samples)
                all_samples.extend(extracted)
                end = len(all_samples)
                entry_ranges.append((start, end))
                per_entry_extracted.append(extracted)

            # -----------------------------------------------------------------
            # Step 4: Batch-check all extracted samples across the chunk
            # -----------------------------------------------------------------
            if use_checker and all_samples:
                # We need a checker — use the first entry's category to look up
                # one (all entries may have different categories, so we build
                # per-entry checkers and call them on their own slices below).
                # Here we collect all check results in one batch pass, using
                # each sample's own checker determined by entry index.

                # Build the flat checker list parallel to all_samples.
                # Cache key is (category, entry_type) — benign and harmful
                # entries for the same category need different checkers.
                sample_checkers: list[Callable[[str], list[dict]]] = []
                for (entry_idx, (_, entry)), (start, end) in zip(
                    enumerate(batch), entry_ranges
                ):
                    category = str(entry.get("source_category", "unknown"))
                    entry_type = str(entry.get("entry_type", "harmful"))
                    subcategory = str(entry.get("constitution_subcategory", ""))
                    cache_key = (category, entry_type)
                    if cache_key not in checker_cache:
                        checker_cache[cache_key] = _build_constitution_checker(
                            category, entry_type, subcategory
                        )
                    checker = checker_cache[cache_key]
                    sample_checkers.extend([checker] * (end - start))

                # Single batch_check_samples call — but checkers may differ per
                # sample. batch_check_samples takes one build_check_messages
                # function, so we inline the batching here to support per-sample
                # checkers:
                messages_to_check = [
                    checker(sample)
                    for checker, sample in zip(sample_checkers, all_samples)
                ]
                if messages_to_check:
                    check_responses = self.input_pipeline.check_backend.batch_generate(
                        messages_to_check, self.input_pipeline.check_model
                    )
                    flat_check_results = []
                    for response in check_responses:
                        accepted = any(
                            response.strip().lower().startswith(p)
                            for p in ("yes", "ok", "accept", "pass")
                        )
                        flat_check_results.append((accepted, "" if accepted else response))
                else:
                    flat_check_results = []
            else:
                # No checker: accept all
                flat_check_results = [(True, "") for _ in all_samples]

            # -----------------------------------------------------------------
            # Step 5: Redistribute results back to entries, save, update stats
            # -----------------------------------------------------------------
            for entry_idx, (_, entry) in enumerate(batch):
                global_idx = batch_start + entry_idx
                start, end = entry_ranges[entry_idx]
                extracted = per_entry_extracted[entry_idx]

                category = str(entry.get("source_category", "unknown"))
                sample_desc = str(entry.get("sample_description", ""))

                if verbose:
                    print(
                        f"\n  [{global_idx + 1}/{total}] "
                        f"{category} | {entry.get('entry_type', '?')} | "
                        f"{sample_desc[:60]}{'...' if len(sample_desc) > 60 else ''}"
                    )

                if not extracted:
                    result.skipped_entries += 1
                    if verbose:
                        print(f"    -> No samples extracted, skipping")
                    continue

                sample_results = [
                    SampleResult(
                        text=sample_text,
                        accepted=accepted,
                        reasoning=reasoning,
                        turn=0,
                    )
                    for sample_text, (accepted, reasoning) in zip(
                        extracted, flat_check_results[start:end]
                    )
                ]

                if save:
                    self._save_results(entry, sample_results, style)

                accepted_list = [r for r in sample_results if r.accepted]
                rejected_list = [r for r in sample_results if not r.accepted]

                result.total_entries_processed += 1
                result.total_prompts_generated += len(sample_results)
                result.total_prompts_accepted += len(accepted_list)
                result.total_prompts_rejected += len(rejected_list)

                for sr in sample_results:
                    prohibited.add(sr.text)

                if verbose:
                    print(
                        f"    -> {len(accepted_list)} accepted, "
                        f"{len(rejected_list)} rejected"
                    )

        # Summary
        if verbose:
            print(f"\n{'='*60}")
            print(f"Summary")
            print(f"{'='*60}")
            print(f"  Entries processed: {result.total_entries_processed}")
            print(f"  Entries skipped:   {result.skipped_entries}")
            print(f"  Prompts generated: {result.total_prompts_generated}")
            print(f"  Prompts accepted:  {result.total_prompts_accepted}")
            print(f"  Prompts rejected:  {result.total_prompts_rejected}")
            print(f"  Acceptance rate:   {result.acceptance_rate:.0%}")
            if save:
                print(f"  Saved to:          {self.output_dir}/")

        return result
