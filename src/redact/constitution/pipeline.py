"""Constitution generation pipeline for constitutional classifiers.

Generates structured category hierarchies (constitutions) across 4 severity
levels: absolutely harmful, dual-use harmful, dual-use benign, and absolutely
benign. Each constitution entry later seeds N input samples for classifier
training.

Usage:
    from redact.llms import get_backend, RateLimiter
    from redact.constitution import ConstitutionPipeline, EntryType

    backend = get_backend("claude-opus-4-6")
    pipeline = ConstitutionPipeline(backend, "claude-opus-4-6", RateLimiter())

    result = pipeline.run(
        taxonomy=load_taxonomy("content_moderation_categories"),
        num_categories=10,
    )
    df = result.to_dataframe()
"""

import csv
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from ..llms.base import LLMBackend
from ..llms.calls import generate_sample
from ..llms.extraction import parse_constitution as _parse_raw
from ..llms.prompts import load_prompt, build_messages
from ..llms.wrappers import RateLimiter
from ..dataset.taxonomy import iter_categories, get_subcategories

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry type enum
# ---------------------------------------------------------------------------


class EntryType(str, Enum):
    """Severity level for a constitution entry.

    Spectrum:
        HARMFUL <-> DUAL_USE_HARMFUL <-> DUAL_USE_BENIGN <-> BENIGN
    """

    HARMFUL = "harmful"
    DUAL_USE_HARMFUL = "dual_use_harmful"
    DUAL_USE_BENIGN = "dual_use_benign"
    BENIGN = "benign"


ALL_ENTRY_TYPES = list(EntryType)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConstitutionEntry:
    """A single parsed + tagged constitution entry."""

    category: str  # LLM-generated category (## heading)
    subcategory: str  # LLM-generated subcategory (### heading)
    sample: str  # Sample description (- (...) line)
    entry_type: str  # "harmful" / "benign" / "dual_use_benign" / "dual_use_harmful"
    source_category: str  # Taxonomy category that seeded this
    source_group_tag: str  # group_tag from taxonomy


@dataclass
class ConstitutionResult:
    """Result of a constitution generation run."""

    entries: list[ConstitutionEntry] = field(default_factory=list)
    raw_outputs: dict[str, str] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entries to a DataFrame."""
        if not self.entries:
            return pd.DataFrame(columns=_CSV_COLUMNS)
        return pd.DataFrame(
            [
                {
                    "constitution_category": e.category,
                    "constitution_subcategory": e.subcategory,
                    "sample_description": e.sample,
                    "entry_type": e.entry_type,
                    "source_category": e.source_category,
                    "source_group_tag": e.source_group_tag,
                }
                for e in self.entries
            ]
        )


# ---------------------------------------------------------------------------
# [END] marker handling
# ---------------------------------------------------------------------------

_END_MARKER = re.compile(r"\n?\[END\]\s*$")


def _strip_end_marker(text: str) -> tuple[str, bool]:
    """Strip [END] marker from LLM output.

    Returns:
        (clean_text, was_complete): was_complete is True if [END] was found,
        meaning the model finished normally rather than hitting the token limit.
    """
    if _END_MARKER.search(text):
        return _END_MARKER.sub("", text), True
    return text, False


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "constitution_category",
    "constitution_subcategory",
    "sample_description",
    "entry_type",
    "source_category",
    "source_group_tag",
    "model",
]


def _save_entries_csv(
    entries: list[ConstitutionEntry],
    path: Path,
    model: str,
    append: bool = False,
) -> None:
    """Save constitution entries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    write_header = mode == "w" or not path.exists()

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "constitution_category": e.category,
                    "constitution_subcategory": e.subcategory,
                    "sample_description": e.sample,
                    "entry_type": e.entry_type,
                    "source_category": e.source_category,
                    "source_group_tag": e.source_group_tag,
                    "model": model,
                }
            )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConstitutionPipeline:
    """Generate constitution entries from taxonomy categories.

    For each taxonomy category, makes up to 4 LLM calls (one per EntryType),
    parses the 3-layer markdown via parse_constitution(), tags entries, and
    saves type-based CSVs.

    Attributes:
        backend: LLM backend for generation.
        model: Model identifier (default: claude-opus-4-6).
        rate_limiter: Optional shared rate limiter.
        output_dir: Directory for saving CSVs (Data_cache/constitution/).
    """

    def __init__(
        self,
        backend: LLMBackend,
        model: str = "claude-opus-4-6",
        rate_limiter: RateLimiter | None = None,
        output_dir: str | Path | None = None,
    ):
        self.backend = backend
        self.model = model
        self.rate_limiter = rate_limiter

        if output_dir is None:
            from redact import get_output_dir

            output_dir = get_output_dir() / "Data_cache" / "constitution"
        self.output_dir = Path(output_dir)

    def _format_category_info(
        self, category_name: str, category_info: dict
    ) -> dict[str, str]:
        """Format taxonomy category info as template kwargs."""
        subcats = category_info.get("subcategories", [])
        if isinstance(subcats, dict):
            subcats_str = ", ".join(subcats.keys())
        elif isinstance(subcats, list):
            subcats_str = ", ".join(subcats)
        else:
            subcats_str = str(subcats)

        return {
            "Category": category_name,
            "description": category_info.get("description", ""),
            "subcategories": subcats_str,
            "group_tag": category_info.get("group_tag", "general"),
        }

    def generate_for_type(
        self,
        category_name: str,
        category_info: dict,
        entry_type: EntryType,
        num_categories: int = 10,
    ) -> tuple[list[ConstitutionEntry], str]:
        """Generate constitution entries of one type for one taxonomy category.

        Makes an LLM call with up to 2 retries on failure, parses the 3-layer
        markdown output, and tags each entry with metadata.

        Args:
            category_name: Taxonomy category name (e.g. "CBRN").
            category_info: Dict with description, subcategories, group_tag.
            entry_type: Which severity level to generate.
            num_categories: Number of constitution categories to request.

        Returns:
            (entries, raw_output): List of tagged entries and raw LLM text.
            Returns ([], "") if all attempts fail.
        """
        prompt_config = load_prompt("constitution", entry_type.value)
        kwargs = self._format_category_info(category_name, category_info)
        kwargs["num_categories"] = str(num_categories)
        messages = build_messages(prompt_config, **kwargs)
        group_tag = category_info.get("group_tag", "general")

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_output = generate_sample(
                    self.backend,
                    self.model,
                    messages,
                    rate_limiter=self.rate_limiter,
                    max_tokens=10000,
                )

                # Strip [END] marker and check completeness
                clean_output, was_complete = _strip_end_marker(raw_output)
                if not was_complete:
                    logger.warning(
                        "Constitution output for %s/%s may be truncated "
                        "(no [END] marker).",
                        category_name,
                        entry_type.value,
                    )

                # Parse 3-layer markdown into entries
                parsed = _parse_raw(clean_output)

                entries = [
                    ConstitutionEntry(
                        category=p.category,
                        subcategory=p.subcategory,
                        sample=p.sample,
                        entry_type=entry_type.value,
                        source_category=category_name,
                        source_group_tag=group_tag,
                    )
                    for p in parsed
                ]

                if not entries:
                    raise ValueError(
                        f"No entries parsed (raw length: {len(raw_output)} chars)"
                    )

                return entries, raw_output

            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Attempt %d/%d failed for %s/%s: %s. Retrying...",
                        attempt + 1,
                        max_attempts,
                        category_name,
                        entry_type.value,
                        e,
                    )
                else:
                    logger.error(
                        "All %d attempts failed for %s/%s: %s. Skipping.",
                        max_attempts,
                        category_name,
                        entry_type.value,
                        e,
                    )
                    return [], ""

    def generate_for_category(
        self,
        category_name: str,
        category_info: dict,
        entry_types: list[EntryType] | None = None,
        num_categories: int = 10,
    ) -> ConstitutionResult:
        """Generate all entry types for one taxonomy category.

        Args:
            category_name: Taxonomy category name.
            category_info: Dict with description, subcategories, group_tag.
            entry_types: Which types to generate. Default: all four.
            num_categories: Number of constitution categories per type.

        Returns:
            ConstitutionResult with all entries and raw outputs.
        """
        if entry_types is None:
            entry_types = ALL_ENTRY_TYPES

        result = ConstitutionResult()

        for entry_type in entry_types:
            entries, raw = self.generate_for_type(
                category_name, category_info, entry_type, num_categories
            )
            result.entries.extend(entries)
            key = f"{category_name}/{entry_type.value}"
            result.raw_outputs[key] = raw

        return result

    def _save_category_entries(
        self,
        cat_result: ConstitutionResult,
    ) -> None:
        """Append a single category's entries to the per-type and merged CSVs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        by_type: dict[str, list[ConstitutionEntry]] = {}
        for e in cat_result.entries:
            by_type.setdefault(e.entry_type, []).append(e)

        for entry_type_val, type_entries in by_type.items():
            _save_entries_csv(
                type_entries,
                self.output_dir / f"{entry_type_val}.csv",
                self.model,
                append=True,
            )

        _save_entries_csv(
            cat_result.entries,
            self.output_dir / "merged.csv",
            self.model,
            append=True,
        )

    def run(
        self,
        taxonomy: dict,
        entry_types: list[EntryType] | None = None,
        num_categories: int = 10,
        num_taxonomy_categories: int | None = None,
        include_standalone_benign: bool = False,
        standalone_benign_categories: int = 10,
        save: bool = True,
        verbose: bool = True,
    ) -> ConstitutionResult:
        """Run constitution generation across taxonomy categories.

        Args:
            taxonomy: Loaded taxonomy dict.
            entry_types: Which types to generate. Default: all four.
            num_categories: Constitution categories per type per taxonomy category.
            num_taxonomy_categories: Limit to first N taxonomy categories (None = all).
            include_standalone_benign: If True, also generate benign entries for
                ALL taxonomy categories (not just those selected for the main run).
                This broadens hard-negative coverage for classifier training.
            standalone_benign_categories: Number of benign constitution categories
                to generate per taxonomy category in standalone benign mode.
            save: Whether to save CSVs to output_dir.
            verbose: Print progress.

        Returns:
            ConstitutionResult with all entries across all categories.
        """
        if entry_types is None:
            entry_types = ALL_ENTRY_TYPES

        all_categories = list(iter_categories(taxonomy))
        categories = all_categories
        if num_taxonomy_categories is not None:
            categories = categories[:num_taxonomy_categories]

        if verbose:
            print(f"\n{'='*60}")
            print("Constitution Generation")
            print(f"{'='*60}")
            print(f"  Model: {self.model}")
            print(f"  Entry types: {[t.value for t in entry_types]}")
            print(f"  Categories per type: {num_categories}")
            print(f"  Taxonomy categories: {len(categories)}")
            total_calls = len(categories) * len(entry_types)
            if include_standalone_benign:
                # Standalone benign covers all taxonomy categories not already
                # getting benign entries in the main run
                standalone_extra = (
                    len(all_categories) - len(categories)
                    if EntryType.BENIGN in entry_types
                    else len(all_categories)
                )
                total_calls += standalone_extra
                print(f"  Standalone benign: {len(all_categories)} taxonomy categories"
                      f" x {standalone_benign_categories} categories each")
            print(f"  Total LLM calls: ~{total_calls}")
            print()

        # Clear existing CSVs for a fresh run
        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for f in self.output_dir.glob("*.csv"):
                f.unlink()

        all_result = ConstitutionResult()
        skipped: list[str] = []

        for i, (cat_name, cat_info) in enumerate(categories, 1):
            if verbose:
                print(f"  [{i}/{len(categories)}] {cat_name}")

            cat_result = self.generate_for_category(
                cat_name, cat_info, entry_types, num_categories
            )

            all_result.entries.extend(cat_result.entries)
            all_result.raw_outputs.update(cat_result.raw_outputs)

            # Track skipped types (empty results from failed retries)
            for et in entry_types:
                key = f"{cat_name}/{et.value}"
                if key in cat_result.raw_outputs and cat_result.raw_outputs[key] == "":
                    skipped.append(f"{cat_name}/{et.value}")

            # Count by type for this category
            if verbose:
                type_counts: dict[str, int] = {}
                for e in cat_result.entries:
                    type_counts[e.entry_type] = type_counts.get(e.entry_type, 0) + 1
                counts_str = ", ".join(
                    f"{t}: {c}" for t, c in sorted(type_counts.items())
                )
                print(f"    -> {len(cat_result.entries)} entries ({counts_str})")

            # Save iteratively after each category
            if save and cat_result.entries:
                self._save_category_entries(cat_result)

        # ── Standalone benign generation ──────────────────────────────
        if include_standalone_benign:
            if verbose:
                print(f"\n  --- Standalone Benign Generation ---")

            for i, (cat_name, cat_info) in enumerate(all_categories, 1):
                if verbose:
                    print(f"  [benign {i}/{len(all_categories)}] {cat_name}")

                entries, raw = self.generate_for_type(
                    cat_name, cat_info, EntryType.BENIGN,
                    num_categories=standalone_benign_categories,
                )

                if not entries:
                    skipped.append(f"{cat_name}/benign (standalone)")
                    continue

                cat_result = ConstitutionResult(
                    entries=entries,
                    raw_outputs={f"{cat_name}/benign_standalone": raw},
                )

                all_result.entries.extend(entries)
                all_result.raw_outputs[f"{cat_name}/benign_standalone"] = raw

                if verbose:
                    print(f"    -> {len(entries)} benign entries")

                if save:
                    self._save_category_entries(cat_result)

        if verbose:
            print(f"\n  Total: {len(all_result.entries)} constitution entries")
            if skipped:
                print(f"  Skipped (failed after retries): {len(skipped)}")
                for s in skipped:
                    print(f"    - {s}")
            if save:
                print(f"  Saved to: {self.output_dir}")

        return all_result

    def save(self, result: ConstitutionResult) -> None:
        """Save entries to 4 type-based CSVs + merged.csv.

        Files written to self.output_dir:
            harmful.csv, benign.csv, dual_use_benign.csv, dual_use_harmful.csv,
            merged.csv
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Group entries by type
        by_type: dict[str, list[ConstitutionEntry]] = {}
        for e in result.entries:
            by_type.setdefault(e.entry_type, []).append(e)

        # Save per-type CSVs
        for entry_type_val, entries in by_type.items():
            path = self.output_dir / f"{entry_type_val}.csv"
            _save_entries_csv(entries, path, self.model)
            logger.info("Saved %d entries to %s", len(entries), path)

        # Save merged
        merged_path = self.output_dir / "merged.csv"
        _save_entries_csv(result.entries, merged_path, self.model)
        logger.info("Saved %d total entries to %s", len(result.entries), merged_path)
