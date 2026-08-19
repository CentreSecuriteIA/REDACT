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
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ..content_moderation.checker import build_quality_checker
from ..content_moderation.generation import (
    ConstitutionInputResult,
    InputPipeline,
)
from ..llms.base import LLMBackend
from ..llms.prompts import load_prompt
from ..llms.wrappers import RateLimiter

logger = logging.getLogger(__name__)

# Prompt pipeline path for load_prompt(); from-constitution input templates
# now live under prompts/input/generation/from_constitution/{style}/.
_PROMPT_PIPELINE = "input"

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
    """Backward-compatible alias for the unified entry-type-aware checker.

    Delegates to ``content_moderation.checker.build_quality_checker`` which
    now reads the unified ``prompts/input/quality_check/template.json``.
    Kept here so existing imports of ``_build_constitution_checker`` keep
    working during the migration window.
    """
    return build_quality_checker(
        category=category,
        entry_type=entry_type,
        subcategory=subcategory,
        prompt_dir=str(prompt_dir) if prompt_dir is not None else None,
    )


# ---------------------------------------------------------------------------
# Result data class — now defined in content_moderation/generation.py and
# re-exported here for backward compatibility.
# ---------------------------------------------------------------------------

# (ConstitutionInputResult imported above from content_moderation.generation)


# ---------------------------------------------------------------------------
# Style discovery
# ---------------------------------------------------------------------------


def get_available_styles(prompt_dir: Path | None = None) -> list[str]:
    """Return list of available template style names.

    Auto-discovers styles from
    ``prompts/input/generation/from_constitution/{style}/``. To add a new
    style, create a new subdirectory with a ``template.json`` inside.

    Args:
        prompt_dir: Root prompts directory. Defaults to package prompts/.

    Returns:
        Sorted list of style names (e.g. ["long", "short"]).
    """
    base = (
        (prompt_dir or _DEFAULT_PROMPT_DIR)
        / "input"
        / "generation"
        / "from_constitution"
    )
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

        # Resolve default directories first (single-source path module) so the
        # composed InputPipeline is created pointing at the right output dir.
        from redact import paths
        if constitution_dir is None:
            constitution_dir = paths.constitution_dir()
        self.constitution_dir = Path(constitution_dir)

        if output_dir is None:
            output_dir = paths.constitution_inputs_dir()
        self.output_dir = Path(output_dir)

        # Compose with InputPipeline for generate+extract+check. Pass the
        # resolved output dir up front so per-category CSVs land there instead of
        # defaulting to Datasets/ (was previously patched via a runtime mutation).
        self.input_pipeline = InputPipeline(
            gen_backend=gen_backend,
            gen_model=gen_model,
            check_backend=check_backend,
            check_model=check_model,
            rate_limiter=rate_limiter,
            extraction_style=extraction_style,
            dataset_dir=self.output_dir,
        )

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

        Backward-compatible thin wrapper around
        ``InputPipeline.run_from_constitution()``. Loads the constitution
        DataFrame from disk (``Data_cache/constitution/``), applies the
        ``entry_types`` / ``source_categories`` filters, then delegates to
        the unified pipeline. Prefer calling
        ``InputPipeline.run_from_constitution()`` directly in new code.

        Args:
            style: Template style ("long", "short", or any custom style under
                ``prompts/input/generation/from_constitution/``).
            samples_per_entry: Number of prompts to generate per constitution entry.
            entry_types: Filter to specific entry types
                (e.g. ["harmful", "benign", "dual_use_harmful", "dual_use_benign"]).
            source_categories: Filter to specific taxonomy categories.
            use_checker: Whether to quality-check generated prompts.
            save: Whether to save results to CSV.
            verbose: Print progress.
            batch_size: Entries per LLM engine pass (default 32).

        Returns:
            ConstitutionInputResult with generation statistics.
        """
        prompt_config = load_prompt(
            _PROMPT_PIPELINE, f"generation/from_constitution/{style}"
        )

        constitution_df = self._load_constitution(entry_types, source_categories)
        if constitution_df.empty:
            if verbose:
                logger.info("No constitution entries match filters; nothing to do.")
            return ConstitutionInputResult()

        # (Saving already routes to self.output_dir — the inner InputPipeline was
        # constructed with dataset_dir=self.output_dir in __init__.)

        if verbose:
            logger.info("Constitution-to-Input Generation")
            logger.info("Style: %s | Entries: %d | Samples/entry: %d | Batch: %d",
                        style, len(constitution_df), samples_per_entry, batch_size)
            if entry_types:
                logger.info("Entry types: %s", entry_types)
            if source_categories:
                logger.info("Source categories: %s", source_categories)

        return self.input_pipeline.run_from_constitution(
            constitution_df=constitution_df,
            prompt_config=prompt_config,
            samples_per_entry=samples_per_entry,
            use_checker=use_checker,
            save=save,
            verbose=verbose,
            batch_size=batch_size,
            style=style,
        )

