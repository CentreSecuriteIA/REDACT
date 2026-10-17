"""The deprecated standalone input path: per category, seed prompts, then turns
of generate, check and save until a sample target is met.

``InputPipeline`` (``inputs.py``) takes these methods from
:class:`_StandaloneGenerationMixin`; ``generate_inputs`` without a
constitution reaches them through :func:`run_standalone_inputs`, and warns.
"""

import logging
from collections.abc import Callable
from math import ceil
from pathlib import Path

from .. import paths
from ..dataset import append_samples, get_existing_samples
from ..dataset.taxonomy import get_seed_prompts, load_seeds
from ..llms.client import ModelClient
from ..llms.prompting import extract_and_clean, load_prompt
from ..llm_pipeline import batch_check_samples
from ..llms.router import generate_sample
from ..types import EntryType
from .checker import build_quality_checker
from .metaprompt import generate_category_description, generate_seeds
from .feedback import next_turn_feedback
from .results import CategoryResult, SampleResult, TurnResult

logger = logging.getLogger(__name__)


class _StandaloneGenerationMixin:
    """Standalone (meta-prompt) generation methods for ``InputPipeline``.

    Assumes the attributes set by ``InputPipeline.__init__`` (``gen``,
    ``check``,
    ``extraction_style``, ``dataset_dir``) and the
    ``_build_generation_messages`` method defined on the base class in
    ``inputs.py`` — a standard mixin, not meant to be instantiated on its
    own.
    """

    def generate_batch(
        self,
        prompt_config: dict,
        samples_per_request: int = 5,
        feedback: str = "",
        prohibited: set[str] | None = None,
        **seed_kwargs: str,
    ) -> tuple[str, list[str]]:
        """Generate one batch: single LLM call producing multiple samples.

        Args:
            prompt_config: Loaded prompt JSON config.
            samples_per_request: Number of samples to request.
            feedback: Rejection feedback from previous turn.
            prohibited: Existing samples to avoid.
            **seed_kwargs: Template variables for prompt rendering.

        Returns:
            (raw_output, extracted_samples) tuple.
        """
        messages = self._build_generation_messages(
            prompt_config,
            samples_per_request,
            feedback=feedback,
            prohibited=prohibited,
            **seed_kwargs,
        )

        raw_output = generate_sample(self.gen, messages)

        # A reply with any other number of samples is rejected whole.
        extracted = extract_and_clean(
            raw_output,
            style=self.extraction_style,
            expected_count=samples_per_request,
        )

        return raw_output, extracted

    def check_samples(
        self,
        samples: list[str],
        build_check_messages: Callable[[str, str], list[dict]],
        turn_index: int,
    ) -> list[SampleResult]:
        """Check all samples in a single batched engine pass.

        Delegates to ``batch_check_samples()``, which chunks the samples and
        hands each chunk to the checker client — how that batch actually runs
        (native engine pass vs. fan-out) is the client's business, not this
        pipeline's.

        Rejected samples are returned with ``accepted=False`` and their
        reasoning preserved — no regeneration is attempted here.

        Args:
            samples: List of extracted sample strings.
            build_check_messages: Function(original, sample_text) -> checker
                message list (original is "" — no comparison text at this
                call site).
            turn_index: Current turn index (for tracking).

        Returns:
            List of SampleResult objects in the same order as ``samples``.
        """
        check_results = batch_check_samples(
            self.check,
            samples,
            build_check_messages,
        )
        return [
            SampleResult(
                text=sample_text,
                accepted=accepted,
                reasoning=reasoning,
                turn=turn_index,
            )
            for sample_text, (accepted, reasoning) in zip(samples, check_results)
        ]

    def run_turn(
        self,
        prompt_config: dict,
        build_check_messages: Callable[[str, str], list[dict]],
        turn_index: int,
        samples_per_request: int = 5,
        feedback: str = "",
        category: str = "",
        save: bool = True,
        prohibited: set[str] | None = None,
        entry_type: str | EntryType = "harmful",
        subcategory: str = "",
        source: str = "metaprompt",
        extra_row_metadata: dict | None = None,
        **seed_kwargs: str,
    ) -> TurnResult:
        """Execute a single generation turn: generate, extract, check, save.

        Args:
            prompt_config: Loaded prompt JSON config.
            build_check_messages: Function(original, sample_text) -> checker
                messages (original is "" — no comparison text at this call site).
            turn_index: Turn number (for tracking/seeds).
            samples_per_request: Samples to request per LLM call.
            feedback: Rejection reasoning from prior turn.
            category: Category name (for saving).
            save: Whether to save samples to CSV.
            prohibited: Existing samples to avoid during generation.
            entry_type: Severity tag for every sample saved this turn
                (default "harmful" — backward-compatible).
            subcategory: Constitution subcategory tag, or "".
            source: "metaprompt" / "constitution" / etc. — written to the
                ``source`` CSV column.
            extra_row_metadata: Extra per-turn metadata applied to every row
                (e.g. ``{"source_sample_description": "..."}``).
            **seed_kwargs: Template variables.

        Returns:
            TurnResult with all sample outcomes.
        """
        # Generate + extract
        raw_output, extracted = self.generate_batch(
            prompt_config,
            samples_per_request=samples_per_request,
            feedback=feedback,
            prohibited=prohibited,
            **seed_kwargs,
        )

        logger.info(
            "Turn %d: extracted %d samples from LLM output",
            turn_index,
            len(extracted),
        )

        # Exact dedup against existing samples before checking
        if prohibited:
            before_count = len(extracted)
            extracted = [s for s in extracted if s not in prohibited]
            dedup_removed = before_count - len(extracted)
            if dedup_removed > 0:
                logger.info(
                    "Turn %d: removed %d duplicates of existing samples",
                    turn_index,
                    dedup_removed,
                )

        # Check each sample individually
        sample_results = self.check_samples(
            extracted, build_check_messages, turn_index
        )

        accepted_samples = [r for r in sample_results if r.accepted]
        rejected_samples = [r for r in sample_results if not r.accepted]

        logger.info(
            "Turn %d: %d accepted, %d rejected",
            turn_index,
            len(accepted_samples),
            len(rejected_samples),
        )

        # Save ALL samples (accepted + rejected) with the unified schema
        if save and sample_results and category:
            entry_type_value = (
                entry_type.value
                if isinstance(entry_type, EntryType)
                else str(entry_type)
            )
            base_meta = {
                "entry_type": entry_type_value,
                "subcategory": subcategory,
            }
            if extra_row_metadata:
                base_meta.update(extra_row_metadata)
            all_texts = [r.text for r in sample_results]
            all_accepted = [r.accepted for r in sample_results]
            all_extra = [
                {**base_meta, "rejection_reason": r.reasoning}
                for r in sample_results
            ]
            append_samples(
                all_texts,
                category=category,
                turn=turn_index,
                accepted=all_accepted,
                source=source,
                extra_columns=all_extra,
                dataset_dir=self.dataset_dir,
            )

        return TurnResult(
            turn_index=turn_index,
            raw_output=raw_output,
            extracted_count=len(extracted),
            accepted_count=len(accepted_samples),
            rejected_count=len(rejected_samples),
            samples=sample_results,
        )

    def run_standalone(
        self,
        categories: list[tuple[str, dict]],
        prompt_config: dict,
        samples_per_category: int = 15,
        samples_per_request: int = 5,
        use_metaprompt: bool = True,
        num_seeds: int = 8,
        seeds_db: dict | None = None,
        verbose: bool = True,
    ) -> list[CategoryResult]:
        """Run standalone (meta-prompt) generation across multiple categories.

        For each category: optionally generate a description + seed prompts via
        LLM (``use_metaprompt=True``), or load hand-written seeds from
        ``seeds_db``; then loop generation turns (via :meth:`run_turn`) until
        ``samples_per_category`` accepted samples are reached or a
        safety-capped ``max_turns`` is exhausted.

        It does not warn on its own: ``generate_inputs``, which reaches it
        through :func:`run_standalone_inputs`, already does.

        Args:
            categories: ``(name, info)`` pairs, e.g. from ``iter_categories(taxonomy)``.
            prompt_config: Loaded ``input``/``generation/standalone`` prompt config.
            samples_per_category: Target accepted samples per category.
            samples_per_request: Samples per LLM call per turn.
            use_metaprompt: LLM-generate descriptions+seeds vs. hand-written.
            num_seeds: Number of seed prompts to generate (``use_metaprompt=True``).
            seeds_db: Hand-written seeds DB (required when ``use_metaprompt=False``).
            verbose: Log progress.

        Returns:
            One :class:`CategoryResult` per category (categories with no
            available seeds are skipped and omitted).

        Raises:
            ValueError: If ``use_metaprompt=False`` and ``seeds_db`` is not
                provided — without this check the failure surfaces several
                calls later as an opaque ``AttributeError`` inside
                ``get_seed_prompts()``.
        """
        if not use_metaprompt and seeds_db is None:
            raise ValueError(
                "run_standalone(use_metaprompt=False) requires seeds_db "
                "(hand-written seed prompts) — there's no LLM call to "
                "generate them from in this mode."
            )
        max_turns = ceil(samples_per_category / samples_per_request) * 3

        all_results: list[CategoryResult] = []
        for i, (category_name, category_info) in enumerate(categories):
            if verbose:
                logger.info("[%d/%d] %s", i + 1, len(categories), category_name)

            if use_metaprompt:
                if verbose:
                    logger.debug("Generating description...")
                description = generate_category_description(
                    self.gen, category_name, prompt_dir=self.prompt_dir)
                if verbose:
                    logger.debug("Description: %d chars", len(description))
                    logger.debug("Generating seeds...")
                seed_text = generate_seeds(
                    self.gen, category_name, description,
                    num_seeds=num_seeds, prompt_dir=self.prompt_dir,
                )
                if verbose:
                    logger.debug("Seeds: %d generated", seed_text.count(chr(10)) + 1)
            else:
                description = category_info.get("description", category_name)
                seed_text = get_seed_prompts(seeds_db, category_name)

            if not seed_text:
                if verbose:
                    logger.info("No seeds available, skipping.")
                continue

            seed_kwargs = {
                "Category": category_name,
                "category_description": description,
                "SeedPrompts": seed_text,
            }

            # Run turns until target is reached or max_turns exceeded
            checker = build_quality_checker(category_name, prompt_dir=self.prompt_dir)
            result = CategoryResult(category=category_name)
            feedback = ""
            prohibited = get_existing_samples(category_name, self.dataset_dir)

            for turn_idx in range(max_turns):
                if result.total_accepted >= samples_per_category:
                    break

                if verbose:
                    logger.info("Turn %d: %d/%d accepted, requesting %d...",
                                turn_idx + 1, result.total_accepted, samples_per_category,
                                samples_per_request)

                turn_result = self.run_turn(
                    prompt_config=prompt_config,
                    build_check_messages=checker,
                    turn_index=turn_idx,
                    samples_per_request=samples_per_request,
                    feedback=feedback,
                    category=category_name,
                    save=True,
                    prohibited=prohibited,
                    **seed_kwargs,
                )
                result.turns.append(turn_result)

                # Update prohibited set
                for sr in turn_result.samples:
                    prohibited.add(sr.text)

                # Collect feedback for next turn
                feedback = next_turn_feedback(turn_result)

                if verbose:
                    logger.info("-> %d accepted, %d rejected (%.0f%%)",
                                turn_result.accepted_count, turn_result.rejected_count,
                                turn_result.acceptance_rate * 100)

                # Stop early if no samples were extracted at all (model issue)
                if turn_result.extracted_count == 0:
                    if verbose:
                        logger.info("No samples extracted, stopping category.")
                    break

            all_results.append(result)

            if verbose:
                logger.info("-> %d accepted / %d extracted (%.0f%%)",
                            result.total_accepted, result.total_extracted,
                            result.overall_acceptance_rate * 100)

        # Summary
        if verbose and all_results:
            logger.info("Summary")
            for result in all_results:
                logger.info("%-30s %3d / %3d (%.0f%%)", result.category, result.total_accepted,
                            result.total_extracted, result.overall_acceptance_rate * 100)
            total_accepted = sum(r.total_accepted for r in all_results)
            total_extracted = sum(r.total_extracted for r in all_results)
            logger.info("Total: %d accepted / %d extracted", total_accepted, total_extracted)

        return all_results


def run_standalone_inputs(
    categories: list[tuple[str, dict]],
    client: ModelClient,
    check: ModelClient,
    dataset_dir: Path,
    *,
    prompt_dir: str | Path | None = None,
    samples_per_category: int,
    samples_per_request: int,
    use_metaprompt: bool,
    seeds_name: str,
    num_seeds: int,
    resume: bool = True,
    verbose: bool = True,
) -> list[CategoryResult]:
    """Generate checked samples for each category into ``dataset_dir``'s tree.

    Seeds are generated per category with ``use_metaprompt``, else read from
    the seeds file ``seeds_name``. ``resume=False`` first removes the files
    of these categories. Returns one :class:`CategoryResult` per category
    that had seeds.
    """
    # inputs.py imports this module for the mixin.
    from .inputs import InputPipeline

    pipeline = InputPipeline(
        gen=client,
        check=check,
        extraction_style="numbered",
        dataset_dir=dataset_dir,
        prompt_dir=prompt_dir,
    )

    prompt_config = load_prompt("input", "generation/standalone", prompt_dir=prompt_dir)

    # Seeds (simple mode)
    seeds_db = None
    if not use_metaprompt:
        seeds_db = load_seeds(seeds_name)

    # Clear existing data when not resuming
    if not resume:
        for cat_name, _ in categories:
            csv_path = paths.category_csv(cat_name, root=dataset_dir)
            if csv_path.exists():
                csv_path.unlink()
                if verbose:
                    logger.info("Cleared existing %s", csv_path)

    if verbose:
        mode = (
            "automated (LLM descriptions + seeds)" if use_metaprompt
            else "simple (taxonomy + hand-written seeds)"
        )
        logger.info("Generate Content Moderation Inputs")
        logger.info("Mode: %s", mode)
        logger.info("Categories: %d | Target: %d samples each (%d/request)",
                    len(categories), samples_per_category, samples_per_request)

    return pipeline.run_standalone(
        categories=categories,
        prompt_config=prompt_config,
        samples_per_category=samples_per_category,
        samples_per_request=samples_per_request,
        use_metaprompt=use_metaprompt,
        num_seeds=num_seeds,
        seeds_db=seeds_db,
        verbose=verbose,
    )
