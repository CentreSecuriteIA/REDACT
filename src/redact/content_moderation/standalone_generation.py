"""Standalone (meta-prompt) content moderation input generation — DEPRECATED.

Split out of content_moderation/generation.py, which used to fuse this
deprecated path into the same module/class as the active constitution-seeded
path (``run_from_constitution``, still in ``generation.py``). Both paths
share the ``InputPipeline`` class — ``InputPipeline`` in ``generation.py``
inherits from :class:`_StandaloneGenerationMixin` below, so
``pipeline.run_category(...)`` / ``pipeline.run_standalone(...)`` /
``pipeline.run_turn(...)`` etc. all still work exactly as before; only the
on-disk file layout changed.

Meta-prompt seeds drive per-category multi-turn generation with the
entry-type-aware quality checker. **Deprecated** (``run_category`` and the
standalone ``generate_inputs()`` path both emit ``DeprecationWarning``) in
favour of constitution-seeded generation, which gives better/more adjustable
coverage — but kept fully functional as the cheap "quick eval" path.
"""

import logging
import warnings
from collections.abc import Callable
from math import ceil

from ..dataset.io import append_samples, get_existing_samples
from ..dataset.taxonomy import get_seed_prompts
from ..llms.calls import batch_check_samples, generate_sample
from ..llms.extraction import extract_and_clean
from ..types import EntryType
from .checker import build_quality_checker
from .metaprompt import generate_category_description, generate_seeds
from .results import CategoryResult, SampleResult, TurnResult, _next_turn_feedback

logger = logging.getLogger(__name__)


class _StandaloneGenerationMixin:
    """Standalone (meta-prompt) generation methods for ``InputPipeline``.

    Assumes the attributes set by ``InputPipeline.__init__`` (``gen_backend``,
    ``gen_model``, ``check_backend``, ``check_model``, ``rate_limiter``,
    ``extraction_style``, ``dataset_dir``) and the
    ``_build_generation_messages`` method defined on the base class in
    ``generation.py`` — a standard mixin, not meant to be instantiated on its
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

        raw_output = generate_sample(
            self.gen_backend,
            self.gen_model,
            messages,
            self.rate_limiter,
        )

        extracted = extract_and_clean(
            raw_output,
            style=self.extraction_style,
        )

        return raw_output, extracted

    def check_samples(
        self,
        samples: list[str],
        build_check_messages: Callable[[str], list[dict]],
        turn_index: int,
    ) -> list[SampleResult]:
        """Check all samples in a single batched engine pass.

        Delegates to ``batch_check_samples()`` which sends all checker prompts
        to ``backend.batch_generate()`` at once (one vLLM engine pass per
        chunk of 32). For API backends the call falls back to sequential.

        Rejected samples are returned with ``accepted=False`` and their
        reasoning preserved — no regeneration is attempted here.

        Args:
            samples: List of extracted sample strings.
            build_check_messages: Function(sample_text) -> checker message list.
            turn_index: Current turn index (for tracking).

        Returns:
            List of SampleResult objects in the same order as ``samples``.
        """
        check_results = batch_check_samples(
            self.check_backend,
            self.check_model,
            samples,
            build_check_messages,
            rate_limiter=self.rate_limiter,
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
        build_check_messages: Callable[[str], list[dict]],
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
            build_check_messages: Function(sample_text) -> checker messages.
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

    def run_category(
        self,
        category: str,
        prompt_config: dict,
        build_check_messages: Callable[[str], list[dict]],
        num_turns: int = 10,
        samples_per_request: int = 5,
        use_feedback: bool = True,
        use_prohibited: bool = True,
        seed_kwargs_per_turn: list[dict[str, str]] | None = None,
        save: bool = True,
        entry_type: str | EntryType = "harmful",
        subcategory: str = "",
        source: str = "metaprompt",
    ) -> CategoryResult:
        """Run the full generation pipeline for a single category.

        Executes num_turns generation turns, optionally passing rejection
        feedback from each turn to the next and injecting existing samples
        as prohibited outputs.

        A lower-level, single-category primitive than :meth:`run_standalone`
        — genuinely different, not accidentally overlapping: this method
        runs a *fixed* ``num_turns`` (no early-stop-on-target logic) using
        seed kwargs the *caller* already prepared per turn, and has no
        per-category metaprompt (description/seed generation) step of its
        own. ``run_standalone`` is a target-driven, multi-category driver
        that generates its own seeds and stops once each category hits its
        sample target — it does not call this method. Both share the same
        per-turn feedback computation (:func:`redact.content_moderation.
        results._next_turn_feedback`).

        Args:
            category: Harm category name.
            prompt_config: Loaded prompt JSON config for this category.
            build_check_messages: Function(sample_text) -> checker messages.
            num_turns: Number of generation turns.
            samples_per_request: Samples per LLM call per turn.
            use_feedback: Whether to pass rejection reasoning to next turn.
            use_prohibited: Whether to inject existing samples as "do not
                repeat" in the generation prompt.
            seed_kwargs_per_turn: Optional list of per-turn seed dictionaries.
                If provided, must have length >= num_turns. Each dict is
                unpacked as **kwargs into the prompt template.
                If None, only Category=category is passed each turn.
            save: Whether to save to CSV.
            entry_type: Severity tag for every sample saved this run
                (default ``"harmful"`` — preserves prior behaviour).
            subcategory: Constitution subcategory tag, or ``""``.
            source: ``"metaprompt"`` (standalone CM) or ``"constitution"``
                (constitution-seeded). Written to the ``source`` CSV column.

        Returns:
            CategoryResult with all turn outcomes.

        .. deprecated::
            Standalone meta-prompt input generation is deprecated in favour of
            constitution-seeded generation, which gives better/more adjustable
            coverage — use ``run_from_constitution`` /
            ``generate_inputs(constitution_df=...)``. This method remains fully
            functional.
        """
        warnings.warn(
            "InputPipeline.run_category (standalone meta-prompt input generation) is "
            "deprecated in favour of constitution-seeded generation "
            "(generate_inputs(constitution_df=...)); it remains functional.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = CategoryResult(category=category)
        feedback = ""

        # Load existing samples for prohibited list + dedup
        prohibited: set[str] | None = None
        if use_prohibited:
            prohibited = get_existing_samples(category, self.dataset_dir)

        for turn_idx in range(num_turns):
            # Determine seed kwargs for this turn
            if seed_kwargs_per_turn and turn_idx < len(seed_kwargs_per_turn):
                seed_kwargs = seed_kwargs_per_turn[turn_idx]
            else:
                seed_kwargs = {"Category": category}

            logger.info("[Turn %d/%d] Category=%s, seeds=%s",
                        turn_idx + 1, num_turns, category, list(seed_kwargs.keys()))

            turn_result = self.run_turn(
                prompt_config=prompt_config,
                build_check_messages=build_check_messages,
                turn_index=turn_idx,
                samples_per_request=samples_per_request,
                feedback=feedback if use_feedback else "",
                category=category,
                save=save,
                prohibited=prohibited,
                entry_type=entry_type,
                subcategory=subcategory,
                source=source,
                **seed_kwargs,
            )

            result.turns.append(turn_result)

            # Update prohibited set with newly generated samples
            if prohibited is not None:
                for sr in turn_result.samples:
                    prohibited.add(sr.text)

            # Collect feedback for next turn
            if use_feedback:
                feedback = _next_turn_feedback(turn_result)

            logger.info("-> %d accepted, %d rejected (rate: %.0f%%)",
                        turn_result.accepted_count, turn_result.rejected_count,
                        turn_result.acceptance_rate * 100)

        logger.info("Category '%s' complete: %d/%d accepted (%.0f%%)",
                    category, result.total_accepted, result.total_extracted,
                    result.overall_acceptance_rate * 100)

        return result

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

        This is a different (target-driven, break-on-empty-extraction,
        generates-its-own-seeds) loop than :meth:`run_category`'s
        (fixed-``num_turns``, caller-supplied seeds) loop — evaluated merging
        them and deliberately kept separate; see :meth:`run_category`'s
        docstring for the reasoning. Moved here from ``pipelines.py``
        verbatim so that module stays a thin dispatcher; it is the
        implementation behind the deprecated standalone ``generate_inputs()``
        path and does not warn on its own, since its only caller already
        does.

        Args:
            categories: ``(name, info)`` pairs, e.g. from ``iter_categories(taxonomy)``.
            prompt_config: Loaded ``input``/``generation/standalone`` prompt config.
            samples_per_category: Target accepted samples per category.
            samples_per_request: Samples per LLM call per turn.
            use_metaprompt: LLM-generate descriptions+seeds vs. hand-written.
            num_seeds: Number of seed prompts to generate (``use_metaprompt=True``).
            seeds_db: Hand-written seeds DB (required when ``use_metaprompt=False``).
            verbose: Print progress.

        Returns:
            One :class:`CategoryResult` per category (categories with no
            available seeds are skipped and omitted).
        """
        max_turns = ceil(samples_per_category / samples_per_request) * 3

        all_results: list[CategoryResult] = []
        for i, (category_name, category_info) in enumerate(categories):
            if verbose:
                logger.info("[%d/%d] %s", i + 1, len(categories), category_name)

            if use_metaprompt:
                if verbose:
                    logger.debug("Generating description...")
                description = generate_category_description(
                    self.gen_backend, self.gen_model, category_name, self.rate_limiter
                )
                if verbose:
                    logger.debug("Description: %d chars", len(description))
                    logger.debug("Generating seeds...")
                seed_text = generate_seeds(
                    self.gen_backend, self.gen_model, category_name, description,
                    num_seeds=num_seeds, rate_limiter=self.rate_limiter,
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
            checker = build_quality_checker(category_name)
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
                feedback = _next_turn_feedback(turn_result)

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
