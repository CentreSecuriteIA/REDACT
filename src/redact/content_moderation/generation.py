"""Content moderation input sample generation pipeline.

Generates multi-sample batches via LLM, extracts individual samples via
regex, checks each sample individually, and saves all samples (accepted
and rejected) incrementally to per-category CSVs.

Usage:
    from redact.llms import get_backend, RateLimiter, load_prompt
    from redact.content_moderation import InputPipeline
    from redact.content_moderation.checker import build_quality_checker

    backend = get_backend("venice-uncensored")
    limiter = RateLimiter()
    prompt_config = load_prompt("content_moderation_input", "violence")

    pipeline = InputPipeline(
        gen_backend=backend, gen_model="venice-uncensored",
        check_backend=backend, check_model="venice-uncensored",
        rate_limiter=limiter,
    )
    result = pipeline.run_category(
        category="violence",
        prompt_config=prompt_config,
        build_check_messages=build_quality_checker("violence"),
        num_turns=10, samples_per_request=5,
    )
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..llms.base import LLMBackend
from ..llms.calls import generate_sample, check_sample
from ..llms.prompts import build_messages
from ..llms.extraction import get_format_instruction, extract_and_clean
from ..llms.wrappers import RateLimiter
from ..dataset.io import append_samples, get_existing_samples, _default_dataset_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Result of checking a single extracted sample."""

    text: str
    accepted: bool
    reasoning: str  # empty if accepted, checker feedback if rejected
    turn: int


@dataclass
class TurnResult:
    """Result of a single generation turn."""

    turn_index: int
    raw_output: str
    extracted_count: int
    accepted_count: int
    rejected_count: int
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.extracted_count == 0:
            return 0.0
        return self.accepted_count / self.extracted_count


@dataclass
class CategoryResult:
    """Aggregate result for a full category generation run."""

    category: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def total_extracted(self) -> int:
        return sum(t.extracted_count for t in self.turns)

    @property
    def total_accepted(self) -> int:
        return sum(t.accepted_count for t in self.turns)

    @property
    def total_rejected(self) -> int:
        return sum(t.rejected_count for t in self.turns)

    @property
    def overall_acceptance_rate(self) -> float:
        if self.total_extracted == 0:
            return 0.0
        return self.total_accepted / self.total_extracted


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class InputPipeline:
    """Multi-sample content moderation input generation pipeline.

    Workflow per turn:
    1. Build messages from prompt config + seed kwargs + format instruction
    2. Optionally inject prohibited outputs (already-generated samples)
    3. Call generator LLM (single request producing N samples)
    4. Extract individual samples via regex (numbered list by default)
    5. Exact-dedup against existing samples (skip duplicates before checking)
    6. Check each sample individually via checker LLM
    7. Save all samples (accepted + rejected) to category CSV with metadata
    8. Collect rejection feedback for next turn
    """

    def __init__(
        self,
        gen_backend: LLMBackend,
        gen_model: str,
        check_backend: LLMBackend,
        check_model: str,
        rate_limiter: RateLimiter | None = None,
        extraction_style: str = "numbered",
        dataset_dir: str | Path | None = None,
    ):
        """Create an InputPipeline.

        Args:
            gen_backend: LLM backend for generation.
            gen_model: Model identifier for generation.
            check_backend: LLM backend for checking (can be same as gen).
            check_model: Model identifier for checking (can differ from gen).
            rate_limiter: Optional shared rate limiter.
            extraction_style: "numbered", "structured_qa", or "delimiter".
            dataset_dir: Root directory for saving CSVs. Defaults to
                redact/Datasets/ (package-relative).
        """
        self.gen_backend = gen_backend
        self.gen_model = gen_model
        self.check_backend = check_backend
        self.check_model = check_model
        self.rate_limiter = rate_limiter
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
        # Deep copy to avoid mutating the original config
        config = dict(prompt_config)

        # Inject format instruction into system prompt
        format_instr = get_format_instruction(
            style=self.extraction_style,
            num_samples=samples_per_request,
        )
        config["system_prompt"] = config.get("system_prompt", "") + format_instr

        # Build the base message list
        messages = build_messages(
            config,
            num_samples=str(samples_per_request),
            **seed_kwargs,
        )

        # Append prohibited outputs so the LLM avoids repeating them
        if prohibited:
            # Show a sample of existing outputs (cap to avoid prompt bloat)
            sample_list = list(prohibited)[:20]
            prohibited_text = "\n".join(f"- {s}" for s in sample_list)
            prohibited_msg = (
                f"\n\nIMPORTANT: The following samples already exist in the "
                f"dataset. Do NOT generate anything similar to these:\n"
                f"{prohibited_text}"
            )
            if len(prohibited) > 20:
                prohibited_msg += (
                    f"\n(... and {len(prohibited) - 20} more. "
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
        """Check each sample individually via the checker LLM.

        Args:
            samples: List of extracted sample strings.
            build_check_messages: Function(sample_text) -> checker message list.
            turn_index: Current turn index (for tracking).

        Returns:
            List of SampleResult objects.
        """
        results: list[SampleResult] = []
        for sample_text in samples:
            accepted, reasoning = check_sample(
                self.check_backend,
                self.check_model,
                sample_text,
                build_check_messages,
                self.rate_limiter,
            )
            results.append(
                SampleResult(
                    text=sample_text,
                    accepted=accepted,
                    reasoning=reasoning,
                    turn=turn_index,
                )
            )
        return results

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

        # Save ALL samples (accepted + rejected) with metadata
        if save and sample_results and category:
            all_texts = [r.text for r in sample_results]
            all_accepted = [r.accepted for r in sample_results]
            all_extra = [
                {"reasoning": r.reasoning} if r.reasoning else {}
                for r in sample_results
            ]
            append_samples(
                all_texts,
                category=category,
                turn=turn_index,
                accepted=all_accepted,
                source="generated",
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
    ) -> CategoryResult:
        """Run the full generation pipeline for a single category.

        Executes num_turns generation turns, optionally passing rejection
        feedback from each turn to the next and injecting existing samples
        as prohibited outputs.

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

        Returns:
            CategoryResult with all turn outcomes.
        """
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

            print(
                f"  [Turn {turn_idx + 1}/{num_turns}] "
                f"Category={category}, seeds={list(seed_kwargs.keys())}"
            )

            turn_result = self.run_turn(
                prompt_config=prompt_config,
                build_check_messages=build_check_messages,
                turn_index=turn_idx,
                samples_per_request=samples_per_request,
                feedback=feedback if use_feedback else "",
                category=category,
                save=save,
                prohibited=prohibited,
                **seed_kwargs,
            )

            result.turns.append(turn_result)

            # Update prohibited set with newly generated samples
            if prohibited is not None:
                for sr in turn_result.samples:
                    prohibited.add(sr.text)

            # Collect feedback for next turn
            if use_feedback:
                rejections = [
                    r.reasoning
                    for r in turn_result.samples
                    if not r.accepted and r.reasoning
                ]
                if rejections:
                    feedback = "\n".join(
                        f"- {reason[:200]}" for reason in rejections[:3]
                    )
                else:
                    feedback = ""

            print(
                f"    -> {turn_result.accepted_count} accepted, "
                f"{turn_result.rejected_count} rejected "
                f"(rate: {turn_result.acceptance_rate:.0%})"
            )

        print(
            f"\n  Category '{category}' complete: "
            f"{result.total_accepted}/{result.total_extracted} accepted "
            f"({result.overall_acceptance_rate:.0%})"
        )

        return result
