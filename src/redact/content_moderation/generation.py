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

import pandas as pd

from ..llms.base import LLMBackend
from ..llms.calls import generate_sample, check_sample, batch_check_samples
from ..llms.prompts import build_messages
from ..llms.extraction import get_format_instruction, extract_and_clean
from ..llms.wrappers import RateLimiter
from ..dataset.io import append_samples, get_existing_samples, _default_dataset_dir
from ..types import EntryType
from .checker import build_quality_checker

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


@dataclass
class ConstitutionInputResult:
    """Aggregate result of constitution-seeded input generation.

    Returned by ``InputPipeline.run_from_constitution()``. Tracks per-batch
    statistics so callers can detect mode collapse or model degradation.
    """

    total_entries_processed: int = 0
    skipped_entries: int = 0
    total_prompts_generated: int = 0
    total_prompts_accepted: int = 0
    total_prompts_rejected: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_prompts_generated == 0:
            return 0.0
        return self.total_prompts_accepted / self.total_prompts_generated


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
        """Check all samples in a single batched engine pass.

        Delegates to ``batch_check_samples()`` which sends all checker prompts
        to ``backend.batch_generate()`` at once (one vLLM engine pass per
        chunk of 32). For API backends the call falls back to sequential.

        This method is shared by both the content moderation pipeline
        (``run_turn()`` / ``run_category()``) and the constitution input
        pipeline (``ConstitutionInputPipeline`` wraps an ``InputPipeline``
        and calls this method directly). Rejected samples are returned with
        ``accepted=False`` and their reasoning preserved — no regeneration
        is attempted here.

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
    ) -> ConstitutionInputResult:
        """Run constitution-seeded input generation, batched across entries.

        Each row in ``constitution_df`` becomes a generation request. The
        per-entry feedback loop is handled inside the LLM checker — failed
        samples are saved with their rejection reason but no regeneration
        loop runs here (this mode prioritises throughput over per-sample
        retries). For per-sample retry use ``generate_with_check`` directly.

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
            verbose: Print per-entry progress.
            batch_size: Entries per LLM engine pass.

        Returns:
            ConstitutionInputResult with generation statistics.
        """
        if constitution_df.empty:
            logger.warning("run_from_constitution: empty constitution_df, nothing to do")
            return ConstitutionInputResult()

        result = ConstitutionInputResult()
        prohibited: set[str] = set()
        checker_cache: dict[tuple[str, str, str], Callable[[str], list[dict]]] = {}

        entries = list(constitution_df.iterrows())
        total = len(entries)

        if verbose:
            print(f"\n  Constitution-seeded generation: {total} entries, "
                  f"batch_size={batch_size}, samples_per_entry={samples_per_entry}")

        for batch_start in range(0, total, batch_size):
            batch = entries[batch_start : batch_start + batch_size]

            # 1. Build per-entry generation messages
            messages_list = []
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

            # 2. Single batch_generate for the whole chunk
            raw_outputs = self.gen_backend.batch_generate(
                messages_list, self.gen_model
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
                    flat_check_msgs.extend(checker(s) for s in extracted)

            flat_check_results: list[tuple[bool, str]]
            if use_checker and flat_check_msgs:
                responses = self.check_backend.batch_generate(
                    flat_check_msgs, self.check_model
                )
                flat_check_results = []
                for response in responses:
                    accepted = any(
                        response.strip().lower().startswith(p)
                        for p in ("yes", "ok", "accept", "pass")
                    )
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
                    print(
                        f"    [{global_idx + 1}/{total}] {category} | {entry_type} | "
                        f"{sample_desc[:60]}{'...' if len(sample_desc) > 60 else ''}"
                    )

                if not extracted:
                    result.skipped_entries += 1
                    if verbose:
                        print("       -> no samples extracted")
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

                accepted_count = sum(1 for sr in sample_results if sr.accepted)
                rejected_count = len(sample_results) - accepted_count

                result.total_entries_processed += 1
                result.total_prompts_generated += len(sample_results)
                result.total_prompts_accepted += accepted_count
                result.total_prompts_rejected += rejected_count

                for sr in sample_results:
                    prohibited.add(sr.text)

                if verbose:
                    print(f"       -> {accepted_count} accepted, {rejected_count} rejected")

        if verbose:
            print(
                f"\n  Constitution-seeded run complete: "
                f"{result.total_prompts_accepted}/{result.total_prompts_generated} "
                f"accepted ({result.acceptance_rate:.0%})"
            )

        return result
