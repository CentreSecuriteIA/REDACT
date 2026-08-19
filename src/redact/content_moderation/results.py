"""Result data classes for content moderation input generation.

Split out of content_moderation/generation.py so both the active
constitution-seeded path (``generation.py``) and the deprecated standalone
path (``standalone_generation.py``) can depend on these without either one
importing the other — mirrors ``redact/types.py``'s "keep it dependency-free"
role, just scoped to this subpackage.
"""

from dataclasses import dataclass, field


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


def _next_turn_feedback(
    turn_result: TurnResult, max_rejections: int = 3, max_chars: int = 200
) -> str:
    """Build the rejection-feedback string fed into the next generation turn.

    Shared by ``InputPipeline.run_category()`` and ``.run_standalone()``
    (both in ``standalone_generation.py``) — both turn a completed turn's
    rejection reasoning into feedback for the next turn's generation prompt
    the same way. "" when nothing was rejected (or nothing had reasoning
    attached).
    """
    rejections = [
        r.reasoning for r in turn_result.samples if not r.accepted and r.reasoning
    ]
    if not rejections:
        return ""
    return "\n".join(f"- {r[:max_chars]}" for r in rejections[:max_rejections])


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
