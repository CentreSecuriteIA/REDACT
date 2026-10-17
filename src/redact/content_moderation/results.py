"""What the input stages return: per sample, turn and category on the
standalone path, and the seeded stage's totals."""

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
    """The totals of one ``InputPipeline.run_from_constitution()`` run."""

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
