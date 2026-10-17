"""What the inputs stage returns: its totals."""

from dataclasses import dataclass


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
