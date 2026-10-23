"""What the inputs stage returns: its totals."""

from dataclasses import dataclass


@dataclass
class ConstitutionInputResult:
    """The totals of one ``InputPipeline.run_from_constitution()`` run.

    The names date from when an entry was one call; each counts units, an
    entry being ``samples_per_entry`` of them.

    Attributes:
        total_entries_processed: Units that wrote a row, accepted or not.
        skipped_entries: Units acked with no row: rejected on every
            attempt, a prompt too long, or a sample the folder held already.
        total_prompts_generated: Samples written, one per unit with a row.
        total_prompts_accepted: Those the checker accepted.
        total_prompts_rejected: Those it rejected.
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
