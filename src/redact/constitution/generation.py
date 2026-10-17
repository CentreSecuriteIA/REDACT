"""The constitution stage: generate each unit, save its rows, then ack it."""

import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    check_columns,
    resume_state,
)
from ..llm_pipeline import drive_units, extracted
from ..llms.client import ModelClient
from ..types import EntryType
from .units import _COLUMNS, Unit, file_of, messages_of, plan_units, rows_of

logger = logging.getLogger(__name__)

# The reply limit of a call: a prompt asks for up to about 90 entries.
_REPLY_TOKENS = 10000
# Generations of a unit before it is given up.
_ATTEMPTS = 3


def read_constitution(output_dir: str | Path) -> pd.DataFrame:
    """The saved constitution: every cell as text, an empty one as ``""``.

    No ``merged.csv`` gives an empty frame with the constitution's columns.
    """
    merged = Path(output_dir) / paths.CONSTITUTION_MERGED_FILENAME
    if not merged.exists():
        return pd.DataFrame(columns=_COLUMNS)
    return pd.read_csv(merged, dtype=str, keep_default_na=False)


class ConstitutionPipeline:
    """Generate constitution entries from a taxonomy into ``output_dir``.

    Each unit's rows go to ``{entry_type}.csv``, or ``general_benign.csv``
    for a benign-taxonomy unit, and to ``merged.csv``. The ledger
    ``constitution.state.jsonl`` beside them is plain: no lock, no recorded
    sizes. A crash between a unit's two appends leaves it un-acked, and the
    next run appends its rows to the type's file again.
    """

    def __init__(
        self,
        client: ModelClient,
        output_dir: str | Path | None = None,
        prompt_dir: str | Path | None = None,
    ):
        """``output_dir`` defaults to ``paths.constitution_dir()``; ``prompt_dir``
        is the prompt overlay, the packaged prompts when None."""
        self.client = client
        self.model = client.model
        self.output_dir = Path(
            output_dir if output_dir is not None else paths.constitution_dir())
        self.prompt_dir = prompt_dir

    def _unit(self, unit: Unit):
        """The unit's call, made again for a reply that is rejected whole.

        The generator returns ``(rows, attempts)``, ``rows`` empty when
        every attempt was rejected.
        """
        messages = messages_of(unit, self.prompt_dir)
        return extracted(
            self.client, lambda: messages,
            lambda reply: rows_of(unit, reply, self.model),
            max_tokens=_REPLY_TOKENS, max_attempts=_ATTEMPTS,
        )

    def _append(self, rows: list[dict]) -> None:
        """Append the rows to their own CSV (:func:`file_of`), then to
        ``merged.csv``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        by_file: dict[str, list[dict]] = {}
        for row in rows:
            by_file.setdefault(file_of(row), []).append(row)
        for name, its_rows in by_file.items():
            append_rows(self.output_dir / f"{name}.csv", its_rows)
        append_rows(self.output_dir / paths.CONSTITUTION_MERGED_FILENAME, rows)

    def run(
        self,
        taxonomy: dict,
        entry_types: list[EntryType] | None = None,
        num_categories: int = 10,
        num_taxonomy_categories: int | None = None,
        include_general_benign: bool = False,
        general_benign_categories: int = 10,
        benign_taxonomy: dict | None = None,
        resume: bool = True,
        verbose: bool = True,
        read_back: bool = True,
    ) -> pd.DataFrame:
        """Generate the constitution, one call per unit (:func:`plan_units`).

        Each taxonomy category gets ``num_categories`` constitution
        categories per entry type (all four by default);
        ``include_general_benign`` adds one call per category of
        ``benign_taxonomy`` for ``general_benign_categories`` benign ones,
        saved to ``general_benign.csv``. Each taxonomy gives its first
        ``num_taxonomy_categories`` categories. ``resume=False`` removes the
        CSVs, the ledger and the plan first; a resume of a constitution
        saved without a column the rows now have raises before any call.

        The units run one at a time, in plan order. A reply without its
        ``[END]`` marker, or with no entry that parses, is rejected and the
        call made again, up to three times; a unit rejected every time is
        acked ``rejected`` with no rows and skipped on resume. A failed call
        stops the run; the units saved before it stay acked.

        Returns:
            The saved constitution (:func:`read_constitution`), rows of earlier
            runs included, or an empty frame when ``read_back`` is False.
        """
        units = plan_units(
            taxonomy, entry_types, num_categories, num_taxonomy_categories,
            include_general_benign, general_benign_categories, benign_taxonomy)
        merged = self.output_dir / paths.CONSTITUTION_MERGED_FILENAME
        if resume:
            try:
                check_columns(merged, [dict.fromkeys(_COLUMNS)])
            except ValueError as error:
                raise ValueError(
                    f"{error} It was saved before the column existed; run with "
                    "resume=False to generate the constitution again.") from None
        ledger = Ledger.sidecar(merged, ("unit",), {"unit": str},
                                name="constitution.state.jsonl")
        manifest = Manifest.sidecar(merged, name="constitution.manifest.jsonl")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            for csv_file in self.output_dir.glob("*.csv"):
                csv_file.unlink()
        done = resume_state(ledger, resume=resume, manifest=manifest)
        manifest.write(unit.manifest_row for unit in units)
        pending = [unit for unit in units if unit.key not in done]

        if verbose:
            logger.info("Constitution Generation")
            logger.info("Model: %s | Units: %d | Categories per unit: %d",
                        self.model, len(units), num_categories)
            if done:
                logger.info("Resume: %d units already completed",
                            len(units) - len(pending))

        written = finished = 0
        rejected: list[str] = []

        def _on_done(key: str, value: tuple[list[dict], int]) -> None:
            nonlocal written, finished
            rows, attempts = value
            finished += 1
            if verbose:
                logger.info("[%d/%d] %s -> %d entries",
                            finished, len(pending), key, len(rows))
            if not rows:
                # Acked: un-acked, it would be generated, and paid for, again
                # on every resume.
                buf.add([], {"unit": key, "status": "rejected", "attempts": attempts})
                rejected.append(key)
                return
            buf.add(rows, {"unit": key})
            written += len(rows)

        with CommitBuffer(append=self._append, ledger=ledger, every=1) as buf:
            drive_units(
                ((unit.key, self._unit(unit)) for unit in pending),
                on_done=_on_done,
                pool_size=1,
                progress="constitution" if verbose else None,
            )

        if rejected:
            logger.warning(
                "%d units rejected after %d attempt(s) each: %s. They are acked "
                "and skipped on resume; run with resume=False, or delete their "
                "lines from %s, to try them again.",
                len(rejected), _ATTEMPTS, ", ".join(rejected), ledger.path.name,
            )
        if verbose:
            logger.info("Generated this run: %d entries, saved to %s",
                        written, self.output_dir)

        return read_constitution(self.output_dir) if read_back else pd.DataFrame()
