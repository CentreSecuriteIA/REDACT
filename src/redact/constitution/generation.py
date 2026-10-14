"""The constitution stage: generate each unit, save its rows, then ack it.

Usage::

    from redact.constitution import ConstitutionPipeline
    from redact.dataset import load_taxonomy
    from redact.llms import ModelClient

    pipeline = ConstitutionPipeline(ModelClient.create("claude-opus-4-6"))
    constitution = pipeline.run(load_taxonomy("content_moderation_categories"))
"""

import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..dataset import Ledger, Manifest, append_rows, resume_state
from ..llms.client import ModelClient
from ..llms.router import generate_sample
from ..types import EntryType
from .units import _COLUMNS, Unit, messages_of, plan_units, rows_of

logger = logging.getLogger(__name__)


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

    Each unit's rows go to ``{entry_type}.csv`` and ``merged.csv``. The
    ledger ``constitution.state.jsonl`` beside them is plain: no lock, no
    recorded sizes. A crash between a unit's two appends leaves it un-acked,
    and the next run appends its rows to the type's file again.
    """

    def __init__(self, client: ModelClient, output_dir: str | Path | None = None):
        """``output_dir`` defaults to ``paths.constitution_dir()``."""
        self.client = client
        self.model = client.model
        self.output_dir = Path(
            output_dir if output_dir is not None else paths.constitution_dir())

    def _rows(self, unit: Unit) -> list[dict]:
        """Generate the unit's rows, up to three times; ``[]`` if each attempt fails."""
        messages = messages_of(unit)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                reply = generate_sample(self.client, messages, max_tokens=10000)
                rows = rows_of(unit, reply, self.model)
                if not rows:
                    raise ValueError(
                        f"No entries parsed (raw length: {len(reply)} chars)")
                return rows
            except Exception as e:
                if attempt < max_attempts:
                    logger.warning("Attempt %d/%d failed for %s: %s. Retrying...",
                                   attempt, max_attempts, unit.key, e)
                else:
                    logger.error("All %d attempts failed for %s: %s. Skipping.",
                                 max_attempts, unit.key, e)
        return []

    def _append(self, rows: list[dict]) -> None:
        """Append the rows to their entry type's CSV, then to ``merged.csv``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        by_type: dict[str, list[dict]] = {}
        for row in rows:
            by_type.setdefault(row["entry_type"], []).append(row)
        for entry_type, its_rows in by_type.items():
            append_rows(self.output_dir / f"{entry_type}.csv", its_rows)
        append_rows(self.output_dir / paths.CONSTITUTION_MERGED_FILENAME, rows)

    def run(
        self,
        taxonomy: dict,
        entry_types: list[EntryType] | None = None,
        num_categories: int = 10,
        num_taxonomy_categories: int | None = None,
        include_standalone_benign: bool = False,
        standalone_benign_categories: int = 10,
        resume: bool = True,
        verbose: bool = True,
        read_back: bool = True,
    ) -> pd.DataFrame:
        """Generate the constitution, one call per ``(source_category, entry_type)``.

        Each taxonomy category (the first ``num_taxonomy_categories``) gets
        ``num_categories`` constitution categories per entry type (all four
        by default); ``include_standalone_benign`` adds one call for
        ``standalone_benign_categories`` benign ones with no taxonomy
        category, saved to ``general_benign.csv``. ``resume=False`` removes
        the CSVs, the ledger and the plan first.

        Returns:
            The saved constitution (:func:`read_constitution`), rows of earlier
            runs included, or an empty frame when ``read_back`` is False.
        """
        units = plan_units(
            taxonomy, entry_types, num_categories, num_taxonomy_categories,
            include_standalone_benign, standalone_benign_categories)
        merged = self.output_dir / paths.CONSTITUTION_MERGED_FILENAME
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

        written = 0
        failed: list[str] = []
        for i, unit in enumerate(pending, 1):
            rows = self._rows(unit)
            if verbose:
                logger.info("[%d/%d] %s -> %d entries",
                            i, len(pending), unit.key, len(rows))
            if not rows:
                failed.append(unit.key)
                continue
            self._append(rows)
            ledger.record([{"unit": unit.key}])
            written += len(rows)

        if verbose:
            logger.info("Generated this run: %d entries, saved to %s",
                        written, self.output_dir)
            if failed:
                logger.info("Skipped (failed after retries): %s", ", ".join(failed))

        return read_constitution(self.output_dir) if read_back else pd.DataFrame()
