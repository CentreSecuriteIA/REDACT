"""The two resume invariants every batched stage must follow identically.

Both are easy to get wrong in ways no test catches until a real crash, and
both were previously restated in four places:

1. **A fresh run clears the artifact *and* its ledger.** Clearing only the
   artifact leaves stale acks behind, so the next run skips every unit and
   writes nothing — silently producing an empty output that looks like a
   successful no-op.
2. **Append before acking.** A unit is recorded only once its rows are on
   disk, so a crash between the two re-runs that unit rather than losing it.
   Reversed, a crash drops work permanently while claiming it was done.

This is deliberately **not** a sweep driver. The loops these live inside
differ in ways that matter — per-unit acking in constitution vs per-chunk
elsewhere, DataFrame masks vs list comprehensions vs tuple-key filters, and
rows that are not 1:1 with units (paraphrase drops deduped rows while still
acking them). Extracting the whole loop would mean a function that is mostly
callbacks. These two steps are the parts that are genuinely identical, so
they are the parts that are shared. ``CommitBuffer`` applies 2 in batches for
stages driven unit by unit (``llm_pipeline.drive_units``): finished units
queue up and are committed every N and at the end.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .sidecar import Ledger


def resume_state(
    ledger: Ledger,
    *,
    resume: bool,
    artifact: str | Path | None = None,
    extra: Iterable[Any] | None = None,
) -> set[Any]:
    """Completed-unit keys for this run, clearing both sources when not resuming.

    Args:
        ledger: The stage's sidecar ledger.
        resume: False means start over — the artifact and the ledger are both
            removed, and the returned set is empty.
        artifact: Output file to delete on a fresh run. ``None`` when the
            caller removes it itself (several stages have more than one).
        extra: Additional completed keys to union in — the back-compat path
            for runs created before this stage had a ledger (jailbreak reads
            them out of the output CSV). Ignored when ``resume`` is False.

    Returns:
        The set of unit keys to skip. Shape matches ``Ledger.completed()``:
        a scalar per unit for a single key field, a tuple for several.
    """
    if not resume:
        if artifact is not None:
            path = Path(artifact)
            if path.exists():
                path.unlink()
        ledger.reset()
        return set()

    done = ledger.completed()
    return done | set(extra) if extra else done


def commit(
    rows: Sequence[Mapping[str, Any]],
    *,
    append: Callable[[Sequence[Mapping[str, Any]]], None],
    ledger: Ledger,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    """Write rows, then ack their units — never the other way round.

    Args:
        rows: Output rows for this unit or chunk. Empty is allowed and skips
            the append: a unit can legitimately produce no rows and still need
            acking, or it would be retried forever.
        append: Writes the rows. Whatever the stage does — append to one CSV,
            or fan out per category. Raising here means nothing is acked, so
            the units re-run.
        ledger: The stage's ledger.
        records: Explicit ack records, when they are not derivable from
            ``rows`` — paraphrase acks deduped units that produced no row, and
            carries a ``status`` field; :class:`CommitBuffer` always passes
            them. Defaults to the ledger's key fields projected out of
            ``rows``, which is what every other stage builds by hand.
    """
    if rows:
        append(rows)
    if records is None:
        records = [{k: r[k] for k in ledger.key_fields} for r in rows]
    ledger.record(records)


class CommitBuffer:
    """Commit finished units in batches: their rows first, their acks after.

    ``add()`` queues one unit. Every ``every`` units, and on leaving the
    ``with`` block, the queued rows are appended and the queued records
    acked through :func:`commit`. The exit flush runs whatever ended the
    block: a finished unit is complete, so a crash loses only the live ones.

    Usage::

        with CommitBuffer(append=write_rows, ledger=ledger) as buf:
            drive_units(source, pool_size=32,
                        on_done=lambda key, value: buf.add(rows_of(value), {"input_id": key}))
    """

    def __init__(self, *, append, ledger: Ledger, every: int = 32):
        """
        Args:
            append: Writes a batch of rows, as for :func:`commit`.
            ledger: The stage's ledger.
            every: Units per flush, at least 1.
        """
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}.")
        self._append, self._ledger, self.every = append, ledger, every
        self._rows: list = []
        self._records: list[Mapping[str, Any]] = []
        self.committed = 0   # units committed so far

    def add(self, rows: Sequence[Any], record: Mapping[str, Any]) -> None:
        """Queue one finished unit: its rows (possibly none) and its ack record."""
        self._rows.extend(rows)
        self._records.append(record)
        if len(self._records) >= self.every:
            self.flush()

    def flush(self) -> None:
        """Commit everything queued.

        The queue is emptied before the write, so a failed append leaves
        its units un-acked and never re-appended.
        """
        if not self._records:
            return
        rows, records = self._rows, self._records
        self._rows, self._records = [], []
        commit(rows, append=self._append, ledger=self._ledger, records=records)
        self.committed += len(records)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.flush()
        return False
