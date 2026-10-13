"""Resume and commit: what a run skips, and rows before acks.

A fresh run clears the artifact together with its ledger: acks left behind
would make the next run skip every unit and write nothing. A unit is acked
only once its rows are on disk, so a crash between the two runs it again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .rows import keys_in
from .sidecar import Ledger, Manifest


def resume_state(
    ledger: Ledger,
    *,
    resume: bool,
    artifact: str | Path | None = None,
    manifest: Manifest | None = None,
    extra: Iterable[Any] | None = None,
) -> set[Any]:
    """The unit keys this run skips; a fresh run clears what an earlier one left.

    Args:
        ledger: The stage's ledger.
        resume: False starts over: the artifact, the ledger and the manifest
            are removed and the result is empty.
        artifact: The stage's output CSV. A fresh run deletes it. A resumed
            run adds the unit keys of its rows, so a unit whose rows were
            written but not acked is not run again. ``None`` when the stage
            clears and reads its own output.
        manifest: The stage's manifest, removed on a fresh run.
        extra: More finished keys to add on a resumed run.

    Returns:
        The keys as ``Ledger.completed()`` shapes them: one value for a
        single key field, a tuple for several.
    """
    path = Path(artifact) if artifact is not None else None
    if not resume:
        if path is not None and path.exists():
            path.unlink()
        ledger.reset()
        if manifest is not None:
            manifest.reset()
        return set()

    done = ledger.completed()
    if path is not None and path.is_file():
        done |= keys_in(path, ledger.key_fields, ledger.casters)
    return done | set(extra) if extra else done


def commit(
    rows: Sequence[Mapping[str, Any]],
    *,
    append: Callable[[Sequence[Mapping[str, Any]]], None],
    ledger: Ledger,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    """Write rows, then ack their units, never the other way round.

    Args:
        rows: Output rows for this unit or chunk. Empty skips the append: a
            unit that produced no rows is still acked, or it would be
            retried forever.
        append: Writes the rows. If it raises, nothing is acked and the
            units run again.
        ledger: The stage's ledger.
        records: The ack records. Defaults to the ledger's key fields taken
            from each row; pass them when a unit has no row or its record
            carries more than the key.
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
        """Set up a buffer that commits through ``append`` and ``ledger``.

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
