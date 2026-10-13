"""Rows before acks: the save of a stage's finished units.

A save is two synced appends: the rows of the finished units, then their
acks, the last of which records the size and a checksum of every file of the
stage (the ledger's ``_saved``). A unit is acked only once its rows are on
disk, and a save counts only once its ``_saved`` line is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .resume import _opened, _saved, release_lock
from .sidecar import Ledger


def commit(
    rows: Sequence[Mapping[str, Any]],
    *,
    append: Callable[[Sequence[Mapping[str, Any]]], Mapping[Path, int] | None],
    ledger: Ledger,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    """Write rows, then ack their units, never the other way round.

    Args:
        rows: Output rows for this unit or chunk. Empty skips the append: a
            unit that produced no rows is still acked, or it would be
            retried forever.
        append: Writes the rows. If it raises, nothing is acked and the
            units run again. On a ledger that :func:`resume_state` opened
            with its artifact it returns the size of each file it wrote, by
            path, as ``append_rows`` does.
        ledger: The stage's ledger.
        records: The ack records. Defaults to the ledger's key fields taken
            from each row; pass them when a unit has no row or its record
            carries more than the key.

    Raises:
        ValueError: Rows are given with no record; nothing is written, since
            rows on disk that no unit owns would be acked by the next save.
            Or the ledger holds ``_saved`` and no resume opened it: the rows
            are appended, their acks are refused, and the next resume cuts
            the rows off.
        RuntimeError: On an opened ledger, a file changed since the last
            save, or the append reported no size. Nothing is acked.
    """
    if records is None:
        records = [{k: r[k] for k in ledger.key_fields} for r in rows]
    records = list(records)
    if not records:
        if rows:
            raise ValueError(f"{len(rows)} rows to commit and no record to ack them.")
        return
    if _opened(ledger):
        _save(rows, records, append, ledger)
    else:
        if rows:
            append(rows)
        ledger.record(records)


def _save(rows, records, append, ledger: Ledger) -> None:
    """One save on a ledger that :func:`resume_state` opened with its artifact.

    Each file must still have the size the ledger last recorded. A new
    ledger gets its opening line, synced, before the first rows: rows on
    disk beside no ledger would read as a folder to take over. The last ack
    carries the size and a checksum of every file of the stage.
    """
    _check_sizes(ledger)
    if ledger.saved is None:
        ledger.record([], saved={})
    sizes = append(rows) if rows else {}
    if rows and not sizes:
        # Acked without their size, the rows would be cut off by the next resume.
        raise RuntimeError(
            f"The append of {len(rows)} rows reported no file size, so "
            f"{ledger.path} cannot record them. Return what append_rows returns."
        )
    ledger.record(records, saved=_saved(ledger, sizes))


def _check_sizes(ledger: Ledger) -> None:
    """Stop a save unless each file still has the size the ledger recorded.

    A file that changed since the last save has another writer. The check
    asks nothing of the storage, so it also holds on a volume that ignores
    the lock.
    """
    for name, (size, _) in (ledger.saved or {}).items():
        path = ledger.path.parent / name
        actual = path.stat().st_size if path.is_file() else None
        if actual != size:
            now = "missing" if actual is None else f"{actual} bytes"
            raise RuntimeError(
                f"{path} is {now} and {ledger.path.name} recorded {size} bytes "
                "after the last save: something else changed the file. Nothing "
                "was saved. Stop the other writer, then resume."
            )


class CommitBuffer:
    """Commit finished units in batches: their rows first, their acks after.

    ``add()`` queues one unit. Every ``every`` units, and on leaving the
    ``with`` block, the queued rows are appended and the queued records
    acked through :func:`commit`. The exit flush runs whatever ended the
    block: a finished unit is complete, so a crash loses only the live ones.
    Leaving the block also gives back the lock :func:`resume_state` took.

    Usage::

        done = resume_state(ledger, resume=True, artifact=out)
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
        self._units: list[tuple[list, Mapping[str, Any]]] = []   # (rows, record)
        self.committed = 0   # units committed so far

    def add(self, rows: Iterable[Any], record: Mapping[str, Any]) -> None:
        """Queue one finished unit: its rows (possibly none) and its ack record.

        The two are queued as one object, so an interrupt cannot leave a
        unit's rows in a save that does not ack it.
        """
        self._units.append((list(rows), record))
        if len(self._units) >= self.every:
            self.flush()

    def flush(self) -> None:
        """Commit everything queued.

        The queue is emptied before the write, so a failed append leaves
        its units un-acked and never re-appended.
        """
        if not self._units:
            return
        units, self._units = self._units, []
        commit(
            [row for rows, _ in units for row in rows],
            append=self._append,
            ledger=self._ledger,
            records=[record for _, record in units],
        )
        self.committed += len(units)

    def __enter__(self):
        """Refuse a ledger that records sizes and that no resume opened.

        Saving onto it would leave what an unfinished save left in the file.
        """
        if self._ledger.saved is not None and not _opened(self._ledger):
            raise RuntimeError(
                f"{self._ledger.path} records the size of each file, and "
                "resume_state has not opened it in this process. Call "
                "resume_state(ledger, resume=..., artifact=...) before the "
                "CommitBuffer."
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.flush()
        finally:
            release_lock(self._ledger)
        return False
