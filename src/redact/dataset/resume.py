"""Resume and commit: what a run skips, and rows before acks.

A save is two synced appends: the rows of the finished units, then their
acks, the last of which records the size and a checksum of every file of the
stage (the ledger's ``_saved``). A unit is acked only once its rows are on
disk, and a save counts only once its ``_saved`` line is.

A stage that names its output to :func:`resume_state` resumes from the
ledger alone. The resume takes the stage's lock, checks each file against
what the ledger recorded, cuts off what a save that did not finish left
behind, and stops on a file it cannot vouch for. It reads no row of the
output, except once in a folder written before the ledger recorded sizes. A
stage that names no output keeps a plain ledger: no lock, no sizes, nothing
cut.

A stage of several output files names their folder and the pattern they
match. A file that matches and that no finished save recorded was first
written by a save that did not finish, and is moved aside whole.

A fresh run clears the artifact together with its ledger: acks left behind
would make the next run skip every unit and write nothing. A fresh run of
part of a stage removes that part's rows and acks alone
(:func:`drop_units`).

Known limits:

- A change to an output that keeps its size and lies before its last 4 kB
  is not noticed.
- Two writers are stopped by the lock and by the size check before each
  save. On a volume that ignores the lock, two runs that save within the
  same millisecond are not.
- Readers other than the stage itself read a stage file as it is, with what
  an unfinished save left in it.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from filelock import FileLock, Timeout

from .rows import _retry_held, keys_in
from .sidecar import Ledger, LedgerState, Manifest, _replace

logger = logging.getLogger(__name__)

# The ``units_in`` of :func:`resume_state`.
_UnitsIn = Callable[[Path], Mapping[Any, Iterable[str]]]

# A checksum covers this many bytes at the end of what a save left in a file.
_CHECKSUM_BYTES = 4096

# The lock of each stage this process is the writer of, by its lock file.
_locks: dict[Path, FileLock] = {}

# The lock file of each ledger that was asked for it, as ``_lock_path`` says.
_lock_paths: WeakKeyDictionary[Ledger, Path] = WeakKeyDictionary()


def resume_state(
    ledger: Ledger,
    *,
    resume: bool,
    artifact: str | Path | None = None,
    pattern: str | None = None,
    manifest: Manifest | None = None,
    extra: Iterable[Any] | None = None,
    units_in: _UnitsIn | None = None,
) -> set[Any]:
    """The unit keys this run skips; a fresh run clears what an earlier one left.

    With ``artifact`` the call also makes this process the one writer of the
    stage, until the stage's ``CommitBuffer`` exits or :func:`release_lock`
    is called, and a resumed run puts the stage's output and the ledger back
    at the last finished save. Call it before the stage writes anything in
    the folder.

    Args:
        ledger: The stage's ledger.
        resume: False starts over: the output, its ``.cut`` files, the
            ledger and the manifest are removed and the result is empty.
        artifact: The stage's output CSV, or with ``pattern`` the folder its
            output files are under. ``None`` for a stage that clears and
            reads its own output: its ledger is read as it is, nothing is
            locked and nothing is written.
        pattern: For a stage of several output files, the glob of those
            files under ``artifact``, such as ``*/samples.csv``.
        manifest: The stage's manifest, removed on a fresh run.
        extra: More finished keys to add on a resumed run.
        units_in: Reads one output file when a folder written before the
            ledger recorded sizes is taken over: each unit key in the file,
            as the ledger casts it, with the content ids of its rows. The
            default takes the ledger's key fields as columns and gives no
            ids.

    Returns:
        The keys as ``Ledger.completed()`` shapes them: one value for a
        single key field, a tuple for several.

    Raises:
        RuntimeError: Another run holds the stage's lock, a file is not what
            the ledger recorded, or another program holds a file that has to
            be cut back.
        PermissionError: Another program holds a file that a fresh run has
            to remove.
    """
    if artifact is None:
        if not resume:
            _clear(ledger, manifest)
            return set()
        done = ledger.completed()
    else:
        path = Path(artifact)
        _lock(ledger)
        try:
            if not resume:
                _clear(ledger, manifest, *_outputs(path, pattern),
                       *_outputs(path, pattern, ".cut"))
                return set()
            done = _restore(ledger, _outputs(path, pattern), units_in)
        except BaseException:
            release_lock(ledger)    # a resume that stops holds nothing
            raise
    return done | set(extra) if extra else done


def release_lock(ledger: Ledger) -> None:
    """Give back the lock :func:`resume_state` took for the ledger's stage.

    ``CommitBuffer`` does this on exit. A stage that resumed and finds
    nothing to save opens no buffer, and calls this itself.
    """
    lock = _locks.pop(_lock_path(ledger), None)
    if lock is not None:
        lock.release()


def _lock(ledger: Ledger) -> None:
    """Make this process the one writer of the ledger's stage, or raise.

    The operating system holds the lock, so a run that died holds nothing.
    A lock this process already holds is kept, so a stage that stopped
    before its ``CommitBuffer`` exited can be called again. Within one
    process the lock therefore does not stop a second call of the same
    stage while the first still runs; only the size check before each save
    does.
    """
    path = _lock_path(ledger)
    if path in _locks:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # No waiting. Not thread-local: the thread that leaves the buffer may
    # not be the one that resumed.
    lock = FileLock(path, timeout=0, thread_local=False)
    try:
        lock.acquire()
    except Timeout:
        raise RuntimeError(
            f"{path} is locked: another run, or this same process, is writing "
            "this stage's output. One process writes an output at a time."
        ) from None
    _locks[path] = lock


def _lock_path(ledger: Ledger) -> Path:
    """The lock file of the ledger's stage: ``<ledger name>.lock`` beside it.

    Made absolute the first time it is asked for, and kept: worked out again
    after a change of directory, a relative path would name another file and
    the lock would never be given back.
    """
    if ledger not in _lock_paths:
        name = ledger.path.with_name(ledger.path.name + ".lock")
        _lock_paths[ledger] = Path(os.path.abspath(name))
    return _lock_paths[ledger]


def _opened(ledger: Ledger) -> bool:
    """Whether :func:`resume_state` opened the ledger with its artifact, and
    the stage has not given the lock back since."""
    return _lock_path(ledger) in _locks


def _clear(ledger: Ledger, manifest: Manifest | None, *files: Path) -> None:
    """Remove what an earlier run left: ``files``, the ledger, the manifest."""
    for file in files:
        file.unlink(missing_ok=True)
    ledger.reset()
    if manifest is not None:
        manifest.reset()


def _outputs(artifact: Path, pattern: str | None, suffix: str = "") -> list[Path]:
    """The stage's output files, or with ``suffix`` the files kept beside them.

    The artifact itself, or the files under it that match ``pattern``. The
    files beside them are looked for by name, so the ``.cut`` file of an
    output that is gone is found too.
    """
    if pattern is None:
        return [artifact.with_name(artifact.name + suffix)]
    return sorted(artifact.glob(pattern + suffix))


def _restore(
    ledger: Ledger, outputs: Sequence[Path], units_in: _UnitsIn | None
) -> set[Any]:
    """Put the stage's files back at its last finished save; the finished units."""
    state = ledger.read()
    if state.saved is None:
        return _adopt(ledger, state, outputs, units_in)
    _put_back(ledger, state.saved, outputs)
    _truncate(ledger.path, state.end)
    return state.done


def _put_back(
    ledger: Ledger,
    saved: Mapping[str, Sequence],
    outputs: Iterable[Path],
    shorter_ok: bool = False,
) -> None:
    """Cut the stage's output back to what its last finished save left.

    Every file is checked before any is cut, so a resume that stops on one
    file has changed none. An output that no save recorded was first written
    by a save that did not finish: all of it is a tail.

    ``shorter_ok`` leaves a recorded file alone that :func:`drop_units`
    left shorter, or removed.
    """
    recorded = {ledger.path.parent / name: entry for name, entry in saved.items()}
    if shorter_ok:
        recorded = {path: entry for path, entry in recorded.items()
                    if not _left_shorter(path, entry[0])}
    for path, (size, checksum) in recorded.items():
        _verify(ledger, path, size, checksum)
    for path, (size, _) in recorded.items():
        _cut_back(path, size)
    for output in outputs:
        if _name(ledger, output) not in saved:
            _cut_back(output, 0)


def _verify(ledger: Ledger, path: Path, size: int, checksum: str) -> None:
    """Stop unless the first ``size`` bytes of ``path`` are those a save left.

    A longer file passes: what follows ``size`` is the tail of a save that
    did not finish. The checksum is taken at the recorded size either way,
    so a file that another program saved again, and made longer, is not cut.
    """
    if not path.is_file():
        found = "is missing"
    elif path.stat().st_size < size:
        found = f"is {path.stat().st_size} bytes"
    elif _checksum(path, size) != checksum:
        found = "holds other bytes there"
    else:
        return
    raise RuntimeError(
        f"Cannot resume {path}: {ledger.path.name} recorded {size} bytes of it "
        f"and the file {found}. It was replaced, edited or copied incompletely. "
        "Restore the file. Or delete the ledger: the next resume then takes "
        "over the rows the file holds, and units without rows run again. Or "
        "start over with resume=False."
    )


def _cut_back(path: Path, size: int) -> None:
    """Cut ``path`` back to ``size`` bytes; what is cut off goes to ``<path>.cut``.

    The bytes after ``size`` are what a save that did not finish left. They
    are appended to the ``.cut`` file, after any earlier tail, and synced
    there before the file is cut. A file cut back to nothing is removed, so
    the next save writes its header.
    """
    if not path.exists():
        return
    extra = path.stat().st_size - size
    if extra > 0:
        cut = _cut_file(path)
        with path.open("rb") as source, cut.open("ab") as target:
            source.seek(size)
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        logger.warning(
            "Cut %d bytes that no finished save recorded off %s; they are kept in %s.",
            extra, path, cut.name,
        )
    if size:
        _truncate(path, size)
    else:
        _retry_cut(path, path.unlink)


def _cut_file(path: Path) -> Path:
    """Where the tails cut off ``path`` are kept: ``<path>.cut``."""
    return path.with_name(path.name + ".cut")


def _truncate(path: Path, size: int) -> None:
    """Cut ``path`` to ``size`` bytes through a file handle, synced and checked.

    A cut by path (``os.truncate``) returns without an error and without
    cutting while another program has the file mapped on Windows. Through a
    handle it raises ``PermissionError`` then, and is tried again.
    """
    if path.stat().st_size == size:
        return

    def cut() -> None:
        with path.open("r+b") as fh:
            fh.truncate(size)
            os.fsync(fh.fileno())

    _retry_cut(path, cut)
    if path.stat().st_size != size:
        raise RuntimeError(
            f"{path} is {path.stat().st_size} bytes after being cut back to "
            f"{size}: another program holds it. Close that program and resume."
        )


def _retry_cut(path: Path, cut: Callable[[], None]) -> None:
    """Run ``cut``, which cuts or removes ``path``, again while the file is held.

    A file that stays held stops the resume, with a message that says why.
    """
    try:
        _retry_held(cut)
    except PermissionError as error:
        raise RuntimeError(
            f"{path} could not be cut back: another program holds it. Close "
            "that program and resume."
        ) from error


def _adopt(
    ledger: Ledger,
    state: LedgerState,
    outputs: Iterable[Path],
    units_in: _UnitsIn | None,
) -> set[Any]:
    """Take over a folder whose ledger records no sizes; the finished units.

    Each output file is read once for its unit keys and the ids of their
    rows. One ledger write then acks the keys found only in the output,
    records the ids with the unit they belong to, and records each file's
    size, so the next resume reads the ledger alone. A folder that holds
    neither rows nor acks is left as it is.
    """
    found: dict[Any, list[str]] = {}
    sizes: dict[Path, int] = {}
    for output in outputs:
        if output.exists() and not output.stat().st_size:
            # A first save that died before its first byte: no row to take over.
            _cut_back(output, 0)
        if output.exists():
            for key, ids in _read_units(ledger, output, units_in).items():
                found.setdefault(key, []).extend(ids)
            sizes[output] = output.stat().st_size
    if ledger.exists():
        # A last line cut off mid-write would run into the next one.
        _truncate(ledger.path, state.end)
    if sizes or state.end:
        fields = ledger.key_fields
        # A unit with ids is acked again: its old ack does not carry them.
        acks = [
            {**dict(zip(fields, key if len(fields) > 1 else (key,))),
             **({"ids": ids} if ids else {})}
            for key, ids in found.items() if ids or key not in state.done
        ]
        ledger.record(acks, saved=_saved(ledger, sizes))
    return state.done | set(found)


def _read_units(
    ledger: Ledger, output: Path, units_in: _UnitsIn | None
) -> Mapping[Any, Iterable[str]]:
    """The unit keys in an output being taken over, each with its rows' ids.

    Raises:
        RuntimeError: The file ends inside a row or does not parse. Nothing
            recorded says where its whole rows end.
    """
    try:
        # pandas reads a last row that lost its end as a short row.
        if not _ends_with_a_row(output):
            raise ValueError("it does not end with a line break")
        if units_in is not None:
            return units_in(output)
        return dict.fromkeys(keys_in(output, ledger.key_fields, ledger.casters), ())
    except ValueError as error:    # pandas' parser errors are ValueErrors too
        raise RuntimeError(
            f"Cannot resume {output}: {error}. The file was written before "
            "the ledger recorded sizes, so nothing says where its whole rows "
            "end. Repair the file, or start over with resume=False."
        ) from error


def _ends_with_a_row(path: Path) -> bool:
    """Whether the CSV at ``path`` ends with a line break, as a whole row does."""
    with path.open("rb") as fh:
        fh.seek(max(path.stat().st_size - 1, 0))
        return fh.read() == b"\n"


def drop_units(
    ledger: Ledger,
    *,
    artifact: str | Path,
    rows: Callable[[Mapping[str, str]], bool],
    acks: Callable[[Mapping[str, Any]], bool],
    pattern: str | None = None,
    units_in: _UnitsIn | None = None,
) -> None:
    """Take some units out of a stage: their rows and their acks.

    For a fresh run of part of a stage: what is not picked keeps its rows
    and its acks. The call makes this process the one writer of the stage,
    as :func:`resume_state` does, and the resume that follows keeps the
    lock.

    What a save that did not finish left is cut off first, as on a resume.
    Each output file is then written again without the rows, and last the
    ledger without the acks, ending in one line that records the new sizes.
    Each goes through a temporary file and a rename. Stopped in between, the
    call leaves files shorter than the ledger records: a resume stops on
    them, and the call, made again, finishes the job. So a file that is
    gone, or shorter than recorded and whole to its last row, is taken here
    as it is.

    Args:
        ledger: The stage's ledger.
        artifact: The stage's output, as for :func:`resume_state`.
        rows: Picks the rows to remove, given a row's cells by column name.
        acks: Picks the ledger records to remove.
        pattern: As for :func:`resume_state`.
        units_in: As for :func:`resume_state`: a folder written before the
            ledger recorded sizes is taken over first.

    Raises:
        RuntimeError: As for :func:`resume_state`, or an output file does
            not parse as CSV.
        PermissionError: Another program holds a file that has to be
            replaced.
    """
    outputs = _outputs(Path(artifact), pattern)
    _lock(ledger)
    try:
        state = ledger.read()
        if state.saved is None:
            _adopt(ledger, state, outputs, units_in)
        else:
            _put_back(ledger, state.saved, outputs, shorter_ok=True)
        for name in ledger.saved or {}:
            _rewrite_without(ledger.path.parent / name, rows)
        if ledger.exists():
            ledger.rewrite(acks, _measured(ledger))
    except BaseException:
        release_lock(ledger)    # a call that stops holds nothing
        raise


def _left_shorter(path: Path, size: int) -> bool:
    """Whether ``path`` is gone, or shorter than ``size`` and whole to its last row.

    That is how :func:`drop_units` leaves a file when it stops half way.
    """
    if not path.is_file():
        return True
    return path.stat().st_size < size and _ends_with_a_row(path)


def _rewrite_without(path: Path, drop: Callable[[Mapping[str, str]], bool]) -> None:
    """Write the CSV at ``path`` again without the rows ``drop`` picks.

    The rows that stay are the bytes they were. They are written to a file
    beside ``path``, which is synced and renamed onto it, so the file holds
    its old rows or the ones that stay. A file with no such row is left
    alone, as is one that is gone. A file left with no row is removed, so
    the next save writes its header.

    Raises:
        RuntimeError: The file does not parse as CSV.
    """
    if not path.is_file():
        return
    temporary = path.with_name(path.name + ".tmp")
    dropped = kept = 0
    try:
        with (
            path.open(encoding="utf-8", newline="") as old,
            temporary.open("w", encoding="utf-8", newline="") as new,
        ):
            records = _records(old)
            text, header = next(records, ("", []))
            new.write(text)
            # A spreadsheet saves the file with a byte-order mark.
            columns = [name.lstrip("\ufeff") for name in header]
            for text, cells in records:
                if drop(dict(zip(columns, cells))):
                    dropped += 1
                else:
                    new.write(text)
                    kept += 1
            if dropped and kept:    # it is going to be the file
                new.flush()
                os.fsync(new.fileno())
        if dropped and kept:
            _replace(temporary, path)
        elif dropped:
            _retry_cut(path, path.unlink)
    except (csv.Error, UnicodeError) as error:
        raise RuntimeError(
            f"Cannot write {path} again without some of its rows: {error}. "
            "Repair the file, or start the whole stage over."
        ) from error
    finally:
        # Still there unless it was renamed onto the file.
        temporary.unlink(missing_ok=True)


def _records(lines: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
    """Each record of a CSV read line by line: its text in the file, its cells.

    A record is several lines when a cell holds a line break.
    """
    taken: list[str] = []

    def take() -> Iterator[str]:
        for line in lines:
            taken.append(line)
            yield line

    # strict: a file that ends inside a quoted cell raises, not a short record.
    for cells in csv.reader(take(), strict=True):
        yield "".join(taken), cells
        taken.clear()


def _measured(ledger: Ledger) -> dict[str, list]:
    """The ``_saved`` of the files the ledger records, as they are on disk now.

    A file that is gone is left out.
    """
    saved = {}
    for name in ledger.saved or {}:
        path = ledger.path.parent / name
        if path.is_file():
            size = path.stat().st_size
            saved[name] = [size, _checksum(path, size)]
    return saved


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


def _saved(ledger: Ledger, sizes: Mapping[Path, int]) -> dict[str, list]:
    """The ``_saved`` of a save that left the files it wrote at ``sizes``.

    It names every file of the stage: one the save did not write keeps what
    the ledger last recorded.
    """
    saved = dict(ledger.saved or {})
    for path, size in sizes.items():
        saved[_name(ledger, path)] = [size, _checksum(path, size)]
    return saved


def _name(ledger: Ledger, path: str | Path) -> str:
    """The name ``_saved`` gives a file: its path from the ledger's folder.

    Written with ``/`` on every platform, so a folder copied to another
    machine resumes there. A file the ledger records under another spelling
    keeps that one: on Windows ``OUT.csv`` and ``out.csv`` are one file.
    """
    folder = ledger.path.parent
    name = Path(os.path.relpath(path, folder)).as_posix()
    recorded = ledger.saved or {}
    if name not in recorded:
        for known in recorded:
            with suppress(OSError):     # one of the two is not on disk
                if os.path.samefile(folder / known, path):
                    return known
    return name


def _checksum(path: str | Path, size: int) -> str:
    """crc32 of the last bytes of ``path`` before ``size``, as 8 hex characters.

    At most ``_CHECKSUM_BYTES`` are read, so the cost does not grow with the
    file.
    """
    start = max(0, size - _CHECKSUM_BYTES)
    with open(path, "rb") as fh:
        fh.seek(start)
        return f"{zlib.crc32(fh.read(size - start)):08x}"


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
