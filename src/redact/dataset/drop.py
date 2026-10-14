"""Starting part of a stage over.

A fresh run of part of a stage removes that part's rows and acks alone
(:func:`drop_units`), and leaves its name beside the ledger until it is
done: no other run goes on in a folder where rows are gone and their acks
are not.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .resume import (
    _adopt,
    _checksum,
    _drop_marker,
    _lock,
    _outputs,
    _put_back,
    _retry_cut,
    _stop_on_unfinished_drop,
    _UnitsIn,
    release_lock,
)
from .rows import _replace, _retry_held
from .sidecar import Ledger


def drop_units(
    ledger: Ledger,
    *,
    artifact: str | Path,
    name: str,
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

    The stage's files are first put back at the last finished save, as on a
    resume. Each output file is then written again without the rows, and
    last the ledger without the acks, ending in one line that records the
    new sizes. Each goes through a temporary file and a rename.

    In between, rows are gone whose acks are not. So before the first file
    is replaced the call leaves its ``name`` beside the ledger, and removes
    it once the ledger is written. While the name is there a resume stops,
    and so does a call under another name. The same call, made again,
    finishes the job: it alone takes a file as it is that is gone, or
    shorter than recorded and whole to its last row.

    Args:
        ledger: The stage's ledger.
        artifact: The stage's output, as for :func:`resume_state`.
        name: What is started over, in words a later run can be told to
            repeat. The call that finishes this one gives the same name.
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
        continued = _stop_on_unfinished_drop(ledger, unless=name)
        state = ledger.read()
        if state.saved is None:
            _adopt(ledger, state, outputs, units_in)
        else:
            _put_back(ledger, state.saved, outputs, shorter_ok=continued)
        if ledger.exists():
            if not continued:
                _mark_drop(ledger, name)
            for recorded in ledger.saved or {}:
                _rewrite_without(ledger.path.parent / recorded, rows)
            ledger.rewrite(acks, _measured(ledger))
        _retry_held(lambda: _drop_marker(ledger).unlink(missing_ok=True))
    except BaseException:
        release_lock(ledger)    # a call that stops holds nothing
        raise


def _mark_drop(ledger: Ledger, name: str) -> None:
    """Leave ``name`` beside the ledger, synced, as a :func:`drop_units` under way."""
    with _drop_marker(ledger).open("wb") as fh:
        fh.write(name.encode("utf-8") + b"\n")
        fh.flush()
        os.fsync(fh.fileno())


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
