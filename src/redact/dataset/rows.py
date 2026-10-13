"""Dataset rows in one CSV: append, read back, and the unit keys already written."""

import csv
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

# Rows per read of a file that is not read whole.
_CHUNK_ROWS = 50_000

_HELD_ATTEMPTS = 20
_HELD_WAIT_SECONDS = 0.1

_T = TypeVar("_T")


def _retry_held(action: Callable[[], _T]) -> _T:
    """Call ``action``, again for a few seconds while it raises ``PermissionError``.

    On Windows a file that another program holds, such as a virus scanner, a
    sync client or a spreadsheet, cannot be opened for writing, cut or
    removed until that program lets go. The last failure is raised.
    """
    for _ in range(_HELD_ATTEMPTS - 1):
        try:
            return action()
        except PermissionError:
            time.sleep(_HELD_WAIT_SECONDS)
    return action()


def _header(path: Path) -> list[str] | None:
    """The column names of the CSV at ``path``, ``None`` when it has no header yet."""
    if not path.exists():
        return None
    # utf-8-sig: a spreadsheet saves the file with a byte-order mark.
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return next(csv.reader(fh), None)


def append_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[Path, int]:
    """Append ``rows`` to the CSV at ``path`` in one write, synced to disk.

    A file without a header takes the rows' column order and gets the header.
    Otherwise each row is written in the order of the file's header, with an
    empty cell for a column the row lacks. A file that another program holds
    is opened again for a few seconds before any byte is written.

    Returns:
        The size of the file after the append, by its path.

    Raises:
        ValueError: A row has a column the file's header lacks. Nothing is
            written.
    """
    path = Path(path)
    frame = pd.DataFrame(rows)
    header = _header(path)
    if header is not None:
        unknown = [repr(c) for c in frame.columns if c not in header]
        if unknown:
            raise ValueError(
                f"Cannot append to {path}: its header has no column "
                f"{', '.join(unknown)}."
            )
        if list(frame.columns) != header:
            frame = frame.reindex(columns=header)
    # The bytes to_csv writes to a path, line endings included.
    data = frame.to_csv(None, header=header is None, index=False).encode("utf-8")
    with _retry_held(lambda: path.open("ab")) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
        return {path: os.fstat(fh.fileno()).st_size}


def read_rows(path: str | Path) -> pd.DataFrame:
    """Every row of the CSV at ``path``, or an empty frame when there is no file."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def iter_rows(
    path: str | Path, columns: Iterable[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Yield the rows of the CSV at ``path`` as dicts, reading it in chunks.

    ``columns`` limits the read to those columns. No file gives no rows.
    Values are typed chunk by chunk, so a column can come out as ``int`` in
    one chunk and ``float`` in another that has an empty cell.
    """
    path = Path(path)
    if not path.exists():
        return
    usecols = None if columns is None else list(columns)
    with pd.read_csv(path, usecols=usecols, chunksize=_CHUNK_ROWS) as chunks:
        for chunk in chunks:
            yield from chunk.to_dict("records")


def count_rows(path: str | Path) -> int:
    """The number of rows in the CSV at ``path``, 0 when there is no file.

    Reads one column in chunks, never the whole file.
    """
    path = Path(path)
    if not path.exists():
        return 0
    with pd.read_csv(path, usecols=[0], chunksize=_CHUNK_ROWS) as chunks:
        return sum(len(chunk) for chunk in chunks)


def keys_in(
    path: str | Path,
    key_fields: Iterable[str],
    casters: Mapping[str, Callable[[Any], Any]] | None = None,
) -> set:
    """The unit keys of the rows in the CSV at ``path``.

    A key is the value of the one key field, or a tuple of the values of
    several, each passed through its caster. A file that is missing, has no
    rows or lacks a key column gives no keys. A cast that fails raises. The
    file is read in chunks and only the keys are kept.
    """
    path = Path(path)
    if not path.exists():
        return set()
    key_fields = tuple(key_fields)
    casters = casters or {}
    keys: set = set()
    # Every column is parsed: with usecols pandas no longer raises on a row
    # that has more cells than the header.
    with pd.read_csv(path, chunksize=_CHUNK_ROWS) as chunks:
        for chunk in chunks:
            if not set(key_fields) <= set(chunk.columns):
                return set()
            columns = []
            for field in key_fields:
                cast = casters.get(field)
                columns.append([cast(v) if cast else v for v in chunk[field]])
            keys.update(columns[0] if len(key_fields) == 1 else zip(*columns))
    return keys
