"""Dataset rows in one CSV: append, read back, the unit keys already written, and
one CSV written from the rows of several."""

import csv
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import pandas as pd

# Rows per read of a file that is not read whole.
_CHUNK_ROWS = 50_000

# The cells pandas reads as true when it reads a column of verdicts whole.
_TRUE_CELLS = ("True", "TRUE", "true")

_HELD_ATTEMPTS = 20
_HELD_WAIT_SECONDS = 0.1
_REPLACE_ATTEMPTS = 5
_REPLACE_WAIT_SECONDS = 0.2

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


def _replace(source: Path, target: Path) -> None:
    """Rename ``source`` onto ``target``, replacing it.

    On Windows the rename raises ``PermissionError`` while another process,
    such as a sync client, has ``target`` open. It is tried again a few times
    and the last failure is raised.
    """
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_WAIT_SECONDS)


def _has_bytes(path: Path) -> bool:
    """Whether ``path`` is a file with a byte in it.

    A save killed between opening a new file and writing to it leaves 0
    bytes; a reader takes that as no file.
    """
    return path.is_file() and path.stat().st_size > 0


def _header(path: Path) -> list[str] | None:
    """The column names of the CSV at ``path``, ``None`` when it has no header yet."""
    if not path.exists():
        return None
    # utf-8-sig: a spreadsheet saves the file with a byte-order mark.
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return next(csv.reader(fh), None)


def _header_for(path: Path, columns: Iterable[str]) -> list[str] | None:
    """The header of the CSV at ``path``, once it is seen to hold ``columns``.

    Raises:
        ValueError: The header lacks one of the columns.
    """
    header = _header(path)
    if header is not None:
        unknown = [repr(c) for c in columns if c not in header]
        if unknown:
            raise ValueError(
                f"Cannot append to {path}: its header has no column "
                f"{', '.join(unknown)}."
            )
    return header


def check_columns(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Raise as :func:`append_rows` would on rows the CSV at ``path`` cannot take.

    For a save over several files: each file's rows are checked before the
    first file is written.

    Raises:
        ValueError: A row has a column the file's header lacks.
    """
    _header_for(Path(path), dict.fromkeys(column for row in rows for column in row))


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
    header = _header_for(path, frame.columns)
    if header is not None and list(frame.columns) != header:
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
    return pd.read_csv(path) if _has_bytes(path) else pd.DataFrame()


def iter_rows(
    path: str | Path, columns: Iterable[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Yield the rows of the CSV at ``path`` as dicts, reading it in chunks.

    ``columns`` limits the read to those columns. No file gives no rows.
    Values are typed chunk by chunk, so a column can come out as ``int`` in
    one chunk and ``float`` in another that has an empty cell.
    """
    path = Path(path)
    if not _has_bytes(path):
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
    if not _has_bytes(path):
        return 0
    with pd.read_csv(path, usecols=[0], chunksize=_CHUNK_ROWS) as chunks:
        return sum(len(chunk) for chunk in chunks)


def has_rows(path: str | Path) -> bool:
    """Whether the CSV at ``path`` holds a row below its header, ``False`` when
    there is no file."""
    path = Path(path)
    return _has_bytes(path) and not pd.read_csv(path, nrows=1, dtype=str).empty


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
    if not _has_bytes(path):
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


class Part(NamedTuple):
    """Rows that :func:`write_concat` writes.

    Attributes:
        source: A CSV file, or a frame already in memory.
        dataset_type: The value of the rows' ``dataset_type`` column. ``None``
            adds no such column.
        accepted_only: Copy only the rows of a file whose ``accepted`` cell
            reads true. A file without the column is copied whole.
        label: What the log calls the part.
    """

    source: Path | pd.DataFrame
    dataset_type: str | None = None
    accepted_only: bool = False
    label: str = ""


def _part_columns(part: Part) -> list[str]:
    """The columns a part's rows are written with."""
    if isinstance(part.source, pd.DataFrame):
        columns = list(part.source.columns)
    elif _has_bytes(part.source):
        columns = list(pd.read_csv(part.source, nrows=0).columns)
    else:
        columns = []
    if part.dataset_type is not None and "dataset_type" not in columns:
        columns.append("dataset_type")
    return columns


def _part_chunks(part: Part) -> Iterator[pd.DataFrame]:
    """Yield a part's rows, a file's ``_CHUNK_ROWS`` at a time.

    The cells of a file are read as text, so a whole number is written as
    it stands, also where the rows of another part leave its column empty.
    A cell pandas reads as a missing value (``nan``, ``N/A``, ...) is
    written empty, as it is when the file is read whole.
    """
    tag = {} if part.dataset_type is None else {"dataset_type": part.dataset_type}
    if isinstance(part.source, pd.DataFrame):
        frame = part.source
        # A column of whole numbers that a merge left gaps in is float. As
        # integers it prints 1 and not 1.0, as the cells of a file do.
        whole = [name for name in frame.columns
                 if pd.api.types.is_float_dtype(frame[name])
                 and (frame[name].dropna() % 1 == 0).all()]
        yield frame.astype(dict.fromkeys(whole, "Int64")).assign(**tag)
        return
    if not _has_bytes(part.source):
        return
    with pd.read_csv(part.source, dtype=str, chunksize=_CHUNK_ROWS) as chunks:
        for chunk in chunks:
            if part.accepted_only and "accepted" in chunk.columns:
                chunk = chunk[chunk["accepted"].isin(_TRUE_CELLS)]
            yield chunk.assign(**tag)


def write_concat(
    target: str | Path, parts: Sequence[Part], *, keep_empty: bool = True,
) -> list[int]:
    """Write the rows of ``parts``, one part after another, as the CSV at ``target``.

    The columns are those of every part in the order they first appear, and
    a row is empty in the columns its part lacks: the file ``pd.concat`` of
    the parts would give, without the parts in memory. The rows go to a
    temporary file beside ``target`` that then replaces it, so ``target``
    holds its old rows or all the new ones.

    Args:
        keep_empty: False leaves ``target`` as it is when no part has a row.

    Returns:
        The number of rows written of each part.
    """
    target = Path(target)
    columns = list(dict.fromkeys(c for part in parts for c in _part_columns(part)))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    written = []
    try:
        # newline="": to_csv ends each row itself, as it does writing to a path.
        with temporary.open("w", encoding="utf-8", newline="") as fh:
            pd.DataFrame(columns=columns).to_csv(fh, index=False)
            for part in parts:
                rows = 0
                for chunk in _part_chunks(part):
                    chunk = chunk.reindex(columns=columns, fill_value="")
                    chunk.to_csv(fh, header=False, index=False)
                    rows += len(chunk)
                written.append(rows)
        if keep_empty or any(written):
            _replace(temporary, target)
    finally:
        # Still there only when it did not replace the target.
        temporary.unlink(missing_ok=True)
    return written
