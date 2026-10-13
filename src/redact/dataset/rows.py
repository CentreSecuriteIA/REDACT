"""Dataset rows in one CSV: append, read back, and the unit keys already written."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def append_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append ``rows`` to the CSV at ``path``; the first append writes the header."""
    path = Path(path)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def read_rows(path: str | Path) -> pd.DataFrame:
    """Every row of the CSV at ``path``, or an empty frame when there is no file."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def keys_in(
    path: str | Path,
    key_fields: Iterable[str],
    casters: Mapping[str, Callable[[Any], Any]] | None = None,
) -> set:
    """The unit keys of the rows in the CSV at ``path``.

    A key is the value of the one key field, or a tuple of the values of
    several, each passed through its caster. A file that is missing, has no
    rows or lacks a key column gives no keys. A cast that fails raises.
    """
    path = Path(path)
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    key_fields = tuple(key_fields)
    if frame.empty or not set(key_fields) <= set(frame.columns):
        return set()
    casters = casters or {}
    columns = []
    for field in key_fields:
        cast = casters.get(field)
        columns.append([cast(v) if cast else v for v in frame[field]])
    return set(columns[0]) if len(key_fields) == 1 else set(zip(*columns))
