"""The two JSONL files beside a stage's artifact: the ledger of finished units
and the manifest of planned ones.

Both hold one JSON object per line and share :class:`JsonlSidecar`: the path
beside the artifact, the reset, and a read that skips a line it cannot parse.
A stage writes the manifest whole before it generates, and appends a unit to
the ledger only after that unit's rows are on disk.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class JsonlSidecar:
    """A sidecar JSONL file next to an artifact. Subclasses set ``_SUFFIX``."""

    _SUFFIX = ".jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def _sidecar_path(cls, artifact_path: str | Path, name: str | None = None) -> Path:
        """Resolve the sidecar path beside ``artifact_path``.

        Default: ``<artifact-stem><_SUFFIX>`` in the artifact's directory. Pass
        ``name`` to override just the filename (kept in the same directory).
        """
        artifact = Path(artifact_path)
        if name:
            return artifact.with_name(name)
        return artifact.with_name(artifact.stem + cls._SUFFIX)

    def exists(self) -> bool:
        return self.path.exists()

    def reset(self) -> None:
        """Delete the file if it exists (used by ``resume=False``)."""
        if self.path.exists():
            self.path.unlink()

    def _iter_records(self) -> Iterator[dict]:
        """Yield one parsed dict per line, skipping blank/corrupt lines."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _write(self, rows: Iterable[Mapping[str, Any]], *, mode: str) -> None:
        """Write ``rows`` as JSON lines.

        ``mode`` is ``"a"`` (append, for the ledger) or ``"w"`` (overwrite, for
        the manifest). An empty ``rows`` is a no-op — no file is created or
        truncated — so both ``.record([])`` and ``.write([])`` leave the disk
        untouched.
        """
        rows = list(rows)
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open(mode, encoding="utf-8") as fh:
            for r in rows:
                # ensure_ascii=False keeps unicode readable in the sidecar file;
                # the file is always opened as utf-8 and read back via json.loads.
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


class Ledger(JsonlSidecar):
    """The finished units of a stage, one record each, in ``*.state.jsonl``.

    A unit's key is the value of its one key field, or a tuple of the values
    of several. ``casters`` convert key values on read, so they compare equal
    to the keys the stage builds. A record may carry fields beyond the key;
    they are written as given and ignored on read.
    """

    _SUFFIX = ".state.jsonl"

    def __init__(
        self,
        path: str | Path,
        key_fields: Iterable[str] = ("id",),
        casters: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        super().__init__(path)
        self.key_fields: tuple[str, ...] = tuple(key_fields)
        if not self.key_fields:
            raise ValueError("Ledger needs at least one key field.")
        self.casters: dict[str, Callable[[Any], Any]] = dict(casters or {})

    @classmethod
    def sidecar(
        cls,
        artifact_path: str | Path,
        key_fields: Iterable[str] = ("id",),
        casters: Mapping[str, Callable[[Any], Any]] | None = None,
        *,
        name: str | None = None,
    ) -> Ledger:
        """The ledger beside ``artifact_path``; ``name`` replaces its file name."""
        return cls(cls._sidecar_path(artifact_path, name), key_fields, casters)

    def _key(self, record: Mapping[str, Any]) -> Any:
        """The record's key: one cast value, or a tuple of them."""
        vals = tuple(
            self.casters.get(f, _identity)(record[f]) for f in self.key_fields
        )
        return vals[0] if len(self.key_fields) == 1 else vals

    def completed(self) -> set:
        """Return the set of unit keys already recorded (empty if no file)."""
        done: set = set()
        for rec in self._iter_records():
            try:
                done.add(self._key(rec))
            except (KeyError, TypeError, ValueError):
                continue
        return done

    def record(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Append completed-unit records (each must contain the key fields)."""
        self._write(records, mode="a")


def _identity(x: Any) -> Any:
    return x


class Manifest(JsonlSidecar):
    """The planned units of a run, one row each, in ``*.manifest.jsonl``.

    Rows carry whatever fields the stage needs; the manifest does not key or
    interpret them.
    """

    _SUFFIX = ".manifest.jsonl"

    @classmethod
    def sidecar(cls, artifact_path: str | Path, *, name: str | None = None) -> Manifest:
        """The manifest beside ``artifact_path``; ``name`` replaces its file name."""
        return cls(cls._sidecar_path(artifact_path, name))

    def write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Write the full plan, overwriting any existing manifest (idempotent).

        An empty ``rows`` is a no-op (leaves any existing file untouched).
        """
        self._write(rows, mode="w")

    def load(self) -> list[dict]:
        """Read every planned row back (one dict per line, bad lines skipped)."""
        return list(self._iter_records())
