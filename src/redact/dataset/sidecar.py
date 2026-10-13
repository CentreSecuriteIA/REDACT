"""The two JSONL files beside a stage's artifact: the ledger of finished units
and the manifest of planned ones.

Both hold one JSON object per line and share :class:`JsonlSidecar`: the path
beside the artifact, the reset, and a read that skips a line it cannot parse.
A stage writes the manifest whole before it generates, and appends a unit to
the ledger only after that unit's rows are on disk.

The last ledger line of a save carries ``_saved``: the size and a checksum of
each file of the stage after that save. A save whose ``_saved`` line is not
on disk did not finish, and its units are not finished. Such a ledger opens
with an empty ``_saved``, written before the first rows::

    {"_saved": {}}
    {"input_id": "3f2a", "iteration": 0}
    {"input_id": "91bc", "iteration": 1, "_saved": {"out.csv": [5284, "5d41402a"]}}

A stage that drops repeated texts puts the content ids of a unit's rows on
its record, as ``ids``, and gets those of every finished unit back as one set.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from itertools import chain, islice
from pathlib import Path
from typing import Any, NamedTuple

from .rows import _retry_held

_REPLACE_ATTEMPTS = 5
_REPLACE_WAIT_SECONDS = 0.2

# Stands for what a ledger's file records until the file is read.
_UNREAD: Any = object()


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

    def _write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Replace the file with ``rows`` as JSON lines.

        ``rows`` are taken one at a time into a temporary file beside the
        target, which is then renamed onto the target, so the file holds its
        old rows or all the new ones. An empty ``rows`` is a no-op: no file
        is created or truncated.
        """
        rows = iter(rows)
        first = list(islice(rows, 1))
        if not first:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False keeps unicode readable in the sidecar file;
        # the file is always opened as utf-8 and read back via json.loads.
        lines = (json.dumps(r, ensure_ascii=False) + "\n" for r in chain(first, rows))
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as fh:
                fh.writelines(lines)
            _replace(temporary, self.path)
        finally:
            # Still there only when the write or the rename failed.
            temporary.unlink(missing_ok=True)


class LedgerState(NamedTuple):
    """What a ledger file holds, as :meth:`Ledger.read` gives it.

    Attributes:
        done: The keys of the finished units.
        saved: Each file's ``[size, checksum]`` on the last ``_saved`` line,
            by the file's name. ``None`` when no line carries one.
        end: The byte offset where the lines that count end.
    """

    done: set
    saved: dict[str, Any] | None
    end: int


class Ledger(JsonlSidecar):
    """The finished units of a stage, one record each, in ``*.state.jsonl``.

    A unit's key is the value of its one key field, or a tuple of the values
    of several. ``casters`` convert key values on read, so they compare equal
    to the keys the stage builds. A record may carry fields beyond the key;
    they are written as given and ignored on read.

    The file is read and written as bytes, each line ending in ``\\n`` on
    every platform, so an offset into it means the same wherever it is read.
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
        self._saved = self._ids = _UNREAD

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

    def read(self) -> LedgerState:
        """The finished units and what the last save recorded, in one pass.

        A line counts only if it ends with a newline: a write that was cut
        off is not a line. Lines after the last one that carries ``_saved``
        belong to a save that did not finish, so their units are not
        finished and their ``ids`` do not count. A ledger with no ``_saved``
        line counts every line.
        """
        done: set = set()
        ids: set = set()
        pending: set = set()        # units since the last ``_saved`` line
        pending_ids: set = set()    # and the ids on their lines
        saved, end, offset = None, 0, 0
        if self.path.exists():
            with self.path.open("rb") as fh:
                for line in fh:
                    if not line.endswith(b"\n"):
                        break
                    offset += len(line)
                    record = _parse(line)
                    if record is None:
                        continue
                    # A line without the key fields holds no unit.
                    with suppress(KeyError, TypeError, ValueError):
                        pending.add(self._key(record))
                    with suppress(TypeError):   # not a list of ids
                        pending_ids.update(record.get("ids") or ())
                    if "_saved" in record:
                        saved, end = record["_saved"], offset
                        done |= pending
                        ids |= pending_ids
                        pending.clear()
                        pending_ids.clear()
        if saved is None:
            done, ids, end = pending, pending_ids, offset
        self._saved, self._ids = saved, ids
        return LedgerState(done, saved, end)

    def completed(self) -> set:
        """The keys of the finished units, as :meth:`read` counts them."""
        return self.read().done

    @property
    def saved(self) -> dict[str, Any] | None:
        """Each file's ``[size, checksum]`` as the last save recorded them.

        ``None`` for a ledger with no ``_saved`` line. It is what the last
        :meth:`read` or :meth:`record` saw; the file is read when neither
        has run.
        """
        if self._saved is _UNREAD:
            self.read()
        return self._saved

    @property
    def ids(self) -> set[str]:
        """The content ids on the records of the finished units, as one set.

        Kept as :attr:`saved` is: what the last :meth:`read` found and what
        :meth:`record` added since. It is the ledger's own set; copy it
        before adding to it.
        """
        if self._ids is _UNREAD:
            self.read()
        return self._ids

    def record(
        self,
        records: Iterable[Mapping[str, Any]],
        saved: Mapping[str, Any] | None = None,
    ) -> None:
        """Append the records of finished units in one write, synced to disk.

        Each record must contain the key fields. ``saved`` is each file's
        ``[size, checksum]`` after the save these units belong to. It goes
        on a copy of the last record as ``_saved``, or on a line of its own
        when there are no records, and marks the save as finished. No
        records and no ``saved`` write nothing.

        Raises:
            ValueError: ``saved`` is missing and the ledger holds a
                ``_saved`` line: the records would come after it, where no
                reader counts them.
        """
        # A list before the file is opened: records that raise append nothing.
        records = list(records)
        if saved is not None:
            records[-1:] = [{**(records[-1] if records else {}), "_saved": dict(saved)}]
        if not records:
            return
        if saved is None and self.saved is not None:
            raise ValueError(
                f"{self.path} records the size of each file with every save "
                "(_saved); records without it would not count."
            )
        # ensure_ascii=False keeps unicode readable; the lines are utf-8.
        self._append(b"".join(
            json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in records
        ))
        if saved is not None:
            self._saved = dict(saved)
        if self._ids is not _UNREAD:
            for record in records:
                self._ids.update(record.get("ids") or ())

    def rewrite(
        self,
        drop: Callable[[Mapping[str, Any]], bool],
        saved: Mapping[str, Any],
    ) -> None:
        """Replace the file with its finished records, less those ``drop`` picks.

        The lines that stay are the bytes they were, and one last line
        records ``saved``. A line that only recorded sizes is left out, so
        doing it again leaves the same file. The new file is written beside
        the old one, synced, and renamed onto it, so the ledger is the old
        one or the new one.
        """
        end = self.read().end
        ids: set = set()
        last = json.dumps({"_saved": dict(saved)}, ensure_ascii=False)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with self.path.open("rb") as old, temporary.open("wb") as new:
                for line in old:
                    end -= len(line)
                    if end < 0:     # what follows belongs to no finished save
                        break
                    record = _parse(line) or {}
                    if drop(record) or set(record) == {"_saved"}:
                        continue
                    new.write(line)
                    with suppress(TypeError):
                        ids.update(record.get("ids") or ())
                new.write(last.encode("utf-8") + b"\n")
                new.flush()
                os.fsync(new.fileno())
            _replace(temporary, self.path)
        finally:
            # Still there only when the write or the rename failed.
            temporary.unlink(missing_ok=True)
        self._saved, self._ids = dict(saved), ids

    def _append(self, data: bytes) -> None:
        """Add ``data`` to the end of the file in one write, synced to disk.

        The write starts on a new line: put onto a last line that lost its
        newline, its first record would not parse. A file that another
        program holds is opened again for a few seconds.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _retry_held(lambda: self.path.open("a+b")) as fh:
            if fh.seek(0, os.SEEK_END):
                fh.seek(-1, os.SEEK_END)
                if fh.read() != b"\n":
                    data = b"\n" + data
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    def reset(self) -> None:
        """Delete the file if it exists, and forget what it recorded."""
        super().reset()
        self._saved, self._ids = None, set()


def _identity(x: Any) -> Any:
    return x


def _parse(line: bytes) -> dict | None:
    """The JSON object on a ledger line, ``None`` when the line holds none."""
    try:
        record = json.loads(line)
    except ValueError:    # not JSON, or not utf-8
        return None
    return record if isinstance(record, dict) else None


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
        self._write(rows)

    def __iter__(self) -> Iterator[dict]:
        """Yield the planned rows one at a time (bad lines skipped)."""
        return self._iter_records()

    def load(self) -> list[dict]:
        """Read every planned row back (one dict per line, bad lines skipped)."""
        return list(self._iter_records())
