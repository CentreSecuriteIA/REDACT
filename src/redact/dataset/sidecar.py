"""Shared foundation for sidecar JSONL files (resume ledger + run manifest).

A resume **ledger** and a run **manifest** are the same primitive — a JSONL file
sitting beside a stage's output artifact, one JSON object per line, created on
write and robust to a half-written/corrupt line on read. This base owns that
primitive so the two concrete types don't duplicate it; each adds only its own
semantics:

- :class:`redact.dataset.ledger.Ledger` — resume *state*: keyed, append-only,
  ``.completed()`` returns the set of finished unit keys.
- :class:`redact.dataset.manifest.Manifest` — the plan written *ahead* of
  generation: an idempotent overwrite, ``.load()`` returns every planned row.

Subclasses set ``_SUFFIX`` (``.state.jsonl`` / ``.manifest.jsonl``); everything
about paths, directory creation, deletion, and bad-line-robust reading lives here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
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
