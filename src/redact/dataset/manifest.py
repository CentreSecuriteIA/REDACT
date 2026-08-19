"""Shared sidecar run-manifest — the plan written *before* generation starts.

A *manifest* enumerates every unit a run will produce, one JSON object per line,
written to ``<artifact>.manifest.jsonl`` before any compute is spent — so the
intended scope of a run is persisted and inspectable up front (and, for stages
that need it, streamed back in chunks to drive execution).

``Manifest`` is the *plan* half of the modular sidecar system: it shares the JSONL
file mechanics with :class:`~redact.dataset.ledger.Ledger` via
:class:`~redact.dataset.sidecar.JsonlSidecar`, and adds idempotent
overwrite/load on top. Planning is idempotent — the same inputs produce the same
plan — so :meth:`write` overwrites rather than appends (contrast with the ledger,
which is append-only). Rows carry whatever fields the stage needs; the manifest
does not key or interpret them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .sidecar import JsonlSidecar


class Manifest(JsonlSidecar):
    """A sidecar ``*.manifest.jsonl`` run plan (one JSON object per unit)."""

    _SUFFIX = ".manifest.jsonl"

    @classmethod
    def sidecar(cls, artifact_path: str | Path, *, name: str | None = None) -> Manifest:
        """Build a manifest sitting beside ``artifact_path``.

        By default the manifest is ``<artifact-stem>.manifest.jsonl`` next to the
        artifact; pass ``name`` to override just the filename.
        """
        return cls(cls._sidecar_path(artifact_path, name))

    def write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Write the full plan, overwriting any existing manifest (idempotent).

        An empty ``rows`` is a no-op (leaves any existing file untouched).
        """
        self._write(rows, mode="w")

    def load(self) -> list[dict]:
        """Read every planned row back (one dict per line, bad lines skipped)."""
        return list(self._iter_records())
