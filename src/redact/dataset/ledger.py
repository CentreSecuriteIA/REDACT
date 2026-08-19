"""Shared sidecar resume-ledger — one JSON object per completed unit.

Every batched generation / scoring stage in the library records completed units
in a sidecar ``<artifact>.state.jsonl`` file, so a crash loses at most one chunk
and resume never has to trust a large (or hand-edited) output CSV. This module is
the **single implementation** behind that pattern; the constitution, input,
output, paraphrase, jailbreak, conversation, and evaluation pipelines all use it
so the on-disk format and resume semantics never drift between stages.

``Ledger`` is the *state* half of the modular sidecar system: it shares the JSONL
file mechanics (path/mkdir/reset/bad-line-robust read) with :class:`Manifest` via
:class:`~redact.dataset.sidecar.JsonlSidecar`, and adds keyed, append-only
completion tracking on top.

A *unit* is identified by one or more **key fields** (e.g. ``("input_id",
"iteration")``). :meth:`Ledger.completed` returns the set of keys already done —
a scalar when there is a single key field, a tuple otherwise — matching how the
caller builds its pending keys. Records may carry extra fields beyond the key
(e.g. ``status``, ``paraphrase_model``); they are written verbatim and ignored on
read, so a status-carrying ledger and a bare completion ledger share one class.

Casters normalize key-field types on read (e.g. ``{"input_id": str,
"iteration": int}``) so the completed-set matches keys the caller derives from a
DataFrame or a Python range. Malformed lines are skipped, never fatal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .sidecar import JsonlSidecar


class Ledger(JsonlSidecar):
    """A sidecar ``*.state.jsonl`` resume ledger keyed on one or more fields."""

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
        """Build a ledger sitting beside ``artifact_path``.

        By default the ledger is ``<artifact-stem>.state.jsonl`` next to the
        artifact; pass ``name`` to override just the filename (kept in the same
        directory).
        """
        return cls(cls._sidecar_path(artifact_path, name), key_fields, casters)

    # -- keying ---------------------------------------------------------------
    def _key(self, record: Mapping[str, Any]) -> Any:
        vals = tuple(
            self.casters.get(f, _identity)(record[f]) for f in self.key_fields
        )
        return vals[0] if len(self.key_fields) == 1 else vals

    # -- read/write -----------------------------------------------------------
    def completed(self) -> set:
        """Return the set of unit keys already recorded (empty if no file)."""
        done: set = set()
        for rec in self._iter_records():
            try:
                done.add(self._key(rec))
            except (KeyError, TypeError, ValueError):
                continue
        return done

    def record(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append completed-unit records (each must contain the key fields)."""
        self._write(rows, mode="a")


def _identity(x: Any) -> Any:
    return x
