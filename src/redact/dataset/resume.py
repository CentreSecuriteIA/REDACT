"""The two resume invariants every batched stage must follow identically.

Both are easy to get wrong in ways no test catches until a real crash, and
both were previously restated in four places:

1. **A fresh run clears the artifact *and* its ledger.** Clearing only the
   artifact leaves stale acks behind, so the next run skips every unit and
   writes nothing — silently producing an empty output that looks like a
   successful no-op.
2. **Append before acking.** A unit is recorded only once its rows are on
   disk, so a crash between the two re-runs that unit rather than losing it.
   Reversed, a crash drops work permanently while claiming it was done.

This is deliberately **not** a sweep driver. The loops these live inside
differ in ways that matter — per-unit acking in constitution vs per-chunk
elsewhere, DataFrame masks vs list comprehensions vs tuple-key filters, and
rows that are not 1:1 with units (paraphrase drops deduped rows while still
acking them). Extracting the whole loop would mean a function that is mostly
callbacks. These two steps are the parts that are genuinely identical, so
they are the parts that are shared.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .ledger import Ledger


def resume_state(
    ledger: Ledger,
    *,
    resume: bool,
    artifact: str | Path | None = None,
    extra: Iterable[Any] | None = None,
) -> set[Any]:
    """Completed-unit keys for this run, clearing both sources when not resuming.

    Args:
        ledger: The stage's sidecar ledger.
        resume: False means start over — the artifact and the ledger are both
            removed, and the returned set is empty.
        artifact: Output file to delete on a fresh run. ``None`` when the
            caller removes it itself (several stages have more than one).
        extra: Additional completed keys to union in — the back-compat path
            for runs created before this stage had a ledger (jailbreak reads
            them out of the output CSV). Ignored when ``resume`` is False.

    Returns:
        The set of unit keys to skip. Shape matches ``Ledger.completed()``:
        a scalar per unit for a single key field, a tuple for several.
    """
    if not resume:
        if artifact is not None:
            path = Path(artifact)
            if path.exists():
                path.unlink()
        ledger.reset()
        return set()

    done = ledger.completed()
    return done | set(extra) if extra else done


def commit(
    rows: Sequence[Mapping[str, Any]],
    *,
    append: Callable[[Sequence[Mapping[str, Any]]], None],
    ledger: Ledger,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    """Write rows, then ack their units — never the other way round.

    Args:
        rows: Output rows for this unit or chunk. Empty is allowed and skips
            the append: a unit can legitimately produce no rows and still need
            acking, or it would be retried forever.
        append: Writes the rows. Whatever the stage does — append to one CSV,
            or fan out per category. Raising here means nothing is acked, so
            the units re-run.
        ledger: The stage's ledger.
        records: Explicit ack records, when they are not derivable from
            ``rows`` — paraphrase acks deduped units that produced no row, and
            carries a ``status`` field. Defaults to the ledger's key fields
            projected out of ``rows``, which is what every other stage builds
            by hand.
    """
    if rows:
        append(rows)
    if records is None:
        records = [{k: r[k] for k in ledger.key_fields} for r in rows]
    ledger.record(records)
