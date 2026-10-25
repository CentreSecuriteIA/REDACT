"""The scenarios table of a jailbreak kind: the scene each cognitive or persona
technique wrote its prompt on, beside the rows it made.

A ``cognitive`` or ``persona`` technique has a model write a scenario, checks
it, then has the model build the prompt on it; its info names the category
only (``category=<key>``), and the row holds the built prompt, not the scene.
With the stage's ``scenarios`` keyword on, each unit collects the scene its
technique built on (:func:`redact.jailbreak.chain.run_chain`'s
``scenarios``), and the stage appends it to ``<stem>.scenarios.csv`` beside
the kind's manifest in the same save as the unit's row, before its ack
(:func:`rows_of`), so a model-written scenario can be read without growing
the row's ``technique_info``.

The table is not one of the stage's checked files: a save that stops after
its scenario rows and before its acks leaves rows of units that run again.
When the units are done, :func:`write_scenarios` keeps only the rows whose
``(input_id, iteration, sample_id)`` the kind's output holds, the last of
each, so the table matches the output whatever stopped before. A fresh run
(``resume=False``) removes it; a resumed one keeps the rows of earlier runs.
"""

from pathlib import Path

import pandas as pd

from redact.dataset import Part, has_rows, write_concat

#: The columns of the scenarios table.
COLUMNS = ["input_id", "iteration", "sample_id", "technique", "scenario"]
#: What identifies a row of the kind's output.
_KEY = ["input_id", "iteration", "sample_id"]
#: The rows read at a time, as the dataset's readers do.
_CHUNK_ROWS = 50_000


def rows_of(unit: dict, sample_id: str) -> list[dict]:
    """The scenarios table's rows of a finished unit whose output row has
    ``sample_id``: one per ``(technique, scenario)`` its chain recorded
    (``unit["scenarios"]``, absent or ``None`` when the stage records
    none)."""
    return [{"input_id": str(unit["id"]), "iteration": int(unit["iteration"]),
             "sample_id": sample_id, "technique": technique, "scenario": scene}
            for technique, scene in unit.get("scenarios") or ()]


def write_scenarios(out_path, path) -> Path | None:
    """Keep in the scenarios table at ``path`` the rows whose output row the
    kind's CSV at ``out_path`` holds (by ``input_id``, ``iteration`` and
    ``sample_id``), the last of rows that repeat one with its technique, in
    their order. The table is written again
    (:func:`redact.dataset.write_concat`, through ``<path>.tmp``) only when
    a row goes. Returns its path, or None when there is no table.
    """
    path = Path(path)
    if not path.exists():
        return None
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    keys = set(zip(*(table[column] for column in _KEY), strict=True))
    held: set = set()
    if keys and has_rows(out_path):
        with pd.read_csv(out_path, usecols=_KEY, dtype=str, keep_default_na=False,
                         chunksize=_CHUNK_ROWS) as chunks:
            for chunk in chunks:
                held |= keys & set(zip(*(chunk[c] for c in _KEY), strict=True))
    kept = table[[key in held for key in zip(*(table[c] for c in _KEY), strict=True)]]
    kept = kept.drop_duplicates([*_KEY, "technique"], keep="last")
    if len(kept) < len(table):
        write_concat(path, [Part(kept[COLUMNS])])
    return path
