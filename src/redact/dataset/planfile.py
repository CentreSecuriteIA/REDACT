"""The plan file and the tables beside a stage's manifest.

The plan file ``<stem>.plan.json`` records what a stage's plan was made
with, so a resume can tell when it would plan otherwise. The coverage table
``<stem>.coverage.csv`` counts what the plan holds; the jailbreak stage's
variants table ``<stem>.variants.csv`` counts the template variants its
rows drew, and its scenarios table ``<stem>.scenarios.csv`` holds the scene
each cognitive or persona technique built on, when the stage records them.
"""

import json
import os
from pathlib import Path

from .rows import _write_atomic

_MANIFEST = ".manifest.jsonl"


def _beside(manifest_path, suffix: str) -> Path:
    """The file ``<stem><suffix>`` beside the manifest at ``manifest_path``."""
    manifest_path = Path(manifest_path)
    name = manifest_path.name
    stem = name[:-len(_MANIFEST)] if name.endswith(_MANIFEST) else manifest_path.stem
    return manifest_path.with_name(f"{stem}{suffix}")


def coverage_path(manifest_path) -> Path:
    """The coverage CSV beside the manifest at ``manifest_path``."""
    return _beside(manifest_path, ".coverage.csv")


def variants_path(manifest_path) -> Path:
    """The variants CSV beside the manifest at ``manifest_path``."""
    return _beside(manifest_path, ".variants.csv")


def scenarios_path(manifest_path) -> Path:
    """The scenarios CSV beside the manifest at ``manifest_path``."""
    return _beside(manifest_path, ".scenarios.csv")


def plan_path(manifest_path) -> Path:
    """The plan file beside the manifest at ``manifest_path``."""
    return _beside(manifest_path, ".plan.json")


def plan_record(**made_with) -> dict:
    """``made_with`` as JSON reads it back, so it compares equal to the
    record a plan file holds."""
    return json.loads(json.dumps(made_with))


def write_plan(manifest_path, record: dict) -> Path:
    """Write ``record`` (:func:`plan_record`) as the plan file beside the
    manifest at ``manifest_path``; returns its path.

    The file goes through ``<name>.tmp``, synced, and a rename, so it holds
    the old record or the new one. Its lines end as a text write ends them.
    """
    path = plan_path(manifest_path)
    text = json.dumps(record, indent=1) + "\n"
    _write_atomic(path, [text.replace("\n", os.linesep).encode("utf-8")])
    return path


def read_plan(manifest_path) -> dict | None:
    """The record the plan file beside the manifest at ``manifest_path``
    holds; ``None`` when there is no plan file.

    Raises:
        ValueError: The file holds no record: it is not JSON (cut off as an
            earlier version wrote it, or edited), or not an object. The
            message names the file and the remedies.
    """
    path = plan_path(manifest_path)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:     # not JSON, or not utf-8
        found = f"it does not parse: {error}"
    else:
        if isinstance(record, dict):
            return record
        found = f"it holds a {type(record).__name__}, not an object"
    raise ValueError(
        f"The plan file {path} holds no plan record ({found}). Delete it to "
        "resume without comparing this run's settings with the plan's (a "
        "WARNING says so), or run with resume=False to start the stage over.")


def changed_plan(manifest_path, current: dict) -> list[str] | None:
    """The keys of ``current`` (:func:`plan_record`) whose values differ from
    those the plan at ``manifest_path`` was made with; ``None`` when it has no
    plan file to compare with."""
    made_with = read_plan(manifest_path)
    if made_with is None:
        return None
    return [key for key, value in current.items() if made_with.get(key) != value]
