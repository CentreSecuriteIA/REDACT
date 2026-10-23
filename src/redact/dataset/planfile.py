"""The plan file and the tables beside a stage's manifest.

The plan file ``<stem>.plan.json`` records what a stage's plan was made
with, so a resume can tell when it would plan otherwise. The coverage table
``<stem>.coverage.csv`` counts what the plan holds; the jailbreak stage's
variants table ``<stem>.variants.csv`` counts the template variants its
rows drew.
"""

import json
from pathlib import Path

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


def plan_path(manifest_path) -> Path:
    """The plan file beside the manifest at ``manifest_path``."""
    return _beside(manifest_path, ".plan.json")


def plan_record(**made_with) -> dict:
    """``made_with`` as JSON reads it back, so it compares equal to the
    record a plan file holds."""
    return json.loads(json.dumps(made_with))


def write_plan(manifest_path, record: dict) -> Path:
    """Write ``record`` (:func:`plan_record`) as the plan file beside the
    manifest at ``manifest_path``; returns its path."""
    path = plan_path(manifest_path)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return path


def read_plan(manifest_path) -> dict | None:
    """The record the plan file beside the manifest at ``manifest_path``
    holds; ``None`` when there is no plan file."""
    path = plan_path(manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def changed_plan(manifest_path, current: dict) -> list[str] | None:
    """The keys of ``current`` (:func:`plan_record`) whose values differ from
    those the plan at ``manifest_path`` was made with; ``None`` when it has no
    plan file to compare with."""
    made_with = read_plan(manifest_path)
    if made_with is None:
        return None
    return [key for key, value in current.items() if made_with.get(key) != value]
