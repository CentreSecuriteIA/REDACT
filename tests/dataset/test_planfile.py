"""The plan file and the coverage table beside a manifest."""

import os

import pytest

from redact.dataset import rows
from redact.dataset.planfile import (
    changed_plan,
    coverage_path,
    plan_path,
    plan_record,
    read_plan,
    write_plan,
)


def test_the_files_sit_beside_the_manifest_under_its_stem(tmp_path):
    manifest = tmp_path / "constitution_inputs.manifest.jsonl"
    assert plan_path(manifest) == tmp_path / "constitution_inputs.plan.json"
    assert coverage_path(manifest) == tmp_path / "constitution_inputs.coverage.csv"
    assert plan_path(tmp_path / "x.jsonl") == tmp_path / "x.plan.json"


def test_a_record_round_trips_through_its_file(tmp_path):
    manifest = tmp_path / "x.manifest.jsonl"
    record = plan_record(seed=42, samples_per_entry=3,
                         axes={"length": {"short": 1, "long": 0.5}}, rounds=(1, 2))
    assert record["rounds"] == [1, 2]       # as JSON reads it back
    write_plan(manifest, record)
    assert read_plan(manifest) == record
    assert changed_plan(manifest, record) == []


def test_a_changed_key_is_named(tmp_path):
    manifest = tmp_path / "x.manifest.jsonl"
    write_plan(manifest, plan_record(seed=42, samples_per_entry=3, axes=None))
    current = plan_record(seed=42, samples_per_entry=4, axes={"length": {}})
    assert changed_plan(manifest, current) == ["samples_per_entry", "axes"]


def test_without_a_plan_file_nothing_is_compared(tmp_path):
    assert read_plan(tmp_path / "x.manifest.jsonl") is None
    assert changed_plan(tmp_path / "x.manifest.jsonl", plan_record(seed=1)) is None


@pytest.mark.parametrize("text, found", [
    ("", "does not parse"),                     # created, killed before a byte
    ('{"seed": 42, "samples_', "does not parse"),   # cut off mid-write
    ("[1, 2]", "holds a list"),
])
def test_a_plan_file_without_a_record_raises_naming_it_and_the_remedies(
        tmp_path, text, found):
    manifest = tmp_path / "x.manifest.jsonl"
    plan_path(manifest).write_text(text, encoding="utf-8")
    for read in (lambda: read_plan(manifest),
                 lambda: changed_plan(manifest, plan_record(seed=1))):
        with pytest.raises(ValueError, match="resume=False") as raised:
            read()
        assert str(plan_path(manifest)) in str(raised.value)
        assert found in str(raised.value) and "Delete it" in str(raised.value)


def test_the_record_is_synced_before_it_replaces_the_file(tmp_path, monkeypatch):
    """After a power loss the file holds the old record or the new one."""
    manifest = tmp_path / "x.manifest.jsonl"
    write_plan(manifest, plan_record(seed=1))
    synced = []
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(
        (os.fstat(fd).st_size, read_plan(manifest))))
    write_plan(manifest, plan_record(seed=2))
    assert synced == [(plan_path(manifest).stat().st_size, {"seed": 1})]
    assert read_plan(manifest) == {"seed": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.plan.json"]


def test_a_write_that_fails_keeps_the_old_record(tmp_path, monkeypatch):
    manifest = tmp_path / "x.manifest.jsonl"
    write_plan(manifest, plan_record(seed=1))

    def refuse(source, target):
        raise PermissionError("held by another process")

    monkeypatch.setattr(rows.os, "replace", refuse)
    monkeypatch.setattr(rows.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        write_plan(manifest, plan_record(seed=2))
    assert read_plan(manifest) == {"seed": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.plan.json"]
