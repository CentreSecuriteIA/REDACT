"""The plan file and the coverage table beside a manifest."""

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
