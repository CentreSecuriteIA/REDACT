"""Tests for the shared sidecar run-manifest (dataset/manifest.py).

Mirrors tests/test_ledger.py: round-trip write/load, idempotent overwrite,
bad-line robustness, the sidecar-path constructor (default + override), the
empty-write no-op, reset, and the empty/missing cases. Also checks the shared
JsonlSidecar base so Ledger and Manifest agree on file conventions.
"""

import json

from redact.dataset import JsonlSidecar, Ledger, Manifest


def test_write_then_load_roundtrip(tmp_path):
    man = Manifest(tmp_path / "run.manifest.jsonl")
    assert man.load() == []                       # missing file -> empty
    rows = [{"input_id": "a", "iteration": 0}, {"input_id": "a", "iteration": 1}]
    man.write(rows)
    assert man.load() == rows


def test_write_overwrites_not_appends(tmp_path):
    man = Manifest(tmp_path / "run.manifest.jsonl")
    man.write([{"id": "1"}, {"id": "2"}])
    man.write([{"id": "3"}])                       # planning is idempotent -> replace
    assert man.load() == [{"id": "3"}]


def test_arbitrary_fields_preserved(tmp_path):
    man = Manifest(tmp_path / "p.manifest.jsonl")
    row = {"sample_id": "z", "iteration": 0, "combination": ["to_rot13"],
           "settings": {"seed": 42}, "status": "planned"}
    man.write([row])
    assert man.load() == [row]


def test_bad_lines_are_skipped(tmp_path):
    p = tmp_path / "b.manifest.jsonl"
    p.write_text(
        json.dumps({"input_id": "good1"}) + "\n"
        + "not json at all\n"
        + "\n"                                     # blank line
        + json.dumps({"input_id": "good2"}) + "\n",
        encoding="utf-8",
    )
    man = Manifest(p)
    assert man.load() == [{"input_id": "good1"}, {"input_id": "good2"}]


def test_empty_write_is_noop_no_file(tmp_path):
    man = Manifest(tmp_path / "sub" / "e.manifest.jsonl")
    man.write([])
    assert not man.exists()


def test_empty_write_leaves_existing_file_untouched(tmp_path):
    man = Manifest(tmp_path / "run.manifest.jsonl")
    man.write([{"id": "1"}])
    man.write([])                                  # no-op, does not truncate
    assert man.load() == [{"id": "1"}]


def test_write_creates_parent_dir(tmp_path):
    man = Manifest(tmp_path / "nested" / "deep" / "m.manifest.jsonl")
    man.write([{"id": "1"}])
    assert man.exists() and man.load() == [{"id": "1"}]


def test_reset_removes_file(tmp_path):
    man = Manifest(tmp_path / "r.manifest.jsonl")
    man.write([{"id": "1"}])
    assert man.exists()
    man.reset()
    assert not man.exists()
    man.reset()                                    # idempotent on missing file
    assert man.load() == []


def test_sidecar_default_name(tmp_path):
    art = tmp_path / "jailbreaks.csv"
    man = Manifest.sidecar(art)
    assert man.path == tmp_path / "jailbreaks.manifest.jsonl"


def test_sidecar_explicit_name(tmp_path):
    art = tmp_path / "Data_cache" / "constitution" / "merged.csv"
    man = Manifest.sidecar(art, name="constitution.manifest.jsonl")
    assert man.path == tmp_path / "Data_cache" / "constitution" / "constitution.manifest.jsonl"


def test_manifest_and_ledger_share_conventions(tmp_path):
    # Both are JsonlSidecars beside the same artifact, differing only by suffix.
    art = tmp_path / "output_responses.csv"
    man, led = Manifest.sidecar(art), Ledger.sidecar(art, key_fields=("input_id",))
    assert isinstance(man, JsonlSidecar) and isinstance(led, JsonlSidecar)
    assert man.path == tmp_path / "output_responses.manifest.jsonl"
    assert led.path == tmp_path / "output_responses.state.jsonl"
