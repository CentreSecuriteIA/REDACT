"""Tests for the shared sidecar resume-ledger (dataset/ledger.py).

Covers the behaviours every pipeline relies on: single- vs composite-key units,
type casting on read, extra (status) fields, bad-line robustness, reset, the
sidecar-path constructor, and the empty/missing cases.
"""

import json

import pytest

from redact.dataset import Ledger


def test_single_key_roundtrip(tmp_path):
    led = Ledger(tmp_path / "x.state.jsonl", key_fields=("unit",))
    assert led.completed() == set()          # missing file -> empty
    led.record([{"unit": "a::b"}, {"unit": "c::d"}])
    assert led.completed() == {"a::b", "c::d"}


def test_composite_key_with_casters(tmp_path):
    led = Ledger(
        tmp_path / "p.state.jsonl",
        key_fields=("input_id", "iteration"),
        casters={"input_id": str, "iteration": int},
    )
    # write ints as ints; caster keeps them ints so callers comparing (str,int) match
    led.record([{"input_id": "abc", "iteration": 0}, {"input_id": "abc", "iteration": 1}])
    assert led.completed() == {("abc", 0), ("abc", 1)}


def test_casters_normalize_numeric_ids_to_str(tmp_path):
    led = Ledger(tmp_path / "c.state.jsonl", key_fields=("id",), casters={"id": str})
    # even if an id round-trips as an int, the caster makes the key a str
    led.path.write_text(json.dumps({"id": 123}) + "\n", encoding="utf-8")
    assert led.completed() == {"123"}


def test_extra_status_fields_are_written_and_ignored_on_read(tmp_path):
    led = Ledger(tmp_path / "s.state.jsonl", key_fields=("input_id", "iteration"),
                 casters={"input_id": str, "iteration": int})
    led.record([{"input_id": "z", "iteration": 0, "status": "dropped", "model": "m"}])
    # extra fields persisted verbatim...
    line = json.loads(led.path.read_text(encoding="utf-8").strip())
    assert line["status"] == "dropped" and line["model"] == "m"
    # ...but the key is still just (input_id, iteration)
    assert led.completed() == {("z", 0)}


def test_bad_lines_are_skipped(tmp_path):
    p = tmp_path / "b.state.jsonl"
    p.write_text(
        json.dumps({"unit": "good1"}) + "\n"
        + "not json at all\n"
        + "\n"                                   # blank line
        + json.dumps({"other": "missing-key"}) + "\n"  # missing key field
        + json.dumps({"unit": "good2"}) + "\n",
        encoding="utf-8",
    )
    led = Ledger(p, key_fields=("unit",))
    assert led.completed() == {"good1", "good2"}


def test_record_empty_is_noop_no_file(tmp_path):
    led = Ledger(tmp_path / "sub" / "e.state.jsonl", key_fields=("id",))
    led.record([])
    assert not led.exists()


def test_record_creates_parent_dir(tmp_path):
    led = Ledger(tmp_path / "nested" / "deep" / "l.state.jsonl", key_fields=("id",))
    led.record([{"id": "1"}])
    assert led.exists() and led.completed() == {"1"}


def test_append_accumulates(tmp_path):
    led = Ledger(tmp_path / "a.state.jsonl", key_fields=("id",))
    led.record([{"id": "1"}])
    led.record([{"id": "2"}])
    assert led.completed() == {"1", "2"}


def test_reset_removes_file(tmp_path):
    led = Ledger(tmp_path / "r.state.jsonl", key_fields=("id",))
    led.record([{"id": "1"}])
    assert led.exists()
    led.reset()
    assert not led.exists()
    led.reset()  # idempotent on missing file
    assert led.completed() == set()


def test_sidecar_default_name(tmp_path):
    art = tmp_path / "conversations.csv"
    led = Ledger.sidecar(art, key_fields=("input_id", "iteration"))
    assert led.path == tmp_path / "conversations.state.jsonl"


def test_sidecar_explicit_name(tmp_path):
    art = tmp_path / "Datasets" / "out.csv"
    led = Ledger.sidecar(art, key_fields=("unit",), name="constitution.state.jsonl")
    assert led.path == tmp_path / "Datasets" / "constitution.state.jsonl"


def test_empty_key_fields_rejected(tmp_path):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "x.state.jsonl", key_fields=())
