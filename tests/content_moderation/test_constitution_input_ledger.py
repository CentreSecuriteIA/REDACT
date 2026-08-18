"""Resume-ledger tests for constitution-seeded input generation.

Mirrors the ledger tests for the constitution / output / paraphrase stages:
a completed ``(sample_description, entry_type, style)`` unit is acked to
``constitution_inputs.state.jsonl`` and skipped on a resume re-run; ``fresh``
clears it. Fully offline (MockBackend).
"""

import pandas as pd

from tests.conftest import MockBackend
from redact.content_moderation.generation import (
    InputPipeline,
    _constitution_inputs_ledger,
    _constitution_inputs_manifest,
)

_PROMPT = {"system_prompt": "sys", "template": "Generate items about {sample_description}"}


def _pipeline(tmp_path):
    return InputPipeline(
        gen_backend=MockBackend("1. alpha sample\n2. beta sample"),
        gen_model="m",
        check_backend=MockBackend("Yes"),
        check_model="m",
        dataset_dir=tmp_path,
    )


def _const_df():
    return pd.DataFrame([
        {"source_category": "Violence", "sample_description": "desc one",
         "constitution_subcategory": "Sub", "entry_type": "harmful"},
        {"source_category": "Privacy", "sample_description": "desc two",
         "constitution_subcategory": "Sub", "entry_type": "benign"},
    ])


def test_ledger_written_and_resume_skips(tmp_path):
    pipe = _pipeline(tmp_path)
    r1 = pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert r1.total_entries_processed == 2

    # ledger recorded both entries
    led = _constitution_inputs_ledger(tmp_path)
    assert led.completed() == {("desc one", "harmful", ""), ("desc two", "benign", "")}

    # resume run: every entry already acked -> nothing processed, no gen calls
    pipe2 = _pipeline(tmp_path)
    r2 = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert r2.total_entries_processed == 0
    assert pipe2.gen_backend.calls == []


def test_manifest_written_before_generation(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                               style="long", verbose=False)
    man = _constitution_inputs_manifest(tmp_path)
    rows = man.load()
    # one planned row per constitution entry, keyed like the ledger.
    assert {(r["sample_description"], r["entry_type"], r["style"]) for r in rows} == {
        ("desc one", "harmful", "long"), ("desc two", "benign", "long"),
    }
    assert all(r["status"] == "planned" for r in rows)


def test_fresh_clears_ledger_and_regenerates(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert _constitution_inputs_ledger(tmp_path).completed()

    pipe2 = _pipeline(tmp_path)
    r = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                                    fresh=True, verbose=False)
    assert r.total_entries_processed == 2          # regenerated despite prior ledger
    assert pipe2.gen_backend.calls != []


def test_resume_falls_back_to_csv_when_ledger_absent(tmp_path):
    # Simulate a pre-ledger run: generate under a named style, then delete the
    # ledger. Resume must still skip via the existing-CSV set (back-compat).
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                               style="long", verbose=False)
    _constitution_inputs_ledger(tmp_path).reset()   # drop the ledger file

    pipe2 = _pipeline(tmp_path)
    r = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                                    style="long", verbose=False)
    assert r.total_entries_processed == 0
    assert pipe2.gen_backend.calls == []
