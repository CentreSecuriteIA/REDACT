"""Offline end-to-end tests for the jailbreak pipeline (plan + execute + resume).

Uses ``pure_only=True`` with only the obfuscation layer so no technique needs an
LLM — the engine completes every sample at prime time and the router is never
called. This exercises planning, chunked execution, manifest, the prompt-content
id, and resume without any network access.
"""

import json

import pandas as pd
import pytest

from redact.pipelines import generate_jailbreaks
from redact.jailbreak.manifest import compute_sample_id, load_plan


PURE_KW = dict(
    pure_only=True,
    include_hacking=False,
    include_manipulation=False,
    include_obfuscation=True,
    include_requests=False,
    auto_generate_benign=False,
    verbose=False,
)


def _inputs():
    prompts = [f"prompt number {i} about a topic" for i in range(5)]
    return pd.DataFrame({"prompt": prompts, "category": ["Cyber"] * 5,
                         "entry_type": ["harmful"] * 5})


def test_plan_then_execute(tmp_path):
    out = tmp_path / "jailbreaks.csv"
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **PURE_KW)

    # Manifest written before generation, one unit per sample.
    manifest = out.with_name(out.stem + ".manifest.jsonl")
    assert manifest.exists()
    plan = load_plan(manifest)
    assert len(plan) == 5
    assert all(set(r) >= {"sample_id", "iteration", "combination", "settings"} for r in plan)

    # One output row per sample, linked by prompt-content id.
    assert len(df) == 5
    ids = {compute_sample_id(p) for p in _inputs()["prompt"]}
    assert set(df["input_id"].astype(str)) == ids
    assert "jailbreak" in df.columns and "combination_spec_version" in df.columns


def test_reproducible_assignment(tmp_path):
    out1 = tmp_path / "a" / "jb.csv"
    out2 = tmp_path / "b" / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out1, seed=42, **PURE_KW)
    generate_jailbreaks(inputs=_inputs(), output_path=out2, seed=42, **PURE_KW)

    p1 = {(r["sample_id"], r["iteration"]): r["combination"]
          for r in load_plan(out1.with_name("jb.manifest.jsonl"))}
    p2 = {(r["sample_id"], r["iteration"]): r["combination"]
          for r in load_plan(out2.with_name("jb.manifest.jsonl"))}
    assert p1 == p2  # same seed + same prompts -> identical combinations


def test_iterations_dedup(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=3,
                        iterations=3, max_obfuscations=1, **PURE_KW)
    plan = load_plan(out.with_name("jb.manifest.jsonl"))
    assert len(plan) == 15  # 5 samples x 3 iterations

    # Per sample, the 3 assigned combinations are distinct (pool is large enough).
    by_sample: dict[str, list] = {}
    for r in plan:
        by_sample.setdefault(r["sample_id"], []).append(tuple(r["combination"]))
    for combos in by_sample.values():
        assert len(set(combos)) == len(combos)


def test_resume_skips_completed(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    first = pd.read_csv(out)
    assert len(first) == 5

    # Re-run with resume=True: nothing new should be appended (all done).
    again = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1,
                                resume=True, **PURE_KW)
    assert len(again) == 5  # no duplicate rows
    assert len(pd.read_csv(out)) == 5
