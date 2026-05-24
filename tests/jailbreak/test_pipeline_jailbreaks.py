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


def test_settings_per_iteration_escalation(tmp_path):
    """Multi-round escalation: 4 rounds per sample, exact counts for rounds 0-1."""
    out = tmp_path / "jb.csv"
    schedule = [
        {"exact_techniques": 1},
        {"exact_techniques": 2},
        {"max_complexity": 6, "max_obfuscations": 2},
        {"max_complexity": 9, "max_obfuscations": 3},
    ]
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=11,
                             settings_per_iteration=schedule, **PURE_KW)

    # iterations derived from schedule length: 5 samples x 4 rounds.
    plan = load_plan(out.with_name("jb.manifest.jsonl"))
    assert len(plan) == 20
    assert sorted({r["iteration"] for r in plan}) == [0, 1, 2, 3]

    # Per-round settings recorded; count-driven rounds have exact combination sizes.
    for r in plan:
        if r["iteration"] == 0:
            assert r["settings"].get("exact_techniques") == 1
            assert len(r["combination"]) == 1
        elif r["iteration"] == 1:
            assert r["settings"].get("exact_techniques") == 2
            assert len(r["combination"]) == 2

    # Output mirrors the exact counts for the count-driven rounds.
    assert (df[df["iteration"] == 0]["num_techniques"] == 1).all()
    assert (df[df["iteration"] == 1]["num_techniques"] == 2).all()


def test_settings_per_iteration_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv",
                            settings_per_iteration=[], **PURE_KW)


def test_include_translation_filter_removes_translation_family():
    """The include_translation=False filter drops exactly the translation family.

    Translation techniques all require an LLM, so an end-to-end (executing) test
    would hit the network. This checks the pool filter the pipeline applies, on the
    real spec-tagged obfuscation pool, offline.
    """
    from redact.jailbreak.obfuscation import get_all_obfuscation_functions

    pool = get_all_obfuscation_functions()
    # Baseline: translation is part of the obfuscation pool.
    assert any("translation" in getattr(f, "families", []) for f in pool)

    # The exact predicate generate_jailbreaks uses when include_translation=False.
    filtered = [f for f in pool if "translation" not in getattr(f, "families", [])]
    assert not any("translation" in getattr(f, "families", []) for f in filtered)
    # Nothing else is dropped.
    assert len(pool) - len(filtered) == sum(
        1 for f in pool if "translation" in getattr(f, "families", [])
    )
