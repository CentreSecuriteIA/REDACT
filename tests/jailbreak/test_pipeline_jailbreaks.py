"""Offline end-to-end tests for the jailbreak pipeline (plan + execute + resume).

Uses ``pure_only=True`` with only the obfuscation layer so no technique needs an
LLM — the engine completes every sample at prime time and the router is never
called. This exercises planning, execution, manifest, the prompt-content
id, and resume without any network access. The ``pool`` tests at the end run
LLM-backed fake techniques over a hand-written plan against a fake backend.
"""


import pandas as pd
import pytest

from redact.jailbreak.manifest import compute_sample_id, load_plan
from redact.jailbreak.protocol import LLMRequest
from redact.llms.backends import ComputeConfig
from redact.pipelines import generate_jailbreaks
from tests.conftest import MockBackend, make_client

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
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=3,
                             iterations=3, max_obfuscations=1, **PURE_KW)
    plan = load_plan(out.with_name("jb.manifest.jsonl"))
    assert len(plan) == 15  # 5 samples x 3 iterations

    # Per sample, the 3 assigned combinations are distinct (pool is large enough).
    by_sample: dict[str, list] = {}
    for r in plan:
        by_sample.setdefault(r["sample_id"], []).append(tuple(r["combination"]))
    for combos in by_sample.values():
        assert len(set(combos)) == len(combos)

    # Regression: sample_id (this row's own id) used to be silently inherited
    # from input_id, so every iteration of the same input shared one value.
    # Each of the 3 iterations per input_id must now have its own distinct
    # sample_id (jailbreak/engine.py's _finalize()).
    for input_id, group in df.groupby("input_id"):
        assert group["sample_id"].nunique() == len(group)
        assert not (group["sample_id"] == input_id).any()


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


def test_completion_ledger_written_and_drives_resume(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)

    # Sidecar ledger has one (input_id, iteration) line per written unit.
    from redact.dataset import Ledger
    led = Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                         casters={"input_id": str, "iteration": int})
    assert len(led.completed()) == 5

    # Ledger alone drives resume: delete the output CSV, keep the ledger. A
    # resume run must still skip everything (nothing regenerated, no new rows).
    out.unlink()
    again = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1,
                                resume=True, **PURE_KW)
    assert len(again) == 0  # all units already acked in the ledger -> no output


def test_fresh_run_clears_ledger(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    from redact.dataset import Ledger
    led = Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                         casters={"input_id": str, "iteration": int})
    assert led.completed()

    # resume=False regenerates from scratch: ledger reset then re-acked to 5.
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, resume=False, **PURE_KW)
    assert len(led.completed()) == 5
    assert len(pd.read_csv(out)) == 5  # not doubled


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


# --- refill pool: LLM techniques over a hand-written plan -------------------

GEN = "jb-gen"


def one_step(text, *, gen_model=None, **kwargs):
    """LLM technique: one call."""
    reply = yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    return reply, "one_step"


def two_step(text, *, gen_model=None, **kwargs):
    """LLM technique: two calls in sequence."""
    a = yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    b = yield LLMRequest(gen_model, [{"role": "user", "content": "step2 " + a}])
    return b, "two_step"


def broken(text, *, gen_model=None, **kwargs):
    """LLM technique whose chain raises after its call."""
    yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    raise ValueError("chain failed")


class _StepBackend(MockBackend):
    """Native-batching fake: ``step1 x`` -> ``S1:x``, ``step2 x`` -> ``S2:x``.

    ``batches`` holds the request contents of each ``generate()`` call.
    """

    def __init__(self):
        super().__init__(model=GEN)
        self.batches: list[list[str]] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=True)

    def generate(self, messages_list, **kwargs):
        super().generate(messages_list, **kwargs)
        contents = [m[-1]["content"] for m in messages_list]
        self.batches.append(contents)
        return [c.replace("step1 ", "S1:").replace("step2 ", "S2:") for c in contents]


def _llm_run(tmp_path, monkeypatch, plan, backend, **kwargs):
    """Run the stage on ``plan``, ``[(prompt, technique), ...]``, written as its manifest."""
    import redact.pipelines as P
    from redact.dataset import Manifest

    monkeypatch.setattr(P, "get_all_obfuscation_functions", lambda: [one_step, two_step, broken])
    for name in ("get_all_hacking_functions", "get_all_manipulation_functions",
                 "get_all_request_functions"):
        monkeypatch.setattr(P, name, lambda: [])
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))

    out = tmp_path / "jb.csv"
    Manifest(out.with_name("jb.manifest.jsonl")).write(
        {"sample_id": compute_sample_id(p), "iteration": 0, "combination": [t.__name__],
         "settings": {}, "status": "planned"}
        for p, t in plan
    )
    prompts = [p for p, _ in plan]
    inputs = pd.DataFrame({"prompt": prompts, "category": ["Cyber"] * len(prompts),
                           "entry_type": ["harmful"] * len(prompts)})
    generate_jailbreaks(inputs=inputs, output_path=out, model=GEN, resume=True,
                        verbose=False, **kwargs)
    return out


def test_pool_run_without_retries_writes_the_same_rows(tmp_path, monkeypatch):
    from redact.dataset.io import _hash_text
    from redact.jailbreak import load_spec

    techniques = [two_step, one_step, broken, one_step, one_step, one_step]
    plan = [(f"zq{i}", t) for i, t in enumerate(techniques)]
    out = _llm_run(tmp_path, monkeypatch, plan, _StepBackend(), batch_size=2)

    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(df.columns) == [
        "prompt", "category", "entry_type", "sample_id", "input_id", "jailbreak",
        "technique", "technique_info", "complexity", "num_techniques", "is_noop",
        "accepted", "reasoning", "iteration", "combination_spec_version",
    ]

    def row(prompt, technique, text, info, accepted, reasoning):
        return {
            "prompt": prompt, "category": "Cyber", "entry_type": "harmful",
            "sample_id": _hash_text(text), "input_id": compute_sample_id(prompt),
            "jailbreak": text, "technique": technique, "technique_info": info,
            "complexity": "0", "num_techniques": "1", "is_noop": str(text == prompt),
            "accepted": str(accepted), "reasoning": reasoning, "iteration": "0",
            "combination_spec_version": str(load_spec().get("version", "")),
        }

    expected = [
        row("zq0", "two_step", "S2:S1:zq0", "two_step", True, ""),
        row("zq2", "broken", "zq2", "ERROR: chain failed", False, "chain failed"),
    ] + [row(f"zq{i}", "one_step", f"S1:zq{i}", "one_step", True, "") for i in (1, 3, 4, 5)]
    assert {(r["input_id"], r["iteration"]): r for r in df.to_dict("records")} == {
        (r["input_id"], r["iteration"]): r for r in expected
    }
    assert len(df) == 6


def test_pool_fills_a_call_with_a_later_step_and_a_new_unit(tmp_path, monkeypatch):
    backend = _StepBackend()
    plan = [("zq0", two_step)] + [(f"zq{i}", one_step) for i in range(1, 6)]
    _llm_run(tmp_path, monkeypatch, plan, backend, batch_size=2)

    # A chunk loop sends zq0's second step alone ([2, 1, 2, 2]).
    assert [len(b) for b in backend.batches] == [2, 2, 2, 1]
    assert backend.batches[2] == ["step2 S1:zq0", "step1 zq4"]


def test_pool_resume_after_a_failed_append_runs_only_unfinished_units(tmp_path, monkeypatch):
    plan = [(f"zq{i}", one_step) for i in range(70)]
    real_to_csv = pd.DataFrame.to_csv
    appends = []

    def failing_second_append(self, *args, **kwargs):
        appends.append(len(self))
        if len(appends) == 2:
            raise OSError("disk full")
        return real_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_second_append)
    with pytest.raises(OSError, match="disk full"):
        _llm_run(tmp_path, monkeypatch, plan, _StepBackend(), batch_size=4)
    monkeypatch.setattr(pd.DataFrame, "to_csv", real_to_csv)
    out = tmp_path / "jb.csv"
    assert appends == [32, 32] and len(pd.read_csv(out)) == 32

    second = _StepBackend()
    _llm_run(tmp_path, monkeypatch, plan, second, batch_size=4)
    df = pd.read_csv(out)
    assert sum(len(b) for b in second.batches) == 70 - 32
    assert sorted(df["prompt"]) == sorted(p for p, _ in plan)
