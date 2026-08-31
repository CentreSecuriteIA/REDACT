"""Offline tests for generate_paraphrases (mapping/resume/dedup/check-drop).

The paraphrase + check LLM calls are monkeypatched to deterministic transforms,
so this exercises planning, model-grouped execution, dedup, resume, and both
targets without any network access.
"""

import pandas as pd
import pytest

import redact.content_moderation.paraphrase as CP
from redact import generate_paraphrases, paths
from redact.dataset.io import append_samples
from tests.conftest import MockBackend, make_client


@pytest.fixture()
def patched(monkeypatch):
    """Patch backend + paraphrase + check to deterministic offline behavior.

    The plan/execute loop these patch points live in now runs in
    content_moderation.paraphrase (moved out of pipelines.py), so the
    monkeypatches target that module.
    """
    monkeypatch.setattr(CP.ModelClient, "create", lambda m: make_client(MockBackend(), m))
    monkeypatch.setattr(
        CP, "paraphrase_batch",
        lambda client, texts, **kw: [f"PARA::{t}" for t in texts],
    )
    # accept everything by default
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda cc, payloads, checker, **kw: [(True, "") for _ in payloads],
    )
    return monkeypatch


def _seed_inputs(tmp_path):
    append_samples(
        ["alpha prompt", "beta prompt"], category="Cyber", turn=0,
        accepted=[True, True], dataset_dir=paths.datasets(tmp_path),
    )


def test_paraphrase_inputs_dedup_and_manifest(tmp_path, patched):
    _seed_inputs(tmp_path)
    with pytest.warns(UserWarning):  # K>1 with a single paraphraser
        res = generate_paraphrases(
            data_dir=tmp_path, target="inputs", paraphrases_per_sample=2, verbose=False,
        )
    df = res["inputs"]
    # 2 inputs x K=2 = 4 units, but PARA::<text> is identical across k -> dedup to 1/input.
    assert set(df["sample"]) == {"PARA::alpha prompt", "PARA::beta prompt"}
    assert bool((df["accepted"] == True).all())  # noqa: E712
    assert set(df["paraphrase_model"]) == {"venice-paraphraser"}
    out = paths.paraphrases_inputs_csv(tmp_path)
    assert out.exists()
    assert out.with_name("paraphrases_inputs.manifest.jsonl").exists()
    # sample_id is this row's own content-hash identity, distinct from input_id
    # (which points back to the origin sample).
    from redact.dataset.io import _hash_text
    assert "sample_id" in df.columns
    for _, row in df.iterrows():
        assert row["sample_id"] == _hash_text(row["sample"])
        assert row["sample_id"] != row["input_id"]


def test_paraphrase_resume_is_idempotent(tmp_path, patched):
    _seed_inputs(tmp_path)
    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    n1 = len(pd.read_csv(paths.paraphrases_inputs_csv(tmp_path)))
    generate_paraphrases(data_dir=tmp_path, target="inputs", resume=True, verbose=False)
    n2 = len(pd.read_csv(paths.paraphrases_inputs_csv(tmp_path)))
    assert n1 == n2 == 2  # no duplicate rows on re-run


def test_paraphrase_check_drop(tmp_path, monkeypatch, patched):
    _seed_inputs(tmp_path)
    # Reject everything -> all dropped -> empty / no artifact rows.
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda cc, payloads, checker, **kw: [(False, "no") for _ in payloads],
    )
    res = generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    assert res["inputs"].empty or (res["inputs"]["accepted"] == False).all()  # noqa: E712


def test_paraphrase_outputs_accepted_only(tmp_path, patched):
    # Seed output_responses.csv: one accepted, one refused (rejected).
    resp = pd.DataFrame({
        "input_id": ["id1", "id2"],
        "output_response": ["a real answer", "I cannot help"],
        "accepted": [True, False],
        "category": ["Cyber", "Cyber"],
        "entry_type": ["harmful", "harmful"],
    })
    paths.output_responses_csv(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    resp.to_csv(paths.output_responses_csv(tmp_path), index=False)

    res = generate_paraphrases(data_dir=tmp_path, target="outputs", verbose=False)
    df = res["outputs"]
    assert list(df["sample"]) == ["PARA::a real answer"]   # only the accepted output paraphrased
    assert list(df["input_id"]) == ["id1"]


def test_ledger_prevents_reattempt_of_dropped_units(tmp_path, monkeypatch):
    _seed_inputs(tmp_path)
    calls = {"n": 0}

    def fake_para(client, texts, **kw):
        calls["n"] += len(texts)
        return ["SAME" for _ in texts]  # identical -> 2nd unit is a dedup-drop

    monkeypatch.setattr(CP.ModelClient, "create", lambda m: make_client(MockBackend(), m))
    monkeypatch.setattr(CP, "paraphrase_batch", fake_para)
    monkeypatch.setattr(CP, "batch_check_samples",
                        lambda c, samples, chk, **k: [(True, "") for _ in samples])

    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    assert calls["n"] == 2  # 2 units attempted (1 kept, 1 deduped-dropped)
    state = paths.paraphrases_inputs_csv(tmp_path).with_name("paraphrases_inputs.state.jsonl")
    assert state.exists()  # ledger recorded both attempts, including the drop

    # Resume: the dropped unit is in the ledger, so nothing is re-attempted.
    generate_paraphrases(data_dir=tmp_path, target="inputs", resume=True, verbose=False)
    assert calls["n"] == 2  # no new paraphrase calls


def test_prompt_dir_threads_to_paraphrase_and_check(tmp_path, monkeypatch):
    _seed_inputs(tmp_path)
    seen = {}
    monkeypatch.setattr(CP.ModelClient, "create", lambda m: make_client(MockBackend(), m))
    monkeypatch.setattr(
        CP, "paraphrase_batch",
        lambda client, texts, prompt_dir=None, **kw: (
            seen.__setitem__("para", prompt_dir) or [f"P::{t}" for t in texts]
        ),
    )
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda c, samples, chk, **k: [(True, "") for _ in samples],
    )
    # capture the checker's prompt_dir without loading real prompt files
    monkeypatch.setattr(
        CP, "build_paraphrase_checker",
        lambda prompt_dir=None: seen.__setitem__("check", prompt_dir) or (lambda s: []),
    )
    custom = str(tmp_path / "myprompts")
    generate_paraphrases(data_dir=tmp_path, target="inputs", prompt_dir=custom, verbose=False)
    assert seen["para"] == custom   # paraphrase prompt dir threaded
    assert seen["check"] == custom  # paraphrase-check prompt dir threaded


def _seed_paraphrase_artifact(tmp_path):
    para = pd.DataFrame({
        "sample_id": ["x1"], "input_id": ["b1"], "iteration": [0], "sample": ["PARA::p"],
        "category": ["Cyber"], "entry_type": ["harmful"], "paraphrase_model": ["m"],
        "accepted": [True], "reasoning": [""], "source": ["paraphrase_inputs"],
    })
    paths.paraphrases_inputs_csv(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    para.to_csv(paths.paraphrases_inputs_csv(tmp_path), index=False)


def test_build_dataset_training_merges_paraphrases(tmp_path):
    _seed_inputs(tmp_path)
    _seed_paraphrase_artifact(tmp_path)
    from redact import build_dataset
    train = build_dataset(data_dir=tmp_path, mode="training", verbose=False)
    assert bool((train["dataset_type"] == "content_moderation_paraphrase").any())
    assert not paths.paraphrased_csv(tmp_path).exists()  # merged, not a separate file


def test_build_dataset_eval_separates_paraphrases(tmp_path):
    _seed_inputs(tmp_path)
    _seed_paraphrase_artifact(tmp_path)
    from redact import build_dataset
    ev = build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
    assert not bool((ev["dataset_type"] == "content_moderation_paraphrase").any())  # excluded
    assert paths.paraphrased_csv(tmp_path).exists()                                 # separate file


def test_run_pipeline_paraphrase_stage(tmp_path, patched):
    _seed_inputs(tmp_path)
    from redact import run_pipeline
    from redact.runconfig import read_manifest
    summary = run_pipeline(
        {"dataset_type": "training", "data_dir": str(tmp_path),
         "stages": ["paraphrase", "build"]},
        params={"paraphrase": {"target": "inputs"}}, verbose=False,
    )
    assert summary["stages"]["paraphrase"]["counts"]["rows"] >= 1
    assert paths.paraphrases_inputs_csv(tmp_path).exists()
    assert read_manifest("paraphrase", tmp_path) is not None
