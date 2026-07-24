"""Offline tests for generate_paraphrases (mapping/resume/dedup/check-drop).

The paraphrase + check LLM calls are monkeypatched to deterministic transforms,
so this exercises planning, model-grouped execution, dedup, resume, and both
targets without any network access.
"""

import pandas as pd
import pytest

import redact.pipelines as P
from redact import generate_paraphrases, paths
from redact.dataset.io import append_samples
from tests.conftest import MockBackend


@pytest.fixture()
def patched(monkeypatch):
    """Patch backend + paraphrase + check to deterministic offline behavior."""
    monkeypatch.setattr(P, "get_backend", lambda m: MockBackend())
    monkeypatch.setattr(
        P, "paraphrase_batch",
        lambda backend, model, texts, **kw: [f"PARA::{t}" for t in texts],
    )
    # accept everything by default
    monkeypatch.setattr(
        P, "batch_check_samples",
        lambda cb, cm, payloads, checker, **kw: [(True, "") for _ in payloads],
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
    assert set(df["paraphrase_model"]) == {"venice-uncensored"}
    out = paths.paraphrases_inputs_csv(tmp_path)
    assert out.exists()
    assert out.with_name("paraphrases_inputs.manifest.jsonl").exists()


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
        P, "batch_check_samples",
        lambda cb, cm, payloads, checker, **kw: [(False, "no") for _ in payloads],
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


def _seed_paraphrase_artifact(tmp_path):
    para = pd.DataFrame({
        "id": ["x1"], "input_id": ["b1"], "iteration": [0], "sample": ["PARA::p"],
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
