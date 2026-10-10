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
    # A residency plan must not read an HF config over the network.
    monkeypatch.setattr("redact.llms.resources.estimate._load_config", lambda _id: None)
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


def test_rows_in_the_csv_are_not_reparaphrased(tmp_path, patched):
    """Units whose rows reached the CSV but whose ack didn't (a crash in between,
    or a pre-ledger run) must not be paraphrased again — that appends a second
    row per unit."""
    _seed_inputs(tmp_path)
    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    out = paths.paraphrases_inputs_csv(tmp_path)
    before = pd.read_csv(out)
    out.with_name("paraphrases_inputs.state.jsonl").unlink()   # ack lost

    generate_paraphrases(data_dir=tmp_path, target="inputs", resume=True, verbose=False)
    after = pd.read_csv(out)
    assert len(after) == len(before)
    assert list(after["sample"]) == list(before["sample"])


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


def test_previous_paraphraser_is_unreferenced_before_unload(tmp_path, patched):
    """When paraphrasers do not co-fit, the previous client must be gone
    before its engine's cache entry is dropped, or the engine stays alive
    while the next one loads. The check model is kept."""
    import weakref
    from types import SimpleNamespace

    import redact.llms.backends.vllm as vllm_module
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model
    from redact import residency

    register_model("_para_second", backend_type="vllm", roles=["paraphraser"],
                   vllm=VLLMConfig(hf_model_id="org/second"))
    created: list = []     # (model, weak reference to its client)

    def create(model):
        # A preload finishes during the stage: its engine must be kept too.
        patched.setitem(vllm_module._engines, late, object())
        client = make_client(MockBackend(), model)
        created.append((model, weakref.ref(client)))
        return client

    late = vllm_module._engine_key("org/late-preload", None, {})
    unloads = []
    events = []

    def unload_local(keep=None, engines=None):
        paraphrasers = [ref for model, ref in created if model != "venice-uncensored"]
        unloads.append((keep, [ref() is None for ref in paraphrasers]))
        kept_engines.append(engines)
        events.append("unload")

    # Loaded before the stage: another stage's engine, and a pool paraphraser.
    kept_engines = []
    other = vllm_module._engine_key("org/other-stage", None, {})
    preloaded = vllm_module._engine_key("org/second", None, {})
    patched.setitem(vllm_module._engines, other, object())
    patched.setitem(vllm_module._engines, preloaded, object())

    plan = SimpleNamespace(sequential=True)
    patched.setattr(CP.ModelClient, "create", create)
    patched.setattr(residency, "plan_residency", lambda models: plan)
    patched.setattr(residency, "unload_local", unload_local)
    patched.setattr(residency, "apply_plan",
                    lambda p, replace=True: events.append((p is plan, replace)))
    _seed_inputs(tmp_path)
    try:
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", paraphraser="distribution",
            paraphrases_per_sample=2, verbose=False,
        )
    finally:
        MODEL_REGISTRY.pop("_para_second", None)
    # One unload, between the two paraphrasers: the first client is gone.
    assert unloads == [(["venice-uncensored"], [True])]
    # Other stages' engines survive the unload; the pool's own does not.
    assert kept_engines == [{"vllm": {other, late}, "introspect": set()}]
    # Its plan never replaces a run's, and is applied once: it outlives the unload.
    assert events == [(True, False), "unload"]


def test_co_fitting_paraphrasers_load_with_planned_grants(tmp_path, patched):
    """Two paraphrasers that share a card get the plan's fractions, and
    nothing is unloaded between them."""
    import redact.llms.backends.vllm as vllm_module
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model
    from redact import residency
    from redact.llms.resources import estimate

    register_model("_para_small", backend_type="vllm", roles=["paraphraser"],
                   vllm=VLLMConfig(hf_model_id="org/small", vram_gb=4.3))
    patched.setattr(estimate, "estimate_kv_gib", lambda *a, **k: 0.0)
    patched.setattr("redact.llms.resources.measure.detect_gpus", lambda: ("FakeGPU", 1))
    patched.setattr("redact.llms.resources.measure.detect_gpu_memory_gib", lambda: 80.0)
    patched.setattr(residency, "unload_local",
                    lambda keep=None, engines=None: pytest.fail("unloaded"))
    vllm_module.VLLMBackend.clear_cache()
    _seed_inputs(tmp_path)
    try:
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", paraphraser="distribution",
            paraphrases_per_sample=2, verbose=False,
        )
        # Pool 72; padded needs 48.89 + 5.33 leave 17.78, so each gets 8.89
        # on top.
        assert vllm_module._planned_utilization == {
            vllm_module._engine_key(
                "dphn/Dolphin-Mistral-24B-Venice-Edition", None, {}): 0.7222,
            vllm_module._engine_key("org/small", None, {}): 0.1777,
        }
    finally:
        MODEL_REGISTRY.pop("_para_small", None)
        vllm_module.VLLMBackend.clear_cache()
        vllm_module.set_planned_utilization({})     # a release keeps the plan


def test_an_unload_between_paraphrasers_shuts_down_only_their_engine(
        tmp_path, monkeypatch):
    """With paraphrasers that do not co-fit, the engine of the one that ran
    first is shut down before the next loads. The check model's engine and
    another stage's stay loaded."""
    import sys
    import weakref
    from types import SimpleNamespace

    import redact.llms.backends.vllm as vllm_module
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model
    from redact import residency

    shutdowns, engines = [], {}

    class _LLM:
        def __init__(self, **kw):
            model = kw["model"]
            engines[model] = weakref.ref(self)
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                shutdown=lambda: shutdowns.append(model)))

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=_LLM))
    monkeypatch.setattr(vllm_module, "_prepare_environment", lambda: None)
    monkeypatch.setattr("redact.llms.resources.estimate._load_config", lambda _id: None)
    monkeypatch.setattr(
        CP, "paraphrase_batch",
        lambda client, texts, **kw: [f"PARA::{t}" for t in texts])
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda cc, payloads, checker, **kw: [(True, "") for _ in payloads])
    monkeypatch.setattr(
        residency, "plan_residency", lambda models: SimpleNamespace(sequential=True))
    monkeypatch.setattr(residency, "apply_plan", lambda plan, replace=True: None)

    register_model("_para_second", backend_type="vllm", roles=["paraphraser"],
                   vllm=VLLMConfig(hf_model_id="org/second"))
    register_model("_para_check", backend_type="vllm",
                   vllm=VLLMConfig(hf_model_id="org/check"))
    pool = {"dphn/Dolphin-Mistral-24B-Venice-Edition", "org/second"}
    vllm_module.VLLMBackend.clear_cache()
    _seed_inputs(tmp_path)
    try:
        vllm_module._engine("org/other-stage", None, {})    # loaded before the stage
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", paraphraser="distribution",
            check_model="_para_check", paraphrases_per_sample=2, verbose=False,
        )
        # One unload, between the two paraphrasers: the first one's engine.
        assert len(shutdowns) == 1 and shutdowns[0] in pool
        assert engines[shutdowns[0]]() is None
        (still_loaded,) = pool - set(shutdowns)
        assert {key[0] for key in vllm_module._engines} == {
            "org/check", "org/other-stage", still_loaded}
    finally:
        MODEL_REGISTRY.pop("_para_second", None)
        MODEL_REGISTRY.pop("_para_check", None)
        vllm_module.VLLMBackend.clear_cache()
