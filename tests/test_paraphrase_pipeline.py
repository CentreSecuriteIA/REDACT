"""Offline tests for generate_paraphrases (mapping/resume/dedup/check-drop).

The paraphrase + check LLM calls are monkeypatched to deterministic transforms,
so this exercises planning, model-grouped execution, dedup, resume, and both
targets without any network access.
"""

import json

import pandas as pd
import pytest

import redact.content_moderation.paraphrase as CP
from redact import generate_paraphrases, paths
from redact.dataset import content_id
from tests.conftest import (
    Killed,
    MockBackend,
    fake_vllm_modules,
    kill_after_rows,
    lock_is_free,
    make_client,
    write_inputs,
)


@pytest.fixture(autouse=True)
def _no_short_text_skip(monkeypatch):
    """The stub texts are a few words long: the paraphrase stage's short-text
    skip would leave every unit unparaphrased. Its own tests set min_words."""
    monkeypatch.setattr(CP, "MIN_WORDS", 0)


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
    write_inputs(paths.constitution_inputs_dir(tmp_path),
                 ["alpha prompt", "beta prompt"])


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
    from redact.dataset import content_id
    assert "sample_id" in df.columns
    for _, row in df.iterrows():
        assert row["sample_id"] == content_id(row["sample"])
        assert row["sample_id"] != row["input_id"]


def test_without_inputs_the_inputs_of_the_run_are_found(tmp_path, patched):
    from tests.test_pipelines import _seed_input_tree

    text = _seed_input_tree(tmp_path)
    res = generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    df = res["inputs"]
    assert list(df["sample"]) == [f"PARA::{text}"]
    assert list(df["input_id"]) == [content_id(text)]


def test_the_stage_paraphrases_the_frame_it_is_given_into_the_file_it_is_given(
        tmp_path, patched):
    source = pd.DataFrame({"sample": ["alpha prompt"], "category": ["Cyber"],
                           "entry_type": ["harmful"]})
    out = tmp_path / "elsewhere" / "paraphrased.csv"
    df = CP.run_paraphrase_target(
        "inputs", source, out, pool=["venice-paraphraser"], check=None, k=1,
        batch_size=8, seed=0, verbose=False,
    )
    assert list(df["sample"]) == ["PARA::alpha prompt"]
    assert list(pd.read_csv(out)["input_id"]) == [content_id("alpha prompt")]
    assert not paths.datasets(tmp_path).exists()


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
    # A file without sample_ids: the response's id is that of its text.
    assert list(df["response_id"]) == [content_id("a real answer")]


def test_the_callers_frames_are_left_as_they_were(tmp_path, patched):
    inputs = pd.DataFrame({
        "sample": ["alpha prompt", "beta prompt"], "category": ["Cyber", "Fraud"],
    }, index=[5, 2])
    outputs = pd.DataFrame({
        "input_id": ["id1", "id2"], "output_response": ["an answer", "a refusal"],
        "accepted": [True, False],
    }, index=[9, 4])
    kept = inputs.copy(), outputs.copy()
    res = generate_paraphrases(
        data_dir=tmp_path, inputs=inputs, outputs=outputs, target="both",
        verbose=False)
    pd.testing.assert_frame_equal(inputs, kept[0])
    pd.testing.assert_frame_equal(outputs, kept[1])
    assert list(res["inputs"]["sample"]) == ["PARA::alpha prompt", "PARA::beta prompt"]
    assert list(res["outputs"]["sample"]) == ["PARA::an answer"]


@pytest.mark.parametrize("verbose", [False, True])
def test_without_read_back_no_artifact_is_read_after_its_last_save(
        tmp_path, patched, verbose):
    """A run that finds the artifact reads it once before it starts, for the
    texts it already holds."""
    _seed_inputs(tmp_path)
    artifact = paths.paraphrases_inputs_csv(tmp_path)
    events, append_rows, read_csv = [], CP.append_rows, pd.read_csv

    def saving(path, rows):
        events.append("save")
        return append_rows(path, rows)

    def reading(path, *args, **kwargs):
        whole = kwargs.get("chunksize") is None and kwargs.get("nrows") is None
        if whole and path == artifact:
            events.append("read")
        return read_csv(path, *args, **kwargs)

    patched.setattr(CP, "append_rows", saving)
    for run, expected in (("first", ["save"]), ("nothing left to do", ["read"])):
        del events[:]
        with patched.context() as patch:
            patch.setattr(pd, "read_csv", reading)
            res = generate_paraphrases(
                data_dir=tmp_path, target="inputs", verbose=verbose, read_back=False)
        assert res["inputs"].empty and len(res["inputs"].columns) == 0, run
        assert events == expected, run
    assert list(pd.read_csv(artifact)["sample"]) == [
        "PARA::alpha prompt", "PARA::beta prompt"]


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
    assert not paths.eval_inputs_csv(tmp_path).exists()


def test_build_dataset_eval_marks_the_paraphrases_among_the_inputs(tmp_path):
    _seed_inputs(tmp_path)
    _seed_paraphrase_artifact(tmp_path)
    from redact import build_dataset
    inputs = build_dataset(data_dir=tmp_path, mode="eval", verbose=False)["inputs"]
    paraphrased = inputs[inputs["paraphrased"]]
    assert list(paraphrased["text"]) == ["PARA::p"]
    assert list(paraphrased["dataset_type"]) == ["content_moderation_paraphrase"]
    assert not paths.complete_dataset_csv(tmp_path).exists()


def test_run_pipeline_paraphrase_stage(tmp_path, patched):
    _seed_inputs(tmp_path)
    from redact import run_pipeline
    from redact.driver import read_record
    summary = run_pipeline(
        {"dataset_type": "training", "data_dir": str(tmp_path),
         "stages": ["paraphrase", "build"]},
        params={"paraphrase": {"target": "inputs"}}, verbose=False,
    )
    assert summary["stages"]["paraphrase"]["counts"]["rows"] >= 1
    assert paths.paraphrases_inputs_csv(tmp_path).exists()
    assert read_record("paraphrase", tmp_path) is not None


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
    patched.setattr(residency, "plan_residency", lambda models, copies=None: plan)
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
    patched.setattr("redact.llms.resources.measure.detect_gpus",
                    lambda visible=False: ("FakeGPU", 1))
    patched.setattr("redact.llms.resources.measure.card_sizes_gib", lambda: [80.0])
    patched.setattr(residency, "unload_local",
                    lambda keep=None, engines=None: pytest.fail("unloaded"))
    vllm_module.VLLMBackend.clear_cache()
    _seed_inputs(tmp_path)
    try:
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", paraphraser="distribution",
            paraphrases_per_sample=2, verbose=False,
        )
        # Pool 72; padded needs 27.99 (24.9 GiB at FP8) + 5.33 leave 38.68,
        # so each gets 19.34 on top.
        assert vllm_module._planned_utilization == {
            vllm_module._engine_key(
                "dphn/Dolphin-Mistral-24B-Venice-Edition", "fp8", {}): 0.5916,
            vllm_module._engine_key("org/small", None, {}): 0.3083,
        }
    finally:
        MODEL_REGISTRY.pop("_para_small", None)
        vllm_module.VLLMBackend.clear_cache()
        vllm_module.set_planned_utilization({})     # a release keeps the plan


@pytest.mark.parametrize("cards", [2, 3])
def test_the_pool_is_planned_with_one_copy_of_each_paraphraser(tmp_path, patched, cards):
    """The pool runs one paraphraser at a time, and a direct call has no
    way to turn copies off. With two cards each paraphraser has one; with
    three, the idle card is not handed to the first as a copy."""
    import redact.llms.backends.vllm as vllm_module
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model
    from redact import residency
    from redact.llms.resources import estimate

    register_model("_para_small", backend_type="vllm", roles=["paraphraser"],
                   vllm=VLLMConfig(hf_model_id="org/small", vram_gb=4.3))
    patched.setattr(estimate, "estimate_kv_gib", lambda *a, **k: 0.0)
    patched.setattr("redact.llms.resources.measure.detect_gpus",
                    lambda visible=False: ("FakeGPU", cards))
    patched.setattr("redact.llms.resources.measure.card_sizes_gib", lambda: [80.0])
    plan_residency, plans = residency.plan_residency, []

    def planned(models, **kwargs):
        plans.append((kwargs, plan_residency(models, **kwargs)))
        return plans[-1][1]

    patched.setattr(residency, "plan_residency", planned)
    _seed_inputs(tmp_path)
    try:
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", paraphraser="distribution",
            paraphrases_per_sample=2, verbose=False,
        )
        (asked, plan), = plans
        assert asked == {"copies": {"venice-paraphraser": 1, "_para_small": 1}}
        assert [(fp.model, fp.devices, fp.copies) for fp in plan.footprints] == [
            ("_para_small", (0,), 1), ("venice-paraphraser", (1,), 1)]
        assert vllm_module._planned_devices == {
            vllm_module._engine_key("org/small", None, {}): (0,),
            vllm_module._engine_key(
                "dphn/Dolphin-Mistral-24B-Venice-Edition", "fp8", {}): (1,),
        }
    finally:
        MODEL_REGISTRY.pop("_para_small", None)


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
        scheduler_config = SimpleNamespace(max_num_seqs=256)

        def __init__(self, **kw):
            model = kw["model"]
            engines[model] = weakref.ref(self)
            self.shutdown = lambda: shutdowns.append(model)

    for name, module in fake_vllm_modules(_LLM).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(vllm_module, "_prepare_environment", lambda: None)
    monkeypatch.setattr("redact.llms.resources.estimate._load_config", lambda _id: None)
    monkeypatch.setattr(
        CP, "paraphrase_batch",
        lambda client, texts, **kw: [f"PARA::{t}" for t in texts])
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda cc, payloads, checker, **kw: [(True, "") for _ in payloads])
    monkeypatch.setattr(
        residency, "plan_residency",
        lambda models, copies=None: SimpleNamespace(sequential=True))
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


# --- how a chunk is saved ----------------------------------------------------

_FOUR = ["alpha prompt", "beta prompt", "gamma prompt", "delta prompt"]


def _seed_four_inputs(tmp_path):
    write_inputs(paths.constitution_inputs_dir(tmp_path), _FOUR)
    # gamma comes back unchanged, so its unit is dropped.
    return lambda client, texts, **kw: [
        t if t.startswith("gamma") else f"PARA::{t}" for t in texts
    ]


def _acks(tmp_path):
    """The ledger's unit lines, without what they record of the file's size."""
    state = paths.paraphrases_inputs_csv(tmp_path).with_name(
        "paraphrases_inputs.state.jsonl")
    if not state.exists():
        return []
    lines = map(json.loads, state.read_text(encoding="utf-8").splitlines())
    acks = [{k: v for k, v in line.items() if k != "_saved"} for line in lines]
    return [ack for ack in acks if ack]


@pytest.mark.parametrize("target", ["inputs", "outputs"])
def test_the_ledger_records_how_much_of_the_output_is_saved(tmp_path, patched, target):
    from redact.dataset import Ledger

    _seed_inputs(tmp_path)
    pd.DataFrame({
        "input_id": ["id1"], "output_response": ["a real answer"], "accepted": [True],
        "category": ["Cyber"], "entry_type": ["harmful"],
    }).to_csv(paths.output_responses_csv(tmp_path), index=False)
    generate_paraphrases(data_dir=tmp_path, target=target, verbose=False)
    out = getattr(paths, f"paraphrases_{target}_csv")(tmp_path)
    size, _checksum = Ledger.sidecar(out).saved[out.name]
    assert size == out.stat().st_size


def test_a_chunk_is_one_append_followed_by_its_acks(tmp_path, patched):
    patched.setattr(CP, "paraphrase_batch", _seed_four_inputs(tmp_path))
    saves, append = [], CP.append_rows

    def spy(path, rows):
        saves.append(([r["sample"] for r in rows], len(_acks(tmp_path))))
        return append(path, rows)

    patched.setattr(CP, "append_rows", spy)
    df = generate_paraphrases(
        data_dir=tmp_path, target="inputs", batch_size=2, verbose=False)["inputs"]

    # Per append: its rows, and the acks on disk when it was made.
    assert saves == [
        (["PARA::alpha prompt", "PARA::beta prompt"], 0), (["PARA::delta prompt"], 2)]
    assert list(df["sample"]) == [
        "PARA::alpha prompt", "PARA::beta prompt", "PARA::delta prompt"]
    statuses = ["accepted", "accepted", "dropped_dedup", "accepted"]
    assert _acks(tmp_path) == [
        {"input_id": content_id(text), "iteration": 0,
         "paraphrase_model": "venice-paraphraser", "status": status}
        for text, status in zip(_FOUR, statuses)
    ]


def test_a_run_killed_between_rows_and_acks_leaves_each_unit_once(tmp_path, patched):
    patched.setattr(CP, "paraphrase_batch", _seed_four_inputs(tmp_path))
    out = paths.paraphrases_inputs_csv(tmp_path)
    state = out.with_name("paraphrases_inputs.state.jsonl")

    kill_after_rows(patched, CP, on_call=2)
    with pytest.raises(Killed):
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", batch_size=2, verbose=False)
    # The second chunk's row landed; its acks did not.
    assert list(pd.read_csv(out)["sample"]) == [
        "PARA::alpha prompt", "PARA::beta prompt", "PARA::delta prompt"]
    assert [ack["input_id"] for ack in _acks(tmp_path)] == [
        content_id(text) for text in _FOUR[:2]]
    assert lock_is_free(state)

    df = generate_paraphrases(
        data_dir=tmp_path, target="inputs", batch_size=2, verbose=False)["inputs"]
    assert list(df["sample"]) == [
        "PARA::alpha prompt", "PARA::beta prompt", "PARA::delta prompt"]
    cut = out.with_name("paraphrases_inputs.csv.cut")
    assert b"PARA::delta prompt" in cut.read_bytes()
    statuses = ["accepted", "accepted", "dropped_dedup", "accepted"]
    assert [ack["status"] for ack in _acks(tmp_path)] == statuses
    assert lock_is_free(state)


def test_a_run_with_nothing_to_do_gives_the_lock_back(tmp_path, patched):
    _seed_inputs(tmp_path)
    state = paths.paraphrases_inputs_csv(tmp_path).with_name(
        "paraphrases_inputs.state.jsonl")
    for _run in range(2):       # the second finds every unit done
        generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
        assert lock_is_free(state)


def test_a_fresh_run_on_no_source_row_clears_what_an_earlier_run_left(tmp_path):
    out = paths.paraphrases_inputs_csv(tmp_path)
    out.parent.mkdir(parents=True)
    names = ["paraphrases_inputs.csv", "paraphrases_inputs.csv.cut",
             "paraphrases_inputs.state.jsonl", "paraphrases_inputs.manifest.jsonl"]
    for name in names:
        out.with_name(name).write_text("left by an earlier run\n", encoding="utf-8")
    returned = CP.run_paraphrase_target(
        "inputs", pd.DataFrame(), out, pool=["para-model"], check=None, k=1,
        resume=False, batch_size=8, seed=0, verbose=False)
    assert returned.empty
    assert [name for name in names if out.with_name(name).exists()] == []
    assert lock_is_free(out.with_name("paraphrases_inputs.state.jsonl"))


def test_a_failed_append_acks_none_of_its_chunk(tmp_path, patched):
    patched.setattr(CP, "paraphrase_batch", _seed_four_inputs(tmp_path))
    outcomes, append = iter([None, OSError("disk full")]), CP.append_rows

    def failing(path, rows):
        error = next(outcomes)
        if error:
            raise error
        return append(path, rows)

    patched.setattr(CP, "append_rows", failing)
    with pytest.raises(OSError, match="disk full"):
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", batch_size=2, verbose=False)

    assert [ack["input_id"] for ack in _acks(tmp_path)] == [
        content_id(text) for text in _FOUR[:2]]
    saved = pd.read_csv(paths.paraphrases_inputs_csv(tmp_path))
    assert list(saved["sample"]) == ["PARA::alpha prompt", "PARA::beta prompt"]


# --- the outputs target: a unit per response ---------------------------------

def _responses(**columns) -> pd.DataFrame:
    """Two accepted responses of one input, in the two modes, and one of another."""
    return pd.DataFrame({
        "input_id": ["in-1", "in-1", "in-2"],
        "sample_id": ["out-a", "out-b", "out-c"],
        "output_response": ["a harmful answer", "a helpful answer", "another answer"],
        "accepted": [True, True, True],
        "category": ["Cyber"] * 3,
        "entry_type": ["dual_use_benign", "dual_use_benign", "harmful"],
        "response_mode": ["harmful", "helpful", "harmful"],
        "output_label": ["harmful", "benign", "harmful"],
        "response_label": ["dual_use_benign_harmful", "dual_use_benign_benign",
                           "harmful_harmful"],
        **columns,
    })


def _outputs_state(tmp_path):
    return paths.paraphrases_outputs_csv(tmp_path).with_name(
        "paraphrases_outputs.state.jsonl")


def test_each_response_of_an_input_is_its_own_unit(tmp_path, patched):
    from redact.dataset import Ledger, Manifest

    paraphrased = []
    patched.setattr(
        CP, "paraphrase_batch",
        lambda client, texts, **kw: paraphrased.extend(texts) or [
            f"PARA{len(paraphrased) - len(texts) + i}::{t}"
            for i, t in enumerate(texts)])
    with pytest.warns(UserWarning):     # K>1 with a single paraphraser
        df = generate_paraphrases(outputs=_responses(), data_dir=tmp_path,
                                  target="outputs", paraphrases_per_sample=2,
                                  verbose=False)["outputs"]
    out = paths.paraphrases_outputs_csv(tmp_path)
    planned = Manifest.sidecar(out).load()
    assert [(r["response_id"], r["input_id"], r["iteration"]) for r in planned] == [
        ("out-a", "in-1", 0), ("out-a", "in-1", 1), ("out-b", "in-1", 0),
        ("out-b", "in-1", 1), ("out-c", "in-2", 0), ("out-c", "in-2", 1)]
    assert len(paraphrased) == len(planned)
    assert Ledger.sidecar(out, ("response_id", "iteration")).completed() == {
        (r["response_id"], r["iteration"]) for r in planned}
    assert list(df.columns) == [
        "sample_id", "response_id", "input_id", "iteration", "sample", "category",
        "entry_type", "response_mode", "output_label", "response_label",
        "paraphrase_model", "accepted", "reasoning", "source"]
    by_response = df.groupby("response_id").first()
    assert by_response.loc[["out-a", "out-b", "out-c"],
                           ["input_id", "output_label", "response_label"]
                           ].values.tolist() == [
        ["in-1", "harmful", "dual_use_benign_harmful"],
        ["in-1", "benign", "dual_use_benign_benign"],
        ["in-2", "harmful", "harmful_harmful"]]
    assert set(df["entry_type"]) == {"dual_use_benign", "harmful"}


def test_a_frame_without_ids_or_labels_writes_the_columns_empty(tmp_path, patched):
    """So a later run from a file that has them appends to the same header."""
    bare = _responses().drop(columns=["sample_id", "response_mode", "output_label",
                                      "response_label"]).iloc[:1]
    first = generate_paraphrases(outputs=bare, data_dir=tmp_path, target="outputs",
                                 verbose=False)["outputs"]
    assert first[["response_id", "response_mode", "output_label", "response_label"]
                 ].values.tolist() == [[content_id("a harmful answer"), "", "", ""]]
    later = generate_paraphrases(outputs=_responses(), data_dir=tmp_path,
                                 target="outputs", verbose=False)["outputs"]
    # out-a's paraphrase is the text the first run wrote: dropped as a repeat.
    assert list(later["response_id"]) == [
        content_id("a harmful answer"), "out-b", "out-c"]
    assert list(later["response_label"].iloc[1:]) == [
        "dual_use_benign_benign", "harmful_harmful"]


def _no_paraphrase(patched):
    def paraphrase(client, texts, **kw):
        raise AssertionError("a paraphrase was asked for")
    patched.setattr(CP, "paraphrase_batch", paraphrase)


def test_a_ledger_keyed_on_the_input_stops_the_outputs_before_any_call(
        tmp_path, patched):
    """The ledger of the stage skips a record without its key: the stop reads
    the old acks through a ledger keyed on the iteration alone."""
    out = paths.paraphrases_outputs_csv(tmp_path)
    out.parent.mkdir(parents=True)
    pd.DataFrame([{"sample_id": "p1", "input_id": "in-1", "iteration": 0,
                   "sample": "PARA::x", "category": "Cyber", "entry_type": "harmful",
                   "paraphrase_model": "m", "accepted": True, "reasoning": "",
                   "source": "paraphrase_outputs"}]).to_csv(out, index=False)
    _outputs_state(tmp_path).write_text(
        '{"input_id": "in-1", "iteration": 0, "paraphrase_model": "m", '
        '"status": "accepted"}\n', encoding="utf-8")
    _no_paraphrase(patched)
    with pytest.raises(ValueError, match="without a response_id") as raised:
        generate_paraphrases(outputs=_responses(), data_dir=tmp_path,
                             target="outputs", verbose=False)
    assert "resume=False" in str(raised.value)
    assert lock_is_free(_outputs_state(tmp_path))


def test_a_file_without_the_response_columns_stops_the_outputs_before_any_call(
        tmp_path, patched):
    """Also with no ledger, which a resume takes over."""
    out = paths.paraphrases_outputs_csv(tmp_path)
    out.parent.mkdir(parents=True)
    pd.DataFrame([{"sample_id": "p1", "input_id": "in-1", "iteration": 0,
                   "sample": "PARA::x", "accepted": True}]).to_csv(out, index=False)
    _no_paraphrase(patched)
    with pytest.raises(ValueError, match="'response_id'") as raised:
        generate_paraphrases(outputs=_responses(), data_dir=tmp_path,
                             target="outputs", verbose=False)
    assert "resume=False" in str(raised.value)
    assert "new data_dir" in str(raised.value)


def _recording_check(patched, verdict=(True, "")):
    """Patch the check call to accept (or give ``verdict``); return its calls."""
    calls = []

    def check(cc, payloads, checker, **kw):
        calls.append((list(payloads), kw))
        return [verdict for _ in payloads]

    patched.setattr(CP, "batch_check_samples", check)
    return calls


_RUN_ON = " and then it goes on into a long story nobody asked for"


def test_a_gated_paraphrase_is_a_rejected_row_and_is_not_checked(tmp_path, patched):
    from redact.llm_pipeline import CHECK_TEMPERATURE

    _seed_inputs(tmp_path)
    patched.setattr(CP, "paraphrase_batch", lambda client, texts, **kw: [
        t + _RUN_ON if t.startswith("beta") else f"PARA::{t}" for t in texts])
    calls = _recording_check(patched)
    df = generate_paraphrases(data_dir=tmp_path, target="inputs",
                              verbose=False)["inputs"]

    by_text = df.set_index("sample")
    assert bool(by_text.loc["PARA::alpha prompt", "accepted"])
    gated = by_text.loc["beta prompt" + _RUN_ON]
    assert not bool(gated["accepted"])
    assert gated["reasoning"].startswith("length ratio")
    assert [ack["status"] for ack in _acks(tmp_path)] == ["accepted", "rejected"]
    # The check model saw the ungated paraphrase alone, at the check temperature.
    assert calls == [(["PARA::alpha prompt"], {
        "originals": ["alpha prompt"], "batch_size": 256, "progress": None,
        "temperature": CHECK_TEMPERATURE})]


def test_a_chunk_whose_paraphrases_are_all_gated_makes_no_check_call(tmp_path, patched):
    _seed_inputs(tmp_path)
    patched.setattr(CP, "paraphrase_batch",
                    lambda client, texts, **kw: [t + _RUN_ON for t in texts])
    calls = _recording_check(patched)
    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    assert calls == []
    assert [ack["status"] for ack in _acks(tmp_path)] == ["rejected", "rejected"]


def test_a_gated_text_is_written_once(tmp_path, patched):
    """It is a row, so the same text from another unit is a repeat."""
    write_inputs(paths.constitution_inputs_dir(tmp_path), ["alpha prompt"])
    patched.setattr(CP, "paraphrase_batch",
                    lambda client, texts, **kw: [t + _RUN_ON for t in texts])
    with pytest.warns(UserWarning):     # K>1 with a single paraphraser
        df = generate_paraphrases(data_dir=tmp_path, target="inputs",
                                  paraphrases_per_sample=2, verbose=False)["inputs"]
    assert len(df) == 1 and not bool(df["accepted"].iloc[0])
    assert [ack["status"] for ack in _acks(tmp_path)] == ["rejected", "dropped_dedup"]


def test_without_the_check_no_gate_runs(tmp_path, patched):
    _seed_inputs(tmp_path)
    patched.setattr(CP, "paraphrase_batch",
                    lambda client, texts, **kw: [t + _RUN_ON for t in texts])
    patched.setattr(CP, "gate_reason",
                    lambda *a: pytest.fail("a gate ran without the check"))
    calls = _recording_check(patched)
    df = generate_paraphrases(data_dir=tmp_path, target="inputs", check=False,
                              verbose=False)["inputs"]
    assert bool(df["accepted"].all()) and calls == []


def test_the_language_gate_rejects_another_language(tmp_path, patched):
    from redact.jailbreak.obfuscation import langid

    _seed_inputs(tmp_path)
    patched.setattr(langid, "label_probability", lambda text: (
        ("deu_Latn", 0.95) if text.startswith("PARA::") else ("eng_Latn", 0.95)))
    df = generate_paraphrases(data_dir=tmp_path, target="inputs",
                              verbose=False)["inputs"]
    assert list(df["reasoning"]) == ["language eng_Latn -> deu_Latn"] * 2
    assert not df["accepted"].astype(bool).any()


def _glotlid_warnings(caplog):
    return [r for r in caplog.records if r.levelname == "WARNING"
            and "GlotLID is not available" in r.getMessage()]


def test_without_glotlid_the_check_warns_once(tmp_path, patched, caplog):
    import logging

    from redact.jailbreak.obfuscation import langid

    _seed_inputs(tmp_path)
    pd.DataFrame({
        "input_id": ["id1"], "output_response": ["a real answer"], "accepted": [True],
        "category": ["Cyber"], "entry_type": ["harmful"],
    }).to_csv(paths.output_responses_csv(tmp_path), index=False)
    patched.setattr(langid, "available", lambda: False)
    with caplog.at_level(logging.WARNING):
        generate_paraphrases(data_dir=tmp_path, target="both", verbose=False)
    (warning,) = _glotlid_warnings(caplog)
    assert "skipped" not in warning.getMessage()    # counted by other tests
    assert "redact[langid]" in warning.getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        generate_paraphrases(data_dir=tmp_path, target="both", check=False,
                             resume=False, verbose=False)
    assert _glotlid_warnings(caplog) == []


def test_without_a_source_row_the_glotlid_warning_is_not_given(
        tmp_path, patched, caplog):
    import logging

    write_inputs(paths.constitution_inputs_dir(tmp_path), ["alpha prompt"],
                 accepted=[False])
    with caplog.at_level(logging.WARNING):
        generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    assert _glotlid_warnings(caplog) == []


# --- the default target: the inputs ------------------------------------------

def _seed_inputs_and_outputs(tmp_path):
    _seed_inputs(tmp_path)
    pd.DataFrame({
        "input_id": ["id1"], "output_response": ["a real answer"], "accepted": [True],
        "category": ["Cyber"], "entry_type": ["harmful"],
    }).to_csv(paths.output_responses_csv(tmp_path), index=False)


def test_by_default_only_the_inputs_are_paraphrased(tmp_path, patched):
    _seed_inputs_and_outputs(tmp_path)
    res = generate_paraphrases(data_dir=tmp_path, verbose=False)
    assert list(res) == ["inputs"]
    assert list(res["inputs"]["sample"]) == ["PARA::alpha prompt", "PARA::beta prompt"]
    assert paths.paraphrases_inputs_csv(tmp_path).exists()
    assert not paths.paraphrases_outputs_csv(tmp_path).exists()


def test_both_targets_paraphrase_the_outputs_too(tmp_path, patched):
    _seed_inputs_and_outputs(tmp_path)
    res = generate_paraphrases(data_dir=tmp_path, target="both", verbose=False)
    assert list(res) == ["inputs", "outputs"]
    assert list(res["outputs"]["sample"]) == ["PARA::a real answer"]


def _stale_warnings(caplog):
    return [r for r in caplog.records if r.levelname == "WARNING"
            and "paraphrases of responses from an earlier run" in r.getMessage()]


def test_a_default_run_warns_of_paraphrases_of_outputs_an_earlier_run_left(
        tmp_path, patched, caplog):
    import logging

    _seed_inputs_and_outputs(tmp_path)
    with caplog.at_level(logging.WARNING):
        generate_paraphrases(data_dir=tmp_path, verbose=False)
    assert _stale_warnings(caplog) == []

    generate_paraphrases(data_dir=tmp_path, target="outputs", verbose=False)
    left = paths.paraphrases_outputs_csv(tmp_path).read_bytes()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        generate_paraphrases(data_dir=tmp_path, verbose=False)
    (warning,) = _stale_warnings(caplog)
    assert "paraphrases_outputs.csv" in warning.getMessage()
    assert 'run target "outputs" with resume: false' in warning.getMessage()
    assert paths.paraphrases_outputs_csv(tmp_path).read_bytes() == left


def test_a_recipe_without_a_target_reads_and_writes_the_inputs_alone(
        tmp_path, patched):
    from redact import run_pipeline
    from redact.driver import read_record
    from redact.telemetry import uninstall

    _seed_inputs_and_outputs(tmp_path)
    try:
        summary = run_pipeline(
            {"dataset_type": "training", "data_dir": str(tmp_path),
             "stages": ["paraphrase"]}, params={}, verbose=False)
    finally:
        uninstall()
    assert summary["stages"]["paraphrase"]["counts"] == {"rows": 2, "accepted": 2}
    sources = read_record("paraphrase", tmp_path)["sources"]
    assert [p for p in sources if "output_responses" in p] == []
    assert sources and all("constitution_inputs" in p for p in sources)
    assert not paths.paraphrases_outputs_csv(tmp_path).exists()


# --- the short-text skip -----------------------------------------------------

_SHORT = "a five word input here"
_LONG = "a fifteen word input that is long enough to be sent to the paraphraser now"


def _recording_paraphrase(patched):
    """Patch the paraphrase call to prefix PARA::; return the texts it got."""
    asked: list[str] = []

    def paraphrase(client, texts, **kw):
        asked.extend(texts)
        return [f"PARA::{t}" for t in texts]

    patched.setattr(CP, "paraphrase_batch", paraphrase)
    return asked


def test_a_short_text_is_acked_skipped_with_no_call(tmp_path, patched):
    from redact.dataset import Manifest

    write_inputs(paths.constitution_inputs_dir(tmp_path), [_SHORT, _LONG])
    asked = _recording_paraphrase(patched)
    df = generate_paraphrases(data_dir=tmp_path, min_words=12, verbose=False)["inputs"]
    assert asked == [_LONG]
    assert list(df["sample"]) == [f"PARA::{_LONG}"]
    assert _acks(tmp_path) == [
        {"input_id": content_id(_SHORT), "iteration": 0,
         "paraphrase_model": "venice-paraphraser", "status": "skipped",
         "reason": "short", "min_words": 12},
        {"input_id": content_id(_LONG), "iteration": 0,
         "paraphrase_model": "venice-paraphraser", "status": "accepted"}]
    planned = Manifest.sidecar(paths.paraphrases_inputs_csv(tmp_path)).load()
    assert {r["input_id"]: r["status"] for r in planned} == {
        content_id(_SHORT): "skipped", content_id(_LONG): "planned"}

    del asked[:]
    generate_paraphrases(data_dir=tmp_path, min_words=12, verbose=False)
    assert asked == []                    # a resume calls nothing for either


def test_a_resume_under_a_lower_threshold_keeps_the_skip_and_warns(
        tmp_path, patched, caplog):
    import logging

    write_inputs(paths.constitution_inputs_dir(tmp_path), [_SHORT, _LONG])
    asked = _recording_paraphrase(patched)
    generate_paraphrases(data_dir=tmp_path, min_words=12, verbose=False)
    del asked[:]
    with caplog.at_level(logging.WARNING):
        generate_paraphrases(data_dir=tmp_path, min_words=0, verbose=False)
    assert asked == []
    (warning,) = [r for r in caplog.records if "earlier min_words" in r.getMessage()]
    assert warning.levelname == "WARNING" and "1 units" in warning.getMessage()
    assert "resume: false" in warning.getMessage()
    assert "skipped" not in warning.getMessage()    # counted by other tests


def test_a_target_of_short_texts_only_writes_acks_and_gives_the_lock_back(
        tmp_path, patched):
    write_inputs(paths.constitution_inputs_dir(tmp_path), [_SHORT, "two words"])
    asked = _recording_paraphrase(patched)
    out = paths.paraphrases_inputs_csv(tmp_path)
    df = generate_paraphrases(data_dir=tmp_path, min_words=12, verbose=False)["inputs"]
    assert asked == [] and df.empty and not out.exists()
    assert [ack["status"] for ack in _acks(tmp_path)] == ["skipped", "skipped"]
    assert lock_is_free(out.with_name("paraphrases_inputs.state.jsonl"))


def test_a_skipped_response_keeps_its_keys_and_resumes(tmp_path, patched):
    responses = _responses(output_response=["too short", _LONG, "also too short"])
    asked = _recording_paraphrase(patched)
    generate_paraphrases(outputs=responses, data_dir=tmp_path, target="outputs",
                         min_words=12, verbose=False)
    acks = [json.loads(line) for line in
            _outputs_state(tmp_path).read_text(encoding="utf-8").splitlines()]
    skipped = [ack for ack in acks if ack.get("status") == "skipped"]
    assert [(a["response_id"], a["input_id"]) for a in skipped] == [
        ("out-a", "in-1"), ("out-c", "in-2")]
    assert asked == [_LONG]
    del asked[:]
    # The resume check of the outputs target passes on these acks.
    generate_paraphrases(outputs=responses, data_dir=tmp_path, target="outputs",
                         min_words=12, verbose=False)
    assert asked == []


def test_a_run_record_counts_the_skipped_units(tmp_path, patched):
    from redact import run_pipeline
    from redact.telemetry import uninstall

    write_inputs(paths.constitution_inputs_dir(tmp_path), [_SHORT, _LONG])
    _recording_paraphrase(patched)
    recipe = {"dataset_type": "training", "data_dir": str(tmp_path),
              "stages": ["paraphrase"]}
    try:
        summary = run_pipeline(recipe, params={"paraphrase": {"min_words": 12}},
                               verbose=False)
        assert summary["stages"]["paraphrase"]["counts"] == {
            "rows": 1, "accepted": 1, "skipped": {"short": 1}}
        summary = run_pipeline({**recipe, "resume": False},
                               params={"paraphrase": {"min_words": 0}}, verbose=False)
    finally:
        uninstall()
    assert "skipped" not in summary["stages"]["paraphrase"]["counts"]


def test_a_resume_after_an_all_skipped_target_calls_nothing_and_a_fresh_run_sends_them(
        tmp_path, patched):
    write_inputs(paths.constitution_inputs_dir(tmp_path), [_SHORT, "two words"])
    asked = _recording_paraphrase(patched)
    generate_paraphrases(data_dir=tmp_path, min_words=12, verbose=False)
    generate_paraphrases(data_dir=tmp_path, min_words=12, verbose=False)
    assert asked == []
    generate_paraphrases(data_dir=tmp_path, min_words=0, resume=False, verbose=False)
    assert sorted(asked) == sorted([_SHORT, "two words"])
    assert [ack["status"] for ack in _acks(tmp_path)] == ["accepted", "accepted"]


def test_one_skip_ack_is_one_unit_however_many_rows_share_its_key(
        tmp_path, patched, caplog):
    import logging

    source = pd.DataFrame({"sample": [_SHORT, _SHORT]})   # one input_id
    out = tmp_path / "paraphrased.csv"

    def run(min_words):
        return CP.run_paraphrase_target(
            "inputs", source, out, pool=["venice-paraphraser"], check=None, k=1,
            batch_size=8, seed=0, verbose=False, min_words=min_words)

    run(12)
    with caplog.at_level(logging.WARNING):
        run(0)
    (warning,) = [r for r in caplog.records if "earlier min_words" in r.getMessage()]
    assert "] 1 units" in warning.getMessage()


def test_on_resume_the_manifest_says_what_the_acks_say(tmp_path, patched):
    from redact.dataset import Manifest

    write_inputs(paths.constitution_inputs_dir(tmp_path), [_SHORT, _LONG])
    _recording_paraphrase(patched)
    generate_paraphrases(data_dir=tmp_path, min_words=12, verbose=False)
    # _SHORT stays acked skipped, _LONG accepted, whatever the threshold says now.
    generate_paraphrases(data_dir=tmp_path, min_words=40, verbose=False)
    planned = Manifest.sidecar(paths.paraphrases_inputs_csv(tmp_path)).load()
    assert {r["input_id"]: r["status"] for r in planned} == {
        content_id(_SHORT): "skipped", content_id(_LONG): "planned"}


def test_a_run_record_counts_the_skips_of_the_file_the_params_name(tmp_path, patched):
    from redact import run_pipeline
    from redact.telemetry import uninstall

    write_inputs(paths.constitution_inputs_dir(tmp_path), [_SHORT, _LONG])
    _recording_paraphrase(patched)
    elsewhere = tmp_path / "elsewhere" / "paraphrased.csv"
    try:
        summary = run_pipeline(
            {"dataset_type": "training", "data_dir": str(tmp_path),
             "stages": ["paraphrase"]},
            params={"paraphrase": {"min_words": 12, "inputs_path": str(elsewhere)}},
            verbose=False)
    finally:
        uninstall()
    assert summary["stages"]["paraphrase"]["counts"]["skipped"] == {"short": 1}
    assert not paths.paraphrases_inputs_csv(tmp_path).exists()


def test_a_check_prompt_that_does_not_build_stops_before_anything_is_cleared(
        tmp_path, patched):
    _seed_inputs(tmp_path)
    generate_paraphrases(data_dir=tmp_path, verbose=False)
    out = paths.paraphrases_inputs_csv(tmp_path)
    before = out.read_bytes()

    def broken(prompt_dir=None):
        raise ValueError("the check prompt does not render")

    patched.setattr(CP, "build_paraphrase_checker", broken)
    with pytest.raises(ValueError, match="does not render"):
        generate_paraphrases(data_dir=tmp_path, resume=False, verbose=False)
    assert out.read_bytes() == before
    assert lock_is_free(out.with_name("paraphrases_inputs.state.jsonl"))


def test_verdicts_go_back_to_their_units_past_a_gated_one(tmp_path, patched):
    write_inputs(paths.constitution_inputs_dir(tmp_path),
                 ["alpha prompt", "beta prompt", "gamma prompt"])
    patched.setattr(CP, "paraphrase_batch", lambda client, texts, **kw: [
        t + _RUN_ON if t.startswith("alpha") else f"PARA::{t}" for t in texts])
    patched.setattr(CP, "batch_check_samples", lambda cc, payloads, checker, **kw: [
        (False, "No: beta changed") if "beta" in t else (True, "") for t in payloads])
    df = generate_paraphrases(data_dir=tmp_path, verbose=False)["inputs"]
    by_input = {text: row for text in ("alpha", "beta", "gamma")
                for _, row in df.iterrows() if row["sample"].removeprefix(
                    "PARA::").startswith(text)}
    assert by_input["alpha"]["reasoning"].startswith("length ratio")
    assert by_input["beta"]["reasoning"] == "No: beta changed"
    assert [bool(by_input[t]["accepted"]) for t in ("alpha", "beta", "gamma")] == [
        False, False, True]


def test_the_stage_takes_min_words_as_a_param(tmp_path):
    from redact.recipe import _DEFAULT_MODELS
    from redact.stages import STAGES, RunContext

    ctx = RunContext(
        data_dir=tmp_path, dataset_type="training", models=dict(_DEFAULT_MODELS),
        prompt_dir=None, taxonomy="harmful", taxonomy_dir=None,
        params={"paraphrase": {"min_words": 0}}, augmentations={},
        resume=True, verbose=False)
    STAGES["paraphrase"].check(ctx)
    assert STAGES["paraphrase"].kwargs(ctx)["min_words"] == 0


def test_the_inputs_target_beside_an_old_outputs_folder_resumes_as_before(
        tmp_path, patched):
    _seed_inputs(tmp_path)
    first = generate_paraphrases(data_dir=tmp_path, target="inputs",
                                 verbose=False)["inputs"]
    _outputs_state(tmp_path).write_text(
        '{"input_id": "in-1", "iteration": 0, "paraphrase_model": "m", '
        '"status": "accepted"}\n', encoding="utf-8")
    _no_paraphrase(patched)
    df = generate_paraphrases(data_dir=tmp_path, target="inputs",
                              verbose=False)["inputs"]
    assert "response_id" not in df.columns
    pd.testing.assert_frame_equal(df, first)
