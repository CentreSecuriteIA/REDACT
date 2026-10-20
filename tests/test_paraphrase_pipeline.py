"""Offline tests for generate_paraphrases (mapping/resume/dedup/check-drop).

Every model is a ParaphraseBackend (tests/conftest.py), which answers a
paraphrase and a check prompt by deterministic rules, so this exercises
planning, the unit pool, dedup, resume and both targets without any network
access.
"""

import json

import pandas as pd
import pytest

import redact.content_moderation.paraphrase as CP
from redact import generate_paraphrases, paths
from redact.dataset import content_id
from tests.conftest import (
    Killed,
    ParaphraseBackend,
    fake_vllm_modules,
    kill_after_rows,
    lock_is_free,
    make_client,
    pinned_pool,
    write_inputs,
)


@pytest.fixture(autouse=True)
def _no_short_text_skip(monkeypatch):
    """The stub texts are a few words long: the paraphrase stage's short-text
    skip would leave every unit unparaphrased. Its own tests set min_words."""
    monkeypatch.setattr(CP, "MIN_WORDS", 0)


def _serve(monkeypatch, paraphrase=None, check=None) -> list[dict]:
    """Serve every model with a ParaphraseBackend (by default ``PARA::<text>``,
    and every check says Yes); return the shared log of its requests."""
    requests: list[dict] = []
    monkeypatch.setattr(CP.ModelClient, "create", lambda model, **kw: make_client(
        ParaphraseBackend(paraphrase=paraphrase, check=check, requests=requests),
        model))
    return requests


def _asked(requests) -> list[str]:
    """The texts the paraphrasers were sent, in order."""
    return [r["sample"] for r in requests if r["kind"] == "paraphrase"]


def _checks(requests) -> list[dict]:
    """The check requests, in order."""
    return [r for r in requests if r["kind"] == "check"]


def _saves_every(monkeypatch, units: int) -> None:
    """Save every ``units`` finished units, not the stage's 32, so a test of
    a few units sees several saves."""
    real = CP.CommitBuffer
    monkeypatch.setattr(CP, "CommitBuffer", lambda **kw: real(**{**kw, "every": units}))


@pytest.fixture()
def patched(monkeypatch):
    """Every model a ParaphraseBackend: a paraphrase is ``PARA::<text>``, and
    every check says Yes."""
    _serve(monkeypatch)
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


def test_a_paraphrase_equal_to_its_padded_original_stripped_is_dropped(
        tmp_path, patched):
    _serve(patched, paraphrase=lambda model, text: text)   # the stripped sample
    out = tmp_path / "paraphrased.csv"
    CP.run_paraphrase_target(
        "inputs", pd.DataFrame({"sample": ["  alpha prompt \n"]}), out,
        pool=["venice-paraphraser"], check=None, k=1, batch_size=8, seed=0,
        verbose=False)
    assert not out.exists()
    (ack,) = [json.loads(line) for line in
              out.with_name("paraphrased.state.jsonl").read_text().splitlines()
              if "status" in line]
    assert ack["status"] == "dropped_dedup"


def test_a_broken_humanizer_prompt_stops_the_stage_before_any_client(
        tmp_path, patched):
    from redact.llms.prompting import load_prompt

    _seed_inputs(tmp_path)
    broken = load_prompt("paraphrase", "generation/humanizer-12b")
    broken["system_prompt"] += " Keep it short."
    folder = tmp_path / "prompts" / "paraphrase" / "generation" / "humanizer-12b"
    folder.mkdir(parents=True)
    (folder / "template.json").write_text(json.dumps(broken), encoding="utf-8")
    patched.setattr(CP.ModelClient, "create",
                    lambda *a, **k: pytest.fail("a client was created"))
    with pytest.raises(ValueError, match="does not render"):
        generate_paraphrases(data_dir=tmp_path, paraphraser="humanizer-12b",
                             prompt_dir=tmp_path / "prompts", verbose=False)
    assert not paths.paraphrases_inputs_csv(tmp_path).exists()


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
    _serve(monkeypatch, check=lambda original, paraphrase: "No")
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
    # identical -> the 2nd unit to finish is a dedup-drop
    requests = _serve(monkeypatch, paraphrase=lambda model, text: "SAME")

    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    # 2 units attempted (1 kept, 1 deduped-dropped)
    assert sorted(_asked(requests)) == ["alpha prompt", "beta prompt"]
    state = paths.paraphrases_inputs_csv(tmp_path).with_name("paraphrases_inputs.state.jsonl")
    assert state.exists()  # ledger recorded both attempts, including the drop

    # Resume: the dropped unit is in the ledger, so nothing is re-attempted.
    generate_paraphrases(data_dir=tmp_path, target="inputs", resume=True, verbose=False)
    assert sorted(_asked(requests)) == ["alpha prompt", "beta prompt"]  # no new calls


def test_prompt_dir_threads_to_paraphrase_and_check(tmp_path, monkeypatch):
    _seed_inputs(tmp_path)
    seen = {}
    _serve(monkeypatch)
    prompt, checker = CP.paraphrase_prompt, CP.build_paraphrase_checker

    def paraphrase_prompt(model, prompt_dir=None):
        seen["para"] = prompt_dir
        return prompt(model, prompt_dir)

    def build_paraphrase_checker(prompt_dir=None):
        seen["check"] = prompt_dir
        return checker(prompt_dir=prompt_dir)

    monkeypatch.setattr(CP, "paraphrase_prompt", paraphrase_prompt)
    monkeypatch.setattr(CP, "build_paraphrase_checker", build_paraphrase_checker)
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
    pinned_pool(patched, "venice-paraphraser", "_para_second")
    created: list = []     # (model, weak reference to its client)

    def create(model):
        # A preload finishes during the stage: its engine must be kept too.
        patched.setitem(vllm_module._engines, late, object())
        client = make_client(ParaphraseBackend(), model)
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
    pinned_pool(patched, "venice-paraphraser", "_para_small")
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
    pinned_pool(patched, "venice-paraphraser", "_para_small")
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

    # The fake engines take no request: a unit returns its paraphrase at once.
    def unit(client, check, checker, template, original, **kw):
        return f"PARA::{original}", True, ""
        yield   # a generator, as the pool takes one

    monkeypatch.setattr(CP, "_unit_gen", unit)
    monkeypatch.setattr(
        residency, "plan_residency",
        lambda models, copies=None: SimpleNamespace(sequential=True))
    monkeypatch.setattr(residency, "apply_plan", lambda plan, replace=True: None)

    register_model("_para_second", backend_type="vllm", roles=["paraphraser"],
                   vllm=VLLMConfig(hf_model_id="org/second"))
    pinned_pool(monkeypatch, "venice-paraphraser", "_para_second")
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


# --- how finished units are saved ---------------------------------------------

_FOUR = ["alpha prompt", "beta prompt", "gamma prompt", "delta prompt"]


# The order the four units finish in, at batch_size 2: gamma's unchanged
# paraphrase needs no check call, so it finishes first; then the other
# three, as their check calls return.
_FINISHED = ["gamma prompt", "alpha prompt", "beta prompt", "delta prompt"]


def _seed_four_inputs(tmp_path):
    write_inputs(paths.constitution_inputs_dir(tmp_path), _FOUR)
    # gamma comes back unchanged, so its unit is dropped.
    return lambda model, text: text if text.startswith("gamma") else f"PARA::{text}"


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


def test_a_save_is_one_append_followed_by_its_acks(tmp_path, patched):
    _serve(patched, paraphrase=_seed_four_inputs(tmp_path))
    _saves_every(patched, 2)
    saves, append = [], CP.append_rows

    def spy(path, rows):
        saves.append(([r["sample"] for r in rows], len(_acks(tmp_path))))
        return append(path, rows)

    patched.setattr(CP, "append_rows", spy)
    df = generate_paraphrases(
        data_dir=tmp_path, target="inputs", batch_size=2, verbose=False)["inputs"]

    # Per append: its rows, and the acks on disk when it was made. The first
    # save is gamma's drop and alpha's row.
    assert saves == [
        (["PARA::alpha prompt"], 0), (["PARA::beta prompt", "PARA::delta prompt"], 2)]
    assert list(df["sample"]) == [
        "PARA::alpha prompt", "PARA::beta prompt", "PARA::delta prompt"]
    statuses = ["dropped_dedup", "accepted", "accepted", "accepted"]
    assert _acks(tmp_path) == [
        {"input_id": content_id(text), "iteration": 0,
         "paraphrase_model": "venice-paraphraser", "status": status}
        for text, status in zip(_FINISHED, statuses)
    ]


def test_a_run_killed_between_rows_and_acks_leaves_each_unit_once(tmp_path, patched):
    _serve(patched, paraphrase=_seed_four_inputs(tmp_path))
    _saves_every(patched, 2)
    out = paths.paraphrases_inputs_csv(tmp_path)
    state = out.with_name("paraphrases_inputs.state.jsonl")

    kill_after_rows(patched, CP, on_call=2)
    with pytest.raises(Killed):
        generate_paraphrases(
            data_dir=tmp_path, target="inputs", batch_size=2, verbose=False)
    # The second save's rows landed; its acks did not.
    assert list(pd.read_csv(out)["sample"]) == [
        "PARA::alpha prompt", "PARA::beta prompt", "PARA::delta prompt"]
    assert [ack["input_id"] for ack in _acks(tmp_path)] == [
        content_id(text) for text in _FINISHED[:2]]
    assert lock_is_free(state)

    df = generate_paraphrases(
        data_dir=tmp_path, target="inputs", batch_size=2, verbose=False)["inputs"]
    assert list(df["sample"]) == [
        "PARA::alpha prompt", "PARA::beta prompt", "PARA::delta prompt"]
    cut = out.with_name("paraphrases_inputs.csv.cut")
    assert b"PARA::delta prompt" in cut.read_bytes()
    statuses = ["dropped_dedup", "accepted", "accepted", "accepted"]
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


def test_a_failed_append_acks_none_of_its_save(tmp_path, patched):
    _serve(patched, paraphrase=_seed_four_inputs(tmp_path))
    _saves_every(patched, 2)
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
        content_id(text) for text in _FINISHED[:2]]
    saved = pd.read_csv(paths.paraphrases_inputs_csv(tmp_path))
    assert list(saved["sample"]) == ["PARA::alpha prompt"]


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

    def paraphrase(model, text):
        paraphrased.append(text)
        return f"PARA{len(paraphrased) - 1}::{text}"

    _serve(patched, paraphrase=paraphrase)
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
    def paraphrase(model, text):
        raise AssertionError("a paraphrase was asked for")
    _serve(patched, paraphrase=paraphrase)


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


_RUN_ON = " and then it goes on into a long story nobody asked for"


def test_a_gated_paraphrase_is_a_rejected_row_and_is_not_checked(tmp_path, patched):
    from redact.llm_pipeline import CHECK_TEMPERATURE

    _seed_inputs(tmp_path)
    requests = _serve(patched, paraphrase=lambda model, text: (
        text + _RUN_ON if text.startswith("beta") else f"PARA::{text}"))
    df = generate_paraphrases(data_dir=tmp_path, target="inputs",
                              verbose=False)["inputs"]

    by_text = df.set_index("sample")
    assert bool(by_text.loc["PARA::alpha prompt", "accepted"])
    gated = by_text.loc["beta prompt" + _RUN_ON]
    assert not bool(gated["accepted"])
    assert gated["reasoning"].startswith("length ratio")
    # In the order the units finished: the gated one needed no check call.
    assert {ack["input_id"]: ack["status"] for ack in _acks(tmp_path)} == {
        content_id("alpha prompt"): "accepted", content_id("beta prompt"): "rejected"}
    # The check model saw the ungated paraphrase alone, at the check temperature.
    assert [(r["original"], r["paraphrase"], r["max_tokens"], r["temperature"])
            for r in _checks(requests)] == [
        ("alpha prompt", "PARA::alpha prompt", None, CHECK_TEMPERATURE)]


def test_a_chunk_whose_paraphrases_are_all_gated_makes_no_check_call(tmp_path, patched):
    _seed_inputs(tmp_path)
    requests = _serve(patched, paraphrase=lambda model, text: text + _RUN_ON)
    generate_paraphrases(data_dir=tmp_path, target="inputs", verbose=False)
    assert _checks(requests) == []
    assert [ack["status"] for ack in _acks(tmp_path)] == ["rejected", "rejected"]


def test_an_unchanged_or_repeated_paraphrase_gets_no_check_call(tmp_path, patched):
    """It is dropped as before (dropped_dedup, no row), with no check call."""
    requests = _serve(patched, paraphrase=_seed_four_inputs(tmp_path))
    generate_paraphrases(data_dir=tmp_path, verbose=False)
    # gamma came back unchanged: no check request for it.
    assert sorted(r["original"] for r in _checks(requests)) == [
        "alpha prompt", "beta prompt", "delta prompt"]
    assert {ack["input_id"]: ack["status"] for ack in _acks(tmp_path)} == {
        content_id(text): "dropped_dedup" if text.startswith("gamma") else "accepted"
        for text in _FOUR}

    # A new unit whose paraphrase is a text already written: no check either.
    del requests[:]
    generate_paraphrases(
        inputs=pd.DataFrame({"sample_id": ["another-id"], "sample": ["alpha prompt"]}),
        data_dir=tmp_path, verbose=False)
    assert _asked(requests) == ["alpha prompt"] and _checks(requests) == []
    assert _acks(tmp_path)[-1] == {"input_id": "another-id", "iteration": 0,
                                   "paraphrase_model": "venice-paraphraser",
                                   "status": "dropped_dedup"}


def test_a_gated_text_is_written_once(tmp_path, patched):
    """It is a row, so the same text from another unit is a repeat."""
    write_inputs(paths.constitution_inputs_dir(tmp_path), ["alpha prompt"])
    _serve(patched, paraphrase=lambda model, text: text + _RUN_ON)
    with pytest.warns(UserWarning):     # K>1 with a single paraphraser
        df = generate_paraphrases(data_dir=tmp_path, target="inputs",
                                  paraphrases_per_sample=2, verbose=False)["inputs"]
    assert len(df) == 1 and not bool(df["accepted"].iloc[0])
    assert [ack["status"] for ack in _acks(tmp_path)] == ["rejected", "dropped_dedup"]


def test_without_the_check_no_gate_runs(tmp_path, patched):
    _seed_inputs(tmp_path)
    requests = _serve(patched, paraphrase=lambda model, text: text + _RUN_ON)
    patched.setattr(CP, "gate_reason",
                    lambda *a: pytest.fail("a gate ran without the check"))
    df = generate_paraphrases(data_dir=tmp_path, target="inputs", check=False,
                              verbose=False)["inputs"]
    assert bool(df["accepted"].all()) and _checks(requests) == []


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


# --- the unit pool --------------------------------------------------------------

def test_each_paraphrase_request_carries_its_own_reply_cap(tmp_path, patched):
    from redact.llm_pipeline import CHECK_TEMPERATURE

    texts = ["a short text", " ".join(["a much longer text"] * 20)]
    write_inputs(paths.constitution_inputs_dir(tmp_path), texts)
    requests = _serve(patched)
    generate_paraphrases(data_dir=tmp_path, paraphraser="humanizer-12b", verbose=False)
    asked = [r for r in requests if r["kind"] == "paraphrase"]
    assert {r["sample"]: (r["max_tokens"], r["temperature"]) for r in asked} == {
        text: (CP.reply_cap(text, 2048), None) for text in texts}
    checks = _checks(requests)
    assert len(checks) == len(texts)
    assert {(r["max_tokens"], r["temperature"]) for r in checks} == {
        (None, CHECK_TEMPERATURE)}


@pytest.mark.parametrize("paraphraser, ceiling", [
    ("venice-paraphraser", 2000), ("humanizer-12b", 2048)])
def test_a_long_text_is_capped_at_the_paraphrasers_default_max_tokens(
        tmp_path, patched, paraphraser, ceiling):
    from redact.llms.model_config import get_model_config

    text = " ".join(["a long input text"] * 600)       # over 10,000 characters
    assert len(text) > 10_000     # noqa: PLR2004
    write_inputs(paths.constitution_inputs_dir(tmp_path), [text])
    requests = _serve(patched)
    generate_paraphrases(data_dir=tmp_path, paraphraser=paraphraser, verbose=False)
    (asked,) = [r for r in requests if r["kind"] == "paraphrase"]
    assert asked["max_tokens"] == ceiling == get_model_config(
        paraphraser).default_max_tokens


def test_a_check_prompt_too_long_on_the_async_lane_acks_its_unit(
        tmp_path, patched):
    from redact.exceptions import PromptTooLongError
    from redact.llms.backends import ComputeConfig

    class Lane(ParaphraseBackend):
        """On the async lane; the check of a beta paraphrase is too long."""

        compute_config = ComputeConfig(supports_async_calls=True)

        async def agenerate(self, messages, *, max_tokens=None, temperature=None,
                            internals_id=None, **kwargs):
            content = messages[-1]["content"]
            if content.startswith("ORIGINAL:\n") and "beta" in content:
                raise PromptTooLongError("the check of beta is over the context")
            return self._answer(messages, max_tokens, temperature)

    patched.setattr(CP.ModelClient, "create", lambda m, **kw: make_client(Lane(), m))
    _seed_inputs(tmp_path)
    df = generate_paraphrases(data_dir=tmp_path, verbose=False)["inputs"]
    assert list(df["sample"]) == ["PARA::alpha prompt"]
    assert {ack["input_id"]: ack["status"] for ack in _acks(tmp_path)} == {
        content_id("alpha prompt"): "accepted",
        content_id("beta prompt"): "prompt_too_long"}


def test_each_paraphrasers_units_are_saved_before_the_next_is_loaded(
        tmp_path, patched):
    from types import SimpleNamespace

    from redact import residency
    from redact.dataset import Manifest
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model

    register_model("_para_second", backend_type="vllm", roles=["paraphraser"],
                   vllm=VLLMConfig(hf_model_id="org/second"))
    pinned_pool(patched, "venice-paraphraser", "_para_second")
    saved_at_unload = []
    patched.setattr(residency, "plan_residency",
                    lambda models, copies=None: SimpleNamespace(sequential=True))
    patched.setattr(residency, "apply_plan", lambda plan, replace=True: None)
    patched.setattr(residency, "unload_local", lambda keep=None, engines=None: (
        saved_at_unload.append([ack["paraphrase_model"] for ack in _acks(tmp_path)])))
    _seed_inputs(tmp_path)
    try:
        generate_paraphrases(data_dir=tmp_path, paraphraser="distribution",
                             paraphrases_per_sample=2, verbose=False)
    finally:
        MODEL_REGISTRY.pop("_para_second", None)
    planned = Manifest.sidecar(paths.paraphrases_inputs_csv(tmp_path)).load()
    first = planned[0]["paraphrase_model"]
    # At the one unload, every unit of the first paraphraser was acked on disk.
    assert saved_at_unload == [
        [first] * sum(r["paraphrase_model"] == first for r in planned)]


def test_a_prompt_too_long_on_the_async_lane_acks_its_unit_and_the_others_finish(
        tmp_path, patched, caplog):
    import logging

    from redact.exceptions import PromptTooLongError
    from redact.llms.backends import ComputeConfig

    class Lane(ParaphraseBackend):
        """On the async lane; the paraphrase of a beta text is too long."""

        compute_config = ComputeConfig(supports_async_calls=True)

        async def agenerate(self, messages, *, max_tokens=None, temperature=None,
                            internals_id=None, **kwargs):
            content = messages[-1]["content"]
            if "beta" in content and not content.startswith("ORIGINAL:\n"):
                raise PromptTooLongError("beta is over the context")
            return self._answer(messages, max_tokens, temperature)

    patched.setattr(CP.ModelClient, "create", lambda m, **kw: make_client(Lane(), m))
    _seed_inputs(tmp_path)
    with caplog.at_level(logging.WARNING):
        df = generate_paraphrases(data_dir=tmp_path, verbose=False)["inputs"]
    assert list(df["sample"]) == ["PARA::alpha prompt"]
    assert {ack["input_id"]: ack["status"] for ack in _acks(tmp_path)} == {
        content_id("alpha prompt"): "accepted",
        content_id("beta prompt"): "prompt_too_long"}
    (warning,) = [r for r in caplog.records if "too long" in r.getMessage()]
    assert "skipped" not in warning.getMessage()    # counted by other tests
    # Acked: a resume does not send it again.
    generate_paraphrases(data_dir=tmp_path, verbose=False)
    assert [ack["status"] for ack in _acks(tmp_path)].count("prompt_too_long") == 1


def test_each_paraphrasers_units_go_to_the_pool_with_both_clients(tmp_path, patched):
    seen = {}
    real_drive, real_size = CP.drive_units, CP.pool_size

    def pool_size(batch_size, *clients):
        seen["clients"] = [client.model for client in clients]
        return real_size(batch_size, *clients)

    def drive_units(source, **kwargs):
        seen.update(progress=kwargs["progress"], cap=kwargs["cap"])
        return real_drive(source, **kwargs)

    patched.setattr(CP, "pool_size", pool_size)
    patched.setattr(CP, "drive_units", drive_units)
    _seed_inputs(tmp_path)
    generate_paraphrases(data_dir=tmp_path, batch_size=4, verbose=True)
    assert seen == {"clients": ["venice-paraphraser", "venice-uncensored"],
                    "progress": "paraphrase:inputs (venice-paraphraser)", "cap": 4}


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
    """Serve paraphrases ``PARA::<text>``; return the texts asked for."""
    asked: list[str] = []

    def paraphrase(model, text):
        asked.append(text)
        return f"PARA::{text}"

    _serve(patched, paraphrase=paraphrase)
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
    _serve(patched,
           paraphrase=lambda model, text: (
               text + _RUN_ON if text.startswith("alpha") else f"PARA::{text}"),
           check=lambda original, paraphrase: (
               "No: beta changed" if "beta" in paraphrase else "Yes"))
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
