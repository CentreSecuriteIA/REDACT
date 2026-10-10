"""Tests for the config-driven run layer (recipe/params/manifests + driver).

The end-to-end driver test uses the jailbreaks + build stages with
``pure_only`` augmentations, so no technique needs an LLM — it exercises stage
dispatch, manifest writing, and cross-stage chaining fully offline.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from redact import paths, telemetry
from redact.dataset.io import append_samples
from redact.runconfig import (
    load_params,
    load_recipe,
    read_manifest,
    run_pipeline,
    write_manifest,
)

# --- manifests ---------------------------------------------------------------

def test_manifest_roundtrip(tmp_path):
    write_manifest(
        "inputs", data_dir=tmp_path,
        params={"samples_per_category": 15}, models={"gen": None},
        counts={"rows": 42, "accepted": 30}, sources=["a.csv"],
    )
    m = read_manifest("inputs", tmp_path)
    assert m["stage"] == "inputs"
    assert m["counts"] == {"rows": 42, "accepted": 30}
    assert m["params"]["samples_per_category"] == 15
    assert "timestamp" in m
    # lands under Datasets/
    assert (paths.datasets(tmp_path) / "inputs.run.json").exists()


def test_read_missing_manifest_returns_none(tmp_path):
    assert read_manifest("outputs", tmp_path) is None


def test_write_manifest_with_extra_and_spec(tmp_path):
    write_manifest("jailbreaks", data_dir=tmp_path, extra={"note": "hi"}, spec_version="1.0")
    m = read_manifest("jailbreaks", tmp_path)
    assert m["note"] == "hi" and m["spec_version"] == "1.0"


def test_counts_none_is_empty():
    from redact.runconfig import _counts
    assert _counts(None) == {}


def test_run_pipeline_loads_params_file(tmp_path, monkeypatch):
    import redact
    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    (tmp_path / "params.json").write_text(json.dumps({"build": {}}))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "dataset_type": "eval", "data_dir": str(tmp_path),
        "stages": ["build"], "params_file": "params.json",
    }))
    from redact import run_pipeline
    summary = run_pipeline(str(recipe), verbose=False)  # params=None -> load params_file
    assert "build" in summary["stages"]


# --- recipe / params loading -------------------------------------------------

def test_load_recipe_defaults_and_base_dir(tmp_path):
    p = tmp_path / "recipe.json"
    p.write_text(json.dumps({"dataset_type": "eval", "data_dir": "x"}))
    r = load_recipe(p)
    assert r["stages"] == ["inputs", "outputs", "jailbreaks", "build"]
    assert r["_base_dir"] == str(tmp_path)


def test_load_recipe_rejects_bad_dataset_type():
    with pytest.raises(ValueError):
        load_recipe({"dataset_type": "bogus"})


def test_load_recipe_rejects_unknown_stage():
    with pytest.raises(ValueError):
        load_recipe({"dataset_type": "eval", "stages": ["inputs", "nope"]})


def test_eval_cannot_include_constitution_stage():
    with pytest.raises(ValueError):
        load_recipe({"dataset_type": "eval", "stages": ["constitution", "inputs"]})


def test_load_params_resolves_relative_to_base_dir(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"inputs": {"num_seeds": 3}}))
    assert load_params("params.json", base_dir=tmp_path) == {"inputs": {"num_seeds": 3}}
    assert load_params(None) == {}


# --- end-to-end driver (offline) --------------------------------------------

def _seed_inputs(data_dir):
    append_samples(
        [f"prompt number {i} about a topic" for i in range(3)],
        category="Cyber", turn=0, accepted=[True, True, True],
        dataset_dir=paths.datasets(data_dir),
    )


def test_run_pipeline_eval_jailbreaks_and_build(tmp_path):
    _seed_inputs(tmp_path)
    recipe = {
        "dataset_type": "eval",
        "data_dir": str(tmp_path),
        "stages": ["jailbreaks", "build"],
        "augmentations": {
            "pure_only": True,
            "include_hacking": False,
            "include_manipulation": False,
            "include_obfuscation": True,
            "include_requests": False,
            "auto_generate_benign": False,
        },
        "resume": True,
    }
    params = {"jailbreaks": {"seed": 7}}
    summary = run_pipeline(recipe, params=params, verbose=False)

    # jailbreaks stage produced one row per seeded input + a manifest with spec version
    assert summary["stages"]["jailbreaks"]["counts"]["rows"] == 3
    assert paths.jailbreaks_csv(tmp_path).exists()
    jb_manifest = read_manifest("jailbreaks", tmp_path)
    assert jb_manifest["counts"]["rows"] == 3
    assert "spec_version" in jb_manifest
    assert jb_manifest["params"]["pure_only"] is True  # augmentation recorded

    # build stage merged inputs + jailbreaks into the complete dataset + manifest
    assert paths.complete_dataset_csv(tmp_path).exists()
    assert read_manifest("build", tmp_path) is not None
    assert summary["stages"]["build"]["counts"]["rows"] >= 3


def test_run_pipeline_all_stages_mocked(tmp_path, monkeypatch):
    """Cover every stage dispatch (incl. constitution/inputs/outputs/paraphrase) by
    mocking the generate_* functions."""
    import redact

    calls = []

    def mk(name, ret):
        def f(*a, **k):
            calls.append(name)
            return ret
        return f

    const_df = pd.DataFrame({"entry_type": ["harmful"], "source_category": ["c"]})
    monkeypatch.setattr(redact, "generate_constitution", mk("constitution", const_df))
    monkeypatch.setattr(redact, "generate_inputs", mk("inputs", pd.DataFrame({"sample": ["s"], "accepted": [True]})))
    monkeypatch.setattr(redact, "generate_outputs", mk("outputs", pd.DataFrame({"output_response": ["o"], "accepted": [True]})))
    monkeypatch.setattr(redact, "generate_paraphrases", mk("paraphrase", {"inputs": pd.DataFrame({"sample": ["p"]}), "outputs": pd.DataFrame()}))
    monkeypatch.setattr(redact, "generate_jailbreaks", mk("jailbreaks", pd.DataFrame({"jailbreak": ["j"], "accepted": [True]})))
    monkeypatch.setattr(redact, "build_dataset", mk("build", pd.DataFrame({"x": [1, 2]})))

    from redact import run_pipeline
    summary = run_pipeline(
        {"dataset_type": "training", "data_dir": str(tmp_path),
         "stages": ["constitution", "inputs", "outputs", "paraphrase", "jailbreaks", "build"],
         "augmentations": {"include_translation": False}},
        params={"inputs": {"num_categories": 1}, "paraphrase": {"target": "inputs"}},
        verbose=True,
    )
    assert calls == ["constitution", "inputs", "outputs", "paraphrase", "jailbreaks", "build"]
    assert set(summary["stages"]) == {"constitution", "inputs", "outputs", "paraphrase", "jailbreaks", "build"}
    assert summary["stages"]["paraphrase"]["counts"]["rows"] == 1  # concat of the non-empty frame


def test_training_inputs_without_constitution_stage_raises(tmp_path):
    # 'training' + inputs stage but no preceding constitution stage -> clear error.
    recipe = {
        "dataset_type": "training",
        "data_dir": str(tmp_path),
        "stages": ["inputs"],
        "resume": True,
    }
    with pytest.raises(ValueError, match="constitution"):
        run_pipeline(recipe, params={}, verbose=False)


# --- telemetry wiring --------------------------------------------------------

def test_run_pipeline_writes_a_trace_and_summarises(tmp_path, monkeypatch):
    """A run leaves a machine-readable record beside its other sidecars."""
    import redact
    from redact import telemetry

    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    recipe = {
        "dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["build"],
    }
    try:
        summary = run_pipeline(recipe, params={"build": {}}, verbose=False)

        assert "telemetry" in summary
        trace = Path(summary["trace"])
        assert trace.parent.name == "Datasets"     # same home as manifest/state
        events = [json.loads(ln) for ln in trace.read_text(encoding="utf-8").splitlines()]
        stages = [e for e in events if e["ev"] == "stage"]
        assert [e["stage"] for e in stages] == ["build"]
        assert stages[0]["duration_s"] >= 0
    finally:
        telemetry.uninstall()


def test_run_pipeline_reports_residency_before_running(tmp_path, monkeypatch, caplog):
    """The plan is surfaced at minute zero, not after the budget is spent."""
    import logging

    import redact
    from redact import telemetry

    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    recipe = {
        "dataset_type": "eval", "data_dir": str(tmp_path),
        "stages": ["inputs", "build"],
        "models": {"gen": "llama-3.2-3b-debug", "check": "llama-3.2-3b-debug"},
    }
    monkeypatch.setattr(redact, "generate_inputs", lambda **k: pd.DataFrame({"x": [1]}))
    try:
        from redact import residency

        with caplog.at_level(logging.INFO, logger=residency.logger.name):
            with patch("redact.llms.ModelClient.create"):
                run_pipeline(recipe, params={"inputs": {}, "build": {}}, verbose=True)
        assert "[residency]" in caplog.text
        assert "llama-3.2-3b-debug" in caplog.text
    finally:
        telemetry.uninstall()


def test_run_pipeline_reports_prompt_overrides(tmp_path, monkeypatch, caplog):
    """The overlay makes "which prompt ran?" non-obvious, so the setup block
    says. Sits next to the residency plan, under the same verbose flag."""
    import json
    import logging

    import redact

    over = tmp_path / "my_prompts" / "input" / "quality_check"
    over.mkdir(parents=True)
    (over / "template.json").write_text(json.dumps({"system_prompt": "MINE"}))

    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    recipe = {
        "dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["build"],
        "prompt_dir": str(tmp_path / "my_prompts"),
    }
    try:
        with caplog.at_level(logging.INFO, logger="redact.llms.prompting.prompts"):
            run_pipeline(recipe, params={"build": {}}, verbose=True)
        assert "1 overridden" in caplog.text
        assert "input/quality_check" in caplog.text
    finally:
        telemetry.uninstall()


def test_empty_stages_list_is_not_the_default_pipeline(tmp_path):
    """`stages: []` is falsy, so `or` turned "run nothing" into the full
    default pipeline — the opposite of what was asked for, silently."""
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": []}
    assert load_recipe(recipe)["stages"] == []


# --- residency wiring --------------------------------------------------------

@pytest.fixture()
def local_models():
    """Register throwaway local vLLM models; clean up after."""
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model

    made = []

    def _make(name, roles=None, **kwargs):
        register_model(name, backend_type="vllm", roles=roles,
                       vllm=VLLMConfig(hf_model_id=f"org/{name}", vram_gb=5.0, **kwargs))
        made.append(name)
        return name

    yield _make
    for name in made:
        MODEL_REGISTRY.pop(name, None)


@pytest.fixture(autouse=True)
def _offline_plans():
    """Plans read no HF config, load no model and leave no planned grant."""
    import redact.llms.backends.vllm as vllm_module

    with patch("redact.llms.resources.estimate._load_config", return_value=None), \
         patch("redact.residency.preload", return_value=None):
        yield
    vllm_module._planned_utilization.clear()


ALL_STAGES = ["constitution", "inputs", "outputs", "paraphrase", "jailbreaks", "build"]


def _models(**overrides):
    from redact.runconfig import _DEFAULT_MODELS
    return {**_DEFAULT_MODELS, **overrides}


def test_stage_models_follow_the_order_the_stages_run():
    """Stage order, not role name: what loads first is planned first."""
    from redact.runconfig import _stage_models

    assert _stage_models(_models(), ALL_STAGES, {}) == [
        "claude-opus-4-6",        # constitution
        "venice-uncensored",      # inputs: gen, and check falls back to it
        "venice-paraphraser",     # paraphrase
        "deepseek-v3.2",          # jailbreaks: translation
    ]
    assert _stage_models(_models(), ["jailbreaks", "paraphrase"], {}) == [
        "venice-uncensored", "deepseek-v3.2", "venice-paraphraser",
    ]
    assert _stage_models(_models(), ["build"], {}) == []


def test_check_resolves_to_the_generation_model_like_the_stages(local_models):
    from redact.runconfig import _stage_models

    gen = local_models("_rc_gen")
    check = local_models("_rc_check")
    assert _stage_models(_models(gen=gen), ["inputs"], {}) == [gen]
    assert _stage_models(_models(gen=gen, check=check), ["outputs"], {}) == [gen, check]


@pytest.mark.parametrize("augmentations", [
    {"include_translation": False},
    {"pure_only": True},
    {"include_obfuscation": False},
])
def test_translation_is_not_planned_when_the_recipe_excludes_it(augmentations):
    from redact.runconfig import _stage_models

    assert _stage_models(_models(), ["jailbreaks"], augmentations) == ["venice-uncensored"]
    assert "deepseek-v3.2" in _stage_models(
        _models(), ["jailbreaks"], {"include_translation": True})


def test_a_paraphraser_pool_plans_every_model_in_it(local_models):
    from redact.runconfig import _stage_models

    extra = local_models("_rc_paraphraser", roles=["paraphraser"])
    wanted = _stage_models(_models(paraphraser="distribution"), ["paraphrase"], {})
    assert wanted == ["venice-paraphraser", extra, "venice-uncensored"]


def _one_card(gib):
    """Plan for one fake card of ``gib`` GiB."""
    return (patch("redact.llms.resources.measure.detect_gpus",
                  return_value=("FakeGPU", 1)),
            patch("redact.llms.resources.measure.detect_gpu_memory_gib",
                  return_value=gib))


def _recorded(calls):
    """Patches that record preload and unload_local calls in ``calls``."""
    from redact import residency
    return (patch.object(residency, "preload",
                         side_effect=lambda names, **kw: calls.append(("preload", names))),
            patch.object(residency, "unload_local",
                         side_effect=lambda keep=None, **kw: calls.append(("unload", keep))))


def test_phases_plan_each_stage_and_preload_what_fits(local_models):
    """Two local models that cannot share a card: the paraphrase stage's plan
    says so, and only the model that fits is loaded ahead."""
    from redact.runconfig import _enter_phase, _plan_phases

    gen = local_models("_rc_zz_gen")              # sorts last by role and name
    paraphraser = local_models("_rc_aa_paraphraser")
    models = _models(gen=gen, paraphraser=paraphraser, paraphrase_check=gen)
    calls = []
    card = _one_card(8.0)                         # 5 + 5 GiB is over 0.9 x 8
    rec = _recorded(calls)
    try:
        with card[0], card[1], rec[0], rec[1]:
            phases = _plan_phases(models, ["inputs", "paraphrase"], verbose=False)
            _enter_phase(phases, 0, set(), verbose=False)
    finally:
        telemetry.uninstall()
    assert [p.resident for p in phases] == [[gen], [gen, paraphraser]]
    assert phases[0].plan.fits and not phases[1].plan.fits
    assert phases[1].plan.as_dict()["groups"] == [[gen], [paraphraser]]
    assert calls == [("preload", [gen])]


def test_phases_apply_one_share_per_engine_for_the_run(local_models):
    """gen is alone in the inputs stage and shares the card in the paraphrase
    stage: it loads once, with the smaller share."""
    import redact.llms.backends.vllm as vllm_module
    from redact.runconfig import _plan_phases

    gen = local_models("_rc_share_gen")
    paraphraser = local_models("_rc_share_para")
    models = _models(gen=gen, paraphraser=paraphraser, paraphrase_check=gen)
    card = _one_card(24.0)
    try:
        with card[0], card[1]:
            phases = _plan_phases(models, ["inputs", "paraphrase"], verbose=False)
    finally:
        telemetry.uninstall()
    assert all(p.plan.fits for p in phases)
    # Pool 21.6; needs 5 + 5 leave 11.6, so each gets 10.8 GiB of the 24.
    assert dict(vllm_module._planned_utilization) == {
        vllm_module._engine_key(f"org/{name}", None, {}): 0.45
        for name in (gen, paraphraser)
    }


def test_entering_a_stage_unloads_what_no_later_stage_calls(local_models):
    """gen is done after the inputs stage: it is released at the transition
    and the paraphrase stage's models are loaded."""
    from redact.runconfig import _enter_phase, _plan_phases

    gen = local_models("_rc_done_gen")
    paraphraser = local_models("_rc_done_para")
    check = local_models("_rc_done_check")
    models = _models(gen=gen, paraphraser=paraphraser, paraphrase_check=check)
    calls, warm = [], set()
    card = _one_card(24.0)
    rec = _recorded(calls)
    try:
        with card[0], card[1], rec[0], rec[1]:
            phases = _plan_phases(models, ["inputs", "paraphrase"], verbose=False)
            _enter_phase(phases, 0, warm, verbose=False)
            _enter_phase(phases, 1, warm, verbose=False)
    finally:
        telemetry.uninstall()
    assert calls == [
        ("preload", [gen]),
        ("unload", [paraphraser, check]),
        ("preload", [paraphraser, check]),
    ]
    assert warm == {paraphraser, check}


def test_the_next_stage_loads_during_a_stage_without_local_models(local_models):
    """constitution calls an API model, so the inputs stage's model loads
    while it runs and is not loaded a second time."""
    from redact.runconfig import _enter_phase, _plan_phases

    gen = local_models("_rc_ahead_gen")
    calls, warm = [], set()
    card = _one_card(24.0)
    rec = _recorded(calls)
    try:
        with card[0], card[1], rec[0], rec[1]:
            phases = _plan_phases(_models(gen=gen), ["constitution", "inputs"],
                                  verbose=False)
            _enter_phase(phases, 0, warm, verbose=False)
            _enter_phase(phases, 1, warm, verbose=False)
    finally:
        telemetry.uninstall()
    assert [p.resident for p in phases] == [[], [gen]]
    assert calls == [("preload", [gen])]


def test_plan_phases_returns_nothing_without_local_models():
    from redact.runconfig import _plan_phases

    assert _plan_phases(_models(), ["build"], verbose=False) == []


def test_run_pipeline_puts_the_plan_in_the_summary(tmp_path, monkeypatch, caplog):
    import logging

    import redact
    from redact import residency

    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(redact, "generate_inputs", lambda **k: pd.DataFrame({"x": [1]}))
    recipe = {
        "dataset_type": "eval", "data_dir": str(tmp_path),
        "stages": ["inputs", "build"],
        "models": {"gen": "llama-3.2-3b-debug"},
    }
    try:
        with caplog.at_level(logging.INFO, logger=residency.logger.name), \
             patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 1)), \
             patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=10.0), \
             patch("redact.llms.resources.estimate.estimate_kv_gib",
                   return_value=0.04), \
             patch("redact.llms.ModelClient.create"):
            summary = run_pipeline(recipe, params={"inputs": {}, "build": {}}, verbose=True)
    finally:
        telemetry.uninstall()
    assert summary["residency"]["fits"] is True
    inputs, build = summary["residency"]["phases"]
    assert (inputs["stage"], build["resident"]) == ("inputs", [])
    plan = inputs
    assert plan["fits"] is True and plan["warnings"] == []
    assert plan["groups"] == [["llama-3.2-3b-debug"]]
    assert plan["models"][0]["devices"] == [0]
    json.dumps(summary["residency"])                 # plain data
    assert "WARNING" not in caplog.text
    # 6.04 x 1.1 + 0.6 = 7.24 GiB, inside the 9.0 a lone engine is granted.
    assert "card 0, vLLM gets 9.0GiB (gpu_memory_utilization=0.90)" in caplog.text

    api_only = run_pipeline(
        {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["build"]},
        params={"build": {}}, verbose=False)
    telemetry.uninstall()
    assert "residency" not in api_only


def test_a_planner_exception_does_not_abort_the_run(tmp_path, monkeypatch, caplog):
    import logging

    import redact
    from redact import residency

    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(redact, "generate_inputs", lambda **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(residency, "plan_phases",
                        lambda *a, **k: 1 / 0)
    recipe = {
        "dataset_type": "eval", "data_dir": str(tmp_path),
        "stages": ["inputs", "build"],
        "models": {"gen": "llama-3.2-3b-debug"},
    }
    try:
        with caplog.at_level(logging.ERROR, logger="redact.runconfig"):
            summary = run_pipeline(recipe, params={"inputs": {}, "build": {}},
                                   verbose=False)
    finally:
        telemetry.uninstall()
    assert list(summary["stages"]) == ["inputs", "build"]
    assert "residency" not in summary
    mine = [r for r in caplog.records if r.name == "redact.runconfig"]
    assert [r.levelname for r in mine] == ["ERROR"]
    assert "planning failed (ZeroDivisionError" in caplog.text
    assert "continues without a plan" in caplog.text


def test_run_pipeline_summary_counts_engines_still_loaded(tmp_path, monkeypatch):
    """Nothing is released before the summary is built, so live time counts."""
    import time

    import redact
    import redact.llms.backends.vllm as vllm_module

    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    key = ("org/still-loaded", None, "[]")
    vllm_module._engine_loaded_at[key] = time.perf_counter() - 120.0
    try:
        summary = run_pipeline(
            {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["build"]},
            params={"build": {}}, verbose=False)
    finally:
        vllm_module._engine_loaded_at.pop(key, None)
        telemetry.uninstall()
    assert summary["telemetry"]["local_s"]["org/still-loaded"] == pytest.approx(120.0, abs=5)
