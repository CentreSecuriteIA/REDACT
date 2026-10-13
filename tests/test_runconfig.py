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
from redact.dataset import append_samples
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


class TestFileCounts:
    """The counts of a stage's files are those of its rows read into one frame."""

    def _write(self, path, accepted, **columns):
        frame = pd.DataFrame({"sample": [f"s{n}" for n in range(len(accepted))],
                              "accepted": accepted, **columns})
        frame.to_csv(path, index=False)
        return frame

    def test_rows_and_accepted_rows_of_several_files(self, tmp_path):
        from redact.runconfig import _counts, _file_counts

        files = [tmp_path / "a.csv", tmp_path / "b.csv"]
        frames = [self._write(files[0], [True, False, True]),
                  self._write(files[1], [False, True])]
        assert _file_counts(files) == {"rows": 5, "accepted": 3}
        assert _file_counts(files) == _counts(pd.concat(frames))

    def test_a_file_is_read_in_chunks(self, tmp_path, monkeypatch):
        import redact.dataset.rows as rows
        from redact.runconfig import _file_counts

        monkeypatch.setattr(rows, "_CHUNK_ROWS", 2)
        self._write(tmp_path / "a.csv", [True, False, True, True, False])
        assert _file_counts([tmp_path / "a.csv"]) == {"rows": 5, "accepted": 3}

    def test_a_cell_without_a_verdict_is_a_row_that_is_not_accepted(self, tmp_path):
        from redact.runconfig import _counts, _file_counts

        frame = self._write(tmp_path / "a.csv", [True, None, False, True])
        assert _file_counts([tmp_path / "a.csv"]) == {"rows": 4, "accepted": 2}
        assert _file_counts([tmp_path / "a.csv"]) == _counts(frame)

    def test_files_without_an_accepted_column_give_rows_only(self, tmp_path):
        from redact.runconfig import _file_counts

        pd.DataFrame({"x": [1, 2, 3]}).to_csv(tmp_path / "a.csv", index=False)
        assert _file_counts([tmp_path / "a.csv"]) == {"rows": 3}

    def test_a_missing_file_has_no_rows(self, tmp_path):
        from redact.runconfig import _file_counts

        assert _file_counts([tmp_path / "none.csv"]) == {"rows": 0}
        assert _file_counts([]) == {"rows": 0}

    def test_keep_limits_the_count_to_rows_with_those_values(self, tmp_path):
        from redact.runconfig import _file_counts

        self._write(tmp_path / "a.csv", [True, True, False, True],
                    template_style=["long", "short", "long", "long"])
        keep = {"accepted": True, "template_style": "long"}
        assert _file_counts([tmp_path / "a.csv"], keep) == {"rows": 2, "accepted": 2}

    def test_keep_does_not_limit_a_file_without_the_column(self, tmp_path):
        from redact.runconfig import _file_counts

        self._write(tmp_path / "a.csv", [True, False, True])
        keep = {"accepted": True, "template_style": "long"}
        assert _file_counts([tmp_path / "a.csv"], keep) == {"rows": 2, "accepted": 2}

    def test_rows_that_are_all_dropped_still_report_the_accepted_count(self, tmp_path):
        from redact.runconfig import _file_counts

        self._write(tmp_path / "a.csv", [False, False])
        assert _file_counts([tmp_path / "a.csv"], {"accepted": True}) == {
            "rows": 0, "accepted": 0}


def test_run_pipeline_asks_no_stage_to_read_its_rows_back(tmp_path, monkeypatch):
    """Only the constitution's rows are used: the inputs stage is seeded from them."""
    import redact

    asked = {}

    def stage(name, result):
        def run(**kwargs):
            asked[name] = kwargs.get("read_back")
            return result
        monkeypatch.setattr(redact, name, run)

    stage("generate_constitution", pd.DataFrame({"entry_type": ["harmful"]}))
    for name in ("generate_inputs", "generate_outputs", "generate_jailbreaks",
                 "build_dataset"):
        stage(name, pd.DataFrame())
    stage("generate_paraphrases", {})
    try:
        summary = run_pipeline(
            {"dataset_type": "training", "data_dir": str(tmp_path),
             "stages": ["constitution", "inputs", "outputs", "paraphrase", "jailbreaks",
                        "build"],
             "augmentations": {"include_translation": False}},
            params={}, verbose=False)
    finally:
        telemetry.uninstall()
    assert asked == {
        "generate_constitution": None, "generate_inputs": False,
        "generate_outputs": False, "generate_paraphrases": False,
        "generate_jailbreaks": False, "build_dataset": False}
    assert summary["stages"]["constitution"]["counts"] == {"rows": 1}


def test_run_pipeline_counts_the_files_a_recipe_names(tmp_path, monkeypatch):
    """A stage whose output path the params override is counted there."""
    import redact

    def writes(key, rows):
        def run(**kwargs):
            pd.DataFrame({"accepted": [True] * rows}).to_csv(kwargs[key], index=False)
            return pd.DataFrame()
        return run

    monkeypatch.setattr(redact, "generate_outputs", writes("output_path", 2))
    monkeypatch.setattr(redact, "generate_jailbreaks", writes("output_path", 3))
    monkeypatch.setattr(redact, "build_dataset", writes("output_path", 4))
    monkeypatch.setattr(redact, "generate_paraphrases", writes("outputs_path", 5))
    params = {
        "outputs": {"output_path": str(tmp_path / "o.csv")},
        "jailbreaks": {"output_path": str(tmp_path / "j.csv")},
        "build": {"output_path": str(tmp_path / "b.csv")},
        "paraphrase": {"outputs_path": str(tmp_path / "p.csv"), "target": "outputs"},
    }
    try:
        summary = run_pipeline(
            {"dataset_type": "eval", "data_dir": str(tmp_path),
             "stages": ["outputs", "paraphrase", "jailbreaks", "build"],
             "augmentations": {"include_translation": False}},
            params=params, verbose=False)
    finally:
        telemetry.uninstall()
    rows = {stage: done["counts"]["rows"] for stage, done in summary["stages"].items()}
    assert rows == {"outputs": 2, "paraphrase": 5, "jailbreaks": 3, "build": 4}


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


# --- models per role: a name, or an object that says how it is served ----------

def _serving_of(models):
    """How a recipe with these ``models`` serves each model, once loaded."""
    from redact.runconfig import _serving

    return _serving(load_recipe({"dataset_type": "eval", "models": models})["models"])


def _models_error(models):
    """What load_recipe says against a recipe's ``models``."""
    with pytest.raises(ValueError) as raised:
        load_recipe({"dataset_type": "eval", "models": models})
    return str(raised.value)


def test_names_and_nulls_load_as_written():
    from redact.runconfig import _role_names

    models = {"gen": "venice-uncensored", "check": None,
              "paraphraser": "distribution", "generator": "not-a-role"}
    recipe = load_recipe({"dataset_type": "eval", "models": models})
    assert recipe["models"] == models
    assert _role_names(recipe["models"]) == models
    assert _serving_of(models) == {}


@pytest.mark.parametrize("name", ["eval_example.json", "training_example.json"])
def test_the_example_recipes_load(name):
    recipe = load_recipe(paths.configs_dir() / "runs" / name)
    assert set(recipe["models"].values()) == {None}
    assert _serving_of(recipe["models"]) == {}


def test_the_object_form_is_read_per_model():
    from redact.runconfig import _role_names

    models = {
        "gen": {"name": "venice-uncensored", "backend_type": "vllm", "copies": 2},
        "check": "llama-3.2-3b-debug",
        "translation": {"name": "deepseek-v3.2", "workers": 4},
    }
    recipe = load_recipe({"dataset_type": "eval", "models": models})
    assert recipe["models"] == models             # kept as written
    assert load_recipe(recipe) == recipe          # so it loads again, as run.py does
    assert _role_names(recipe["models"]) == {
        "gen": "venice-uncensored", "check": "llama-3.2-3b-debug",
        "translation": "deepseek-v3.2"}
    assert _serving_of(models) == {
        "venice-uncensored": {"backend_type": "vllm", "copies": 2},
        "deepseek-v3.2": {"workers": 4}}


@pytest.mark.parametrize("models,serving", [
    ({"gen": {"backend_type": "vllm"}},
     {"venice-uncensored": {"backend_type": "vllm"}}),
    ({"translation": {"workers": 1}}, {"deepseek-v3.2": {"workers": 1}}),
    ({"paraphraser": {"copies": 1}}, {"venice-paraphraser": {"copies": 1}}),
    # check without a name is the generation model.
    ({"gen": "llama-3.2-3b-debug", "check": {"copies": 1}},
     {"llama-3.2-3b-debug": {"copies": 1}}),
])
def test_an_object_without_a_name_is_the_roles_default_model(models, serving):
    from redact.runconfig import _role_names

    assert _serving_of(models) == serving
    assert _role_names(models)[list(models)[-1]] is None


@pytest.mark.parametrize("entry", [
    {}, {"name": None}, {"name": "no-such-model"},
    {"name": "venice-uncensored", "backend_type": None, "copies": None,
     "workers": None},
])
def test_an_object_that_gives_no_serving_is_a_name(entry):
    """Nothing is looked up for it, and a null is a key left out."""
    assert _serving_of({"gen": entry}) == {}


def test_an_unknown_key_is_rejected():
    assert _models_error({"gen": {"name": "venice-uncensored", "backend": "vllm"}}) == (
        "models['gen'] has unknown key(s) ['backend']; write only "
        "['name', 'backend_type', 'copies', 'workers'].")


@pytest.mark.parametrize("key", ["copies", "workers"])
@pytest.mark.parametrize("count", [0, -1, 1.5, "2", True])
def test_copies_and_workers_must_be_whole_numbers_of_at_least_one(key, count):
    assert _models_error({"gen": {"name": "venice-uncensored", key: count}}) == (
        f"models['gen']: {key} must be a whole number of at least 1, got "
        f"{count!r}; write one, or leave {key} out.")


@pytest.mark.parametrize("backend_type", ["api", "telepathy", ["vllm"]])
def test_a_backend_type_the_model_has_no_setup_for_is_rejected(backend_type):
    entry = {"name": "llama-3.2-3b-debug", "backend_type": backend_type}
    assert _models_error({"gen": entry}) == (
        f"models['gen']: backend_type {backend_type!r} is not a setup of "
        "'llama-3.2-3b-debug'; write one of ['vllm'].")


def test_a_model_with_two_setups_and_no_default_needs_a_backend_type():
    from redact.llms.model_config import (
        MODEL_REGISTRY,
        APIConfig,
        VLLMConfig,
        register_model,
    )

    name = "_rc_no_default"
    register_model(
        name, vllm=VLLMConfig(hf_model_id=f"org/{name}"),
        api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                      base_url="https://test.example/v1", rpm=10))
    try:
        assert _models_error({"gen": {"name": name, "copies": 2}}) == (
            f"models['gen']: backend_type None is not a setup of {name!r}; "
            "write one of ['api', 'vllm'].")
        local = {"name": name, "backend_type": "vllm", "copies": 2}
        assert _serving_of({"gen": local}) == {name: {"backend_type": "vllm", "copies": 2}}
    finally:
        MODEL_REGISTRY.pop(name, None)


@pytest.mark.parametrize("name,advice", [
    ("venice-uncensored", 'leave copies out, or write "backend_type": "vllm".'),
    ("deepseek-v3.2", "leave copies out."),
])
def test_copies_are_rejected_for_a_model_not_served_by_vllm(name, advice):
    assert _models_error({"gen": {"name": name, "copies": 2}}) == (
        f"models['gen']: copies are for a model served by vLLM, and {name!r} is "
        f"served by its api setup; {advice}")


@pytest.mark.parametrize("entry,advice", [
    ({"name": "venice-uncensored", "backend_type": "vllm", "workers": 4},
     'leave workers out, or write "backend_type": "api".'),
    ({"name": "llama-3.2-3b-debug", "workers": 4}, "leave workers out."),
])
def test_workers_are_rejected_for_a_model_not_served_by_an_api(entry, advice):
    assert _models_error({"gen": entry}) == (
        f"models['gen']: workers are for a hosted model, and {entry['name']!r} is "
        f"served by its vllm setup, where requests at once are automatic; {advice}")


def test_workers_above_one_are_rejected_for_a_series_only_backend():
    opus = {"name": "claude-opus-4-6", "workers": 2}
    assert _models_error({"constitution": opus}) == (
        "models['constitution']: 'claude-opus-4-6' takes one request at a time "
        "(its anthropic backend makes no parallel calls); write \"workers\": 1, "
        "or leave workers out.")
    assert _serving_of({"constitution": {**opus, "workers": 1}}) == {
        "claude-opus-4-6": {"workers": 1}}


def test_a_model_in_two_roles_with_the_same_serving_is_accepted():
    """A role that gives a name alone takes the serving another role gives."""
    local = {"name": "venice-uncensored", "backend_type": "vllm", "copies": 2}
    models = {"gen": local, "paraphrase_check": dict(local),
              "check": "venice-uncensored"}
    assert _serving_of(models) == {
        "venice-uncensored": {"backend_type": "vllm", "copies": 2}}


def test_a_model_in_two_roles_with_different_serving_is_rejected():
    models = {
        "gen": {"name": "venice-uncensored", "backend_type": "vllm", "copies": 2},
        "paraphrase_check": {"name": "venice-uncensored", "backend_type": "vllm"},
    }
    assert _models_error(models) == (
        "models['gen'] and models['paraphrase_check'] both resolve to "
        "'venice-uncensored' but serve it differently ({'backend_type': 'vllm', "
        "'copies': 2} and {'backend_type': 'vllm'}); one model is served one way "
        "in a run, so write the same keys in both, or in one of them only.")


def test_the_paraphraser_pool_takes_no_object():
    for entry in ({"name": "distribution"}, {"name": "distribution", "copies": 1}):
        assert _models_error({"paraphraser": entry}) == (
            "models['paraphraser'] names the pool of all paraphrasers, which takes "
            "no object; write \"distribution\", or the name of one paraphraser.")
    one = {"name": "venice-paraphraser", "copies": 1}
    assert _serving_of({"paraphraser": one}) == {"venice-paraphraser": {"copies": 1}}


def test_an_object_under_an_unknown_role_is_rejected():
    assert _models_error({"generator": {"name": "venice-uncensored"}}) == (
        "models['generator'] is not a model role; write one of ['gen', 'check', "
        "'translation', 'constitution', 'paraphraser', 'paraphrase_check'].")


def test_serving_is_rejected_for_a_model_that_is_not_registered():
    entry = {"name": "no-such-model", "backend_type": "vllm"}
    assert _models_error({"gen": entry}) == (
        "models['gen']: 'no-such-model' is not a registered model, so how it is "
        "served cannot be set; write a registered \"name\".")
    with patch("redact.llms.model_config.default_model_for_role",
               side_effect=KeyError("uncensored_gen")):
        assert _models_error({"gen": {"copies": 2}}) == (
            "models['gen']: the role's default model is not a registered model, "
            "so how it is served cannot be set; write a registered \"name\".")


def test_the_top_level_copies_key_is_rejected_with_the_new_form():
    with pytest.raises(ValueError) as raised:
        load_recipe({"dataset_type": "eval", "copies": {"gen": 2}})
    assert str(raised.value) == (
        'The top-level "copies" key is no longer read; write each count in its '
        'role\'s entry under "models" instead, e.g. '
        '"gen": {"name": "llama-3.2-3b-debug", "copies": 2}.')
    load_recipe({"dataset_type": "eval", "copies": None})     # a null says nothing


def test_a_stage_is_handed_a_model_name_per_role(tmp_path, monkeypatch):
    """Whatever form the recipe writes a role in."""
    import redact

    handed = {}
    monkeypatch.setattr(
        redact, "generate_inputs",
        lambda **k: handed.update(model=k["model"], check_model=k["check_model"])
        or pd.DataFrame({"x": [1]}))
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": {"name": "deepseek-v3.2", "workers": 1}, "check": {}}}
    try:
        run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    assert handed == {"model": "deepseek-v3.2", "check_model": None}
    assert read_manifest("inputs", tmp_path)["models"] == {
        "gen": "deepseek-v3.2", "check": None, "translation": None,
        "constitution": None, "paraphraser": None, "paraphrase_check": None}


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


def test_a_training_run_reads_the_inputs_it_seeded(tmp_path, monkeypatch):
    """The stages after ``inputs`` used to look only at the standalone tree,
    and a training run stopped at the first of them with no inputs found."""
    import redact.content_moderation.paraphrase as CP
    from redact.llms import ModelClient
    from tests.conftest import MockBackend, make_client

    answers = MockBackend(["answer one", "answer two", "answer three"])
    monkeypatch.setattr(ModelClient, "create", lambda m: make_client(answers, m))
    monkeypatch.setattr(
        CP, "paraphrase_batch",
        lambda client, texts, **kw: [f"PARA::{t}" for t in texts])
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda client, texts, checker, **kw: [(True, "") for _ in texts])
    monkeypatch.setattr("redact.llms.resources.estimate._load_config", lambda _id: None)
    append_samples(
        [f"prompt number {i} about a topic" for i in range(3)],
        category="Cyber", turn=0, dataset_dir=paths.constitution_inputs_dir(tmp_path),
    )
    recipe = {
        "dataset_type": "training",
        "data_dir": str(tmp_path),
        "stages": ["outputs", "paraphrase", "jailbreaks"],
        "augmentations": {
            "pure_only": True, "include_hacking": False, "include_manipulation": False,
            "include_requests": False, "auto_generate_benign": False,
        },
    }
    try:
        summary = run_pipeline(
            recipe, params={"outputs": {"check_outputs": False}}, verbose=False)
    finally:
        telemetry.uninstall()
    rows = {stage: done["counts"]["rows"] for stage, done in summary["stages"].items()}
    # Three inputs: a response each, a paraphrase of each input and response,
    # a jailbreak each.
    assert rows == {"outputs": 3, "paraphrase": 6, "jailbreaks": 3}


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
    def paraphrase(*a, **k):
        # As the stage does: its rows go to the target's file, which is counted.
        calls.append("paraphrase")
        out = paths.paraphrases_inputs_csv(tmp_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample": ["p"]}).to_csv(out, index=False)
        return {"inputs": pd.DataFrame(), "outputs": pd.DataFrame()}

    monkeypatch.setattr(redact, "generate_paraphrases", paraphrase)
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
    # The rows of the target's file.
    assert summary["stages"]["paraphrase"]["counts"]["rows"] == 1


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
    """Register throwaway local vLLM models, of 5 GiB unless a size is
    given; clean up after."""
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model

    made = []

    def _make(name, roles=None, gb=5.0, **kwargs):
        register_model(name, backend_type="vllm", roles=roles,
                       vllm=VLLMConfig(hf_model_id=f"org/{name}", vram_gb=gb, **kwargs))
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


def _loader():
    """What the vLLM loader holds, per model name: ``(share, cards)``."""
    import redact.llms.backends.vllm as vllm_module

    assert set(vllm_module._planned_utilization) == set(vllm_module._planned_devices)
    return {key[0].removeprefix("org/"): (share, vllm_module._planned_devices[key])
            for key, share in vllm_module._planned_utilization.items()}


def test_entering_a_stage_hands_the_loader_the_stages_plan(local_models):
    """The loader holds the shares and cards of the stage entered, and of the
    next one when its models load ahead, before any of them is preloaded."""
    from redact import residency
    from redact.runconfig import _enter_phase, _plan_phases

    gen = local_models("_rc_hand_gen")
    paraphraser = local_models("_rc_hand_para")
    check = local_models("_rc_hand_check")
    models = _models(gen=gen, paraphraser=paraphraser, paraphrase_check=check)
    alone, shared = (0.9, (0,)), (0.45, (0,))
    handed, entered, warm = [], [], set()
    card = _one_card(24.0)
    try:
        with card[0], card[1], patch.object(residency, "unload_local"), \
             patch.object(residency, "preload", side_effect=lambda names, **kw:
                          handed.append((names, _loader()))):
            phases = _plan_phases(
                models, ["constitution", "inputs", "paraphrase"], verbose=False)
            planned = _loader()
            for i in range(3):
                _enter_phase(phases, i, warm, verbose=False)
                entered.append(_loader())
    finally:
        telemetry.uninstall()
    assert [p.resident for p in phases] == [[], [gen], [paraphraser, check]]
    assert planned == {gen: alone, paraphraser: shared, check: shared}
    # gen loads during the constitution stage, and a load still under way
    # when the inputs stage is entered finds the same share and card.
    assert entered == [{gen: alone}, {gen: alone},
                       {paraphraser: shared, check: shared}]
    assert handed == [([gen], {gen: alone}),
                      ([paraphraser, check], {paraphraser: shared, check: shared})]


def test_a_model_loaded_again_loads_under_the_plan_of_its_stage(local_models):
    """Stage b does not hold x. Alone in stage a it has both cards and the
    full share; in stage c it shares card 0, and its second load takes that."""
    from redact import residency
    from redact.runconfig import _enter_phase

    w, x, y, z = (local_models(f"_rc_again_{n}") for n in "wxyz")
    plans = [residency.plan_residency(resident, gpus=2, gib_per_gpu=24.0)
             for resident in ([x], [y], [z, w, x])]
    phases = [residency.Phase(stage, [fp.model for fp in plan.footprints], plan)
              for stage, plan in zip("abc", plans)]
    handed, warm = [], set()
    with patch.object(residency, "unload_local"), \
         patch.object(residency, "preload", side_effect=lambda names, **kw:
                      handed.append((names, _loader()))):
        for i in range(3):
            _enter_phase(phases, i, warm, verbose=False)
    assert handed == [
        ([x], {x: (0.9, (0, 1))}),
        ([y], {y: (0.9, (0, 1))}),
        ([z, w, x], {z: (0.45, (0,)), w: (0.9, (1,)), x: (0.45, (0,))}),
    ]


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


def _planned_copies(tmp_path, monkeypatch, models):
    """The copies ``run_pipeline`` hands the planner for a recipe's ``models``."""
    import redact
    from redact import residency

    seen = []
    monkeypatch.setattr(redact, "generate_inputs", lambda **k: pd.DataFrame({"x": [1]}))
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": models}
    try:
        with patch.object(residency, "plan_phases",
                          side_effect=lambda stage_models, **kw: seen.append(kw) or []):
            run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    (kwargs,) = seen
    return kwargs["copies"]


def test_recipe_copies_reach_the_planner_by_model_name(
        tmp_path, monkeypatch, local_models):
    """A recipe counts copies per role; the planner counts them per model."""
    gen = local_models("_rc_copies_gen")
    counted = {"gen": {"name": gen, "copies": 2}}
    assert _planned_copies(tmp_path, monkeypatch, counted) == {gen: 2}
    assert _planned_copies(tmp_path, monkeypatch, {"gen": gen}) == {}


def test_recipe_copies_reach_the_planner_for_a_roles_default_model(
        tmp_path, monkeypatch):
    models = {"gen": {"backend_type": "vllm", "copies": 2},
              "translation": {"workers": 1}}
    assert _planned_copies(tmp_path, monkeypatch, models) == {"venice-uncensored": 2}


@pytest.mark.parametrize("gen,planned", [
    ("llama-3.2-3b-debug", 2),
    ({"name": "llama-3.2-3b-debug", "copies": 2}, 2),
    ({"name": "llama-3.2-3b-debug", "copies": 1}, 1),
])
def test_run_pipeline_summary_carries_copies(tmp_path, monkeypatch, gen, planned):
    """Two cards and one local model: a copy per card unless the recipe says one."""
    import redact
    import redact.llms.backends.vllm as vllm_module

    seen = []
    monkeypatch.setattr(
        redact, "generate_inputs",
        lambda **k: seen.append(list(vllm_module._planned_devices.values()))
        or pd.DataFrame({"x": [1]}))
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": gen}}
    try:
        with patch("redact.llms.resources.measure.detect_gpus",
                   return_value=("FakeGPU", 2)), \
             patch("redact.llms.resources.measure.detect_gpu_memory_gib",
                   return_value=10.0):
            summary = run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    model = summary["residency"]["phases"][0]["models"][0]
    assert (model["copies"], model["devices"]) == (planned, list(range(planned)))
    assert seen == [[tuple(range(planned))]]           # what the loader was handed
    # Also for a count left to the plan.
    served = {"name": "llama-3.2-3b-debug", "backend_type": "vllm", "copies": planned}
    assert summary["serving"] == {"gen": served, "check": served}


@pytest.fixture()
def hosted_and_local():
    """Register a model whose default setup is hosted and that has local
    weights too; clean up after."""
    from redact.llms.model_config import (
        MODEL_REGISTRY,
        APIConfig,
        VLLMConfig,
        register_model,
    )

    name = "_rc_hosted_and_local"
    register_model(
        name, backend_type="api",
        api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                      base_url="https://test.example/v1", rpm=10),
        vllm=VLLMConfig(hf_model_id=f"org/{name}", vram_gb=5.0))
    yield name
    MODEL_REGISTRY.pop(name, None)


def test_a_run_serves_a_hosted_model_locally_when_the_recipe_says_so(
        tmp_path, monkeypatch, hosted_and_local):
    """The stage's client, the plan and the records follow the recipe, and
    none of it outlasts the run."""
    from types import SimpleNamespace

    import redact
    from redact.llms import ModelClient
    from redact.llms.backends import capabilities
    from redact.llms.model_config import model_compute_config

    name = hosted_and_local
    built = []

    def generate_inputs(**k):
        backend = ModelClient.create(k["model"]).backend
        built.append((k["model"], backend.backend_name, backend.hf_model_id))
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(redact, "generate_inputs", generate_inputs)
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": {"name": name, "backend_type": "vllm"}}}
    engine = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=256))
    card = _one_card(10.0)
    try:
        with card[0], card[1], \
             patch("redact.llms.backends.vllm._engine", return_value=engine):
            summary = run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    assert built == [(name, "vllm", f"org/{name}")]
    (planned,) = summary["residency"]["phases"][0]["models"]
    assert (planned["model"], planned["setup"]) == (name, "vllm")
    served = {"name": name, "backend_type": "vllm", "copies": 1}
    assert summary["serving"] == {"gen": served, "check": served}
    record = read_manifest("inputs", tmp_path)
    assert (record["serving"], record["models"]["gen"]) == (summary["serving"], name)
    # Hosted again once the run is over.
    assert capabilities._run_serving == {}
    assert model_compute_config(name).supports_native_batching is False


@pytest.mark.parametrize("fails,error", [
    ("the plan", KeyboardInterrupt), ("a stage", RuntimeError)])
def test_a_run_that_fails_leaves_no_serving_behind(
        tmp_path, monkeypatch, hosted_and_local, fails, error):
    import redact
    from redact import residency
    from redact.llms.backends import capabilities

    seen = []

    def fail(*args, **kwargs):
        seen.append(dict(capabilities._run_serving))
        raise error("boom")

    if fails == "the plan":
        monkeypatch.setattr(residency, "plan_phases", fail)
    monkeypatch.setattr(redact, "generate_inputs", fail)
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": {"name": hosted_and_local, "backend_type": "vllm"}}}
    card = _one_card(10.0)
    try:
        with card[0], card[1], pytest.raises(error, match="boom"):
            run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    assert seen == [{hosted_and_local: {"backend_type": "vllm"}}]   # set when it failed
    assert capabilities._run_serving == {}


def test_a_run_gives_a_hosted_model_the_recipes_workers(tmp_path, monkeypatch):
    """Two requests at once in the registry, one in this run."""
    import redact
    from redact.llms import ModelClient

    seen = []
    monkeypatch.setattr(
        redact, "generate_inputs",
        lambda **k: seen.append(ModelClient.create(k["model"]).backend.max_workers)
        or pd.DataFrame({"x": [1]}))
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": {"name": "deepseek-v3.2", "workers": 1}}}
    try:
        summary = run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    assert seen == [1]
    assert ModelClient.create("deepseek-v3.2").backend.max_workers == 2
    served = {"name": "deepseek-v3.2", "backend_type": "api", "workers": 1}
    assert summary["serving"] == {"gen": served, "check": served}


def test_a_recipe_of_names_and_nulls_is_recorded_with_how_it_was_served(
        tmp_path, monkeypatch):
    """The registry's defaults are named, per stage for the roles it calls,
    and the record's ``models`` are as they were."""
    import redact
    from redact.runconfig import _DEFAULT_MODELS

    monkeypatch.setattr(redact, "generate_inputs", lambda **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(redact, "build_dataset", lambda **k: pd.DataFrame({"x": [1]}))
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path),
              "stages": ["inputs", "build"]}
    try:
        summary = run_pipeline(recipe, params={"inputs": {}, "build": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    venice = {"name": "venice-uncensored", "backend_type": "api", "workers": 3}
    assert summary["serving"] == {"gen": venice, "check": venice}
    inputs, build = read_manifest("inputs", tmp_path), read_manifest("build", tmp_path)
    assert (inputs["serving"], build["serving"]) == (summary["serving"], {})
    assert inputs["models"] == build["models"] == _DEFAULT_MODELS


def test_served_lists_a_pool_and_names_no_model_for_a_role_no_stage_calls(
        local_models):
    from redact.runconfig import _served

    extra = local_models("_rc_served_para", roles=["paraphraser"])
    models = _models(gen="no-such-model", paraphraser="distribution")
    served = _served(models, ["paraphrase", "jailbreaks"],
                     {"include_translation": False}, {extra: {"copies": 2}}, [])
    assert served == {
        "paraphraser": [
            # No plan: the recipe's count, or none for one left to the plan.
            {"name": "venice-paraphraser", "backend_type": "vllm", "copies": None},
            {"name": extra, "backend_type": "vllm", "copies": 2}],
        "paraphrase_check": {
            "name": "venice-uncensored", "backend_type": "api", "workers": 3},
        "gen": {"name": "no-such-model"},          # its stage reports it
        "translation": None,
    }


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


def test_a_failed_stage_still_releases_the_models_and_logs_the_summary(
        tmp_path, monkeypatch, caplog):
    """The card comes back and the cost is reported, whatever the stage did."""
    import logging
    import time

    import redact
    import redact.llms.backends.vllm as vllm_module
    from redact import residency

    def boom(**k):
        raise RuntimeError("boom")

    monkeypatch.setattr(redact, "generate_inputs", boom)
    key = ("org/left-loaded", None, "[]")
    vllm_module._engine_loaded_at[key] = time.perf_counter() - 120.0
    wanted = []
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": "llama-3.2-3b-debug"}}
    card = _one_card(10.0)
    try:
        with card[0], card[1], \
             patch.object(residency, "preload",
                          side_effect=lambda names, **kw: wanted.append(kw["wanted"])), \
             caplog.at_level(logging.INFO, logger="redact.telemetry"), \
             pytest.raises(RuntimeError, match="boom"):
            run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        released = key not in vllm_module._engine_loaded_at    # read before the cleanup
        vllm_module._engine_loaded_at.pop(key, None)
        telemetry.uninstall()
    assert released                                            # released at the end
    assert wanted == [set()]                                   # the preloader was told to stop
    assert "[telemetry] local org/left-loaded held" in caplog.text


def test_a_run_leaves_no_planned_share_behind(tmp_path, monkeypatch):
    """The plan is the run's own: a later direct call must not inherit its shares."""
    import redact
    import redact.llms.backends.vllm as vllm_module

    seen = []
    monkeypatch.setattr(
        redact, "generate_inputs",
        lambda **k: seen.append(list(vllm_module._planned_utilization.values()))
        or pd.DataFrame({"x": [1]}))
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": "llama-3.2-3b-debug"}}
    card = _one_card(10.0)
    try:
        with card[0], card[1]:
            run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    assert seen == [[0.9]]                                 # applied while the stage ran
    assert vllm_module._planned_utilization == {}          # gone with the run


def test_a_run_leaves_no_planned_card_behind(tmp_path, monkeypatch):
    """The cards are the run's own too."""
    import redact
    import redact.llms.backends.vllm as vllm_module

    seen = []
    monkeypatch.setattr(
        redact, "generate_inputs",
        lambda **k: seen.append(list(vllm_module._planned_devices.values()))
        or pd.DataFrame({"x": [1]}))
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path), "stages": ["inputs"],
              "models": {"gen": "llama-3.2-3b-debug"}}
    card = _one_card(10.0)
    try:
        with card[0], card[1]:
            run_pipeline(recipe, params={"inputs": {}}, verbose=False)
    finally:
        telemetry.uninstall()
    assert seen == [[(0,)]]                                # applied while the stage ran
    assert vllm_module._planned_devices == {}              # gone with the run


def test_a_run_unloads_a_model_for_a_stage_and_loads_it_again(
        tmp_path, monkeypatch, local_models, caplog):
    """Two 24 GiB cards. The inputs stage's two engines need a card each, so
    x, which the constitution and paraphrase stages call, is unloaded for
    it. Its second load takes that lifetime's card, copies and share, and
    the run leaves nothing loaded or planned."""
    import logging
    import os
    from types import SimpleNamespace

    import redact
    import redact.llms.backends.vllm as vllm_module
    from redact.llms import ModelClient
    from tests.conftest import fake_vllm_modules

    x, p = local_models("_rc_twice_x"), local_models("_rc_twice_p")
    g, c = local_models("_rc_twice_g", gb=15.0), local_models("_rc_twice_c", gb=15.0)
    events = []

    class _LLM:
        scheduler_config = SimpleNamespace(max_num_seqs=256)

        def __init__(self, **kw):
            self.model = kw["model"].removeprefix("org/")
            events.append(("load", self.model, kw["gpu_memory_utilization"],
                           os.environ.get("CUDA_VISIBLE_DEVICES"),
                           kw.get("data_parallel_size", 1)))

        def shutdown(self):
            events.append(("unload", self.model))

    def stage(name, *roles, result=None):
        def run(**k):
            events.append(name)
            for role in roles:
                ModelClient.create(k[role])
            return pd.DataFrame({"x": [1]}) if result is None else result

        monkeypatch.setattr(redact, f"generate_{name}", run)

    stage("constitution", "model")
    stage("inputs", "model", "check_model")
    stage("paraphrases", "paraphraser", "check_model", result={})
    stage("jailbreaks", "model")
    recipe = {"dataset_type": "training", "data_dir": str(tmp_path),
              "stages": ["constitution", "inputs", "paraphrase", "jailbreaks"],
              "models": {"constitution": x, "gen": g, "check": c,
                         "paraphraser": p, "paraphrase_check": x}}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    vllm_module.VLLMBackend.clear_cache()
    try:
        with patch("redact.llms.resources.measure.detect_gpus",
                   return_value=("FakeGPU", 2)), \
             patch("redact.llms.resources.measure.detect_gpu_memory_gib",
                   return_value=24.0), \
             patch.dict("sys.modules", fake_vllm_modules(_LLM)), \
             patch.object(vllm_module, "_prepare_environment"), \
             caplog.at_level(logging.INFO, logger="redact.residency.phases"):
            summary = run_pipeline(recipe, params={}, verbose=False)
    finally:
        telemetry.uninstall()
        vllm_module.VLLMBackend.clear_cache()
    assert summary["residency"]["fits"] is True
    assert [phase["resident"] for phase in summary["residency"]["phases"]] == [
        [x], [g, c], [x, g, p], [g]]
    assert f"{x} is unloaded for stage inputs and loaded again for stage " \
           "paraphrase, so that inputs fits." in caplog.text
    assert events == [
        "constitution", ("load", x, 0.9, "0,1", 2),
        ("unload", x),
        "inputs", ("load", g, 0.9, "0", 1), ("load", c, 0.9, "1", 1),
        ("unload", c),
        "paraphrases", ("load", p, 0.45, "1", 1), ("load", x, 0.45, "1", 1),
        ("unload", p), ("unload", x),
        "jailbreaks",
        ("unload", g),
    ]
    assert vllm_module._engines == {} and vllm_module.held_seconds() == {}
    assert vllm_module._planned_utilization == {} == vllm_module._planned_devices
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_load_recipe_rejects_a_repeated_stage():
    from redact.runconfig import load_recipe

    with pytest.raises(ValueError, match="more than once"):
        load_recipe({"dataset_type": "eval", "stages": ["inputs", "inputs"]})


def test_a_failed_model_switch_does_not_abort_the_run(local_models, caplog):
    import logging

    from redact import residency
    from redact.runconfig import _enter_phase, _plan_phases

    gen = local_models("_rc_switch_gen")
    card = _one_card(24.0)
    try:
        with card[0], card[1], \
             patch.object(residency, "preload", side_effect=RuntimeError("boom")), \
             caplog.at_level(logging.ERROR, logger="redact.runconfig"):
            phases = _plan_phases(_models(gen=gen), ["inputs"], verbose=False)
            _enter_phase(phases, 0, set(), verbose=False)
    finally:
        telemetry.uninstall()
    assert "could not switch models (RuntimeError: boom)" in caplog.text
