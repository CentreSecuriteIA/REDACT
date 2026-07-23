"""Tests for the config-driven run layer (recipe/params/manifests + driver).

The end-to-end driver test uses the jailbreaks + build stages with
``pure_only`` augmentations, so no technique needs an LLM — it exercises stage
dispatch, manifest writing, and cross-stage chaining fully offline.
"""

import json

import pandas as pd
import pytest

from redact import paths
from redact.dataset.io import append_samples
from redact.runconfig import (
    load_recipe, load_params, write_manifest, read_manifest, run_pipeline,
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
