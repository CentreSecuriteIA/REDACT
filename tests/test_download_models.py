"""``scripts/download_models.py``: what it would fetch, without fetching."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "download_models", Path(__file__).parents[1] / "scripts" / "download_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_translator_and_the_shared_24b_weights_are_listed_once(script):
    todo = script.checkpoints(["translategemma-27b", "venice-uncensored",
                               "venice-uncensored-fp8", "venice-paraphraser"])
    assert {repo: users for repo, users in todo.items()
            if repo != "cis-lmu/glotlid"} == {
        "dphn/Dolphin-Mistral-24B-Venice-Edition": [
            "venice-uncensored", "venice-uncensored-fp8", "venice-paraphraser"],
        "Infomaniak-AI/vllm-translategemma-27b-it": ["translategemma-27b"],
    }


def test_a_dry_run_names_each_checkpoint_once(script, capsys):
    assert script.main(["--dry-run", "translategemma-27b", "venice-uncensored-fp8",
                        "venice-paraphraser"]) == 0
    out = capsys.readouterr().out
    for repo in ("dphn/Dolphin-Mistral-24B-Venice-Edition",
                 "Infomaniak-AI/vllm-translategemma-27b-it"):
        assert sum(line.startswith(repo) for line in out.splitlines()) == 1


def test_a_machine_translation_model_brings_the_language_identifier(script):
    """GlotLID's one file is fetched with the translator, not its whole
    repository."""
    todo = script.checkpoints(["translategemma-27b"])
    assert todo["cis-lmu/glotlid"] == ["translategemma-27b"]
    assert script.FILES == {"cis-lmu/glotlid": "model_v3.bin"}
    assert "cis-lmu/glotlid" not in script.checkpoints(["venice-uncensored-fp8"])
