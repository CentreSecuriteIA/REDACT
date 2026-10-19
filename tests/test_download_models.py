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


def test_a_checkpoint_with_a_replaced_config_is_fetched_whole(script, monkeypatch):
    """Its tokenizer files come with the snapshot, so the engine's tokenizer
    folder finds them in the hub cache, offline too."""
    import huggingface_hub

    from redact.llms.model_config import MODEL_REGISTRY

    replaced = {c.vllm.hf_model_id for c in MODEL_REGISTRY.values()
                if c.vllm is not None and c.vllm.hf_config is not None}
    fetched = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, **kw: fetched.append((repo_id, kw)) or "/snap")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: "/file")
    names = [n for n, c in MODEL_REGISTRY.items()
             if c.vllm is not None and c.vllm.hf_config is not None]
    assert script.main(names) == 0
    assert replaced and dict(fetched) == dict.fromkeys(replaced, {})
