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


def test_the_humanizer_is_listed_once_with_the_language_identifier(script):
    assert script.checkpoints(["humanizer-12b"]) == {
        "jialinyyzz/humanizer": ["humanizer-12b"],
        "cis-lmu/glotlid": ["humanizer-12b"]}


def test_a_snapshot_leaves_the_gguf_files_out_and_takes_the_pinned_revision(
        script, monkeypatch):
    import huggingface_hub

    from redact.llms.model_config import get_model_config

    fetched = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, **kw: fetched.append((repo_id, kw)) or "/snap")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: "/file")
    assert script.main(["humanizer-12b"]) == 0
    pinned = get_model_config("humanizer-12b").vllm.vllm_kwargs["revision"]
    assert fetched == [("jialinyyzz/humanizer",
                        {"ignore_patterns": ["*.gguf"], "revision": pinned})]


def test_two_pins_of_one_checkpoint_stop_the_download(script, monkeypatch, capsys):
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model

    for name, rev in (("_dl_a", "a" * 40), ("_dl_b", "b" * 40)):
        register_model(name, backend_type="vllm", vllm=VLLMConfig(
            hf_model_id="org/same", vllm_kwargs={"revision": rev}))
    try:
        with pytest.raises(ValueError, match="different revisions"):
            script.revision("org/same", ["_dl_a", "_dl_b"])
        assert script.main(["_dl_a", "_dl_b", "--dry-run"]) == 1
        assert "_dl_a pin " + "a" * 40 in capsys.readouterr().out
    finally:
        MODEL_REGISTRY.pop("_dl_a", None)
        MODEL_REGISTRY.pop("_dl_b", None)


def test_a_tokenizer_loaded_from_elsewhere_is_fetched_too(
        script, monkeypatch, capsys):
    """vLLM loads the tokenizer at tokenizer_revision, from the repository
    tokenizer names; offline at load, its files must be in the cache. A
    tokenizer revision equal to the revision adds nothing."""
    import huggingface_hub

    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model

    fetched = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, **kw: fetched.append((repo_id, kw)) or "/snap")
    entries = {
        "_dl_tok": ("org/tok", {"revision": "a" * 40, "tokenizer_revision": "b" * 40}),
        "_dl_same": ("org/same-tok", {"revision": "c" * 40,
                                      "tokenizer_revision": "c" * 40}),
        "_dl_other": ("org/weights", {"tokenizer": "org/tokenizer"}),
    }
    for name, (repo, kwargs) in entries.items():
        register_model(name, backend_type="vllm",
                       vllm=VLLMConfig(hf_model_id=repo, vllm_kwargs=kwargs))
    try:
        assert script.main(list(entries)) == 0
    finally:
        for name in entries:
            MODEL_REGISTRY.pop(name, None)
    files = [*script.TOKENIZER_FILES, "config.json"]
    assert fetched == [
        ("org/tok", {"ignore_patterns": ["*.gguf"], "revision": "a" * 40}),
        ("org/same-tok", {"ignore_patterns": ["*.gguf"], "revision": "c" * 40}),
        ("org/weights", {"ignore_patterns": ["*.gguf"]}),
        ("org/tok", {"allow_patterns": files, "revision": "b" * 40}),
        ("org/tokenizer", {"allow_patterns": files})]
    out = capsys.readouterr().out
    assert "org/tok  (the tokenizer of _dl_tok)  at " + "b" * 40 in out
    assert "org/tokenizer  (the tokenizer of _dl_other)" in out


def test_a_dry_run_needs_no_huggingface_hub(script, monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert script.main(["--dry-run", "translategemma-27b"]) == 0
    out = capsys.readouterr().out
    assert "huggingface_hub is not installed" in out
    assert "Infomaniak-AI/vllm-translategemma-27b-it" in out
    assert script.main(["translategemma-27b"]) == 1


def test_a_paraphraser_brings_the_language_identifier(script):
    """The paraphrase check's language gate reads GlotLID too; a model named
    for comparison, with no role, brings nothing."""
    assert script.checkpoints(["humanizer-12b"])["cis-lmu/glotlid"] == [
        "humanizer-12b"]
    assert "cis-lmu/glotlid" not in script.checkpoints(["venice-paraphraser"])


def test_every_local_checkpoint_includes_the_default_paraphraser(script):
    assert script.checkpoints()["jialinyyzz/humanizer"] == ["humanizer-12b"]


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
    assert replaced and dict(fetched) == dict.fromkeys(
        replaced, {"ignore_patterns": ["*.gguf"]})
