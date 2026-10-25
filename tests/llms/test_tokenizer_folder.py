"""The tokenizer folder of a checkpoint whose ``config.json`` is replaced.

Nothing here touches the network: ``huggingface_hub.snapshot_download`` is a
fake that returns a snapshot folder written by the test, and the assets
cache is a temporary folder.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from redact import paths
from redact.llms.backends import tokenizer_folder as tf

# A checkpoint's top-level files: what a tokenizer and a processor load from,
# its own config, and files the folder must leave out.
TOKENIZER_NAMES = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "generation_config.json", "preprocessor_config.json", "processor_config.json",
)
OTHER_NAMES = ("config.json", "model-00001-of-00002.safetensors",
               "model.safetensors.index.json", "README.md")


def _snapshot(root: Path, commit: str = "c0ffee", names=TOKENIZER_NAMES) -> Path:
    folder = root / "hub" / "models--org--m" / "snapshots" / commit
    folder.mkdir(parents=True)
    for name in (*names, *OTHER_NAMES):
        (folder / name).write_text(f"{commit}:{name}", encoding="utf-8")
    return folder


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _same_config(folder: Path, config: Path) -> bool:
    name = "config.json"
    return (folder / name).read_bytes() == (config / name).read_bytes()


def _config(root: Path, content: dict) -> Path:
    text = json.dumps(content, sort_keys=True)
    folder = root / "configs" / hashlib.sha256(text.encode()).hexdigest()[:12]
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(content), encoding="utf-8")
    return folder


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """A fake hub: ``hub.snapshots[revision]`` is the folder the download
    returns, and ``hub.calls`` records what it was asked."""
    import huggingface_hub
    from huggingface_hub import constants

    class Hub:
        def __init__(self):
            self.assets = tmp_path / "assets"
            self.snapshots = {None: _snapshot(tmp_path)}
            self.calls = []

        def download(self, repo_id, *, revision=None, allow_patterns=None, **_):
            self.calls.append((repo_id, revision, allow_patterns))
            return str(self.snapshots[revision])

    fake = Hub()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.download)
    monkeypatch.setattr(constants, "HF_ASSETS_CACHE", str(fake.assets))
    return fake


@pytest.fixture()
def config(tmp_path) -> Path:
    return _config(tmp_path, {"model_type": "gemma3", "patched": True})


class TestTheFolder:
    def test_it_holds_the_tokenizer_files_and_the_packaged_config(self, hub, config):
        folder = Path(tf.tokenizer_folder("org/m", str(config)))
        assert sorted(p.name for p in folder.iterdir()) == sorted(
            (*TOKENIZER_NAMES, "config.json"))
        for name in TOKENIZER_NAMES:
            assert _text(folder / name) == f"c0ffee:{name}"
        # The packaged config, never the checkpoint's own.
        assert _same_config(folder, config)

    def test_only_the_tokenizer_files_are_fetched(self, hub, config):
        tf.tokenizer_folder("org/m", str(config))
        assert hub.calls == [("org/m", None, list(tf.TOKENIZER_FILES))]
        assert not any(name in tf.TOKENIZER_FILES for name in OTHER_NAMES)

    def test_it_lives_in_the_assets_cache_named_by_the_checkpoint(self, hub, config):
        folder = Path(tf.tokenizer_folder("org/m", str(config)))
        assert folder.parent == hub.assets / "redact" / "tokenizers"
        assert folder.name.startswith("org--m-")

    def test_files_the_checkpoint_lacks_are_skipped(self, hub, config, tmp_path):
        hub.snapshots[None] = _snapshot(tmp_path / "few", names=(
            "tokenizer.json", "tokenizer_config.json"))
        folder = Path(tf.tokenizer_folder("org/m", str(config)))
        assert sorted(p.name for p in folder.iterdir()) == [
            "config.json", "tokenizer.json", "tokenizer_config.json"]

    def test_a_checkpoint_without_a_tokenizer_file_is_refused(
            self, hub, config, tmp_path):
        hub.snapshots[None] = _snapshot(tmp_path / "none", names=())
        with pytest.raises(FileNotFoundError, match="none of the tokenizer files"):
            tf.tokenizer_folder("org/m", str(config))
        assert not (hub.assets / "redact" / "tokenizers").exists() or not any(
            (hub.assets / "redact" / "tokenizers").iterdir())

    def test_the_revision_is_fetched(self, hub, config, tmp_path):
        hub.snapshots["v2"] = _snapshot(tmp_path / "v2", commit="beef")
        folder = Path(tf.tokenizer_folder("org/m", str(config), revision="v2"))
        assert hub.calls[-1][:2] == ("org/m", "v2")
        assert _text(folder / "tokenizer.json") == "beef:tokenizer.json"

    def test_a_local_checkpoint_is_read_in_place(self, hub, config, tmp_path):
        local = _snapshot(tmp_path / "local", commit="local")
        folder = Path(tf.tokenizer_folder(str(local), str(config)))
        assert hub.calls == []
        assert folder.name.startswith(f"{local.name}-")  # its name, not its path
        assert _text(folder / "tokenizer.json") == "local:tokenizer.json"


class TestLinksAndCopies:
    def test_files_are_linked_where_the_os_allows(self, hub, config, monkeypatch):
        linked = []

        def symlink(source, target):
            linked.append(Path(source))
            Path(target).write_bytes(Path(source).read_bytes())

        monkeypatch.setattr(os, "symlink", symlink)
        tf.tokenizer_folder("org/m", str(config))
        snapshot = hub.snapshots[None]
        assert sorted(linked) == sorted(snapshot / n for n in TOKENIZER_NAMES)

    def test_they_are_copied_where_it_allows_no_link(self, hub, config, monkeypatch):
        def no_symlink(source, target):
            raise OSError(1314, "A required privilege is not held by the client")

        monkeypatch.setattr(os, "symlink", no_symlink)
        folder = Path(tf.tokenizer_folder("org/m", str(config)))
        assert not any(p.is_symlink() for p in folder.iterdir())
        assert _text(folder / "tokenizer.json") == "c0ffee:tokenizer.json"


class TestReuse:
    def test_a_second_build_reuses_the_folder(self, hub, config):
        first = Path(tf.tokenizer_folder("org/m", str(config)))
        (first / "marker").write_text("kept", encoding="utf-8")
        assert Path(tf.tokenizer_folder("org/m", str(config))) == first
        assert _text(first / "marker") == "kept"

    def test_a_changed_packaged_config_gives_a_new_folder(
            self, hub, config, tmp_path):
        first = Path(tf.tokenizer_folder("org/m", str(config)))
        changed = _config(tmp_path, {"model_type": "gemma3", "patched": "again"})
        second = Path(tf.tokenizer_folder("org/m", str(changed)))
        assert second != first and first.is_dir()
        assert _same_config(second, changed)

    def test_a_new_snapshot_gives_a_new_folder(self, hub, config, tmp_path):
        first = Path(tf.tokenizer_folder("org/m", str(config)))
        hub.snapshots[None] = _snapshot(tmp_path / "new", commit="d00d")
        second = Path(tf.tokenizer_folder("org/m", str(config)))
        assert second != first
        assert _text(second / "tokenizer.json") == "d00d:tokenizer.json"

    def test_a_folder_missing_a_file_is_built_again(self, hub, config):
        folder = Path(tf.tokenizer_folder("org/m", str(config)))
        (folder / "config.json").unlink()
        assert Path(tf.tokenizer_folder("org/m", str(config))) == folder
        assert _same_config(folder, config)

    def test_a_folder_another_process_built_first_is_taken(
            self, hub, config, monkeypatch):
        """The rename into place fails when the folder exists by then."""
        built = Path(tf.tokenizer_folder("org/m", str(config)))
        (built / "marker").write_text("theirs", encoding="utf-8")
        aside = built.with_name("aside")
        built.rename(aside)
        link = tf._link

        def other_process_finishes_first(source, target):
            if aside.exists():
                aside.rename(built)
            link(source, target)

        monkeypatch.setattr(tf, "_link", other_process_finishes_first)
        assert Path(tf.tokenizer_folder("org/m", str(config))) == built
        assert _text(built / "marker") == "theirs"
        assert sorted(p.name for p in built.parent.iterdir()) == [built.name]

    def test_a_folder_another_process_finished_meanwhile_is_kept(
            self, hub, config, monkeypatch):
        """Between the check and the build another process renamed its whole
        folder into place; an engine of its may be reading it, so it stays."""
        built = Path(tf.tokenizer_folder("org/m", str(config)))
        (built / "marker").write_text("theirs", encoding="utf-8")
        complete, calls = tf._complete, []

        def not_yet_at_the_first_look(folder):
            calls.append(folder)
            return len(calls) > 1 and complete(folder)

        monkeypatch.setattr(tf, "_complete", not_yet_at_the_first_look)
        assert Path(tf.tokenizer_folder("org/m", str(config))) == built
        assert _text(built / "marker") == "theirs"
        assert len(calls) == 2  # noqa: PLR2004

    def test_threads_build_one_folder_once(self, hub, config, monkeypatch):
        links = []
        link = tf._link
        monkeypatch.setattr(tf, "_link",
                            lambda s, t: links.append(t.name) or link(s, t))
        start = threading.Barrier(4)
        folders = []

        def build():
            start.wait()
            folders.append(tf.tokenizer_folder("org/m", str(config)))

        threads = [threading.Thread(target=build) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(set(folders)) == 1 and len(folders) == 4  # noqa: PLR2004
        assert sorted(links) == sorted(TOKENIZER_NAMES)


def test_transformers_loads_the_config_and_the_tokenizer_from_the_folder(
        tmp_path, monkeypatch):
    """What vLLM does with the folder: reads its config (the packaged one,
    which transformers can read where the checkpoint's own fails) and loads
    the tokenizer from it."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_ASSETS_CACHE", str(tmp_path / "assets"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(checkpoint / "tokenizer.json"))
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}))
    # A config transformers cannot read.
    (checkpoint / "config.json").write_text('{"model_type": "no-such-model-type"}')

    packaged = paths.hf_configs_dir() / "translategemma-27b"
    folder = tf.tokenizer_folder(str(checkpoint), str(packaged))
    assert transformers.AutoConfig.from_pretrained(folder).model_type == "gemma3"
    loaded = transformers.AutoTokenizer.from_pretrained(folder)
    assert loaded("hello world", add_special_tokens=False)["input_ids"] == [1, 2]
