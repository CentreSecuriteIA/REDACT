"""The folder a vLLM engine loads its tokenizer from when its config is replaced.

vLLM reads a config from the folder it loads the tokenizer from, not only
from ``hf_config_path``, so a checkpoint whose own ``config.json`` the
installed transformers cannot read fails there. The folder built here holds
the checkpoint's tokenizer and processor files and a copy of the readable
config, and is passed to the engine as ``tokenizer``.
"""

import hashlib
import os
import re
import shutil
import threading
from fnmatch import fnmatchcase
from pathlib import Path

#: The files a tokenizer and a processor load from, as patterns over a
#: checkpoint's top-level file names. A checkpoint holds some of them.
TOKENIZER_FILES = (
    "tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*",
    "generation_config.json", "preprocessor_config.json", "processor_config.json",
    "vocab*", "merges.txt", "spiece.model", "tekken.json",
)

# What may not appear in a folder name, replaced by "--".
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
# Held while a folder is found or built, so two loads never build one twice.
_lock = threading.Lock()


def tokenizer_folder(
    hf_model_id: str, config_dir: str, *, revision: str | None = None,
) -> str:
    """The folder to load ``hf_model_id``'s tokenizer from, holding the
    ``config.json`` of ``config_dir`` in place of the checkpoint's own.

    The tokenizer files come from ``hf_model_id`` when it is a folder, else
    from its hub snapshot at ``revision``, of which only the files of
    :data:`TOKENIZER_FILES` are fetched (a hub cache that holds them serves
    them offline). Each is linked into the folder, or copied where the OS
    allows no link. The folder is
    ``$HF_HOME/assets/redact/tokenizers/<checkpoint>-<hash>``, hashed from
    the snapshot's path and the config's bytes, so a new snapshot or a
    changed config gets a folder of its own; one built before is reused.

    Raises:
        FileNotFoundError: The checkpoint has none of the tokenizer files.
    """
    from huggingface_hub import constants

    config = (Path(config_dir) / "config.json").read_bytes()
    with _lock:
        if Path(hf_model_id).is_dir():
            source = Path(hf_model_id).resolve()
            label = source.name
        else:
            source, label = _snapshot(hf_model_id, revision), hf_model_id
        digest = hashlib.sha256(os.fsencode(source) + b"\0" + config).hexdigest()[:16]
        name = f"{_UNSAFE.sub('--', label).strip('-.')}-{digest}"
        folder = Path(constants.HF_ASSETS_CACHE) / "redact" / "tokenizers" / name
        if not _complete(folder):
            _build(folder, source, config)
    return str(folder)


def _snapshot(repo_id: str, revision: str | None) -> Path:
    """The hub snapshot of ``repo_id`` holding its tokenizer files."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, revision=revision,
                                  allow_patterns=list(TOKENIZER_FILES)))


def _complete(folder: Path) -> bool:
    """Whether ``folder`` holds its config and every file in it still resolves."""
    return (folder / "config.json").is_file() and all(
        path.exists() for path in folder.iterdir())


def _link(source: Path, target: Path) -> None:
    """Link ``target`` to ``source``, or copy it where the OS allows no link."""
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _build(folder: Path, source: Path, config: bytes) -> None:
    """Fill ``folder`` from ``source`` and ``config``.

    Through a temporary folder renamed into place, so a folder that exists
    is whole. When another process renames its own in first, that one is
    kept.
    """
    names = sorted(path.name for path in source.iterdir() if path.is_file() and any(
        fnmatchcase(path.name, pattern) for pattern in TOKENIZER_FILES))
    if not names:
        raise FileNotFoundError(
            f"{source} holds none of the tokenizer files {list(TOKENIZER_FILES)}, "
            f"so the engine has no tokenizer to load beside its replaced config.")
    # A folder a link of which no longer resolves; checked again, since
    # another process may have renamed a whole one into place meanwhile, and
    # an engine of its may be reading it.
    if _complete(folder):
        return
    shutil.rmtree(folder, ignore_errors=True)
    staging = folder.with_name(f"{folder.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for name in names:
            _link(source / name, staging / name)
        (staging / "config.json").write_bytes(config)
        try:
            staging.rename(folder)
        except OSError:
            if not _complete(folder):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
