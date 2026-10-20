"""Download the weights of every local model in the registry.

    python scripts/download_models.py                      # every local checkpoint
    python scripts/download_models.py llama-3.2-3b-debug   # only the named models
    python scripts/download_models.py --dry-run            # list, download nothing

Reads ``configs/llm/models.json`` through the registry, so a model added there
is picked up here. Models sharing a checkpoint are downloaded once, whole,
GGUF files left out (vLLM reads safetensors), at the revision an entry pins:
the tokenizer files a setup with ``hf_config`` gathers into its tokenizer
folder at load come with it. A machine-translation model or a paraphraser brings
the language identifier its output is checked with. Files land in the
HuggingFace cache (``HF_HOME``), which is where vLLM, transformers and the
identifier look for them at load time.
"""

import argparse
import sys
from pathlib import Path

from redact.jailbreak.obfuscation.langid import GLOTLID_FILE, GLOTLID_REPO
from redact.jailbreak.obfuscation.translation import MACHINE_TRANSLATION_ROLE
from redact.llms.model_config import MODEL_REGISTRY

_LOCAL_SETUPS = ("vllm", "introspect")

#: The roles whose output is checked with GlotLID: a machine translation's
#: language, and a paraphrase's (the paraphrase check's language gate).
_LANGID_ROLES = (MACHINE_TRANSLATION_ROLE, "paraphraser")

#: Repositories of which one file is fetched, not the whole snapshot.
FILES = {GLOTLID_REPO: GLOTLID_FILE}

#: Left out of every snapshot: vLLM loads safetensors, and a repository may
#: hold tens of GB of GGUF files beside them (the humanizer's about 64 GB).
IGNORE_PATTERNS = ["*.gguf"]


def checkpoints(names: list[str] | None = None) -> dict[str, list[str]]:
    """Map each HF repository to fetch to the registered models that need it."""
    unknown = set(names or ()) - set(MODEL_REGISTRY)
    if unknown:
        raise KeyError(f"Not in the registry: {sorted(unknown)}")

    found: dict[str, list[str]] = {}
    for name, config in MODEL_REGISTRY.items():
        if names and name not in names:
            continue
        for setup in _LOCAL_SETUPS:
            local = getattr(config, setup)
            # A path on disk is already downloaded.
            if local is None or Path(local.hf_model_id).exists():
                continue
            users = found.setdefault(local.hf_model_id, [])
            if name not in users:
                users.append(name)
        if any(role in config.roles for role in _LANGID_ROLES):
            found.setdefault(GLOTLID_REPO, []).append(name)
    return found


def revision(repo_id: str, users: list[str]) -> str | None:
    """The revision the vLLM setups of ``users`` pin for ``repo_id``
    (``vllm_kwargs["revision"]``), or ``None`` when none pins one.

    Raises:
        ValueError: Two users pin different revisions of one repository.
    """
    pins: dict[str, list[str]] = {}
    for name in users:
        local = MODEL_REGISTRY[name].vllm
        if local is None or local.hf_model_id != repo_id:
            continue
        pin = (local.vllm_kwargs or {}).get("revision")
        if pin:
            pins.setdefault(pin, []).append(name)
    if len(pins) > 1:
        raise ValueError(
            f"{repo_id}: its users pin different revisions "
            + "; ".join(f"{', '.join(names)} pin {pin}" for pin, names in pins.items())
            + ". Pin one revision for the checkpoint.")
    return next(iter(pins), None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download local model weights.")
    parser.add_argument("models", nargs="*", help="Registry names (default: all).")
    parser.add_argument("--dry-run", action="store_true", help="List only.")
    args = parser.parse_args(argv)

    try:
        todo = checkpoints(args.models)
    except KeyError as exc:
        print(exc.args[0])
        return 1
    if not todo:
        print("No local checkpoints to download.")
        return 0
    try:
        revisions = {repo: revision(repo, users) for repo, users in todo.items()
                     if repo not in FILES}
    except ValueError as exc:
        print(exc)
        return 1

    # Imported after redact, which has loaded .env (HF_HOME, HF_TOKEN) by now.
    from huggingface_hub import constants, hf_hub_download, snapshot_download

    print(f"HF cache: {constants.HF_HUB_CACHE}")
    failed = False
    for repo_id, users in todo.items():
        rev = revisions.get(repo_id)
        pinned = f"  at {rev}" if rev else ""
        print(f"{repo_id}  (used by {', '.join(users)}){pinned}")
        if args.dry_run:
            continue
        try:
            if repo_id in FILES:
                print(f"  -> {hf_hub_download(repo_id, FILES[repo_id])}")
            else:
                print("  -> " + snapshot_download(
                    repo_id, ignore_patterns=IGNORE_PATTERNS,
                    **({"revision": rev} if rev else {})))
        except Exception as exc:  # noqa: BLE001 - report it and try the next one
            print(f"  FAILED: {exc}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
