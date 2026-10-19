"""Download the weights of every local model in the registry.

    python scripts/download_models.py                      # every local checkpoint
    python scripts/download_models.py llama-3.2-3b-debug   # only the named models
    python scripts/download_models.py --dry-run            # list, download nothing

Reads ``configs/llm/models.json`` through the registry, so a model added there
is picked up here. Models sharing a checkpoint are downloaded once, whole: the
tokenizer files a setup with ``hf_config`` gathers into its tokenizer folder
at load come with it. A machine-translation model brings the language
identifier its translations are checked with. Files land in the HuggingFace
cache (``HF_HOME``), which is where vLLM, transformers and the identifier look
for them at load time.
"""

import argparse
import sys
from pathlib import Path

from redact.jailbreak.obfuscation.langid import GLOTLID_FILE, GLOTLID_REPO
from redact.jailbreak.obfuscation.translation import MACHINE_TRANSLATION_ROLE
from redact.llms.model_config import MODEL_REGISTRY

_LOCAL_SETUPS = ("vllm", "introspect")

#: Repositories of which one file is fetched, not the whole snapshot.
FILES = {GLOTLID_REPO: GLOTLID_FILE}


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
        if MACHINE_TRANSLATION_ROLE in config.roles:
            found.setdefault(GLOTLID_REPO, []).append(name)
    return found


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

    # Imported after redact, which has loaded .env (HF_HOME, HF_TOKEN) by now.
    from huggingface_hub import constants, hf_hub_download, snapshot_download

    print(f"HF cache: {constants.HF_HUB_CACHE}")
    failed = False
    for repo_id, users in todo.items():
        print(f"{repo_id}  (used by {', '.join(users)})")
        if args.dry_run:
            continue
        try:
            if repo_id in FILES:
                print(f"  -> {hf_hub_download(repo_id, FILES[repo_id])}")
            else:
                print(f"  -> {snapshot_download(repo_id)}")
        except Exception as exc:  # noqa: BLE001 - report it and try the next one
            print(f"  FAILED: {exc}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
