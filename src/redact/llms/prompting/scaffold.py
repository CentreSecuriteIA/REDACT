"""Write a prompt tree that a user can edit.

``mode="copy"`` writes the real prompts. ``mode="empty"`` writes skeletons
with the same keys and the text replaced by TODOs. Used by
``scripts/scaffold_prompts.py``.

A user's prompt directory only needs the files they changed, because prompts
fall back to the packaged copy per file.
"""

import json
import shutil
from pathlib import Path

from ... import paths

_DEFAULT_PROMPT_DIR = paths.prompts_dir()

def _placeholder_prompt(config: dict) -> dict:
    """Replace one prompt config's text with placeholders, keeping its keys.

    - ``system_prompt``, ``template`` and ``instruction`` (whichever the file
      has) become a TODO that names every ``seed_fields`` entry as a
      ``{placeholder}``.
    - ``few_shot_examples`` is emptied if present.
    - ``metadata`` is set to version ``0.0`` with a placeholder note.

    Every other key, including ``seed_fields``, is kept unchanged.
    """
    stub = dict(config)
    seed_fields = config.get("seed_fields", [])
    fields_note = (
        " ".join(f"{{{f}}}" for f in seed_fields) if seed_fields else "(no seed_fields declared)"
    )
    for key in ("system_prompt", "template", "instruction"):
        if key in stub:
            stub[key] = (
                f"TODO: replace with a real {key}. Available placeholder fields: {fields_note}"
            )
    # Emptied only if present, so the stub gains no key its source lacks.
    if "few_shot_examples" in stub:
        stub["few_shot_examples"] = []
    stub["metadata"] = {
        "version": "0.0",
        "notes": "PLACEHOLDER — replace before use.",
    }
    return stub


def scaffold_prompt_tree(
    target_dir: str | Path,
    source_dir: str | Path | None = None,
    mode: str = "copy",
) -> int:
    """Write an editable copy of a ``prompts/`` tree.

    Delete the files you do not intend to override: a prompt missing from a
    user's directory is read from the packaged copy.

    Args:
        target_dir: Root directory to write into.
        source_dir: Root to copy from. Defaults to the packaged ``prompts/``.
        mode: ``"copy"`` (default) writes the real prompts. ``"empty"``
            writes skeletons whose text is replaced by TODOs naming each
            ``seed_fields`` entry as a ``{placeholder}``.

    Returns:
        Number of files written.

    Raises:
        ValueError: Unknown ``mode``.

    Note:
        ``"empty"`` replaces prompt text only. Directory and category names
        are left as they are, so review the output before publishing.
    """
    if mode not in ("copy", "empty"):
        raise ValueError(f"mode must be 'copy' or 'empty', got {mode!r}")

    source = Path(source_dir) if source_dir is not None else _DEFAULT_PROMPT_DIR
    target = Path(target_dir)
    written = 0
    for src_path in source.rglob("*.json"):
        dst_path = target / src_path.relative_to(source)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(src_path, dst_path)
        else:
            with open(src_path, encoding="utf-8") as f:
                config = json.load(f)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(_placeholder_prompt(config), f, indent=4)
                f.write("\n")
        written += 1
    return written
