"""Write a prompt tree that a user can edit.

``mode="copy"`` writes the real prompts. ``mode="empty"`` writes skeletons
with the same keys, each text field replaced by a TODO that keeps that field's
own placeholders. Used by ``scripts/scaffold_prompts.py``.

A user's prompt directory only needs the files they changed, because prompts
fall back to the packaged copy per file.
"""

import json
import shutil
from pathlib import Path

from ... import paths
from .prompts import _field_roots, _read_prompt

_DEFAULT_PROMPT_DIR = paths.prompts_dir()


def _placeholders(text: object) -> str:
    """The ``{placeholders}`` one prompt text field uses, as stub text."""
    try:
        roots = _field_roots(text) if isinstance(text, str) else []
    except ValueError:
        # A malformed source field has no placeholders to carry over.
        roots = []
    return " ".join(f"{{{r}}}" for r in roots if r.isidentifier()) or "(none)"


def _placeholder_prompt(config: dict) -> dict:
    """Replace one prompt config's text with placeholders, keeping its keys.

    - ``system_prompt``, ``template`` and ``instruction`` (whichever the file
      has) each become a TODO holding the ``{placeholders}`` that field used,
      so the stub renders with the same values as the original.
    - ``few_shot_examples`` is emptied if present.
    - ``metadata``, if present, is set to version ``0.0`` with a placeholder
      note.

    Every other key, including ``seed_fields``, is kept unchanged.
    """
    stub = dict(config)
    for key in ("system_prompt", "template", "instruction"):
        if key in stub:
            stub[key] = (
                f"TODO: replace with a real {key}. "
                f"Placeholders: {_placeholders(config[key])}"
            )
    # Replaced only if present, so the stub gains no key its source lacks.
    if "few_shot_examples" in stub:
        stub["few_shot_examples"] = []
    if "metadata" in stub:
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
            writes skeletons whose text fields are replaced by TODOs that
            keep each field's own ``{placeholders}``.

    Returns:
        Number of files written.

    Raises:
        ValueError: Unknown ``mode``.
        FileNotFoundError: The source directory does not exist.

    Note:
        ``"empty"`` replaces prompt text only. Directory and category names
        are left as they are, so review the output before publishing.
    """
    if mode not in ("copy", "empty"):
        raise ValueError(f"mode must be 'copy' or 'empty', got {mode!r}")

    source = Path(source_dir) if source_dir is not None else _DEFAULT_PROMPT_DIR
    if not source.is_dir():
        raise FileNotFoundError(f"Prompt source directory {source} does not exist.")
    target = Path(target_dir)
    written = 0
    for src_path in source.rglob("*.json"):
        dst_path = target / src_path.relative_to(source)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(src_path, dst_path)
        else:
            config = _read_prompt(src_path, source_dir is not None)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(_placeholder_prompt(config), f, indent=4)
                f.write("\n")
        written += 1
    return written
