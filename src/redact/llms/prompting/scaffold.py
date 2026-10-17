"""Write a prompt tree, or a setup folder of recipes and configs, that a user
can edit.

``scaffold_prompt_tree``: ``mode="copy"`` writes the real prompts,
``mode="empty"`` skeletons with the same keys, each text field replaced by a
TODO that keeps that field's own placeholders. ``scaffold_setup``: two
recipes, their params files, and the config files a recipe's ``config_dir``
overlays. Used by ``scripts/scaffold_prompts.py`` and
``scripts/scaffold_setup.py``.

A user's prompt directory or config folder only needs the files they
changed, because each falls back to the packaged copy per file.
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


# The files of a setup folder copied as they are, by their path in it: from
# the packaged configs/, path for path, or from the example params files.
_SETUP_COPIES = {
    "taxonomy/harmful.json": "taxonomy/harmful.json",
    "taxonomy/benign.json": "taxonomy/benign.json",
    "jailbreak/augmentations.json": "jailbreak/augmentations.json",
    "eval_params.json": "runs/params_eval_example.json",
    "training_params.json": "runs/params_training_example.json",
}
# Each recipe of a setup folder: the packaged example it is written from, and
# its params file.
_SETUP_RECIPES = {
    "eval_recipe.json": ("runs/eval_example.json", "eval_params.json"),
    "training_recipe.json": ("runs/training_example.json", "training_params.json"),
}
_SETUP_README = """\
# Your REDACT setup

Run a recipe from this folder: `python scripts/run.py eval_recipe.json`, or
`training_recipe.json`. Each recipe's `"config_dir": "."` points the run here.

- Every file here wins over the library's packaged copy of the same file:
  `dataset_types/`, `taxonomy/` and `jailbreak/augmentations.json`.
- Delete every file you do not change, so you keep getting the library's
  updates to it.
- Prompts are a separate tier: `python scripts/scaffold_prompts.py ./prompts`
  writes an editable copy, and a recipe's `prompt_dir` points at it.
- The technique spec and templates, `llm/` and `datasets/` come from the
  library only; a file of theirs here is ignored, with a warning.
"""


def scaffold_setup(target_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write a setup folder: an eval and a training recipe with their params
    files, the dataset type files, the two taxonomies, the augmentation
    section's defaults, and a ``README.md``.

    Each recipe is its packaged example with ``"config_dir": "."`` and its
    params file beside it, so it runs as the example does until a file here
    is changed. The other files are byte copies of the packaged ones.

    Returns:
        The paths written.

    Raises:
        FileExistsError: A file to write exists and ``overwrite`` is False;
            the message names every such file. Nothing is written.
    """
    target = Path(target_dir)
    configs = paths.configs_dir()
    copies = {f"dataset_types/{p.name}": f"dataset_types/{p.name}"
              for p in sorted((configs / "dataset_types").glob("*.json"))}
    copies.update(_SETUP_COPIES)
    names = [*copies, *_SETUP_RECIPES, "README.md"]
    existing = [name for name in names if (target / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{target} already holds {existing}; pass overwrite=True to replace "
            "them.")
    written = []
    for name, source in copies.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(configs / source, path)
        written.append(path)
    for name, (example, params) in _SETUP_RECIPES.items():
        recipe = json.loads((configs / example).read_text(encoding="utf-8"))
        recipe = {"dataset_type": recipe.pop("dataset_type"), "config_dir": ".",
                  **recipe, "params_file": params}
        path = target / name
        path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    readme = target / "README.md"
    readme.write_text(_SETUP_README, encoding="utf-8")
    written.append(readme)
    return written
