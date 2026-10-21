"""A user's config folder, overlaid file by file over the packaged ``configs/``.

The folder mirrors ``configs/``: ``dataset_types/eval.json`` and
``training.json``, ``taxonomy/<name>.json``, ``jailbreak/augmentations.json``,
``generation_axes.json`` and ``dimensions.json`` are each read from it when
it has the file, else from the package. Nothing else is: a dataset type of
another name is none, the technique spec and templates under ``jailbreak/``,
``llm/`` and ``datasets/`` are loaded at import or tied to code, and a file
of theirs is warned about as ignored.
"""

import json
import logging
from pathlib import Path

from redact import paths

logger = logging.getLogger(__name__)

# The dataset types a recipe may name, each the name of its file.
DATASET_TYPES = ("eval", "training")
# The folders whose files a config folder may hold, each read one by one.
_READ_FOLDERS = ("dataset_types", "taxonomy")
# The files at the folder's top it may hold.
_READ_FILES = ("generation_axes.json", "dimensions.json")
_AUGMENTATIONS = "jailbreak/augmentations.json"
# The folders read from the package only.
_IGNORED_FOLDERS = ("jailbreak", "llm", "datasets")


def config_file(relative: str, config_dir: str | Path | None = None) -> Path:
    """``configs/<relative>``: the file in ``config_dir`` when it has one,
    else the packaged one."""
    if config_dir is not None:
        mine = Path(config_dir) / relative
        if mine.is_file():
            return mine
    return paths.configs_dir() / relative


def taxonomy_folder(
    name: object,
    taxonomy_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> Path | None:
    """The folder the taxonomy ``name`` is read from: ``taxonomy_dir`` when
    given, else ``<config_dir>/taxonomy`` when it has the file, else
    ``None``, the packaged ``configs/taxonomy``."""
    if taxonomy_dir is not None:
        return Path(taxonomy_dir)
    if config_dir is not None and isinstance(name, str):
        folder = Path(config_dir) / "taxonomy"
        if (folder / (name if name.endswith(".json") else f"{name}.json")).is_file():
            return folder
    return None


def _is_read(relative: str) -> bool:
    folder, _, name = relative.partition("/")
    if folder == "dataset_types":
        return name in {f"{dtype}.json" for dtype in DATASET_TYPES}
    return relative in (_AUGMENTATIONS, *_READ_FILES) or (
        folder in _READ_FOLDERS and "/" not in name and name.endswith(".json"))


def config_sources(config_dir: str | Path) -> dict[str, list[str]]:
    """The files of ``config_dir`` under the folders of ``configs/``, and
    its ``generation_axes.json`` and ``dimensions.json``, as paths relative
    to it.

    Returns:
        ``overridden``: read in place of a packaged file; ``added``: read,
        with no packaged file of that name (a taxonomy of your own);
        ``ignored``: a dataset type other than ``eval`` and ``training``, or
        under ``jailbreak/``, ``llm/`` or ``datasets/``, and not read. Other
        files at the folder's top, such as recipes, are none.
    """
    root = Path(config_dir)
    files = sorted(
        [path.relative_to(root).as_posix()
         for folder in (*_READ_FOLDERS, *_IGNORED_FOLDERS)
         for path in (root / folder).rglob("*") if path.is_file()]
        + [name for name in _READ_FILES if (root / name).is_file()])
    read = [name for name in files if _is_read(name)]
    return {
        "overridden": [name for name in read
                       if (paths.configs_dir() / name).is_file()],
        "added": [name for name in read
                  if not (paths.configs_dir() / name).is_file()],
        "ignored": [name for name in files if name not in read],
    }


def _natural_allow(path: Path) -> list[str] | None:
    """The natural kind's ``families.allow`` in the augmentations file at
    ``path``, or None when it has none (the kind off, or every family
    allowed)."""
    natural = json.loads(path.read_text(encoding="utf-8")).get("natural")
    families = natural.get("families") if isinstance(natural, dict) else None
    allow = families.get("allow") if isinstance(families, dict) else None
    return allow if isinstance(allow, list) else None


def unseen_natural_families(config_dir: str | Path) -> list[str]:
    """The families the packaged ``augmentations.json`` allows the natural
    kind that the copy in ``config_dir`` does not: what a copy made before
    the library added them cannot draw. Empty without a copy, or when either
    file's natural kind has no allow list."""
    mine = Path(config_dir) / _AUGMENTATIONS
    if not mine.is_file():
        return []
    packaged = _natural_allow(paths.configs_dir() / _AUGMENTATIONS)
    copied = _natural_allow(mine)
    if packaged is None or copied is None:
        return []
    return [family for family in packaged if family not in copied]


def report_config_sources(
    config_dir: str | Path | None, *, verbose: bool = True,
) -> dict[str, list[str]]:
    """Warn about the files of ``config_dir`` that are not read and, with
    ``verbose``, log those that are. A copy of ``augmentations.json`` whose
    natural kind does not allow a family the packaged one allows is logged
    at INFO, naming the families (:func:`unseen_natural_families`), so a
    stale copy shows what it cannot draw.

    Returns:
        :func:`config_sources`, each list empty for no ``config_dir``.
    """
    if config_dir is None:
        return {"overridden": [], "added": [], "ignored": []}
    found = config_sources(config_dir)
    if found["ignored"]:
        logger.warning(
            "configs: %s in %s ignored: of dataset_types/ only eval.json and "
            "training.json are read, and the technique spec and templates, llm/ "
            "and datasets/ come from the library only.",
            ", ".join(found["ignored"]), config_dir)
    if verbose:
        read = found["overridden"] + found["added"]
        logger.info(
            "configs: %d file(s) read from %s%s, the rest from the library",
            len(read), config_dir, f" ({', '.join(read)})" if read else "")
    if unseen := unseen_natural_families(config_dir):
        logger.info(
            "configs: the natural kind of %s in %s does not allow %s, which the "
            "library's allows, so its variations do not draw them: add them to "
            "its natural families.allow, delete the copy, or write it again with "
            "scaffold_setup(..., overwrite=True), which replaces your edits.",
            _AUGMENTATIONS, config_dir, ", ".join(unseen))
    return found
