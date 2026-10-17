"""A user's config folder, overlaid file by file over the packaged ``configs/``.

The folder mirrors ``configs/``: ``dataset_types/<type>.json``,
``taxonomy/<name>.json`` and ``jailbreak/augmentations.json`` are each read
from it when it has the file, else from the package. Nothing else is: the
technique spec and templates under ``jailbreak/``, ``llm/`` and
``datasets/`` are loaded at import or tied to code, and a file under them is
warned about as ignored.
"""

import logging
from pathlib import Path

from redact import paths

logger = logging.getLogger(__name__)

# The folders whose files a config folder may hold, each read one by one.
_READ_FOLDERS = ("dataset_types", "taxonomy")
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
    return relative == _AUGMENTATIONS or (
        folder in _READ_FOLDERS and "/" not in name and name.endswith(".json"))


def config_sources(config_dir: str | Path) -> dict[str, list[str]]:
    """The files of ``config_dir`` under the folders of ``configs/``, as
    paths relative to it.

    Returns:
        ``overridden``: read in place of a packaged file; ``added``: read,
        with no packaged file of that name (a taxonomy of your own);
        ``ignored``: under ``jailbreak/``, ``llm/`` or ``datasets/`` and not
        read. Files elsewhere in the folder, such as recipes, are none.
    """
    root = Path(config_dir)
    files = sorted(
        path.relative_to(root).as_posix()
        for folder in (*_READ_FOLDERS, *_IGNORED_FOLDERS)
        for path in (root / folder).rglob("*") if path.is_file())
    read = [name for name in files if _is_read(name)]
    return {
        "overridden": [name for name in read
                       if (paths.configs_dir() / name).is_file()],
        "added": [name for name in read
                  if not (paths.configs_dir() / name).is_file()],
        "ignored": [name for name in files if name not in read],
    }


def report_config_sources(
    config_dir: str | Path | None, *, verbose: bool = True,
) -> dict[str, list[str]]:
    """Warn about the files of ``config_dir`` that are not read and, with
    ``verbose``, log those that are.

    Returns:
        :func:`config_sources`, each list empty for no ``config_dir``.
    """
    if config_dir is None:
        return {"overridden": [], "added": [], "ignored": []}
    found = config_sources(config_dir)
    if found["ignored"]:
        logger.warning(
            "configs: %s in %s ignored: the technique spec and templates, llm/ "
            "and datasets/ come from the library only.",
            ", ".join(found["ignored"]), config_dir)
    if verbose:
        read = found["overridden"] + found["added"]
        logger.info(
            "configs: %d file(s) read from %s%s, the rest from the library",
            len(read), config_dir, f" ({', '.join(read)})" if read else "")
    return found
