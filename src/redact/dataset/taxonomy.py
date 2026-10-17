"""The taxonomy and seed JSON files under ``configs/``: write a taxonomy, load
them, and read what generation needs.

A taxonomy file::

    {
        "name": "...",
        "description": "...",
        "categories": {
            "CategoryName": {"description": "...", "subcategories": [...] or {...}},
            ...
        },
        "aliases": {"AltName": "CanonicalName", ...},
        "groups": {"group_name": ["Cat1", "Cat2"], ...}
    }

A seeds file maps each category to its hand-written prompts under ``"seeds"``.
"""

import json
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

from .. import paths

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = paths.taxonomy_dir()
_DEFAULT_SEEDS_DIR = paths.seeds_dir()

# Former file names, read for one release: former -> current.
_FORMER_TAXONOMIES = {"content_moderation_categories": "harmful"}
_FORMER_SEEDS = {"content_moderation_seeds": "harmful"}


def load_taxonomy(
    name: str,
    config_dir: str | Path | None = None,
) -> dict:
    """Load the taxonomy file ``name``, with or without its ``.json``.

    ``config_dir`` defaults to the packaged ``configs/taxonomy``. The former
    name ``content_moderation_categories`` loads ``harmful``, and a directory
    whose file has the former name is read for ``harmful``, each with a
    ``DeprecationWarning``, for one release.

    Raises:
        FileNotFoundError: The directory holds no such file.
    """
    if config_dir is None:
        config_dir = _DEFAULT_CONFIG_DIR
    return _load_json(config_dir, name, "Taxonomy", _FORMER_TAXONOMIES)


def create_taxonomy(
    name: str,
    categories: dict[str, str | dict],
    description: str = "",
    aliases: dict[str, str] | None = None,
    groups: dict[str, list[str]] | None = None,
    taxonomy_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Write the taxonomy file ``name`` and return the taxonomy it holds.

    A category given as a bare description is written as
    ``{"description": ...}``. ``taxonomy_dir`` defaults to the packaged
    ``configs/taxonomy``, where :func:`load_taxonomy` reads by default.
    ``verbose`` logs the path written.
    """
    taxonomy = {
        "name": name,
        "description": description,
        "categories": {
            category: {"description": info} if isinstance(info, str) else info
            for category, info in categories.items()
        },
        "aliases": aliases or {},
        "groups": groups or {},
    }
    directory = Path(taxonomy_dir or _DEFAULT_CONFIG_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    if verbose:
        logger.info("Taxonomy '%s' saved to %s", name, path)
    return taxonomy


def iter_categories(taxonomy: dict) -> Iterator[tuple[str, dict]]:
    """Yield each category of a taxonomy as ``(name, info)``.

    ``info`` is the category's dict. A category given as a bare description
    yields ``{"description": ...}``.
    """
    for name, info in taxonomy.get("categories", {}).items():
        if isinstance(info, dict):
            yield name, info
        else:
            yield name, {"description": str(info)}


def load_seeds(
    name: str,
    seeds_dir: str | Path | None = None,
) -> dict:
    """Load the seeds file ``name``, with or without its ``.json``.

    ``seeds_dir`` defaults to the packaged ``configs/seeds``. The former name
    ``content_moderation_seeds`` loads ``harmful``, with a
    ``DeprecationWarning``, for one release.

    Raises:
        FileNotFoundError: The directory holds no such file.
    """
    if seeds_dir is None:
        seeds_dir = _DEFAULT_SEEDS_DIR
    return _load_json(seeds_dir, name, "Seeds file", _FORMER_SEEDS)


def get_seed_prompts(
    seeds: dict,
    category: str,
) -> str:
    """A category's seed prompts as a numbered list, one per line.

    The text a generation template takes for ``{SeedPrompts}``. A category
    without seeds gives ``""``.
    """
    prompts = seeds.get("seeds", {}).get(category, [])
    if not prompts:
        return ""
    return "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))


def _load_json(
    directory: str | Path, name: str, what: str, former: dict[str, str],
) -> dict:
    """Parse ``name`` under ``directory``; ``what`` names the file in the error.

    A file missing under ``name`` is read under the name ``former`` pairs it
    with (a former name with its current one, or the other way round), with
    a ``DeprecationWarning``.
    """
    path = Path(directory) / name
    if not path.suffix:
        path = path.with_suffix(".json")

    if not path.exists():
        path = _under_other_name(path, what, former)

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _under_other_name(path: Path, what: str, former: dict[str, str]) -> Path:
    """``path`` under the name ``former`` pairs its name with, with a warning.

    Raises:
        FileNotFoundError: Neither name is a file.
    """
    name = path.stem
    paired = former.get(name) or next(
        (old for old, current in former.items() if current == name), None)
    other = path.with_name(f"{paired}.json")
    if paired is None or not other.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    if name in former:
        message = (f"{what} {name!r} is now {paired!r}; load it by that name. "
                   "The former name is read for this release.")
    else:
        message = (f"{other} has the former name of {name!r}; rename it "
                   f"{path.name}. It is read for this release.")
    warnings.warn(message, DeprecationWarning, stacklevel=4)
    return other
