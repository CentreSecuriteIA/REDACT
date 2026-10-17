"""The dimensions a text varies along, each on one side: ``augmentation``,
which transforms a finished text, or ``generation``, which shapes a text as
it is written.

``configs/dimensions.json`` holds each tag's ``side`` and its one-sentence
``definition``. Each augmentation family and each generation axis carries one
tag as its ``dimension``.
"""

import json
from collections.abc import Mapping
from functools import lru_cache

from . import paths

#: The sides a dimension belongs to.
SIDES = ("generation", "augmentation")

_KEYS = ("side", "definition")


@lru_cache(maxsize=1)
def table() -> dict:
    """Each tag of ``configs/dimensions.json``, without comments, checked.

    Raises:
        ValueError: A tag without a known side or a definition, or with a key
            the table does not hold.
    """
    path = paths.configs_dir() / "dimensions.json"
    tags = {}
    for tag, entry in json.loads(path.read_text(encoding="utf-8")).items():
        if tag.startswith("_"):
            continue
        at = f"{path.name}: {tag}"
        if not isinstance(entry, dict) or set(entry) != set(_KEYS):
            raise ValueError(f"{at} must hold {list(_KEYS)} and nothing else.")
        if entry["side"] not in SIDES:
            raise ValueError(f"{at}: side must be one of {list(SIDES)}.")
        if not isinstance(entry["definition"], str) or not entry["definition"].strip():
            raise ValueError(f"{at}: the definition must be a sentence.")
        tags[tag] = entry
    return tags


def tags_of(side: str) -> list[str]:
    """The tags of ``side``, in the table's order."""
    return [tag for tag, entry in table().items() if entry["side"] == side]


def check_tags(where: str, dimensions: Mapping[str, object], side: str) -> None:
    """Raise unless each value of ``dimensions`` (a name: its tag) is a tag of
    ``side``.

    Raises:
        ValueError: Every name without one, then the tags of ``side``.
    """
    known = tags_of(side)
    wrong = [f"{name} ({tag!r})" if tag is not None else name
             for name, tag in dimensions.items() if tag not in known]
    if wrong:
        raise ValueError(
            f"{where}: {', '.join(wrong)} must carry a dimension of the {side} "
            f"side, one of {known} (configs/dimensions.json).")
