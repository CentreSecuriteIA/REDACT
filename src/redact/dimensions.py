"""The dimensions a text varies along, each on one side: ``augmentation``,
which transforms a finished text, or ``generation``, which shapes a text as
it is written.

``configs/dimensions.json`` holds each tag's ``side`` and its one-sentence
``definition``; a generation tag may name, in ``same_as``, the augmentation
tags that are the same dimension, and in ``undecided`` those the owner has
yet to rule on, checked as the same until then. Each augmentation family and
each generation axis carries one tag as its ``dimension``, and :func:`check`
refuses a generation axis and an augmentation family that vary one dimension.
"""

import json
from collections.abc import Mapping
from functools import lru_cache

from . import paths

#: The sides a dimension belongs to.
SIDES = ("generation", "augmentation")

_KEYS = ("side", "definition")
_RELATIONS = ("same_as", "undecided")


@lru_cache(maxsize=1)
def table() -> dict:
    """Each tag of ``configs/dimensions.json``, without comments, checked.

    Raises:
        ValueError: A tag without a known side or a definition, with a key the
            table does not hold, or naming in ``same_as`` or ``undecided`` a
            tag that is not an augmentation tag, or one in both.
    """
    path = paths.configs_dir() / "dimensions.json"
    tags = {}
    for tag, entry in json.loads(path.read_text(encoding="utf-8")).items():
        if tag.startswith("_"):
            continue
        at = f"{path.name}: {tag}"
        if not isinstance(entry, dict) or not set(_KEYS) <= set(entry) <= {
                *_KEYS, *(_RELATIONS if entry.get("side") == "generation" else ())}:
            raise ValueError(
                f"{at} must hold {list(_KEYS)}, and a generation tag may hold "
                f"{list(_RELATIONS)}; nothing else.")
        if entry["side"] not in SIDES:
            raise ValueError(f"{at}: side must be one of {list(SIDES)}.")
        if not isinstance(entry["definition"], str) or not entry["definition"].strip():
            raise ValueError(f"{at}: the definition must be a sentence.")
        tags[tag] = {**dict.fromkeys(_RELATIONS, []), **entry}
    augmentation = [tag for tag, entry in tags.items()
                    if entry["side"] == "augmentation"]
    for tag, entry in tags.items():
        same, undecided = (entry[key] for key in _RELATIONS)
        if not all(isinstance(names, list) and set(names) <= set(augmentation)
                   for names in (same, undecided)) or set(same) & set(undecided):
            raise ValueError(
                f"{path.name}: {tag}: same_as and undecided must each list "
                f"augmentation tags, none in both; they are {augmentation}.")
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


def relation(generation: str, augmentation: str) -> str | None:
    """``"same"`` when the two tags are one dimension (one tag, or named in
    ``same_as``), ``"undecided"`` when the owner has yet to rule, else None."""
    entry = table()[generation]
    if generation == augmentation or augmentation in entry["same_as"]:
        return "same"
    if augmentation in entry["undecided"]:
        return "undecided"
    return None


def check(varied: Mapping[str, str], drawn: Mapping[str, Mapping[str, str]]) -> None:
    """Raise when a generation axis and an augmentation family vary one dimension.

    ``varied`` maps each axis that varies, as ``stage.axis``, to its tag;
    ``drawn`` maps each kind of the augmentation section to the families it
    draws, each with its tag.

    Raises:
        ValueError: One line per axis and dimension it shares: the axis, the
            families of each kind, and the two ways out.
    """
    problems = []
    for name, tag in varied.items():
        stage, axis = name.split(".", 1)
        for other in tags_of("augmentation"):
            how = relation(tag, other)
            families = {kind: sorted(f for f, t in drawn_tags.items() if t == other)
                        for kind, drawn_tags in drawn.items()}
            families = {kind: names for kind, names in families.items() if names}
            if how is None or not families:
                continue
            by_kind = "; ".join(f"{kind}: {', '.join(names)}"
                                for kind, names in families.items())
            undecided = (" The two are undecided: the owner has yet to rule "
                         "whether they are one dimension, and until then they "
                         "are checked as one." if how == "undecided" else "")
            problems.append(
                f"- the axis {name} ({tag}) and the {other} families ({by_kind})."
                f"{undecided} "
                f'Give the axis one value or leave it out (params {stage}.axes, '
                f'as {{"{axis}": null}}), or deny those families in each kind '
                "(augmentations.<kind>.families.deny).")
    if problems:
        raise ValueError(
            "A generation axis and augmentation families vary the same dimension "
            "(configs/dimensions.json):\n" + "\n".join(problems))
