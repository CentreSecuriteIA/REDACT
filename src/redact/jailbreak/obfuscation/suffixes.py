"""Suffix techniques: a random suffix of punctuation, fragments, symbols or
emoji, and the adversarial suffixes published in the literature.

A noise suffix's ``info`` is ``suffix=<suffix>``. The published suffixes are
data, one row each of ``configs/jailbreak/obfuscation/suffix_strings.json``
(:data:`SUFFIX_FILE`), which the spec's ``suffixes`` generator also reads: a
row makes ``to_adversarial_suffix_<slug>``, which appends the row's suffix
after a space, as the noise suffixes do, and writes
``slug=<slug>;source=<source>``. A row without its fields fails the import,
naming the row (:func:`suffix_rows`).
"""

import random
import re
from collections.abc import Callable
from pathlib import Path

from ..technique import data_rows

#: The data file of the published suffixes, under ``configs/jailbreak/``.
SUFFIX_FILE = "obfuscation/suffix_strings.json"
#: What every row of :data:`SUFFIX_FILE` gives, each a non-empty string.
REQUIRED_FIELDS = ("suffix", "source", "optimized_against", "since")
#: A row's slug: the end of its technique's name, so a Python name's letters.
_SLUG = re.compile(r"[a-z0-9_]+")
#: The noise suffixes' slugs, which a row may not take.
NOISE_SLUGS = ("punctuation", "fragments", "unicode", "emoji")


def to_adversarial_suffix_punctuation(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Add punctuation-based adversarial suffix."""
    rng = rng or random
    punctuation_types = ["!", "_", "=", "*", "/", '"', "~", "|", "bracket"]
    length = rng.randint(8, 40)

    suffix = ""
    i = 0
    while i < length:
        punct_type = rng.choice(punctuation_types)
        repeat_count = rng.randint(1, min(5, length - i))

        if punct_type == "bracket":
            bracket_type = rng.choice(["{", "[", "("])
            close_bracket = (
                "}" if bracket_type == "{" else
                "]" if bracket_type == "[" else ")"
            )
            max_pairs = min(repeat_count, (length - i) // 2)
            if max_pairs > 0:
                suffix += bracket_type * max_pairs + close_bracket * max_pairs
                i += max_pairs * 2
            else:
                i += 1
        else:
            suffix += punct_type * repeat_count
            i += repeat_count

    return prompt + " " + suffix, f"suffix={suffix}"


def to_adversarial_suffix_fragments(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Add word fragment-based adversarial suffix."""
    rng = rng or random
    words = [
        "inst", "direct", "inverse", "desc", "steps",
        "style", "output", "format", "theorem", "logic",
    ]
    length = rng.randint(2, 10)
    suffix = " ".join(rng.choice(words) for _ in range(length))
    return prompt + " " + suffix, f"suffix={suffix}"


def to_adversarial_suffix_unicode(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Add Unicode symbol-based adversarial suffix."""
    rng = rng or random
    symbols = ["\u2200", "\u2203", "\u2211", "\u21d2", "\u222b", "\u03bb", "\u03a9", "\u03b4"]
    length = rng.randint(3, 10)
    suffix = " ".join(rng.choice(symbols) for _ in range(length))
    return prompt + " " + suffix, f"suffix={suffix}"


def to_adversarial_suffix_emoji(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Add emoji-based adversarial suffix."""
    rng = rng or random
    emojis = ["\U0001f513", "\u26a0\ufe0f", "\u2705", "\U0001f680",
              "\U0001f525", "\U0001f916", "\U0001f4ac", "\U0001f4af"]
    length = rng.randint(5, 20)

    suffix = ""
    i = 0
    while i < length:
        emoji = rng.choice(emojis)
        repeat_count = rng.randint(1, min(5, length - i))
        suffix += emoji * repeat_count
        i += repeat_count

    return prompt + " " + suffix, f"suffix={suffix}"


def suffix_rows(folder: str | Path | None = None) -> list[dict]:
    """The rows of :data:`SUFFIX_FILE` under ``folder`` (the packaged
    ``configs/jailbreak/`` by default), each checked: a slug of lower-case
    letters, digits and underscores that is none of :data:`NOISE_SLUGS`,
    and every one of :data:`REQUIRED_FIELDS` a non-empty string, the
    ``source`` without a ``;`` (it goes into the info, whose fields ``;``
    separates); ``notes``, when given, a string.

    Raises:
        ValueError: A row breaks one of these, named by its slug (or its
            place, without one: :func:`~redact.jailbreak.technique.data_rows`).
    """
    rows = data_rows(SUFFIX_FILE, *([Path(folder)] if folder is not None else []))
    for row in rows:
        _check_row(row)
    return rows


def _check_row(row: dict) -> None:
    """Raise unless ``row`` is a published suffix's row (:func:`suffix_rows`)."""
    slug = row["slug"]
    if not _SLUG.fullmatch(slug) or slug in NOISE_SLUGS:
        raise ValueError(
            f"{SUFFIX_FILE}: row {slug!r} must have a slug of lower-case letters, "
            f"digits and underscores, none of {list(NOISE_SLUGS)} (the noise "
            "suffixes' names).")
    missing = [field for field in REQUIRED_FIELDS
               if not isinstance(row.get(field), str) or not row[field].strip()]
    if missing:
        raise ValueError(
            f"{SUFFIX_FILE}: row {slug!r} must give {missing}, each a non-empty "
            f"string; every row gives {list(REQUIRED_FIELDS)}.")
    if ";" in row["source"]:
        raise ValueError(
            f"{SUFFIX_FILE}: the source of row {slug!r} may not hold a ';', which "
            "separates the fields of a technique's info.")
    if not isinstance(row.get("notes", ""), str):
        raise ValueError(
            f"{SUFFIX_FILE}: the notes of row {slug!r} must be a string, got "
            f"{row['notes']!r}.")


def make_suffix_fn(row: dict) -> Callable:
    """The technique of the published suffix ``row``,
    ``to_adversarial_suffix_<slug>``: the prompt, a space, the suffix; its
    info ``slug=<slug>;source=<source>``."""
    suffix, info = row["suffix"], f"slug={row['slug']};source={row['source']}"

    def fn(prompt: str) -> tuple[str, str]:
        return prompt + " " + suffix, info

    fn.__name__ = fn.__qualname__ = f"to_adversarial_suffix_{row['slug']}"
    fn.__doc__ = f"Published adversarial suffix {row['slug']!r} ({row['source']})."
    return fn


#: The four noise suffixes, written out above.
NOISE_SUFFIXES = [
    to_adversarial_suffix_punctuation,
    to_adversarial_suffix_fragments,
    to_adversarial_suffix_unicode,
    to_adversarial_suffix_emoji,
]
# One technique per published suffix, in the file's order:
# to_adversarial_suffix_<slug>.
_PUBLISHED = [make_suffix_fn(row) for row in suffix_rows()]
globals().update({fn.__name__: fn for fn in _PUBLISHED})


def get_suffix_functions() -> list[Callable]:
    """Return all adversarial suffix technique functions: the four noise
    suffixes, then one per row of :data:`SUFFIX_FILE`."""
    return [*NOISE_SUFFIXES, *_PUBLISHED]
