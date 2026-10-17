"""Casual writing: the prompt with the casing, spacing, punctuation, abbreviations
or accents a person typing quickly gives it.

``info`` is empty.
"""

import random
import re
import unicodedata
from collections.abc import Callable

# Share of a prompt's spaces that to_dropped_spaces and to_doubled_spaces change.
_SPACE_SHARE = 0.2

_ABBREVIATIONS = {
    "you": "u", "your": "ur", "are": "r", "please": "pls", "because": "bc",
    "thanks": "thx", "with": "w/", "without": "w/o", "people": "ppl",
    "something": "sth", "really": "rly", "about": "abt", "before": "b4",
    "tomorrow": "tmrw", "message": "msg", "information": "info",
    "though": "tho", "through": "thru", "okay": "ok", "probably": "prob",
}
_ABBREVIATED = re.compile(
    r"\b(" + "|".join(map(re.escape, _ABBREVIATIONS)) + r")\b", re.IGNORECASE)


def to_lowercase(prompt: str) -> tuple[str, str]:
    """Write the prompt in lower case."""
    return prompt.lower(), ""


def to_uppercase(prompt: str) -> tuple[str, str]:
    """Write the prompt in capitals."""
    return prompt.upper(), ""


def to_no_sentence_capitals(prompt: str) -> tuple[str, str]:
    """Start each sentence with a small letter."""
    return re.sub(r"(^|[.!?]\s+)(\w)",
                  lambda m: m.group(1) + m.group(2).lower(), prompt), ""


def _change_spaces(prompt: str, rng, replacement: str) -> str:
    """``prompt`` with a share of its spaces, at least one, made ``replacement``."""
    spaces = [i for i, c in enumerate(prompt) if c == " "]
    if not spaces:
        return prompt
    chosen = set(rng.sample(spaces, max(1, round(len(spaces) * _SPACE_SHARE))))
    return "".join(replacement if i in chosen else c for i, c in enumerate(prompt))


def to_dropped_spaces(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Drop some of the spaces between words."""
    return _change_spaces(prompt, rng or random, ""), ""


def to_doubled_spaces(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Double some of the spaces between words."""
    return _change_spaces(prompt, rng or random, "  "), ""


def to_dropped_punctuation(prompt: str) -> tuple[str, str]:
    """Drop the punctuation, apostrophes, quotes and dashes, except a point
    inside a number."""
    return re.sub(r"[.,;:!?'’‘“”«»—–]+(?!\d)", "", prompt), ""


def to_doubled_punctuation(prompt: str) -> tuple[str, str]:
    """Double each full stop, question mark and exclamation mark that ends a word."""
    return re.sub(r"(?<![.!?])([.!?])(?=\s|$)", r"\1\1", prompt), ""


def to_abbreviations(prompt: str) -> tuple[str, str]:
    """Write common words as their chat abbreviations."""
    return _ABBREVIATED.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], prompt), ""


def to_stripped_accents(prompt: str) -> tuple[str, str]:
    """Write accented letters without their accents."""
    decomposed = unicodedata.normalize("NFD", prompt)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped), ""


def get_casual_writing_functions() -> list[Callable]:
    """Return all casual writing technique functions."""
    return [
        to_lowercase,
        to_uppercase,
        to_no_sentence_capitals,
        to_dropped_spaces,
        to_doubled_spaces,
        to_dropped_punctuation,
        to_doubled_punctuation,
        to_abbreviations,
        to_stripped_accents,
    ]
