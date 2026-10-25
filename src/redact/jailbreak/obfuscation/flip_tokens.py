"""Flip tokens: a token string a guard-in-the-loop search found to flip one
guard's verdict, appended to the prompt (EchoGram, Schulz and Yeung,
HiddenLayer 2025).

One technique per row of ``configs/jailbreak/obfuscation/flip_tokens.json``,
``to_flip_tokens_<slug>``: the spec's ``flip_tokens`` generator names it, and
this module builds the function from the same file, so a row adds a
technique and the import fails while the two disagree. The rows are to come
from a search run against a guard model on a GPU pod, which the library does
not hold yet; the packaged file ships with none, which registers the family
with no technique.

A technique appends its row's ``tokens`` after its ``separator`` (a space
unless the row says otherwise: the one the search appended with) and writes
``slug=<slug>`` as its info; the guard and the counts stay in the data file.
The tokens are specific to the guard they were found against: on another
guard they are a recognizable string at the end of a request, not a working
attack.
"""

import re
from collections.abc import Callable
from pathlib import Path

from redact import paths

from ..technique import data_rows

#: The data file, under ``configs/jailbreak/``.
FLIP_TOKEN_FILE = "obfuscation/flip_tokens.json"
#: The start of every technique name the file makes.
PREFIX = "to_flip_tokens_"
#: What every row gives besides its slug.
REQUIRED_FIELDS = ("tokens", "guard", "flips", "n_harmful", "n_benign", "since")
#: What a row may also give: the benign prompts the tokens turned flagged
#: (of ``n_benign``), the separator the search appended with, free notes,
#: and the keys the spec's generator reads (complexity, deprecated).
OPTIONAL_FIELDS = ("benign_flips", "separator", "notes", "complexity", "deprecated")
#: The separator a row without one is appended with.
SEPARATOR = " "
_SLUG = re.compile(r"[a-z0-9_]+")


def flip_token_rows(folder: str | Path | None = None) -> list[dict]:
    """The rows of the flip-token file under ``folder`` (``configs/jailbreak/``
    by default), each checked.

    Raises:
        ValueError: A row lacks a required field or gives an unknown one, its
            slug is not lower-case letters, digits and underscores, its
            tokens or guard is not a non-blank string, a count is not a whole
            number in range (``flips`` and ``benign_flips`` at most
            ``n_harmful`` and ``n_benign``, ``n_harmful`` at least 1), or its
            separator or notes is not a string. The message names the row.
    """
    folder = Path(folder) if folder is not None else paths.jailbreak_configs_dir()
    rows = data_rows(FLIP_TOKEN_FILE, folder)
    for row in rows:
        _check_row(row)
    return rows


def _check_row(row: dict) -> None:
    """Raise unless ``row`` is a flip-token row (:func:`flip_token_rows`)."""
    slug = row["slug"]
    if not _SLUG.fullmatch(slug):
        raise ValueError(
            f"{FLIP_TOKEN_FILE}: row {slug!r} must have a slug of lower-case "
            "letters, digits and underscores, the end of its technique name.")
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{FLIP_TOKEN_FILE}: row {slug!r} must give {missing}.")
    unknown = sorted(set(row) - {"slug", *REQUIRED_FIELDS, *OPTIONAL_FIELDS})
    if unknown:
        raise ValueError(
            f"{FLIP_TOKEN_FILE}: row {slug!r} gives unknown field(s) {unknown}; "
            f"write only slug, {list(REQUIRED_FIELDS)} and {list(OPTIONAL_FIELDS)}.")
    for field in ("tokens", "guard"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(
                f"{FLIP_TOKEN_FILE}: row {slug!r} must give {field!r} as a "
                f"non-blank string, got {row[field]!r}.")
    for field in ("separator", "notes"):
        if field in row and not isinstance(row[field], str):
            raise ValueError(
                f"{FLIP_TOKEN_FILE}: the {field} of row {slug!r} must be a string, "
                f"got {row[field]!r}.")
    _check_count(row, "n_harmful", low=1)
    _check_count(row, "n_benign", low=0)
    _check_count(row, "flips", low=0, high=row["n_harmful"])
    if "benign_flips" in row:
        _check_count(row, "benign_flips", low=0, high=row["n_benign"])


def _check_count(row: dict, field: str, *, low: int, high: int | None = None) -> None:
    """Raise unless ``row[field]`` is a whole number from ``low`` to ``high``."""
    value = row[field]
    if (type(value) is not int or value < low
            or (high is not None and value > high)):
        top = "" if high is None else f" and at most {high}"
        raise ValueError(
            f"{FLIP_TOKEN_FILE}: the {field} of row {row['slug']!r} must be a whole "
            f"number of at least {low}{top}, got {value!r}.")


def make_flip_token_fn(row: dict) -> Callable[[str], tuple[str, str]]:
    """The technique of one row: the prompt, the row's separator, then its
    tokens; its info names the row's slug."""
    tokens, slug = row["tokens"], row["slug"]
    separator = row.get("separator", SEPARATOR)

    def flip_tokens(prompt: str) -> tuple[str, str]:
        return f"{prompt}{separator}{tokens}", f"slug={slug}"

    flip_tokens.__name__ = flip_tokens.__qualname__ = f"{PREFIX}{slug}"
    return flip_tokens


#: One technique per row of the packaged file, in the file's order.
FLIP_TOKEN_FUNCTIONS = [make_flip_token_fn(row) for row in flip_token_rows()]
globals().update({fn.__name__: fn for fn in FLIP_TOKEN_FUNCTIONS})


def get_flip_token_functions() -> list[Callable]:
    """The flip-token techniques, one per row of the packaged file."""
    return list(FLIP_TOKEN_FUNCTIONS)
