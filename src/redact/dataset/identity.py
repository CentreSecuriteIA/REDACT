"""A row's id: the content hash of its text, and of its text normalised."""

import hashlib
import re
from itertools import repeat

import pandas as pd


def content_id(text: str) -> str:
    """The MD5 of ``text``, truncated to 16 hex characters."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# Neither a letter, a digit, an underscore nor a space, in any script.
_PUNCTUATION = re.compile(r"[^\w\s]")


def normalised_id(text: str) -> str:
    """The content id of ``text`` in lower case, without punctuation, and with
    each run of whitespace one space: two texts that differ only in those
    have the same."""
    return content_id(" ".join(_PUNCTUATION.sub("", text.lower()).split()))


def row_id(row, text_col: str, id_col: str = "sample_id") -> str:
    """The id a row already carries in ``id_col``, else the content id of its text.

    An id is missing when it is ``None``, NaN or ``pd.NA``, or when its
    stripped string is empty or ``"nan"`` in any case. A set id is returned
    as a stripped string.
    """
    value = row.get(id_col)
    existing = "" if pd.isna(value) else str(value).strip()
    if existing and existing.lower() != "nan":
        return existing
    return content_id(str(row[text_col]))


def row_ids(frame: pd.DataFrame, text_col: str, id_col: str = "sample_id") -> list[str]:
    """:func:`row_id` of every row of ``frame``, in order.

    Reads the two columns, without building a row for each id.
    """
    given = frame[id_col] if id_col in frame.columns else repeat(None)
    return [
        row_id({id_col: value, text_col: text}, text_col, id_col)
        for value, text in zip(given, frame[text_col])
    ]
