"""A row's id: the content hash of its text."""

import hashlib

import pandas as pd


def content_id(text: str) -> str:
    """The MD5 of ``text``, truncated to 16 hex characters."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


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
