"""A row's id: the content hash of its text."""

import hashlib


def content_id(text: str) -> str:
    """The MD5 of ``text``, truncated to 16 hex characters."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
