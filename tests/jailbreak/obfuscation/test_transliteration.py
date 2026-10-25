"""Rule-based transliteration: deterministic, lossy, script-only rewrites."""

import hashlib
import itertools
import json

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation.transliteration import (
    _TABLES,
    _transliterate,
    get_transliteration_functions,
)
from redact.jailbreak.technique import applies

_SCRIPTS = sorted(_TABLES)
_SAMPLE = "Make a plan at 3pm, please!"


def _fn(script: str):
    return {fn.__name__: fn
            for fn in get_transliteration_functions()}[f"to_{script}_transliteration"]


def test_there_is_one_technique_per_script():
    assert sorted(fn.__name__ for fn in get_transliteration_functions()) == sorted(
        f"to_{s}_transliteration" for s in _SCRIPTS)


@pytest.mark.parametrize("script", _SCRIPTS)
def test_latin_letters_are_rewritten_and_nothing_else_is(script):
    out = _transliterate(_SAMPLE, _TABLES[script])
    assert out != _SAMPLE
    # No ASCII letter survives; digits, spaces and punctuation are kept.
    assert not any(c.isascii() and c.isalpha() for c in out)
    assert "3" in out
    assert out.count(" ") == _SAMPLE.count(" ")
    assert "," in out and "!" in out


@pytest.mark.parametrize("script", _SCRIPTS)
def test_a_second_pass_changes_nothing(script):
    """The output holds no ASCII letter, so transliteration is idempotent."""
    once = _transliterate(_SAMPLE, _TABLES[script])
    assert _transliterate(once, _TABLES[script]) == once


@pytest.mark.parametrize("script", _SCRIPTS)
def test_text_of_another_script_is_left_as_it_is(script):
    """``needs_latin``: a text with no ASCII letter is returned unchanged,
    so a transliteration after a translation into another script is a
    no-op (the rules keep the two apart besides)."""
    for other in ("привет 5!", "こんにちは。", "مرحبا ٣", "नमस्ते"):
        assert _transliterate(other, _TABLES[script]) == other


@pytest.mark.parametrize("script", _SCRIPTS)
def test_the_technique_applies_only_to_a_text_with_a_latin_letter(script):
    fn = _fn(script)
    assert applies(fn, "abc", "harmful")
    assert not applies(fn, "привет", "harmful")
    assert not applies(fn, "123 ... !", "benign")


def test_it_is_lossy_so_distinct_letters_can_collide():
    """Phonetic, not invertible: in Cyrillic ``c``, ``k`` and ``q`` all map to
    к, so the rendering cannot be reversed to the original letters."""
    table = _TABLES["cyrillic"]
    assert table["c"] == table["k"] == table["q"]
    assert _transliterate("cat", table) == _transliterate("kat", table)


@pytest.mark.parametrize("script", _SCRIPTS)
def test_output_is_entirely_in_the_target_script_or_kept_characters(script):
    """Every mapped grapheme is a non-ASCII character of the target script's
    block (the table's values), so the result is readable text, not mojibake."""
    for value in _TABLES[script].values():
        assert value and not any(c.isascii() and c.isalpha() for c in value)


# --- the tables and what they write, pinned ----------------------------------
# The family is in the eval pool, so its text may move only in an eval
# review: each table is pinned by digest, and so is what it writes for a
# sample that holds every key of every table as a word of its own (longest
# first), then a sentence with case, digraphs inside words, digits,
# punctuation, a non-ASCII letter and braces. An edited or swapped entry moves
# the first digest; a change to the matching (digraphs, case, what is kept)
# moves the second. tests/fixtures/template_draws.json pins the techniques on
# its own probes too, under its eval-review guard.

_KEYS = sorted({key for table in _TABLES.values() for key in table},
               key=lambda key: (-len(key), key))
_PINNED_SAMPLE = (" ".join(_KEYS)
                  + " | The Quick, BROWN fox; Psalm 3pm, x-ray Shchi? café {braces}")

#: Per script: the first 16 hex digits of the SHA-256 of the table (its JSON,
#: keys sorted) and of what it writes for :data:`_PINNED_SAMPLE`.
_PINNED = {
    "arabic": ("720a63b3d09e0600", "e96953bacfe12488"),
    "cyrillic": ("ccbb60a5f2eba369", "16ec51cedb779efe"),
    "devanagari": ("fd236fd0a60b8dd6", "ddcf7ed56057f790"),
    "greek": ("56d8c57ace667f4b", "5d78e382c7fc3743"),
}


def _short_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_every_table_is_pinned():
    assert sorted(_PINNED) == _SCRIPTS


@pytest.mark.parametrize("script", _SCRIPTS)
def test_the_table_and_what_it_writes_are_pinned(script):
    table = _TABLES[script]
    assert (_short_digest(json.dumps(table, sort_keys=True, ensure_ascii=False)),
            _short_digest(_transliterate(_PINNED_SAMPLE, table))) == _PINNED[script]


@pytest.mark.parametrize("script", _SCRIPTS)
def test_every_key_alone_is_written_as_its_entry(script):
    """A digraph or trigraph wins over its letters (``sh`` is one grapheme,
    not ``s`` then ``h``), in either case."""
    table = _TABLES[script]
    for key, value in table.items():
        assert _transliterate(key, table) == value, key
        assert _transliterate(key.upper(), table) == value, key


@pytest.mark.parametrize("script", _SCRIPTS)
def test_swapping_any_two_entries_changes_the_pinned_text(script):
    """The sample holds every key, so the pinned output catches a swap of
    any two entries that write different text."""
    table = _TABLES[script]
    pinned = _transliterate(_PINNED_SAMPLE, table)
    for first, second in itertools.combinations(table, 2):
        if table[first] == table[second]:
            continue
        swapped = {**table, first: table[second], second: table[first]}
        assert _transliterate(_PINNED_SAMPLE, swapped) != pinned, (first, second)
