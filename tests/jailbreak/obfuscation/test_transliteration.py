"""Rule-based transliteration: deterministic, lossy, script-only rewrites."""

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
