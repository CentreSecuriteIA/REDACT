"""The share of a text's letters in a script, which the code-switch gate of a
non-Latin second language reads."""

import pytest

from redact.jailbreak.obfuscation import code_switching as cs
from redact.jailbreak.obfuscation.scripts import _RANGES, in_script, script_share

#: Kanji (Han), hiragana and katakana, one word each, then a kanji word.
_KANJI, _HIRAGANA, _KATAKANA = "天気", "きょう", "テスト"


def test_japanese_is_kanji_hiragana_and_katakana():
    """Written Japanese mixes all three: a kanji is as Japanese as a kana."""
    for word in (_KANJI, _HIRAGANA, _KATAKANA):
        assert script_share(word, "Jpan") == 1.0, word
    assert script_share(_KANJI, "Hani") == 1.0
    assert script_share(_KANJI + _HIRAGANA, "Hira") == pytest.approx(3 / 5)
    assert not in_script("a", "Jpan") and not in_script("д", "Jpan")


def test_only_letters_count_and_no_letter_is_no_share():
    assert script_share("天気 2024!", "Jpan") == 1.0
    assert script_share("2024 !?", "Jpan") == 0.0
    assert script_share("plain words", "Latn") == 1.0
    assert script_share("plain 天気", "Latn") == pytest.approx(5 / 7)


def test_every_code_switch_script_has_its_letters():
    assert {row["script"] for row in cs.LANGUAGES} <= set(_RANGES)


def test_a_japanese_mix_in_kanji_passes_the_code_gate():
    """A rewrite whose Japanese words are kanji, about a quarter of its letters,
    mixes the two languages; without its kanji counted it would read as no
    Japanese at all."""
    japanese = next(row for row in cs.LANGUAGES if row["slug"] == "japanese")
    original = "Please tell me about the weather today."
    mixed = "Please tell me about the 天気 today, 今日の天気予報."
    assert cs._code_gate(mixed, original, japanese) == ""
    assert cs._code_gate(original, original, japanese) == "no Japanese"
