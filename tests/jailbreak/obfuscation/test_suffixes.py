"""Tests for adversarial suffix functions."""

import random

from redact.jailbreak.obfuscation.suffixes import (
    get_suffix_functions,
    to_adversarial_suffix_emoji,
    to_adversarial_suffix_fragments,
    to_adversarial_suffix_punctuation,
    to_adversarial_suffix_unicode,
)

SAMPLE = "test prompt"


class TestSuffixFunctions:
    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_suffix_functions():
            result, suffix = fn(SAMPLE)
            assert result.startswith(SAMPLE)

    def test_suffix_appended(self):
        random.seed(42)
        for fn in get_suffix_functions():
            result, suffix = fn(SAMPLE)
            assert len(result) > len(SAMPLE)
            assert suffix  # info string is non-empty (contains the suffix)

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_suffix_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_count(self):
        assert len(get_suffix_functions()) == 4

    def test_deterministic_with_seed(self):
        for fn in get_suffix_functions():
            random.seed(123)
            r1, _ = fn(SAMPLE)
            random.seed(123)
            r2, _ = fn(SAMPLE)
            assert r1 == r2, f"{fn.__name__} not deterministic with same seed"


class TestPunctuationSuffix:
    def test_contains_punctuation(self):
        random.seed(42)
        _, suffix = to_adversarial_suffix_punctuation(SAMPLE)
        assert any(c in suffix for c in "!_=*/\"~|{}[]()")


class TestFragmentsSuffix:
    def test_contains_words(self):
        random.seed(42)
        _, suffix = to_adversarial_suffix_fragments(SAMPLE)
        assert " " in suffix  # multiple words separated by spaces


class TestUnicodeSuffix:
    def test_contains_unicode(self):
        random.seed(42)
        _, suffix = to_adversarial_suffix_unicode(SAMPLE)
        assert any(ord(c) > 127 for c in suffix)


class TestEmojiSuffix:
    def test_contains_emoji(self):
        random.seed(42)
        _, suffix = to_adversarial_suffix_emoji(SAMPLE)
        assert any(ord(c) > 127 for c in suffix)
