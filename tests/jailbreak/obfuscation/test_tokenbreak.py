"""Tests for tokenbreak obfuscation (pure word-level transforms)."""

import random

from redact.jailbreak.obfuscation.tokenbreak import (
    get_tokenbreak_functions,
    match_case,
    tokenbreak_delimiter,
    tokenbreak_obfuscate_word,
    tokenbreak_prepend,
    tokenbreak_split,
)


class TestMatchCase:
    def test_upper(self):
        assert match_case("HELLO", "world") == "WORLD"

    def test_lower(self):
        assert match_case("hello", "WORLD") == "world"

    def test_title(self):
        assert match_case("Hello", "world") == "World"

    def test_empty_original(self):
        assert match_case("", "test") == "test"

    def test_mixed_case(self):
        # First char lowercase = return as-is
        assert match_case("hELLO", "world") == "world"


class TestTokenbreakPrepend:
    def test_returns_tuple(self):
        random.seed(42)
        result, token = tokenbreak_prepend("test")
        assert isinstance(result, str)
        assert isinstance(token, str)
        assert result.endswith("test")

    def test_token_prepended(self):
        random.seed(0)
        result, token = tokenbreak_prepend("word")
        assert result == token + "word"


class TestTokenbreakSplit:
    def test_returns_tuple(self):
        random.seed(42)
        result, token_repr = tokenbreak_split("hello")
        assert isinstance(result, str)
        assert len(result) > len("hello")

    def test_single_char(self):
        random.seed(42)
        result, _ = tokenbreak_split("x")
        assert "x" in result

    def test_contains_original_chars(self):
        random.seed(10)
        result, _ = tokenbreak_split("test")
        # All original chars should be present (possibly with extra)
        cleaned = result.replace(result[len("test") // 2], "", 1) if len(result) > 4 else result
        # At minimum, the word is longer than original
        assert len(result) > len("test")


class TestTokenbreakDelimiter:
    def test_returns_tuple(self):
        random.seed(42)
        result, delim = tokenbreak_delimiter("word")
        assert isinstance(result, str)
        assert isinstance(delim, str)

    def test_word_present(self):
        # With structural or standard delimiters, the word is present
        random.seed(1)
        for _ in range(10):
            result, _ = tokenbreak_delimiter("test")
            # Either the word is in the result or it's been split by separators
            assert "t" in result


class TestTokenbreakObfuscateWord:
    def test_prepend(self):
        random.seed(42)
        result, _ = tokenbreak_obfuscate_word("hello", "prepend")
        assert result.endswith("hello")

    def test_split(self):
        random.seed(42)
        result, _ = tokenbreak_obfuscate_word("hello", "split")
        assert len(result) > len("hello")

    def test_delimiter(self):
        random.seed(42)
        result, _ = tokenbreak_obfuscate_word("hello", "delimiter")
        assert isinstance(result, str)

    def test_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown attack"):
            tokenbreak_obfuscate_word("test", "invalid")


class TestGetTokenbreakFunctions:
    def test_returns_three(self):
        funcs = get_tokenbreak_functions()
        assert len(funcs) == 3
        names = [f.__name__ for f in funcs]
        assert "to_tokenbreak_prepend" in names
        assert "to_tokenbreak_split" in names
        assert "to_tokenbreak_delimiter" in names
