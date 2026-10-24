"""Tests for tokenbreak obfuscation (pure word-level transforms)."""

import random

import pytest

from redact.jailbreak.obfuscation.tokenbreak import (
    _CHAR_TOKENS,
    _INVISIBLE_CHARS,
    _SEPARATOR_DELIMITERS,
    _VARIATION_SELECTORS,
    _is_mark,
    _is_word,
    _parse_harmful_words,
    get_tokenbreak_functions,
    match_case,
    tokenbreak_delimiter,
    tokenbreak_obfuscate_word,
    tokenbreak_prepend,
    tokenbreak_split,
)

# Greetings whose letters carry combining marks (a vowel sign, a virama, a
# Khmer coeng), which \w does not match.
BENGALI = "নমস্কার"  # namaskar (greeting)
THAI = "สวัสดี"  # sawasdee (hello)
NEPALI = "नमस्ते"  # namaste
KHMER = "សួស្តី"  # suostei (hello)
MARKED_WORDS = pytest.mark.parametrize(
    "word", [BENGALI, THAI, NEPALI, KHMER], ids=["bengali", "thai", "nepali", "khmer"],
)


class _LastChoice(random.Random):
    """A seeded rng whose ``choice`` takes the last option: tokenbreak_split
    inserts U+FE0F, tokenbreak_delimiter takes its separator branch with "|"."""

    def choice(self, seq):
        return seq[-1]


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

    @MARKED_WORDS
    def test_a_split_keeps_each_letter_with_its_marks(self, word):
        for seed in range(10):
            result, _ = tokenbreak_split(word, _LastChoice(seed))
            left, right = result.split("️")
            assert left + right == word
            assert not _is_mark(right[0])

    def test_an_ascii_word_is_split_as_before(self):
        # Each ASCII character is a cluster, so the draw is the one made over
        # characters before the split cut at clusters.
        tokens = {"char": _CHAR_TOKENS, "invisible_char": _INVISIBLE_CHARS,
                  "variation_selector": _VARIATION_SELECTORS}
        for seed in range(20):
            rng = random.Random(seed)
            kind = rng.choice(["char", "invisible_char", "variation_selector"])
            token = rng.choice(tokens[kind])
            position = rng.randint(1, len("hello") - 1)
            expected = "hello"[:position] + token + "hello"[position:]
            assert tokenbreak_split("hello", random.Random(seed))[0] == expected


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

    @MARKED_WORDS
    def test_a_separator_keeps_each_letter_with_its_marks(self, word):
        result, delimiter = tokenbreak_delimiter(word, _LastChoice(0))
        pieces = result.split(delimiter)
        assert "".join(pieces) == word
        assert not any(_is_mark(piece[0]) for piece in pieces)

    def test_a_separator_cuts_a_conjunct_between_its_letters(self):
        # na | ma | sa + virama | ka + vowel sign aa | ra: the conjunct sa-ka is
        # cut after its virama, as to_separator cuts it.
        result, _ = tokenbreak_delimiter(BENGALI, _LastChoice(0))
        assert result == "ন|ম|স্|কা|র"

    def test_an_ascii_word_is_separated_as_before(self):
        # Before the clusters, the separator branch joined the word's characters.
        for seed in range(20):
            rng = random.Random(seed)
            if rng.choice(["standard", "structural", "separator"]) != "separator":
                continue
            expected = rng.choice(_SEPARATOR_DELIMITERS).join(list("hello"))
            assert tokenbreak_delimiter("hello", random.Random(seed))[0] == expected


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


class TestParseHarmfulWords:
    PROMPT = "How do I grow an apple and a pear tree?"

    @pytest.mark.parametrize("reply", [
        "  apple pear \n", "- apple\n- pear", "* apple\n* pear", "• apple\n• pear",
        "1. apple\n2. pear", "1) apple 2) pear", "**apple** **pear**", "apple, pear",
    ])
    def test_a_word_list_in_a_common_form_gives_its_words(self, reply):
        assert _parse_harmful_words(reply, self.PROMPT) == ["apple", "pear"]

    @pytest.mark.parametrize("reply", ["None", "none.", "N/A", ""])
    def test_a_none_reply_gives_no_words(self, reply):
        assert _parse_harmful_words(reply, self.PROMPT) == []

    @pytest.mark.parametrize("reply", [
        "Harmful words: apple", "apple banana", "apple n/a pear", "apple 42",
        " ".join(["apple"] * 11),
    ], ids=["labelled", "not-in-prompt", "placeholder", "number", "rambling"])
    def test_a_malformed_reply_gives_none(self, reply):
        assert _parse_harmful_words(reply, self.PROMPT) is None

    @MARKED_WORDS
    def test_a_word_with_combining_marks_gives_its_word(self, word):
        assert _parse_harmful_words(f"- {word}", f"Say {word} to a friend.") == [word]

    def test_a_token_starting_with_a_mark_gives_none(self):
        # The end of namaskar, from its vowel sign aa: in the prompt, not a word.
        assert _parse_harmful_words("ার", BENGALI) is None

    @pytest.mark.parametrize("reply", [
        "apple Apple APPLE pear", "apple\nAPPLE\npear\nApple", "Apple, apple, pear",
    ])
    def test_a_word_named_twice_in_any_case_counts_once_in_its_first_spelling(
            self, reply):
        words = _parse_harmful_words(reply, self.PROMPT)
        assert words is not None
        assert [w.lower() for w in words] == ["apple", "pear"]
        assert words[0] == reply.replace(",", " ").split()[0]

    @pytest.mark.parametrize("reply", ["a", "apple a", "A", "I", "a pear"])
    def test_a_word_of_one_letter_gives_none(self, reply):
        """Replacing it would change that letter wherever it stands."""
        assert _parse_harmful_words(reply, "How can I grow a pear and an apple?") is None

    @pytest.mark.parametrize("letter", ["я", "é", "e\u0301", "क", "の"])
    def test_a_letter_of_another_script_is_no_more_a_word_than_a_latin_one(self, letter):
        assert _parse_harmful_words(letter, f"Say {letter} twice") is None

    def test_one_chinese_character_is_a_word(self):
        assert _parse_harmful_words("爆", "怎么做爆米花") == ["爆"]

    @pytest.mark.parametrize("word", [
        "می\u200cخواهم",    # Persian, with a zero-width non-joiner
        "র\u200d্যা",       # Bengali, with a zero-width joiner and a virama
        "ab\u200bcd",       # a zero-width space
    ], ids=["zwnj", "zwj", "zwsp"])
    def test_a_word_with_a_zero_width_character_gives_its_word(self, word):
        assert _parse_harmful_words(f"- {word}", f"Say {word} to a friend.") == [word]

    @pytest.mark.parametrize("reply", [
        "\u201capple\u201d \u201cpear\u201d", "\u2018apple\u2019, \u2018pear\u2019",
        "\u201eapple\u201c \u00abpear\u00bb", "apple\u2026 pear\u2026", "\u2026apple pear",
    ], ids=["double", "single", "low-and-guillemets", "ellipsis", "leading-ellipsis"])
    def test_curly_quotes_and_an_ellipsis_around_a_word_are_stripped(self, reply):
        assert _parse_harmful_words(reply, self.PROMPT) == ["apple", "pear"]

    @pytest.mark.parametrize("reply", ["\u201cnone\u201d", "None\u2026", "\u2018N/A\u2019"])
    def test_a_none_reply_in_curly_quotes_gives_no_words(self, reply):
        assert _parse_harmful_words(reply, self.PROMPT) == []

    def test_a_curly_apostrophe_inside_a_word_stays(self):
        assert _parse_harmful_words("don\u2019t", "Say don\u2019t") == ["don\u2019t"]


class TestIsWord:
    @MARKED_WORDS
    def test_a_word_with_combining_marks_is_a_word(self, word):
        assert _is_word(word)

    @pytest.mark.parametrize("token", [
        "ার",  # the end of namaskar, from its vowel sign aa
        "apple pear", "42", "n/a", "_apple", "", "\u200dapple",
    ], ids=["leading-mark", "space", "number", "placeholder", "underscore", "empty",
            "leading-joiner"])
    def test_a_token_that_is_not_a_word_is_refused(self, token):
        assert not _is_word(token)
