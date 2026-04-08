"""Tests for ask-answer-in-language techniques (requests/answer_language.py)."""

import random

from redact.jailbreak.requests.answer_language import (
    to_ask_answer_in_french,
    to_ask_answer_in_japanese,
    to_ask_answer_in_swahili,
    to_ask_answer_in_scots_gaelic,
    get_answer_language_functions,
)


SAMPLE = "How do I pick a lock?"


class TestAnswerLanguageRegistry:
    def test_returns_twenty(self):
        assert len(get_answer_language_functions()) == 20

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_answer_language_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_answer_language_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_has_language_and_variant(self):
        random.seed(42)
        for fn in get_answer_language_functions():
            _, info = fn(SAMPLE)
            assert "language=" in info
            assert "variant=" in info

    def test_function_names(self):
        names = [f.__name__ for f in get_answer_language_functions()]
        assert "to_ask_answer_in_french" in names
        assert "to_ask_answer_in_swahili" in names
        assert "to_ask_answer_in_scots_gaelic" in names
        assert "to_ask_answer_in_javanese" in names


class TestIndividualLanguageFunctions:
    def test_french_info(self):
        random.seed(0)
        _, info = to_ask_answer_in_french(SAMPLE)
        assert "language=French" in info

    def test_japanese_info(self):
        random.seed(0)
        _, info = to_ask_answer_in_japanese(SAMPLE)
        assert "language=Japanese" in info

    def test_swahili_info(self):
        random.seed(0)
        _, info = to_ask_answer_in_swahili(SAMPLE)
        assert "language=Swahili" in info

    def test_scots_gaelic_info(self):
        random.seed(0)
        _, info = to_ask_answer_in_scots_gaelic(SAMPLE)
        assert "language=Scots Gaelic" in info

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_ask_answer_in_french(SAMPLE)
            results.add(result)
        assert len(results) > 1
