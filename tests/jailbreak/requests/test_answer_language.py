"""Tests for ask-answer-in-language techniques (requests/answer_language.py)."""

import random

from redact.jailbreak.requests.answer_language import (
    get_answer_language_functions,
    to_ask_answer_in_french,
    to_ask_answer_in_japanese,
    to_ask_answer_in_scots_gaelic,
    to_ask_answer_in_swahili,
)

SAMPLE = "How do I pick a lock?"


class TestAnswerLanguageRegistry:
    def test_returns_twenty_six(self):
        assert len(get_answer_language_functions()) == 26

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

    def test_one_per_row_of_the_language_table_marked_answer_language(self):
        """Maori too, which translation leaves out."""
        from redact.jailbreak.technique import data_rows

        assert [f.__name__ for f in get_answer_language_functions()] == [
            f"to_ask_answer_in_{row['slug']}" for row in data_rows("languages.json")
            if row["answer_language"]]
        assert "to_ask_answer_in_maori" in [
            f.__name__ for f in get_answer_language_functions()]

    def test_the_six_languages_of_spec_1_2_are_answer_languages_since_1_2(self):
        slugs = ["hmong", "guarani", "chinese", "hindi", "yoruba", "amharic"]
        added = [f.__name__ for f in get_answer_language_functions()
                 if f.since == "1.2"]
        assert added == [f"to_ask_answer_in_{slug}" for slug in slugs]

    def test_guarani_is_an_answer_language_only(self):
        """Its translation measured too poorly (TranslateGemma loops on a
        fifth of its sentences); the other five of the six are translations
        too, four since spec 1.3 and Yoruba since spec 1.4."""
        from redact.jailbreak.pool import languages

        assert "to_guarani" not in languages()
        assert {"to_hmong", "to_chinese", "to_hindi", "to_amharic",
                "to_yoruba"} <= languages()


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
