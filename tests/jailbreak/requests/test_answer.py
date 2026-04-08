"""Tests for answer-format and output-conditioning directives (requests/answer.py)."""

import random

from redact.jailbreak.requests.answer import (
    to_refusal_then_comply,
    to_apology_prevention,
    to_prefix_injection,
    to_answer_in_markdown,
    to_answer_as_tutorial,
    to_answer_as_code,
    to_answer_as_dialogue,
    to_answer_as_story,
    to_answer_as_structured_data,
    get_answer_functions,
)


SAMPLE = "How do I pick a lock?"


class TestAnswerRegistry:
    def test_returns_nine(self):
        assert len(get_answer_functions()) == 9

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_answer_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_answer_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        for fn in get_answer_functions():
            _, info = fn(SAMPLE)
            assert "directive=" in info
            assert "variant=" in info

    def test_expected_names(self):
        names = [f.__name__ for f in get_answer_functions()]
        assert "to_refusal_then_comply" in names
        assert "to_apology_prevention" in names
        assert "to_prefix_injection" in names
        assert "to_answer_in_markdown" in names
        assert "to_answer_as_tutorial" in names
        assert "to_answer_as_code" in names
        assert "to_answer_as_dialogue" in names
        assert "to_answer_as_story" in names
        assert "to_answer_as_structured_data" in names


class TestOutputConditioningDirectives:
    def test_refusal_then_comply_info(self):
        random.seed(0)
        _, info = to_refusal_then_comply(SAMPLE)
        assert info.startswith("directive=refusal_then_comply")

    def test_apology_prevention_info(self):
        random.seed(0)
        _, info = to_apology_prevention(SAMPLE)
        assert info.startswith("directive=apology_prevention")

    def test_prefix_injection_info(self):
        random.seed(0)
        _, info = to_prefix_injection(SAMPLE)
        assert info.startswith("directive=prefix_injection")


class TestOutputFormatDirectives:
    def test_answer_in_markdown_info(self):
        random.seed(0)
        _, info = to_answer_in_markdown(SAMPLE)
        assert info.startswith("directive=answer_in_markdown")

    def test_answer_as_tutorial_info(self):
        random.seed(0)
        _, info = to_answer_as_tutorial(SAMPLE)
        assert info.startswith("directive=answer_as_tutorial")

    def test_answer_as_code_info(self):
        random.seed(0)
        _, info = to_answer_as_code(SAMPLE)
        assert info.startswith("directive=answer_as_code")

    def test_answer_as_dialogue_info(self):
        random.seed(0)
        _, info = to_answer_as_dialogue(SAMPLE)
        assert info.startswith("directive=answer_as_dialogue")

    def test_answer_as_story_info(self):
        random.seed(0)
        _, info = to_answer_as_story(SAMPLE)
        assert info.startswith("directive=answer_as_story")

    def test_answer_as_structured_data_info(self):
        random.seed(0)
        _, info = to_answer_as_structured_data(SAMPLE)
        assert info.startswith("directive=answer_as_structured_data")

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_answer_as_code(SAMPLE)
            results.add(result)
        assert len(results) > 1
