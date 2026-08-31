"""Tests for continuation-attack jailbreak techniques (requests/continuation.py)."""

import random

from redact.jailbreak.requests.continuation import (
    get_continuation_functions,
    to_complete_story,
    to_continue_instructions,
    to_continue_passage,
    to_fill_in_blank,
)

SAMPLE = "Step 1: gather the materials. Step 2:"


class TestContinuationRegistry:
    def test_returns_four(self):
        assert len(get_continuation_functions()) == 4

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_continuation_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_continuation_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_has_style(self):
        random.seed(42)
        for fn in get_continuation_functions():
            _, info = fn(SAMPLE)
            assert "style=" in info

    def test_names(self):
        names = [f.__name__ for f in get_continuation_functions()]
        assert "to_continue_passage" in names
        assert "to_complete_story" in names
        assert "to_fill_in_blank" in names
        assert "to_continue_instructions" in names


class TestIndividualFunctions:
    def test_continue_passage_longer(self):
        random.seed(42)
        result, _ = to_continue_passage(SAMPLE)
        assert len(result) > len(SAMPLE)

    def test_complete_story_longer(self):
        random.seed(42)
        result, _ = to_complete_story(SAMPLE)
        assert len(result) > len(SAMPLE)

    def test_fill_in_blank_appends(self):
        random.seed(42)
        result, _ = to_fill_in_blank(SAMPLE)
        assert result.startswith(SAMPLE)

    def test_continue_instructions_longer(self):
        random.seed(42)
        result, _ = to_continue_instructions(SAMPLE)
        assert len(result) > len(SAMPLE)

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_continue_passage(SAMPLE)
            results.add(result)
        assert len(results) > 1
