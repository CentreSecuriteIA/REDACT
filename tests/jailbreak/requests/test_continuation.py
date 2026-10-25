"""Tests for continuation-attack jailbreak techniques (requests/continuation.py)."""

import random
import re

from redact.jailbreak.directives import read_pools
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

    def test_fill_in_blank_keeps_the_prompt_and_adds_to_it(self):
        random.seed(42)
        result, _ = to_fill_in_blank(SAMPLE)
        assert SAMPLE in result and len(result) > len(SAMPLE)

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


class TestEveryVariantFramesThePromptCoherently:
    """What each directive's text says must fit the text it is put around:
    the prompt is a request, so a variant that speaks of a blank, a story or
    numbered instructions has to give the prompt that place. Flags, not text."""

    _POOLS = read_pools("requests/continuation_templates.json")

    def test_a_fill_in_blank_variant_has_a_blank_to_fill_after_the_prompt(self):
        marker = re.compile(r"_{3,}|\[missing\]|\[Insert")
        without = [v["name"] for v in self._POOLS["fill_in_blank"]
                   if not marker.search(v["suffix"])]
        assert without == []

    def test_a_continue_instructions_variant_makes_the_prompt_the_first_entry(self):
        without = [v["name"] for v in self._POOLS["continue_instructions"]
                   if not v["prefix"].endswith("\n\n1. ")]
        assert without == []

    def test_a_complete_story_variant_says_the_story_opens_with_the_prompt(self):
        opening = re.compile(r"\b(opens|opening|beginning|begins|begun|says)\b")
        without = [v["name"] for v in self._POOLS["complete_story"]
                   if not opening.search(v["prefix"])]
        assert without == []

    def test_every_pool_still_has_its_floor_of_variants(self):
        assert all(len(pool) >= 8 for pool in self._POOLS.values())   # noqa: PLR2004

    def test_no_two_variants_of_a_pool_say_nearly_the_same_thing(self):
        """Two variants of one pool must differ in wording, not only in a
        label: the smaller of their sets of word trigrams (prefix and suffix
        together) shares at most half with the other. The pools keep to 0.24
        or less; ``fill_blank`` and ``blank_answer`` of ``fill_in_blank`` were
        at 0.73 (a question, an answer line, an instruction to fill it in)
        before the second was written as a transcript. A flag, not text."""
        def trigrams(variant: dict) -> set[tuple[str, ...]]:
            words = re.findall(r"\w+", (variant["prefix"] + " "
                                        + variant["suffix"]).lower())
            return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}

        too_alike = []
        for directive, pool in self._POOLS.items():
            for i, first in enumerate(pool):
                for second in pool[i + 1:]:
                    a, b = trigrams(first), trigrams(second)
                    if a and b and len(a & b) / min(len(a), len(b)) > 0.5:  # noqa: PLR2004
                        too_alike.append((directive, first["name"], second["name"]))
        assert too_alike == []
