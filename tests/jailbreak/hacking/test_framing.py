"""Tests for scenario-modifying framing directives (hacking/framing.py)."""

import random

from redact.jailbreak.hacking.framing import (
    get_framing_functions,
    to_fictional_world,
    to_high_stake,
    to_nefarious_goal,
    to_no_moral_constraints,
    to_noble_goal,
)

SAMPLE = "How do I pick a lock?"


class TestFramingFunctions:
    def test_returns_five(self):
        assert len(get_framing_functions()) == 5

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_framing_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_framing_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        for fn in get_framing_functions():
            _, info = fn(SAMPLE)
            assert "directive=" in info
            assert "variant=" in info

    def test_names(self):
        names = [f.__name__ for f in get_framing_functions()]
        assert "to_fictional_world" in names
        assert "to_noble_goal" in names
        assert "to_nefarious_goal" in names
        assert "to_high_stake" in names
        assert "to_no_moral_constraints" in names

    def test_answer_functions_not_present(self):
        names = [f.__name__ for f in get_framing_functions()]
        assert "to_answer_in_markdown" not in names
        assert "to_refusal_then_comply" not in names
        assert "to_prefix_injection" not in names


class TestIndividualDirectives:
    def test_fictional_world_info(self):
        random.seed(42)
        _, info = to_fictional_world(SAMPLE)
        assert info.startswith("directive=fictional_world")

    def test_noble_goal_info(self):
        random.seed(0)
        _, info = to_noble_goal(SAMPLE)
        assert info.startswith("directive=noble_goal")

    def test_nefarious_goal_info(self):
        random.seed(0)
        _, info = to_nefarious_goal(SAMPLE)
        assert info.startswith("directive=nefarious_goal")

    def test_high_stake_urgency(self):
        random.seed(42)
        result, _ = to_high_stake(SAMPLE)
        assert len(result) > len(SAMPLE)

    def test_no_moral_constraints_appends(self):
        random.seed(42)
        result, _ = to_no_moral_constraints(SAMPLE)
        assert len(result) > len(SAMPLE)

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_fictional_world(SAMPLE)
            results.add(result)
        assert len(results) > 1
