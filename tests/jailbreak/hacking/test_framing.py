"""Tests for scenario-modifying framing directives (hacking/framing.py)."""

import random
import re

from redact.jailbreak.directives import read_pools
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


class TestNefariousGoalStatesABadGoal:
    """The directive is named for a bad goal the asker states, so every
    variant says one in the first person, and none is one of the legitimate
    roles (a researcher, a tester, a scholar, a journalist) that the
    ``noble_goal`` pool and the ``cognitive`` techniques cover. The checks
    show names and flags, not text."""

    _POOL = read_pools("hacking/framing_templates.json")["nefarious_goal"]
    _FIRST_PERSON = re.compile(r"\b(I|I'm|I'll|my|My)\b")
    _LEGITIMATE_ROLE = re.compile(
        r"researcher|research |journalist|academic|scholar|authori[sz]ed|"
        r"red[- ]team|tester|educational|to protect|to counter", re.IGNORECASE)

    def test_every_variant_speaks_in_the_first_person(self):
        missing = [v["name"] for v in self._POOL
                   if not self._FIRST_PERSON.search(v["template"])]
        assert missing == []

    def test_no_variant_claims_a_legitimate_role(self):
        claimed = [v["name"] for v in self._POOL
                   if self._LEGITIMATE_ROLE.search(v["template"])]
        assert claimed == []

    def test_the_variants_that_were_roles_are_gone(self):
        names = {v["name"] for v in self._POOL}
        assert not names & {"researcher", "red_team", "academic", "journalist"}

    def test_the_pool_still_has_its_floor_of_variants(self):
        assert len(self._POOL) >= 8     # noqa: PLR2004

    def test_the_technique_writes_one_of_them(self):
        texts = {v["template"].format_map({"prompt": SAMPLE}) for v in self._POOL}
        for seed in range(10):
            assert to_nefarious_goal(SAMPLE, rng=random.Random(seed))[0] in texts
