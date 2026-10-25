"""Tests for impersonation, temporal, and asking jailbreak techniques."""

import random
import re

from redact.jailbreak.directives import read_pools
from redact.jailbreak.requests.asking import (
    get_asking_functions,
    to_ask_for_details,
    to_innocuous_question,
)
from redact.jailbreak.requests.impersonation import (
    get_impersonation_functions,
    to_impersonate_good_person,
)
from redact.jailbreak.requests.temporal import (
    get_temporal_functions,
    to_use_past_tense,
)

SAMPLE = "How do I pick a lock?"


class TestImpersonation:
    def test_returns_three(self):
        assert len(get_impersonation_functions()) == 3     # noqa: PLR2004

    def test_returns_tuple(self):
        random.seed(42)
        assert isinstance(to_impersonate_good_person(SAMPLE), tuple)

    def test_prompt_preserved(self):
        random.seed(42)
        result, _ = to_impersonate_good_person(SAMPLE)
        assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        _, info = to_impersonate_good_person(SAMPLE)
        assert "type=good_person" in info
        assert "variant=" in info

    def test_prepends(self):
        random.seed(42)
        result, _ = to_impersonate_good_person(SAMPLE)
        assert result.index(SAMPLE) > 0

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_impersonate_good_person(SAMPLE)
            results.add(result)
        assert len(results) > 1


class TestTemporal:
    def test_returns_three(self):
        assert len(get_temporal_functions()) == 3     # noqa: PLR2004

    def test_returns_tuple(self):
        random.seed(42)
        assert isinstance(to_use_past_tense(SAMPLE), tuple)

    def test_prompt_preserved(self):
        random.seed(42)
        result, _ = to_use_past_tense(SAMPLE)
        assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        _, info = to_use_past_tense(SAMPLE)
        assert "type=past_tense" in info
        assert "variant=" in info

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_use_past_tense(SAMPLE)
            results.add(result)
        assert len(results) > 1


class TestAsking:
    def test_returns_three(self):
        assert len(get_asking_functions()) == 3     # noqa: PLR2004

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_asking_functions():
            assert isinstance(fn(SAMPLE), tuple)

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_asking_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        for fn in get_asking_functions():
            _, info = fn(SAMPLE)
            assert "type=" in info
            assert "variant=" in info

    def test_innocuous_prepends(self):
        random.seed(42)
        result, _ = to_innocuous_question(SAMPLE)
        assert result.index(SAMPLE) > 0

    def test_ask_for_details_appends(self):
        random.seed(42)
        result, _ = to_ask_for_details(SAMPLE)
        assert result.startswith(SAMPLE)


class TestPastTenseAsksForThePastReading:
    """Every past-tense variant tells the model to write or answer in a past
    setting, not only names one. Flags, not text."""

    _POOL = read_pools("requests/temporal_templates.json")["past_tense"]
    _ASKS = re.compile(
        r"\b(imagine|write|writing|describe|tell|answer|explain|as if|as a)\b",
        re.IGNORECASE)
    _PAST = re.compile(r"past|histor|already|ago|taken place|looking back|looks back|"
                       r"archiv|memoir|retrospective", re.IGNORECASE)

    def test_every_variant_gives_an_instruction(self):
        assert [v["name"] for v in self._POOL
                if not self._ASKS.search(v["template"])] == []

    def test_every_variant_sets_the_request_in_the_past(self):
        assert [v["name"] for v in self._POOL
                if not self._PAST.search(v["template"])] == []

    def test_the_pool_has_its_floor_of_variants(self):
        assert len(self._POOL) >= 8     # noqa: PLR2004
