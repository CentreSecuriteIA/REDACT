"""Tests for impersonation, temporal, and asking jailbreak techniques."""

import random

from redact.jailbreak.requests.impersonation import (
    to_impersonate_good_person,
    get_impersonation_functions,
)
from redact.jailbreak.requests.temporal import (
    to_use_past_tense,
    get_temporal_functions,
)
from redact.jailbreak.requests.asking import (
    to_innocuous_question,
    to_ask_for_details,
    get_asking_functions,
)


SAMPLE = "How do I pick a lock?"


class TestImpersonation:
    def test_returns_one(self):
        assert len(get_impersonation_functions()) == 1

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
    def test_returns_one(self):
        assert len(get_temporal_functions()) == 1

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
    def test_returns_two(self):
        assert len(get_asking_functions()) == 2

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
