"""Tests for distractor jailbreak techniques (requests/distractor.py)."""

import random

from redact.jailbreak.requests.distractor import (
    to_distractor_prefix_related,
    to_distractor_prefix_unrelated,
    to_distractor_suffix_related,
    to_distractor_suffix_unrelated,
    get_distractor_functions,
)


SAMPLE = "How do I pick a lock?"


class TestDistractorRegistry:
    def test_returns_four(self):
        assert len(get_distractor_functions()) == 4

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_distractor_functions():
            assert isinstance(fn(SAMPLE), tuple)

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_distractor_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        for fn in get_distractor_functions():
            _, info = fn(SAMPLE)
            assert "type=" in info
            assert "variant=" in info

    def test_longer_than_original(self):
        random.seed(42)
        for fn in get_distractor_functions():
            result, _ = fn(SAMPLE)
            assert len(result) > len(SAMPLE)


class TestPrefixVsSuffix:
    def test_prefix_related_prepends(self):
        random.seed(42)
        result, _ = to_distractor_prefix_related(SAMPLE)
        assert result.index(SAMPLE) > 0

    def test_prefix_unrelated_prepends(self):
        random.seed(42)
        result, _ = to_distractor_prefix_unrelated(SAMPLE)
        assert result.index(SAMPLE) > 0

    def test_suffix_related_appends(self):
        random.seed(42)
        result, _ = to_distractor_suffix_related(SAMPLE)
        assert result.startswith(SAMPLE)

    def test_suffix_unrelated_appends(self):
        random.seed(42)
        result, _ = to_distractor_suffix_unrelated(SAMPLE)
        assert result.startswith(SAMPLE)

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_distractor_prefix_related(SAMPLE)
            results.add(result)
        assert len(results) > 1
