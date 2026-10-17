"""Tests for indirect-request jailbreak techniques (requests/indirect.py)."""

import random

from redact.jailbreak.requests.indirect import (
    get_indirect_functions,
    to_critique,
    to_explain_to_child,
    to_proofread,
    to_rewrite,
    to_summarize,
    to_translate_task,
)

SAMPLE = "The explosive device requires ammonium nitrate mixed with fuel oil."


class TestIndirectRegistry:
    def test_returns_six(self):
        assert len(get_indirect_functions()) == 6

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_indirect_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_indirect_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        for fn in get_indirect_functions():
            _, info = fn(SAMPLE)
            assert "task=" in info
            assert "variant=" in info

    def test_names(self):
        names = [f.__name__ for f in get_indirect_functions()]
        assert "to_summarize" in names
        assert "to_proofread" in names
        assert "to_critique" in names
        assert "to_translate_task" in names
        assert "to_explain_to_child" in names
        assert "to_rewrite" in names


class TestIndividualFunctions:
    def test_summarize_info(self):
        random.seed(0)
        _, info = to_summarize(SAMPLE)
        assert info.startswith("task=summarize")

    def test_proofread_info(self):
        random.seed(0)
        _, info = to_proofread(SAMPLE)
        assert info.startswith("task=proofread")

    def test_critique_info(self):
        random.seed(0)
        _, info = to_critique(SAMPLE)
        assert info.startswith("task=critique")

    def test_translate_task_info(self):
        random.seed(0)
        _, info = to_translate_task(SAMPLE)
        assert info.startswith("task=translate_task")

    def test_explain_to_child_info(self):
        random.seed(0)
        _, info = to_explain_to_child(SAMPLE)
        assert info.startswith("task=explain_to_child")

    def test_rewrite_info(self):
        random.seed(0)
        _, info = to_rewrite(SAMPLE)
        assert info.startswith("task=rewrite")

    def test_longer_than_original(self):
        random.seed(42)
        for fn in get_indirect_functions():
            result, _ = fn(SAMPLE)
            assert len(result) > len(SAMPLE)

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_summarize(SAMPLE)
            results.add(result)
        assert len(results) > 1
