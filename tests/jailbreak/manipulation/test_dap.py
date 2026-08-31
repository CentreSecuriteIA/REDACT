"""Tests for DAP (Distract and Persuade) manipulation."""

import random

from redact.jailbreak.manipulation.dap import (
    _build_dap_prompt,
    get_dap_functions,
    to_dap_random_long,
    to_dap_random_short,
    to_dap_selected_long,
    to_dap_selected_short,
)
from redact.jailbreak.protocol import run_sync
from tests.conftest import MockBackend


def _make_benign_data():
    pairs = [
        {"prompt": f"q{i}", "answer": f"a{i}"}
        for i in range(10)
    ]
    return {
        "short": pairs,
        "long": pairs,
        "by_subcat_short": {"Cooking": pairs},
        "by_subcat_long": {"Cooking": pairs},
        "all_subcategories": ["Cooking", "Gardening"],
    }


class TestBuildDapPrompt:
    def test_harmful_in_middle(self):
        random.seed(42)
        pairs = [{"prompt": f"q{i}", "answer": f"a{i}"} for i in range(5)]
        result, idx = _build_dap_prompt("harmful?", pairs, 4)
        assert idx >= 2  # never first
        assert "harmful?" in result

    def test_harmful_has_empty_answer(self):
        random.seed(42)
        pairs = [{"prompt": f"q{i}", "answer": f"a{i}"} for i in range(5)]
        result, idx = _build_dap_prompt("harmful?", pairs, 3)
        lines = result.split("\n")
        # Find the harmful Q line and check its answer is empty
        for i, line in enumerate(lines):
            if "harmful?" in line:
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                assert next_line.strip().endswith(":")
                break


class TestDapRandom:
    def test_short(self):
        random.seed(42)
        data = _make_benign_data()
        result, info = to_dap_random_short("harmful prompt", data)
        assert "harmful prompt" in result
        assert "harmful_idx=" in info
        assert "num_shots=" in info

    def test_long(self):
        random.seed(42)
        data = _make_benign_data()
        result, info = to_dap_random_long("harmful prompt", data)
        assert "harmful prompt" in result


def _run(fn, prompt, backend, **kwargs):
    """Drive a technique generator to completion against a MockBackend."""
    def call(request):
        return backend.generate([request.messages])[0]
    return run_sync(fn(prompt, **kwargs), call)


class TestDapSelected:
    def test_short_exact_match(self):
        random.seed(42)
        data = _make_benign_data()
        backend = MockBackend("Cooking")
        result, info = _run(to_dap_selected_short, "harmful prompt", backend, benign_data=data)
        assert "harmful prompt" in result
        assert "sub_category=Cooking" in info
        assert "selection=selected" in info
        assert "harmful_idx=" in info  # DAP includes the insertion position

    def test_long_exact_match(self):
        random.seed(42)
        data = _make_benign_data()
        backend = MockBackend("Cooking")
        result, info = _run(to_dap_selected_long, "harmful prompt", backend, benign_data=data)
        assert "harmful prompt" in result
        assert "harmful_idx=" in info

    def test_fallback_when_unparseable(self):
        random.seed(42)
        data = _make_benign_data()
        backend = MockBackend("totally unrelated response xyz")
        _, info = _run(to_dap_selected_short, "harmful prompt", backend, benign_data=data)
        assert "selection=fallback_random" in info


class TestGetDapFunctions:
    def test_returns_four(self):
        funcs = get_dap_functions()
        assert len(funcs) == 4
