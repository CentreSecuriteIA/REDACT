"""Tests for Few-Shot Hacking (FSH) manipulation."""

import random

from redact.jailbreak.manipulation.fsh import (
    _build_fsh_prompt,
    get_fsh_functions,
    to_fsh_random_long,
    to_fsh_random_short,
    to_fsh_selected_long,
    to_fsh_selected_short,
)
from redact.jailbreak.manipulation.pairs import _select_subcategory_gen
from redact.llm_pipeline import drive_sync
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
        "all_subcategories": ["Cooking", "Gardening", "Fitness"],
    }


def _select(reply: str) -> tuple[str, bool]:
    """The subcategory a model picks with ``reply``, and whether it fell back."""
    gen = _select_subcategory_gen("test", ["Cooking", "Gardening"], gen_model="m")
    return drive_sync(gen, lambda request: reply)


class TestSelectSubcategory:
    def test_exact_match(self):
        assert _select("Cooking") == ("Cooking", False)

    def test_fuzzy_match(self):
        assert _select("I think Cooking & Baking is closest") == ("Cooking", False)

    def test_fallback_random(self):
        subcat, fallback = _select("totally irrelevant response xyz")
        assert subcat in ["Cooking", "Gardening"]
        assert fallback is True


class TestBuildFshPrompt:
    def test_structure(self):
        pairs = [{"prompt": "q1", "answer": "a1"}, {"prompt": "q2", "answer": "a2"}]
        result, extra = _build_fsh_prompt("harmful?", pairs, 2)
        assert extra == {}
        assert "Q1:" in result
        assert "Q3: harmful?" in result
        assert "A3:" in result

    def test_harmful_at_end(self):
        pairs = [{"prompt": f"q{i}", "answer": f"a{i}"} for i in range(5)]
        result, _ = _build_fsh_prompt("bad prompt", pairs, 3)
        lines = result.strip().split("\n")
        # Last non-empty line should be the answer placeholder
        last_lines = [l for l in lines if l.strip()]
        assert last_lines[-1].startswith("A")


class TestFshRandom:
    def test_short(self):
        random.seed(42)
        data = _make_benign_data()
        result, info = to_fsh_random_short("harmful prompt", data)
        assert "harmful prompt" in result
        assert "num_shots=" in info
        assert "sub_category=" in info

    def test_long(self):
        random.seed(42)
        data = _make_benign_data()
        result, info = to_fsh_random_long("harmful prompt", data)
        assert "harmful prompt" in result
        assert "num_shots=" in info


def _run(fn, prompt, backend, **kwargs):
    """Drive a technique generator to completion against a MockBackend."""
    def call(request):
        return backend.generate([request.messages])[0]
    return drive_sync(fn(prompt, **kwargs), call)


class TestFshSelected:
    def test_short_exact_match(self):
        random.seed(42)
        data = _make_benign_data()
        backend = MockBackend("Cooking")
        result, info = _run(to_fsh_selected_short, "harmful prompt", backend, benign_data=data)
        assert "harmful prompt" in result
        assert "sub_category=Cooking" in info
        assert "selection=selected" in info
        assert "harmful_idx" not in info  # FSH has no position field (that's DAP)

    def test_long_exact_match(self):
        random.seed(42)
        data = _make_benign_data()
        backend = MockBackend("Cooking")
        result, info = _run(to_fsh_selected_long, "harmful prompt", backend, benign_data=data)
        assert "harmful prompt" in result
        assert "sub_category=Cooking" in info

    def test_fallback_when_unparseable(self):
        random.seed(42)
        data = _make_benign_data()
        backend = MockBackend("totally unrelated response xyz")
        _, info = _run(to_fsh_selected_short, "harmful prompt", backend, benign_data=data)
        assert "selection=fallback_random" in info


class TestGetFshFunctions:
    def test_returns_four(self):
        funcs = get_fsh_functions()
        assert len(funcs) == 4
