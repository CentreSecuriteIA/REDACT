"""Tests for cognitive/psychological hacking techniques."""

import pytest
from tests.conftest import MockBackend
from redact.jailbreak.hacking.cognitive import (
    HACKING_CATEGORIES,
    CATEGORY_NAMES,
    get_situation,
    create_jailbreak,
    get_hacking_functions,
)


class TestConstants:
    def test_five_categories(self):
        assert len(HACKING_CATEGORIES) == 5

    def test_five_names(self):
        assert len(CATEGORY_NAMES) == 5
        assert "persona_roleplay" in CATEGORY_NAMES
        assert "deep_inception" in CATEGORY_NAMES

    def test_category_tuples(self):
        for name, desc in HACKING_CATEGORIES:
            assert isinstance(name, str)
            assert len(desc) > 20


class TestGetSituation:
    def test_extracts_scenario(self):
        backend = MockBackend("Some preamble\n**Scenario Description**: A dark alley at midnight")
        result = get_situation("test prompt", backend, "model")
        assert "dark alley" in result

    def test_raises_on_no_match(self):
        backend = MockBackend("No scenario here, just random text")
        with pytest.raises(ValueError, match="scenario description"):
            get_situation("test", backend, "model")


class TestCreateJailbreak:
    def test_with_scenario(self):
        backend = MockBackend("jailbreak output text")
        jailbreak, info, scenario = create_jailbreak(
            "harmful prompt", 0, backend, "model", scenario="pre-gen scenario"
        )
        assert jailbreak == "jailbreak output text"
        assert "persona_roleplay" in info
        assert scenario == "pre-gen scenario"

    def test_all_categories(self):
        backend = MockBackend("output")
        for i in range(5):
            _, info, _ = create_jailbreak(
                "prompt", i, backend, "model", scenario="scenario"
            )
            assert CATEGORY_NAMES[i] in info


class TestGetHackingFunctions:
    def test_returns_five(self):
        funcs = get_hacking_functions()
        assert len(funcs) == 5
        names = [f.__name__ for f in funcs]
        assert "to_persona_roleplay" in names
        assert "to_deep_inception" in names
