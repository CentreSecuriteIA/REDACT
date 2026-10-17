"""Tests for cognitive/psychological hacking techniques."""

import pytest

from redact.jailbreak.hacking.cognitive import (
    _extract_scenario,
    get_cognitive_functions,
    to_persona_roleplay,
)
from redact.llm_pipeline import drive_sync


def _drive(technique, replies: list[str], **kwargs):
    """Run ``technique`` with ``replies`` in turn; return its result and requests."""
    requests = []

    def call(request):
        requests.append(request)
        return replies[len(requests) - 1]

    return drive_sync(technique("harmful prompt", gen_model="m", **kwargs), call), requests


class TestScenario:
    def test_extracts_scenario(self):
        reply = "Some preamble\n**Scenario Description**: A dark alley at midnight"
        assert _extract_scenario(reply) == "A dark alley at midnight"

    def test_falls_back_to_full_text_on_no_match(self):
        reply = "No scenario here, just random text"
        assert _extract_scenario(reply) == reply

    def test_the_scenario_reaches_the_construction_request(self):
        replies = ["**Scenario Description**: A dark alley at midnight", "jailbreak"]
        result, requests = _drive(to_persona_roleplay, replies)
        assert result == ("jailbreak", "category=persona_roleplay")
        assert len(requests) == 2
        assert "dark alley" in str(requests[1].messages)


class TestConstruction:
    def test_a_given_scenario_skips_the_scenario_request(self):
        result, requests = _drive(
            to_persona_roleplay, ["jailbreak output text"], scenario="pre-gen scenario")
        assert result == ("jailbreak output text", "category=persona_roleplay")
        assert len(requests) == 1
        assert requests[0].model == "m"

    @pytest.mark.parametrize("technique", get_cognitive_functions())
    def test_each_technique_names_its_category(self, technique):
        (_, info), _ = _drive(technique, ["output"], scenario="scenario")
        assert info == f"category={technique.__name__.removeprefix('to_')}"


def test_an_empty_construction_is_rejected_naming_the_technique():
    from tests.jailbreak.obfuscation.test_typos import row_of_replies

    row = row_of_replies(to_persona_roleplay, ["scenario", ""])
    assert (row["accepted"], row["jailbreak"]) == (False, "harmful prompt")
    assert "technique=to_persona_roleplay" in row["reasoning"]


class TestGetCognitiveFunctions:
    def test_returns_five(self):
        names = [f.__name__ for f in get_cognitive_functions()]
        assert len(names) == 5
        assert "to_persona_roleplay" in names
        assert "to_deep_inception" in names
