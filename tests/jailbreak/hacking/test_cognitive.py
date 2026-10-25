"""Tests for cognitive/psychological hacking techniques."""

import json

import pytest

from redact.jailbreak.hacking.cognitive import (
    _TECHNIQUES,
    MALFORMED_SCENARIO,
    _extract_scenario,
    get_cognitive_functions,
    to_avi,
    to_persona_roleplay,
)
from redact.llm_pipeline import CHECK_TEMPERATURE, drive_sync


def _drive(technique, replies: list[str], **kwargs):
    """Run ``technique`` with ``replies`` in turn; return its result and requests.

    The scenario check (its request is at the check temperature) accepts and is
    not among the requests: ``test_scenario_check.py`` covers it.
    """
    requests = []

    def call(request):
        if request.temperature == CHECK_TEMPERATURE:
            return "Yes"
        requests.append(request)
        return replies[len(requests) - 1]

    return drive_sync(technique("harmful prompt", gen_model="m", **kwargs), call), requests


class TestScenario:
    def test_extracts_scenario(self):
        reply = "Some preamble\n**Scenario Description**: A dark alley at midnight"
        assert _extract_scenario(reply) == "A dark alley at midnight"

    @pytest.mark.parametrize(("reply", "scene"), [
        ("Notes on ideas\n---\nA dark alley at midnight\n---", "A dark alley at midnight"),
        ("Notes\n---\nA dark alley at midnight", "A dark alley at midnight"),
        ("Notes\n---\nfirst idea\n---\nA dark alley\n---\n", "A dark alley"),
        ("Notes\n ---- \n**Scenario Description**:\nA dark alley\n---", "A dark alley"),
        ("Notes\n---\n### Scenario Description\nA dark alley", "A dark alley"),
    ])
    def test_the_scene_is_the_last_block_behind_the_delimiters(self, reply, scene):
        assert _extract_scenario(reply) == scene

    @pytest.mark.parametrize("reply", [
        "No scenario here, just random text",
        "Notes\n---\n  \n---\n",
        "**Scenario Description**:   ",
        "",
    ])
    def test_a_reply_without_a_scene_has_none(self, reply):
        """It is rejected whole, never taken as the scene."""
        assert _extract_scenario(reply) is None

    def test_the_scenario_reaches_the_construction_request(self):
        replies = ["**Scenario Description**: A dark alley at midnight", "jailbreak"]
        result, requests = _drive(to_persona_roleplay, replies)
        assert result == ("jailbreak", "category=persona_roleplay")
        assert len(requests) == 2
        assert "dark alley" in str(requests[1].messages)

    def test_a_reply_without_a_scene_is_asked_again(self):
        replies = ["just notes", "Notes\n---\nA dark alley at midnight\n---", "jailbreak"]
        result, requests = _drive(to_persona_roleplay, replies)
        assert result == ("jailbreak", "category=persona_roleplay")
        assert len(requests) == 3
        assert requests[1].messages == requests[0].messages
        assert requests[1].model == "m"
        assert "dark alley" in str(requests[2].messages)

    def test_three_replies_without_a_scene_discard_the_prompt(self):
        result, requests = _drive(to_avi, ["notes", "more notes", "still notes"])
        assert result == ("harmful prompt", MALFORMED_SCENARIO)
        assert len(requests) == 3


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


class TestConstructionPrompt:
    """The cognitive techniques keep the shared construction prompt, with the
    category's name and description; the persona prompts are the personas'."""

    @staticmethod
    def _overlay(root, name: str, template: str) -> None:
        folder = root / "jailbreak" / name
        folder.mkdir(parents=True)
        (folder / "template.json").write_text(
            json.dumps({"system_prompt": "", "template": template}), encoding="utf-8")

    @pytest.mark.parametrize("technique", get_cognitive_functions())
    def test_each_technique_reads_the_shared_construction_prompt(
            self, technique, tmp_path):
        self._overlay(
            tmp_path, "jailbreak_construction",
            "MARK-SHARED {jailbreak_type_name}|{jailbreak_type_description}|"
            "{prompt}|{scenario}")
        self._overlay(
            tmp_path, "persona_construction",
            "MARK-PERSONA {persona_name}|{persona_description}|{prompt}|{scenario}")
        entry = _TECHNIQUES[technique.__name__.removeprefix("to_")]
        _, requests = _drive(
            technique, ["output"], scenario="the scene", prompt_dir=tmp_path)
        assert requests[0].messages[-1]["content"] == (
            f"MARK-SHARED {entry['display_name']}|{entry['description']}|"
            "harmful prompt|the scene")
        assert "MARK-PERSONA" not in str(requests[0].messages)


def test_an_empty_construction_is_rejected_naming_the_technique():
    from tests.jailbreak.obfuscation.test_typos import row_of_replies

    row = row_of_replies(to_persona_roleplay, ["---\nscenario\n---", "Yes", ""])
    assert (row["accepted"], row["jailbreak"]) == (False, "harmful prompt")
    assert "technique=to_persona_roleplay" in row["reasoning"]


def test_a_discarded_scenario_rejects_the_row_naming_the_technique():
    from tests.jailbreak.obfuscation.test_typos import row_of_replies

    row = row_of_replies(to_avi, ["notes"] * 3)
    assert (row["accepted"], row["jailbreak"]) == (False, "harmful prompt")
    assert "technique=to_avi" in row["reasoning"]
    assert "malformed scenario" in row["reasoning"]


class TestGetCognitiveFunctions:
    def test_returns_five(self):
        names = [f.__name__ for f in get_cognitive_functions()]
        assert len(names) == 5
        assert "to_persona_roleplay" in names
        assert "to_deep_inception" in names
