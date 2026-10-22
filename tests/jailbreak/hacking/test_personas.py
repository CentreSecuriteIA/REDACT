"""Tests for the persona techniques."""

import pytest

import redact.jailbreak.hacking.personas as P
from redact.jailbreak.hacking.cognitive import MALFORMED_SCENARIO
from redact.jailbreak.hacking.personas import (
    display_name,
    get_persona_functions,
    to_invented_persona,
)
from redact.llm_pipeline import CHECK_TEMPERATURE, drive_sync

_SCENE = "Notes\n---\nA quiet lab at night\n---"


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


def _named() -> dict:
    """Each named persona's technique, by the persona's key."""
    return {fn.__name__.removeprefix("to_").removesuffix("_persona"): fn
            for fn in get_persona_functions() if fn is not to_invented_persona}


def test_there_are_one_invented_and_seventeen_named_personas():
    assert len(get_persona_functions()) == 18
    assert sorted(_named()) == sorted(P._NAMED_PERSONAS)


def test_every_persona_of_the_data_file_has_a_technique():
    """A persona added to personas.json without its entry in the registry
    would never be drawn."""
    assert sorted(P._PERSONAS) == sorted(P._NAMED_PERSONAS)


@pytest.mark.parametrize("key", ["mercenary", "underground_hacker",
                                 "anarchist_philosopher"])
def test_the_three_spec_1_3_archetypes_are_attack_only_scenario_checked_personas(key):
    fn = _named()[key]
    assert (fn.families, fn.layer, fn.complexity) == (["persona"], "hacking", 2)
    assert (fn.requires_llm, fn.requires_benign, fn.check, fn.calls) == (
        True, False, "scenario", [3, 9])
    assert (fn.kinds, fn.since) == (["adversarial"], "1.3")


def test_every_named_persona_has_a_display_name_and_a_description():
    for key in P._NAMED_PERSONAS:
        entry = P._PERSONAS[key]
        assert entry["display_name"].strip(), key
        assert entry["description"].strip(), key


@pytest.mark.parametrize("key", P._NAMED_PERSONAS)
def test_the_scenario_prompt_names_the_persona_by_its_display_name(key):
    result, requests = _drive(_named()[key], [_SCENE, "jailbreak"])
    assert result == ("jailbreak", f"category={key}_persona")
    scenario_prompt = str(requests[0].messages)
    assert P._PERSONAS[key]["display_name"] in scenario_prompt
    if "_" in key:
        assert key not in scenario_prompt


def test_the_display_name_falls_back_to_the_key_in_words(monkeypatch):
    monkeypatch.setitem(P._PERSONAS, "cult_leader", {"description": "A leader."})
    assert display_name("cult_leader") == "Cult Leader"
    assert display_name("someone_from_the_future") == "Someone from the Future"


def test_the_scene_reaches_the_construction_request():
    _, requests = _drive(_named()["alien"], [_SCENE, "jailbreak"])
    assert len(requests) == 2
    assert "A quiet lab at night" in str(requests[1].messages)
    assert P._PERSONAS["alien"]["description"] in str(requests[1].messages)


def test_a_scenario_reply_without_a_scene_is_asked_again_then_discards():
    result, requests = _drive(_named()["alien"], ["notes", _SCENE, "jailbreak"])
    assert result == ("jailbreak", "category=alien_persona")
    assert len(requests) == 3
    result, requests = _drive(_named()["alien"], ["notes"] * 3)
    assert result == ("harmful prompt", MALFORMED_SCENARIO)
    assert len(requests) == 3


def test_the_invented_persona_keeps_its_category_and_its_discard():
    result, _ = _drive(to_invented_persona, [_SCENE, "jailbreak"])
    assert result == ("jailbreak", "category=invented_persona")
    result, _ = _drive(to_invented_persona, ["notes"] * 3)
    assert result == ("harmful prompt", MALFORMED_SCENARIO)


def test_a_given_scenario_skips_the_scenario_request():
    result, requests = _drive(_named()["actor"], ["jailbreak"], scenario="a stage")
    assert result == ("jailbreak", "category=actor_persona")
    assert len(requests) == 1
    assert requests[0].model == "m"
