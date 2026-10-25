"""Tests for the persona techniques."""

import json
import string

import pytest

import redact.jailbreak.hacking.personas as P
from redact.jailbreak.hacking.cognitive import MALFORMED_SCENARIO
from redact.jailbreak.hacking.personas import (
    MALFORMED_PERSONA,
    display_name,
    get_persona_functions,
    to_invented_persona,
)
from redact.jailbreak.technique import DISCARDED
from redact.llm_pipeline import CHECK_TEMPERATURE, drive_sync
from redact.llms.prompting import build_messages, load_prompt

_SCENE = "Notes\n---\nA quiet lab at night\n---"
_PERSONA = (
    "Name: Maple the Baker\nDescription: A cheerful baker who talks about bread.")


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


def test_the_invented_persona_asks_for_a_persona_first_then_makes_the_scenario():
    result, requests = _drive(to_invented_persona, [_PERSONA, _SCENE, "jailbreak"])
    assert result == ("jailbreak", "category=invented_persona")
    assert len(requests) == 3
    assert [request.model for request in requests] == ["m"] * 3
    assert "harmful prompt" in str(requests[0].messages)


def test_the_scenario_and_the_construction_get_the_invented_name_and_description():
    _, requests = _drive(to_invented_persona, [_PERSONA, _SCENE, "jailbreak"])
    scenario_prompt, construction_prompt = (
        str(requests[1].messages), str(requests[2].messages))
    assert "Maple the Baker" in scenario_prompt
    assert "A cheerful baker who talks about bread." in scenario_prompt
    assert "A cheerful baker who talks about bread." in construction_prompt
    assert "A quiet lab at night" in construction_prompt


@pytest.mark.parametrize("reply", [
    "Name: Maple the Baker",                                     # no description
    "Description: A baker.",                                     # no name
    "Name:\nDescription: A baker.",                              # an empty name
    "Name: Maple\nDescription:",                                 # an empty description
    "**Name:** Maple\n**Description:** A baker.",                # markdown labels
    "Here is one.\nName: Maple\nDescription: A baker.",          # text before
    "Name: Maple\nDescription: A baker.\nName: Rye\nDescription: A miller.",  # two
    "Maple, a baker",                                            # no labels
], ids=["no description", "no name", "empty name", "empty description", "markdown",
        "text before", "two personas", "no labels"])
def test_a_reply_that_is_not_a_name_then_a_description_is_asked_again(reply):
    result, requests = _drive(
        to_invented_persona, [reply, _PERSONA, _SCENE, "jailbreak"])
    assert result == ("jailbreak", "category=invented_persona")
    assert len(requests) == 4


@pytest.mark.parametrize(("reply", "persona"), [
    (_PERSONA, ["Maple the Baker", "A cheerful baker who talks about bread."]),
    ("\n\nName:   Maple  \n\nDescription:  A baker.\n", ["Maple", "A baker."]),
    ("Name: Maple\r\nDescription: A baker.\r\n", ["Maple", "A baker."]),
    ("Name: Maple\nDescription: A baker,\nwho bakes bread.",
     ["Maple", "A baker,\nwho bakes bread."]),
], ids=["plain", "blank lines and spaces", "CRLF", "a description over lines"])
def test_a_name_then_a_description_is_read_whole(reply, persona):
    assert P._persona_of(reply) == persona


def test_an_invented_persona_whose_replies_never_took_the_form_is_discarded():
    result, requests = _drive(to_invented_persona, ["notes"] * 3)
    assert result == ("harmful prompt", MALFORMED_PERSONA)
    assert MALFORMED_PERSONA.startswith(DISCARDED)
    assert len(requests) == 3


def test_an_invented_persona_keeps_the_scenario_discard():
    result, requests = _drive(to_invented_persona, [_PERSONA, *["notes"] * 3])
    assert result == ("harmful prompt", MALFORMED_SCENARIO)
    assert len(requests) == 4


def test_a_given_scenario_skips_the_scenario_but_not_the_invention():
    result, requests = _drive(
        to_invented_persona, [_PERSONA, "jailbreak"], scenario="a stage")
    assert result == ("jailbreak", "category=invented_persona")
    assert len(requests) == 2
    assert "A cheerful baker who talks about bread." in str(requests[1].messages)


def test_the_invented_persona_declares_its_invention_in_its_calls():
    """The family's 3 to 9 calls, and its persona first: up to 3 replies."""
    assert to_invented_persona.calls == [4, 12]
    assert _named()["alien"].calls == [3, 9]


def test_a_given_scenario_skips_the_scenario_request():
    result, requests = _drive(_named()["actor"], ["jailbreak"], scenario="a stage")
    assert result == ("jailbreak", "category=actor_persona")
    assert len(requests) == 1
    assert requests[0].model == "m"


# --- the persona is the one who answers --------------------------------------
#
# The scenario and the construction prompts get the persona's name, so the
# target is asked to play the persona (the answerer), not to be the person
# who asks. These tests check which prompt is read and which placeholders it
# is given, and where the name stands; they assert no prompt wording.

_PERSONA_PROMPTS = {
    "persona_invention": {"prompt"},
    "persona_scenario_generation": {"prompt", "persona_name", "persona_description"},
    "persona_construction": {
        "prompt", "persona_name", "persona_description", "scenario"},
}


def _fields(config: dict) -> list[str]:
    """Each placeholder in a prompt's system prompt and template, repeats kept."""
    return [name
            for part in (config["system_prompt"], config["template"])
            for _, name, _, _ in string.Formatter().parse(part) if name]


@pytest.mark.parametrize(("name", "expected"), _PERSONA_PROMPTS.items())
def test_a_persona_prompt_holds_the_placeholders_the_code_fills_and_no_other(
        name, expected):
    config = load_prompt("jailbreak", name)
    assert set(_fields(config)) == expected
    assert set(config["seed_fields"]) == expected


@pytest.mark.parametrize(
    "name", ["persona_scenario_generation", "persona_construction"])
def test_the_persona_prompts_use_the_persona_name_more_than_once(name):
    """Once to say who the persona is and once at least for what it does: the
    answerer's part. A prompt that names it once (or never) leaves the part
    to the model."""
    fields = _fields(load_prompt("jailbreak", name))
    assert fields.count("persona_name") > 1


def test_the_construction_holds_the_prompt_the_scene_and_the_description():
    fields = _fields(load_prompt("jailbreak", "persona_construction"))
    for placeholder in ("persona_description", "scenario", "prompt"):
        assert placeholder in fields


def test_the_rendered_construction_holds_each_value_and_the_name_more_than_once():
    values = dict(prompt="PROBE-REQUEST", persona_name="PROBE-NAME",
                  persona_description="PROBE-DESCRIPTION", scenario="PROBE-SCENE")
    text = "\n".join(
        m["content"] for m in build_messages(
            load_prompt("jailbreak", "persona_construction"), **values))
    for value in values.values():
        assert value in text
    assert text.count("PROBE-NAME") > 1
    assert "{" not in text


@pytest.mark.parametrize("key", P._NAMED_PERSONAS)
def test_a_named_persona_is_built_from_the_persona_construction_prompt(key):
    expected = build_messages(
        load_prompt("jailbreak", "persona_construction"),
        prompt="harmful prompt", persona_name=P._PERSONAS[key]["display_name"],
        persona_description=P._PERSONAS[key]["description"],
        scenario="A quiet lab at night")
    _, requests = _drive(_named()[key], [_SCENE, "jailbreak"])
    assert requests[1].messages == expected


def test_an_invented_persona_gives_the_construction_its_invented_name():
    _, requests = _drive(to_invented_persona, [_PERSONA, _SCENE, "jailbreak"])
    assert requests[2].messages == build_messages(
        load_prompt("jailbreak", "persona_construction"),
        prompt="harmful prompt", persona_name="Maple the Baker",
        persona_description="A cheerful baker who talks about bread.",
        scenario="A quiet lab at night")


def test_a_given_scenario_still_gives_the_construction_the_persona_name():
    _, requests = _drive(_named()["actor"], ["jailbreak"], scenario="a stage")
    assert P._PERSONAS["actor"]["display_name"] in str(requests[0].messages)


def _overlay(root, name: str, template: str):
    folder = root / "jailbreak" / name
    folder.mkdir(parents=True)
    (folder / "template.json").write_text(
        json.dumps({"system_prompt": "", "template": template}), encoding="utf-8")


def test_the_persona_construction_is_overlaid_apart_from_the_shared_one(tmp_path):
    """A prompt_dir's ``persona_construction`` reaches the persona techniques,
    and its ``jailbreak_construction`` (the cognitive techniques') does not."""
    _overlay(tmp_path, "persona_construction",
             "MARK-PERSONA {persona_name}|{persona_description}|{prompt}|{scenario}")
    _overlay(tmp_path, "jailbreak_construction",
             "MARK-SHARED {jailbreak_type_name}|{jailbreak_type_description}|"
             "{prompt}|{scenario}")
    description = P._PERSONAS["alien"]["description"]
    _, requests = _drive(_named()["alien"], [_SCENE, "jailbreak"], prompt_dir=tmp_path)
    assert requests[1].messages[-1]["content"] == (
        f"MARK-PERSONA Alien|{description}|harmful prompt|A quiet lab at night")
    assert "MARK-SHARED" not in str(requests[1].messages)
