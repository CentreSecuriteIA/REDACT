"""Tests for the scenario check of the cognitive and persona techniques: a
scenario, then its check, then the construction built on it."""

import pytest

from redact.jailbreak.hacking.cognitive import (
    MALFORMED_SCENARIO,
    to_hypothetical_framing,
)
from redact.jailbreak.hacking.personas import to_artist_persona
from redact.llm_pipeline import CHECK_TEMPERATURE, drive_sync

# to_invented_persona asks for its persona first: test_personas.py covers that
# step and the scenario it shares with the named personas.
TECHNIQUES = [to_hypothetical_framing, to_artist_persona]

PROMPT = "sample request 1"
# A scenario reply: notes, then the scene between two delimiter lines.
ONE = "Notes\n---\nscene one\n---"
TWO = "Notes\n---\nscene two\n---"
NO_SCENE = "no scene in this reply"


def _drive(gen, replies: list[str]):
    """Answer ``gen``'s requests with ``replies`` in turn; return
    ``(requests, result)``."""
    requests, answers = [], iter(replies)

    def answer(request):
        requests.append(request)
        return next(answers)

    return requests, drive_sync(gen, answer)


def _text(request) -> str:
    """Every message of ``request``, joined."""
    return "\n".join(m["content"] for m in request.messages)


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_an_accepted_scenario_takes_a_scenario_a_check_and_the_construction(technique):
    requests, (text, info) = _drive(
        technique(PROMPT, gen_model="gen", check_model="check"),
        [ONE, "Yes", "constructed"])
    assert (text, info.startswith("category=")) == ("constructed", True)
    assert [r.model for r in requests] == ["gen", "check", "gen"]
    check = requests[1]
    assert PROMPT in _text(check) and "scene one" in _text(check)
    assert "Notes" not in _text(check)
    assert (check.temperature, requests[0].temperature) == (CHECK_TEMPERATURE, None)
    assert "scene one" in _text(requests[2])


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_a_rejected_scenario_is_made_again_with_the_verdict(technique):
    requests, (text, _) = _drive(
        technique(PROMPT, gen_model="gen"),
        [ONE, "No: off topic.", TWO, "Yes", "constructed"])
    assert text == "constructed"
    assert "No: off topic." not in _text(requests[0])
    assert "No: off topic." in _text(requests[2])
    assert "scene two" in _text(requests[4]) and "scene one" not in _text(requests[4])


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_a_scenario_rejected_every_time_is_discarded_with_no_construction(technique):
    requests, result = _drive(
        technique(PROMPT, gen_model="gen"), [ONE, "No: a.", TWO, "No: b."])
    assert result == (
        PROMPT, "DISCARDED; reason=scenario check failed; feedback=No: b.")
    assert [r.model for r in requests] == ["gen"] * 4


def test_a_discarded_scenario_is_a_rejected_row():
    from tests.jailbreak.obfuscation.test_typos import row_of_replies

    row = row_of_replies(to_artist_persona, [ONE, "No: a.", TWO, "No: b."], PROMPT)
    assert (row["accepted"], row["jailbreak"]) == (False, PROMPT)
    assert row["reasoning"] == (
        "DISCARDED; technique=to_artist_persona; reason=scenario check failed; "
        "feedback=No: b.")


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_a_reply_without_a_scene_is_asked_again_before_the_check(technique):
    requests, (text, _) = _drive(
        technique(PROMPT, gen_model="gen", check_model="check"),
        [NO_SCENE, ONE, "Yes", "constructed"])
    assert text == "constructed"
    assert [r.model for r in requests] == ["gen", "gen", "check", "gen"]


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_three_replies_without_a_scene_discard_with_no_check(technique):
    requests, result = _drive(
        technique(PROMPT, gen_model="gen", check_model="check"), [NO_SCENE] * 3)
    assert result == (PROMPT, MALFORMED_SCENARIO)
    assert [r.model for r in requests] == ["gen"] * 3


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_a_second_scenario_without_a_scene_is_discarded_as_malformed(technique):
    requests, result = _drive(
        technique(PROMPT, gen_model="gen"),
        [ONE, "No: a.", NO_SCENE, NO_SCENE, NO_SCENE])
    assert result == (PROMPT, MALFORMED_SCENARIO)
    assert [r.model for r in requests] == ["gen"] * 5


@pytest.mark.parametrize("technique", TECHNIQUES)
@pytest.mark.parametrize("replies", [
    [ONE, "Yes", "constructed"],
    [NO_SCENE, NO_SCENE, ONE, "Yes", "constructed"],
    [ONE, "No", TWO, "Yes", "constructed"],
    [ONE, "No", TWO, "No"],
    [NO_SCENE, NO_SCENE, ONE, "No", NO_SCENE, NO_SCENE, TWO, "Yes", "constructed"],
])
def test_an_application_makes_no_more_calls_than_the_spec_says(technique, replies):
    requests, _ = _drive(technique(PROMPT, gen_model="gen"), replies)
    low, high = technique.calls
    assert low <= len(requests) <= high


def test_the_spec_counts_three_to_nine_calls_for_both_families():
    assert [t.calls for t in TECHNIQUES] == [[3, 9]] * len(TECHNIQUES)
    assert {t.check for t in TECHNIQUES} == {"scenario"}


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_check_attempts_bounds_the_scenarios(technique):
    requests, (_, info) = _drive(
        technique(PROMPT, gen_model="gen", check_attempts=1), [ONE, "No"])
    assert [r.model for r in requests] == ["gen", "gen"]
    assert info.startswith("DISCARDED; reason=scenario check failed")


def test_without_a_check_model_the_generation_model_checks():
    requests, _ = _drive(
        to_artist_persona(PROMPT, gen_model="gen"),
        [ONE, "No", TWO, "Yes", "constructed"])
    assert [r.model for r in requests] == ["gen"] * 5


def test_a_check_model_takes_the_check_requests_only():
    requests, _ = _drive(
        to_artist_persona(PROMPT, gen_model="gen", check_model="check"),
        [ONE, "No", TWO, "Yes", "constructed"])
    assert [r.model for r in requests] == ["gen", "check", "gen", "check", "gen"]


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_a_scenario_given_is_neither_asked_for_nor_checked(technique):
    requests, (text, _) = _drive(
        technique(PROMPT, gen_model="gen", scenario="given scene"), ["constructed"])
    assert text == "constructed"
    assert len(requests) == 1 and "given scene" in _text(requests[0])
