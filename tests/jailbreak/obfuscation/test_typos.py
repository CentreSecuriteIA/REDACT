"""Tests for the typo techniques: a rewrite, then its check."""

import pytest

from redact.jailbreak.chain import combine_techniques, run_chain
from redact.jailbreak.obfuscation.typos import (
    get_typos_functions,
    to_rewrite_with_typos_low,
)
from redact.jailbreak.units import row_of
from redact.llm_pipeline import CHECK_TEMPERATURE, drive_sync


def row_of_replies(technique, replies: list[str], prompt: str = "harmful prompt"):
    """The row of one unit of ``technique`` on ``prompt``, its model answering
    ``replies`` in turn."""
    answers = iter(replies)
    value = drive_sync(run_chain([technique], prompt, gen_model="m"),
                       lambda request: next(answers))
    unit = {"id": "in", "prompt": prompt, "iteration": 0,
            "combination": combine_techniques(technique, sort_by_hierarchy=False)}
    return row_of(unit, value, {}, "", gen_model="m")


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


def test_level_descriptions_come_from_the_prompt_file():
    from redact.llms.prompting import load_prompt

    config = load_prompt("jailbreak", "rewrite_with_typos")
    assert set(config["level_descriptions"]) == {"low", "medium", "high", "insane"}


def test_an_accepted_rewrite_takes_a_rewrite_and_a_check():
    requests, result = _drive(
        to_rewrite_with_typos_low("a plain prompt", gen_model="gen",
                                  check_model="check"),
        ["a plian prompt", "Yes"])
    assert result == ("a plian prompt", "level=low")
    assert [r.model for r in requests] == ["gen", "check"]
    check = requests[1]
    assert "a plain prompt" in _text(check) and "a plian prompt" in _text(check)
    assert (check.temperature, requests[0].temperature) == (CHECK_TEMPERATURE, None)


def test_a_rejected_rewrite_is_made_again_with_the_verdict():
    requests, result = _drive(
        to_rewrite_with_typos_low("a plain prompt", gen_model="gen"),
        ["a plian prompt", "No: a word changed.", "a plaim prompt", "Yes"])
    assert result == ("a plaim prompt", "level=low")
    assert "No: a word changed." not in _text(requests[0])
    assert "No: a word changed." in _text(requests[2])


def test_a_rewrite_rejected_every_time_is_discarded_with_the_reason():
    row = row_of_replies(to_rewrite_with_typos_low,
                         ["first", "No: unreadable.", "second", "No: still so."])
    assert (row["accepted"], row["jailbreak"]) == (False, "harmful prompt")
    assert row["reasoning"] == (
        "DISCARDED; technique=to_rewrite_with_typos_low; reason=typo check "
        "failed; level=low; feedback=No: still so.")


@pytest.mark.parametrize("technique", get_typos_functions())
@pytest.mark.parametrize("verdicts", [["Yes"], ["No", "Yes"], ["No", "No"]])
def test_an_application_makes_no_more_calls_than_the_spec_says(technique, verdicts):
    replies = [reply for verdict in verdicts for reply in ("rewrite", verdict)]
    requests, _ = _drive(technique("a plain prompt", gen_model="gen"), replies)
    low, high = technique.calls
    assert low <= len(requests) <= high


def test_check_attempts_bounds_the_rewrites():
    requests, (_, info) = _drive(
        to_rewrite_with_typos_low("a plain prompt", gen_model="gen",
                                  check_attempts=1),
        ["rewrite", "No"])
    assert [r.model for r in requests] == ["gen", "gen"]
    assert info.startswith("DISCARDED")


def test_without_a_check_model_the_generation_model_checks():
    requests, _ = _drive(
        to_rewrite_with_typos_low("a plain prompt", gen_model="gen"),
        ["rewrite", "No", "rewrite", "Yes"])
    assert [r.model for r in requests] == ["gen"] * 4


def test_a_check_model_takes_the_check_requests_only():
    requests, _ = _drive(
        to_rewrite_with_typos_low("a plain prompt", gen_model="gen",
                                  check_model="check"),
        ["rewrite", "No", "rewrite", "Yes"])
    assert [r.model for r in requests] == ["gen", "check", "gen", "check"]


@pytest.mark.parametrize("reply", ["", " \n "])
def test_an_empty_rewrite_the_check_accepts_is_still_rejected(reply):
    row = row_of_replies(to_rewrite_with_typos_low, [reply, "Yes"])
    assert (row["accepted"], row["jailbreak"]) == (False, "harmful prompt")
    assert "technique=to_rewrite_with_typos_low" in row["reasoning"]
    assert "empty text" in row["reasoning"]


def test_an_empty_input_left_empty_stays_a_noop():
    row = row_of_replies(to_rewrite_with_typos_low, ["", "Yes"], prompt="")
    assert (row["accepted"], row["is_noop"]) == (True, True)
