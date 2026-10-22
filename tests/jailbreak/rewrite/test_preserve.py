"""Tests for the preserve check of the persuasion and tense-rewrite
techniques: a rewrite of the whole prompt, then its check.

No test prints a text: they compare flags, names and the messages a request
carries against the prompts' own renderings.
"""

import json

import pytest

from redact.jailbreak.chain import combine_techniques, run_chain
from redact.jailbreak.rewrite.persuasion import to_persuade_scarcity
from redact.jailbreak.rewrite.tense_rewrite import to_rewrite_in_past_tense
from redact.jailbreak.units import row_of
from redact.llm_pipeline import CHECK_TEMPERATURE, drive_sync

#: Each technique with its info label.
TECHNIQUES = [(to_persuade_scarcity, "category=scarcity"),
              (to_rewrite_in_past_tense, "tense=past")]
_CASES = pytest.mark.parametrize(("technique", "label"), TECHNIQUES,
                                 ids=[fn.__name__ for fn, _ in TECHNIQUES])
PROMPT = "a plain prompt"


def drive(gen, replies: list[str]):
    """Answer ``gen``'s requests with ``replies`` in turn; return
    ``(requests, result)``."""
    requests, answers = [], iter(replies)

    def answer(request):
        requests.append(request)
        return next(answers)

    return requests, drive_sync(gen, answer)


def text_of(request) -> str:
    """Every message of ``request``, joined."""
    return "\n".join(m["content"] for m in request.messages)


@_CASES
def test_an_accepted_rewrite_takes_a_rewrite_and_a_check(technique, label):
    requests, result = drive(
        technique(PROMPT, gen_model="gen", check_model="check"),
        ["  the rewrite \n", "Yes"])
    assert result == ("the rewrite", label)
    assert [r.model for r in requests] == ["gen", "check"]
    check = requests[1]
    assert PROMPT in text_of(check) and "the rewrite" in text_of(check)
    assert (check.temperature, requests[0].temperature) == (CHECK_TEMPERATURE, None)


@_CASES
def test_a_rejected_rewrite_is_made_again_with_the_verdict(technique, label):
    requests, result = drive(
        technique(PROMPT, gen_model="gen"),
        ["first", "No: a word changed.", "second", "Yes"])
    assert result == ("second", label)
    assert "No: a word changed." not in text_of(requests[0])
    assert "No: a word changed." in text_of(requests[2])


@_CASES
def test_a_rewrite_rejected_every_time_is_discarded_with_the_reason(technique, label):
    _, (text, info) = drive(
        technique(PROMPT, gen_model="gen"),
        ["first", "No: unreadable.", "second", "No: still so."])
    assert text == "second"
    assert info == ("DISCARDED; reason=preserve check failed; "
                    f"{label}; feedback=No: still so.")


@_CASES
def test_a_discarded_rewrite_leaves_a_row_with_the_prompt_and_the_reason(
        technique, label):
    answers = iter(["first", "No", "second", "No"])
    value = drive_sync(run_chain([technique], PROMPT, gen_model="m"),
                       lambda request: next(answers))
    unit = {"id": "in", "prompt": PROMPT, "iteration": 0,
            "combination": combine_techniques(technique, sort_by_hierarchy=False)}
    row = row_of(unit, value, {}, "", gen_model="m")
    assert (row["accepted"], row["jailbreak"]) == (False, PROMPT)
    assert row["reasoning"] == (
        f"DISCARDED; technique={technique.__name__}; reason=preserve check "
        f"failed; {label}; feedback=No")


@_CASES
@pytest.mark.parametrize("verdicts", [["Yes"], ["No", "Yes"], ["No", "No"]])
def test_an_application_makes_no_more_calls_than_the_spec_says(
        technique, label, verdicts):
    replies = [reply for verdict in verdicts for reply in ("rewrite", verdict)]
    requests, _ = drive(technique(PROMPT, gen_model="gen"), replies)
    low, high = technique.calls
    assert (technique.check, low <= len(requests) <= high) == ("preserve", True)


@_CASES
def test_check_attempts_bounds_the_rewrites(technique, label):
    requests, (_, info) = drive(
        technique(PROMPT, gen_model="gen", check_attempts=1), ["rewrite", "No"])
    assert [r.model for r in requests] == ["gen", "gen"]
    assert info.startswith("DISCARDED")


@_CASES
def test_without_a_check_model_the_generation_model_checks(technique, label):
    requests, _ = drive(technique(PROMPT, gen_model="gen"),
                        ["rewrite", "No", "rewrite", "Yes"])
    assert [r.model for r in requests] == ["gen"] * 4


@_CASES
def test_a_check_model_takes_the_check_requests_only(technique, label):
    requests, _ = drive(technique(PROMPT, gen_model="gen", check_model="check"),
                        ["rewrite", "No", "rewrite", "Yes"])
    assert [r.model for r in requests] == ["gen", "check", "gen", "check"]


@_CASES
@pytest.mark.parametrize("reply", ["", " \n "])
def test_an_empty_rewrite_the_check_accepts_is_still_rejected(
        technique, label, reply):
    answers = iter([reply, "Yes"])
    _, info = drive_sync(run_chain([technique], PROMPT, gen_model="m"),
                         lambda request: next(answers))
    assert info == f"DISCARDED; technique={technique.__name__}; reason=empty text"


@_CASES
def test_the_check_reads_the_prompt_dir_overlay_of_its_prompt(
        technique, label, tmp_path):
    overlay = tmp_path / "jailbreak" / "preserve_check"
    overlay.mkdir(parents=True)
    (overlay / "template.json").write_text(json.dumps({
        "system_prompt": "", "template": "OVERLAID {original} | {rewritten}",
        "seed_fields": ["original", "rewritten"], "few_shot_examples": [],
        "metadata": {"version": "1.0", "notes": ""},
    }), encoding="utf-8")
    requests, _ = drive(
        technique(PROMPT, gen_model="gen", prompt_dir=tmp_path), ["rewrite", "Yes"])
    assert text_of(requests[1]) == f"OVERLAID {PROMPT} | rewrite"
