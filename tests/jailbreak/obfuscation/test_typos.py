"""Tests for the typo techniques."""

import pytest

from redact.jailbreak.chain import combine_techniques, run_chain
from redact.jailbreak.obfuscation.typos import to_rewrite_with_typos_low
from redact.jailbreak.units import row_of
from redact.llm_pipeline import drive_sync


def row_of_replies(technique, replies: list[str], prompt: str = "harmful prompt"):
    """The row of one unit of ``technique`` on ``prompt``, its model answering
    ``replies`` in turn."""
    answers = iter(replies)
    value = drive_sync(run_chain([technique], prompt, gen_model="m"),
                       lambda request: next(answers))
    unit = {"id": "in", "prompt": prompt, "iteration": 0,
            "combination": combine_techniques(technique, sort_by_hierarchy=False)}
    return row_of(unit, value, {}, "", gen_model="m")


def test_level_descriptions_come_from_the_prompt_file():
    from redact.llms.prompting import load_prompt

    config = load_prompt("jailbreak", "rewrite_with_typos")
    assert set(config["level_descriptions"]) == {"low", "medium", "high", "insane"}


@pytest.mark.parametrize("reply", ["", " \n "])
def test_an_empty_rewrite_is_rejected_naming_the_technique(reply):
    row = row_of_replies(to_rewrite_with_typos_low, [reply])
    assert (row["accepted"], row["jailbreak"]) == (False, "harmful prompt")
    assert "technique=to_rewrite_with_typos_low" in row["reasoning"]
    assert "empty text" in row["reasoning"]


def test_an_empty_input_left_empty_stays_a_noop():
    row = row_of_replies(to_rewrite_with_typos_low, [""], prompt="")
    assert (row["accepted"], row["is_noop"]) == (True, True)
