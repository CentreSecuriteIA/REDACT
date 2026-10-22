"""Tests for the model-distractor techniques: a model writes sentences, which
are joined to the prompt before it or after it. No check model reads them.

These tests compare the messages a request carries with the prompt's own
rendering, and never print a text.
"""

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.rewrite.model_distractors import (
    EMPTY_DISTRACTOR,
    get_model_distractor_functions,
    to_model_distractor_prefix,
    to_model_distractor_suffix,
)
from redact.llms.prompting import build_messages, load_prompt
from tests.jailbreak.rewrite.test_preserve import PROMPT, drive

SENTENCES = "two plain sentences. they end here."


@pytest.mark.parametrize(("technique", "position", "joined"), [
    (to_model_distractor_prefix, "before", f"{SENTENCES} {PROMPT}"),
    (to_model_distractor_suffix, "after", f"{PROMPT} {SENTENCES}"),
])
def test_the_sentences_are_joined_before_or_after_the_prompt(
        technique, position, joined):
    requests, result = drive(
        technique(PROMPT, gen_model="gen"), [f"  {SENTENCES}\n"])
    assert result == (joined, f"position={position}")
    assert len(requests) == 1
    assert requests[0].messages == build_messages(
        load_prompt("jailbreak", "model_distractor"), position=position,
        prompt=PROMPT)


@pytest.mark.parametrize("technique", [to_model_distractor_prefix,
                                       to_model_distractor_suffix])
def test_a_reply_with_no_text_is_asked_for_again(technique):
    requests, (text, info) = drive(
        technique(PROMPT, gen_model="gen"), ["", " \n ", SENTENCES])
    assert [r.model for r in requests] == ["gen"] * 3
    assert SENTENCES in text and PROMPT in text and info.startswith("position=")


@pytest.mark.parametrize("technique", [to_model_distractor_prefix,
                                       to_model_distractor_suffix])
def test_a_prompt_whose_replies_never_held_text_is_discarded_unchanged(technique):
    requests, (text, info) = drive(technique(PROMPT, gen_model="gen"), ["", "", " "])
    assert len(requests) == 3   # noqa: PLR2004
    assert (text, info) == (PROMPT, EMPTY_DISTRACTOR)
    assert info == "DISCARDED; reason=empty distractor"


@pytest.mark.parametrize("fn", get_model_distractor_functions(),
                         ids=lambda fn: fn.__name__)
def test_a_model_distractor_is_a_model_written_technique_with_no_check(fn):
    assert (fn.layer, fn.families, fn.requires_llm, fn.kinds) == (
        "rewrite", ["model_distractors"], True, ["adversarial"])
    assert (fn.check, fn.calls, fn.complexity, fn.since) == (
        None, [1, 3], 2, "1.3")
