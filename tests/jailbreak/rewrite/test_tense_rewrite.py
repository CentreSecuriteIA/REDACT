"""Tests for the tense-rewrite techniques: a model rewrites the prompt in the
past or the future tense.

The check of the rewrite is ``test_preserve.py``'s. These tests compare the
messages a request carries with the prompt's own rendering, and never print a
text.
"""

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.rewrite.tense_rewrite import (
    get_tense_rewrite_functions,
    to_rewrite_in_future_tense,
    to_rewrite_in_past_tense,
)
from redact.llms.prompting import build_messages, load_prompt
from tests.jailbreak.rewrite.test_preserve import PROMPT, drive, text_of


@pytest.mark.parametrize(("technique", "tense"), [
    (to_rewrite_in_past_tense, "past"), (to_rewrite_in_future_tense, "future")])
def test_a_technique_fills_the_prompt_with_its_tense(technique, tense):
    requests, (_, info) = drive(technique(PROMPT, gen_model="gen"), ["rewrite", "Yes"])
    assert requests[0].messages == build_messages(
        load_prompt("jailbreak", "tense_rewrite"), tense=tense, prompt=PROMPT)
    assert info == f"tense={tense}"


def test_the_two_techniques_send_different_requests():
    sent = [text_of(drive(fn(PROMPT, gen_model="gen"), ["rewrite", "Yes"])[0][0])
            for fn in get_tense_rewrite_functions()]
    assert len(set(sent)) == len(sent) == 2     # noqa: PLR2004


@pytest.mark.parametrize("fn", get_tense_rewrite_functions(),
                         ids=lambda fn: fn.__name__)
def test_a_tense_rewrite_is_a_model_written_technique_of_the_rewrite_layer(fn):
    assert (fn.layer, fn.families, fn.requires_llm, fn.kinds) == (
        "rewrite", ["tense_rewrite"], True, ["adversarial"])
    assert (fn.check, fn.calls, fn.complexity, fn.since) == (
        "preserve", [2, 4], 2, "1.3")
