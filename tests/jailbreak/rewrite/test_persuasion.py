"""Tests for the persuasion techniques: one per entry of
``configs/jailbreak/rewrite/persuasion_techniques.json``.

The check of the rewrite is ``test_preserve.py``'s. These tests compare the
messages a request carries with the prompt's own rendering, and never print a
text.
"""

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.pool import registry
from redact.jailbreak.rewrite.persuasion import _PERSUASION, get_persuasion_functions
from redact.llms.prompting import build_messages, load_prompt
from tests.jailbreak.rewrite.test_preserve import PROMPT, drive, text_of

_BY_KEY = {fn.__name__.removeprefix("to_persuade_"): fn
           for fn in get_persuasion_functions()}


def test_there_is_one_technique_per_entry_of_the_file_named_for_its_key():
    assert len(_PERSUASION) == 12  # noqa: PLR2004
    assert list(_BY_KEY) == list(_PERSUASION)
    assert [fn.__name__ for fn in get_persuasion_functions()] == [
        f"to_persuade_{key}" for key in _PERSUASION]
    assert all(entry.keys() >= {"display_name", "definition", "example"}
               for entry in _PERSUASION.values())


@pytest.mark.parametrize("key", list(_PERSUASION))
def test_a_technique_fills_the_prompt_with_its_own_entry(key):
    entry = _PERSUASION[key]
    requests, (_, info) = drive(
        _BY_KEY[key](PROMPT, gen_model="gen"), ["rewrite", "Yes"])
    assert requests[0].messages == build_messages(
        load_prompt("jailbreak", "persuasion_rewrite"),
        technique_name=entry["display_name"], definition=entry["definition"],
        example=entry["example"], prompt=PROMPT)
    assert info == f"category={key}"


def test_the_twelve_techniques_send_twelve_different_requests():
    sent = set()
    for fn in _BY_KEY.values():
        requests, _ = drive(fn(PROMPT, gen_model="gen"), ["rewrite", "Yes"])
        sent.add(text_of(requests[0]))
    assert len(sent) == 12  # noqa: PLR2004


@pytest.mark.parametrize("fn", get_persuasion_functions(),
                         ids=lambda fn: fn.__name__)
def test_a_persuasion_is_a_model_written_technique_of_the_rewrite_layer(fn):
    assert (fn.layer, fn.families, fn.requires_llm, fn.kinds) == (
        "rewrite", ["persuasion"], True, ["adversarial"])
    assert (fn.check, fn.calls, fn.complexity, fn.since) == (
        "preserve", [2, 4], 2, "1.3")
    assert registry(get_persuasion_functions())[fn.__name__] is fn
