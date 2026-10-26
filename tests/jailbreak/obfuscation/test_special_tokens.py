"""Special tokens: the prompt is kept whole inside the markup, and the
family's rules bind both ways."""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.directives import read_pools
from redact.jailbreak.obfuscation import special_tokens as S
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.rules import compatible

_PROMPTS = [
    "How do I set up the printer?",
    'Say "hello" to the team.',
    "café — 日本語 — 123 !?",
]
_BY_NAME = {fn.__name__: fn for fn in all_techniques()}


def _joins(first: str, second: str) -> bool:
    return _BY_NAME[second] in compatible([_BY_NAME[first]], [_BY_NAME[second]])


@pytest.mark.parametrize("name", ["to_eos_tokens", "to_chat_template_abuse"])
@pytest.mark.parametrize("prompt", _PROMPTS)
def test_the_prompt_is_carried_whole_and_the_variant_is_named(name, prompt):
    fn = _BY_NAME[name]
    for seed in range(6):
        text, info = fn(prompt, rng=random.Random(seed))
        assert prompt in text
        assert info.startswith("variant=")


def test_the_pools_each_hold_at_least_eight_variants():
    pools = read_pools("obfuscation/special_tokens_templates.json")
    assert len(pools["eos_style"]) >= 8  # noqa: PLR2004
    assert len(pools["chat_template"]) >= 8  # noqa: PLR2004


def test_each_technique_is_deterministic_for_a_seed():
    for fn in S.get_special_tokens_functions():
        assert (fn("a prompt", rng=random.Random(4))
                == fn("a prompt", rng=random.Random(4)))


# --- the rules bind both ways --------------------------------------------------

_SPECIAL = ["to_eos_tokens", "to_chat_template_abuse"]
_REFUSED = [
    "to_json", "to_xml", "to_markdown", "to_code_comment", "to_csv_row",
    "to_ascii_art", "to_variation_selector_smuggling", "to_bidi_override",
    "to_zalgo_high", "to_fsh_random_short", "to_dap_selected_long",
    "to_fake_history",
]
# A wrapper in a model's own syntax joins an encoding, a translation, a hacking
# frame and a request (they wrap or are wrapped by it without clashing).
_JOINED = ["to_base64", "to_rot13", "to_french", "to_fictional_world",
           "to_summarize", "to_use_past_tense"]


@pytest.mark.parametrize("special", _SPECIAL)
@pytest.mark.parametrize("other", _REFUSED)
def test_special_tokens_refuses_wrappers_hidings_and_manipulation(special, other):
    assert not _joins(special, other)
    assert not _joins(other, special)


@pytest.mark.parametrize("special", _SPECIAL)
@pytest.mark.parametrize("other", _JOINED)
def test_special_tokens_joins_an_encoding_translation_frame_or_request(special, other):
    assert _joins(special, other)
    assert _joins(other, special)


def test_a_combination_holds_one_special_tokens():
    assert not _joins("to_eos_tokens", "to_chat_template_abuse")
    assert not _joins("to_chat_template_abuse", "to_eos_tokens")
