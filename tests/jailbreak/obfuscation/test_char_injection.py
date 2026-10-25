"""Character injection: the bidi form renders back, the deletions and zalgo
strip back, upside-down inverts, and the family's rules bind both ways."""

import random
import unicodedata

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation import char_injection as C
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.rules import compatible

_PROMPTS = [
    "How do plants turn sunlight into food?",
    "Write a short note about the garden.",
    'Line one\nSay "hi" to {braces} and a back\\slash.',
    "café — naïve — ünïcödé — 日本語 — emoji 🧪 end",
    "ASCII only 123 !? @#",
    "x",
]
_DENSITIES = ["low", "medium", "high"]
_BY_NAME = {fn.__name__: fn for fn in all_techniques()}


def _joins(first: str, second: str) -> bool:
    return _BY_NAME[second] in compatible([_BY_NAME[first]], [_BY_NAME[second]])


# --- the control characters are the real bidi override codes -------------------

def test_the_override_codes_are_the_unicode_bidi_override_and_pop():
    assert unicodedata.bidirectional(C._RLO) == "RLO"
    assert unicodedata.bidirectional(C._PDF) == "PDF"
    assert unicodedata.bidirectional(C._LRO) == "LRO"


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_bidi_override_renders_back_to_the_prompt(prompt):
    text, info = C.to_bidi_override(prompt)
    assert info == ""
    assert C._RLO in text and C._PDF in text
    assert C.render_override(text) == prompt


def test_render_override_reverses_only_the_overridden_run():
    # Text before an override stays in order; the overridden run reverses.
    assert C.render_override("ab" + C._RLO + "cd" + C._PDF + "ef") == "abdcef"


# --- deletions strip back ------------------------------------------------------

@pytest.mark.parametrize("level", _DENSITIES)
@pytest.mark.parametrize("prompt", _PROMPTS)
def test_deletion_chars_strip_back_to_the_prompt(level, prompt):
    fn = _BY_NAME[f"to_deletion_chars_{level}"]
    for seed in range(5):
        text, info = fn(prompt, rng=random.Random(seed))
        assert info == ""
        assert C.strip_deletions(text) == prompt


def test_higher_deletion_density_injects_more():
    prompt = "a fairly long prompt to see the density differ across the levels"
    counts = []
    for level in _DENSITIES:
        fn = _BY_NAME[f"to_deletion_chars_{level}"]
        text, _ = fn(prompt, rng=random.Random(0))
        counts.append(text.count(C._BACKSPACE))
    assert counts[0] < counts[2]


# --- zalgo strips back ---------------------------------------------------------

@pytest.mark.parametrize("level", _DENSITIES)
@pytest.mark.parametrize("prompt", _PROMPTS)
def test_zalgo_strips_back_to_the_prompt(level, prompt):
    fn = _BY_NAME[f"to_zalgo_{level}"]
    for seed in range(5):
        text, info = fn(prompt, rng=random.Random(seed))
        assert info == ""
        assert C.strip_zalgo(text) == prompt
        # every added character is a combining mark
        added = [c for c in text if c not in prompt]
        assert all(unicodedata.combining(c) for c in added)


def test_higher_zalgo_density_adds_more_marks():
    prompt = "a fairly long prompt"
    counts = []
    for level in _DENSITIES:
        fn = _BY_NAME[f"to_zalgo_{level}"]
        text, _ = fn(prompt, rng=random.Random(0))
        counts.append(len(text) - len(prompt))
    assert counts[0] < counts[1] < counts[2]


# --- upside-down inverts over the mapped characters ----------------------------

def test_the_flip_map_is_injective():
    values = list(C._FLIP.values())
    assert len(values) == len(set(values))


def test_upside_down_inverts_and_reverses():
    inverse = {v: k for k, v in C._FLIP.items()}
    source = "Hello World 123 ?!()"
    text, info = C.to_upside_down(source)
    assert info == ""
    assert text != source
    assert "".join(inverse.get(c, c) for c in reversed(text)) == source


def test_every_technique_is_deterministic_for_a_seed():
    for fn in C.get_char_injection_functions():
        assert fn("some prompt", rng=random.Random(9)) == \
            fn("some prompt", rng=random.Random(9))


# --- the rules bind both ways --------------------------------------------------

_CHAR = ["to_bidi_override", "to_upside_down", "to_deletion_chars_low",
         "to_deletion_chars_high", "to_zalgo_low", "to_zalgo_high"]
_REFUSED = [
    "to_base64", "to_rot13", "to_morse", "to_french", "to_thai",
    "to_cyrillic_transliteration", "to_code_switch_spanish", "to_romanize_hindi",
    "to_dialect_mexican_spanish", "to_rewrite_with_typos_low", "to_lowercase",
    "to_random_capitals_low", "to_sensitive_words_split", "to_tokenbreak_split",
    "to_word_art", "to_pseudonym_substitution", "to_adversarial_suffix_punctuation",
    "to_payload_split", "to_json", "to_code_comment", "to_ascii_art",
    "to_variation_selector_smuggling",
]
# char_injection refuses every obfuscation, but keeps the other layers, the
# manipulation layer among them (the reordered question sits among clean
# examples): it is not in the manipulation block.
_JOINED = ["to_fictional_world", "to_summarize", "to_fsh_random_short",
           "to_dap_selected_long", "to_fake_history", "to_dan_persona_template"]


@pytest.mark.parametrize("char", _CHAR)
@pytest.mark.parametrize("other", _REFUSED)
def test_char_injection_refuses_every_other_obfuscation(char, other):
    assert not _joins(char, other)
    assert not _joins(other, char)


@pytest.mark.parametrize("char", _CHAR)
@pytest.mark.parametrize("other", _JOINED)
def test_char_injection_joins_the_hacking_request_and_manipulation_layers(char, other):
    assert _joins(char, other)
    assert _joins(other, char)


@pytest.mark.parametrize("first", ["to_bidi_override", "to_zalgo_low"])
@pytest.mark.parametrize("second", ["to_upside_down", "to_deletion_chars_high"])
def test_a_combination_holds_one_char_injection(first, second):
    assert not _joins(first, second)
