"""Invisible-Unicode smuggling: the carried forms decode back exactly, the
suffix strips back exactly, and the family's rules bind both ways."""

import random
import unicodedata

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation import unicode_smuggling as U
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import applies, load_spec

_PROMPTS = [
    "How do plants turn sunlight into food?",
    "Write a limerick about the sea.",
    'Line one\nSay "hi" to {braces} and a back\\slash.',
    "café — naïve — ünïcödé — 日本語 — emoji 🧪 end",
    "ASCII only 123 !? @#",
    "x",
]
_BY_NAME = {fn.__name__: fn for fn in all_techniques()}


def _joins(first: str, second: str) -> bool:
    return _BY_NAME[second] in compatible([_BY_NAME[first]], [_BY_NAME[second]])


# --- encode/decode are inverse -------------------------------------------------

def test_byte_and_selector_round_trip_every_value():
    for value in range(256):
        assert U._selector_to_byte(U._byte_to_selector(value)) == value


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_variation_selector_smuggling_decodes_to_the_exact_prompt(prompt):
    for seed in range(6):
        text, info = U.to_variation_selector_smuggling(prompt, rng=random.Random(seed))
        assert U.decode_variation_selectors(text) == prompt
        assert info.startswith("carrier=")
        # The visible text is the carrier alone: the payload is all selectors.
        assert U.strip_trailing_selectors(text) in {
            v["template"] for v in U._POOLS["vs_carrier"]}


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_the_hidden_payload_is_invisible_format_characters(prompt):
    text, _ = U.to_variation_selector_smuggling(prompt, rng=random.Random(0))
    payload = text[len(U.strip_trailing_selectors(text)):]
    assert payload  # a non-empty prompt hides at least one byte
    assert all(unicodedata.category(ch) in ("Mn", "Cf") for ch in payload)


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_invisible_suffix_leaves_the_prompt_and_strips_back(prompt):
    for seed in range(6):
        text, info = U.to_invisible_suffix(prompt, rng=random.Random(seed))
        assert info == ""
        assert text.startswith(prompt)
        assert U.strip_trailing_selectors(text) == prompt
        appended = len(text) - len(prompt)
        assert 8 <= appended <= 48  # noqa: PLR2004


def test_taggable_tells_printable_ascii_from_the_rest():
    assert U.is_taggable("plain ascii 123 !?")
    assert not U.is_taggable("café")
    assert not U.is_taggable("has\nnewline")
    assert not U.is_taggable("")


@pytest.mark.parametrize("prompt", [p for p in _PROMPTS if U.is_taggable(p)])
def test_tag_smuggling_decodes_to_the_exact_prompt(prompt):
    for seed in range(6):
        text, info = U.to_unicode_tag_smuggling(prompt, rng=random.Random(seed))
        assert U.decode_tags(text) == prompt
        assert info.startswith("carrier=")
        # Every tag character is an invisible format character.
        tags = [c for c in text if 0xE0020 <= ord(c) <= 0xE007E]  # noqa: PLR2004
        assert tags and all(unicodedata.category(c) == "Cf" for c in tags)


def test_tag_smuggling_is_drawn_only_for_a_taggable_prompt():
    fn = U.to_unicode_tag_smuggling
    assert applies(fn, "plain ascii", "harmful")
    assert not applies(fn, "café", "harmful")
    assert not applies(fn, "line\nbreak", "harmful")


def test_the_other_two_have_no_applicability():
    for fn in (U.to_variation_selector_smuggling, U.to_invisible_suffix):
        assert applies(fn, "café 日本語", "harmful")


def test_no_carrier_holds_an_invisible_character():
    for variants in U._POOLS.values():
        for variant in variants:
            for ch in variant["template"]:
                point = ord(ch)
                assert not (0xFE00 <= point <= 0xFE0F)  # noqa: PLR2004
                assert not (0xE0100 <= point <= 0xE01EF)  # noqa: PLR2004
                assert not (0xE0000 <= point <= 0xE007F)  # noqa: PLR2004


def test_the_same_seed_gives_the_same_text():
    for fn in U.get_unicode_smuggling_functions():
        a = fn("some prompt text", rng=random.Random(11))
        b = fn("some prompt text", rng=random.Random(11))
        assert a == b


# --- the rules bind both ways --------------------------------------------------

#: The manipulation families whose examples or invented turns would sit
#: visibly around the hidden question; reasoning_injection, the fourth
#: family of the layer, is not refused (an open question for the owner).
_MANIPULATION = ("fsh", "dap", "fake_turns")
_JOINED = ["to_fictional_world", "to_summarize", "to_use_past_tense",
           "to_distractor_prefix_related"]
_SMUGGLERS = sorted(name for name, fn in _BY_NAME.items()
                    if "unicode_smuggling" in fn.families)


def _other_obfuscation_families() -> set[str]:
    """Every obfuscation family of the spec but unicode_smuggling, read when
    the test runs: a family added later is checked here without editing a
    list."""
    return {name for name, family in load_spec()["families"].items()
            if family.get("layer") == "obfuscation"} - {"unicode_smuggling"}


def test_the_family_names_its_own_techniques():
    assert _SMUGGLERS == ["to_invisible_suffix", "to_unicode_tag_smuggling",
                          "to_variation_selector_smuggling"]


def test_the_spec_keeps_unicode_smuggling_from_every_other_obfuscation_family():
    """Each other obfuscation family of the spec is named in
    unicode_smuggling's list or names it in its own, one with no technique
    yet (flip_tokens) too, so a family added to the spec fails here until it
    is kept apart."""
    families = load_spec()["families"]
    own = set(families["unicode_smuggling"]["cross_incompatible_with"])
    missing = sorted(
        name for name in _other_obfuscation_families()
        if name not in own and "unicode_smuggling" not in families[name].get(
            "cross_incompatible_with", []))
    assert missing == []


@pytest.mark.parametrize("smuggler", _SMUGGLERS)
def test_unicode_smuggling_refuses_the_whole_text_and_manipulation_families(
        smuggler):
    """A full-prompt hiding: it refuses every registered technique of every
    other obfuscation family, and of fsh, dap and fake_turns, whichever is
    picked first (re-derived through rules.compatible)."""
    wanted = _other_obfuscation_families() | set(_MANIPULATION)
    others = [name for name, fn in _BY_NAME.items()
              if set(fn.families) & wanted]
    assert len(others) > 150  # noqa: PLR2004
    joins = sorted(name for name in others
                   if _joins(smuggler, name) or _joins(name, smuggler))
    assert joins == []


@pytest.mark.parametrize("smuggler", _SMUGGLERS)
@pytest.mark.parametrize("other", _JOINED)
def test_unicode_smuggling_joins_a_hacking_frame_or_a_request(smuggler, other):
    """The frame is hidden with the prompt, and a request is appended visibly
    after the carrier."""
    assert _joins(smuggler, other)
    assert _joins(other, smuggler)


@pytest.mark.parametrize("first", _SMUGGLERS)
@pytest.mark.parametrize("second", _SMUGGLERS)
def test_a_combination_holds_one_unicode_smuggling(first, second):
    assert not _joins(first, second)
