"""Bijection learning (obfuscation/bijection.py, spec 1.5, Huang 2410.01294):
one pure obfuscation technique that teaches a per-unit random letter cipher in
context, then shows the encoded prompt. The frame wording is not reproduced
here; the tests assert on shape, the slot, the info, the floor, the spec tags,
the pool membership, the applicability, the rules (both ways) and — the key
mechanism — that the encoded prompt decodes back with the shown mapping."""

import json
import random
from pathlib import Path

import pytest

import redact.jailbreak.obfuscation.bijection as BJ
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.obfuscation.bijection import (
    _SCHEMES,
    _canon,
    _decode,
    _encode,
    _mapping,
    get_bijection_functions,
    to_bijection_learning,
)
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import applies, load_spec
from redact.types import ALL_ENTRY_TYPES

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, cafe.',
    "a  b   c",
    "MixedCASE text with 123 and !?",
    "",
]
_FLOOR = 8
_DIRECTIVE = "bijection_learning"


def _pool() -> list[dict]:
    return read_pools(BJ._FILE)[_DIRECTIVE]


def test_there_is_one_technique():
    names = [fn.__name__ for fn in get_bijection_functions()]
    assert names == ["to_bijection_learning"]


def test_the_pool_is_one_the_technique_can_draw():
    check_pools([to_bijection_learning.variant_pool])


def test_the_pool_holds_its_floor_of_variants_each_named_once():
    names = [variant["name"] for variant in _pool()]
    assert len(names) == len(set(names)) >= _FLOOR


def test_there_are_at_least_twenty_teaching_sentences():
    assert len(BJ._SENTENCES) >= 20  # noqa: PLR2004


# --- the cipher is a bijection and decodes back ---------------------------------

@pytest.mark.parametrize("scheme", _SCHEMES)
def test_each_scheme_builds_a_bijection(scheme):
    """Every image is distinct (invertible), over all 26 letters."""
    for seed in range(8):
        mapping = _mapping(scheme, random.Random(seed))
        assert len(mapping) == 26  # noqa: PLR2004
        assert len(set(mapping.values())) == 26  # noqa: PLR2004


@pytest.mark.parametrize("scheme", _SCHEMES)
@pytest.mark.parametrize("text", TEXTS)
def test_the_encoded_text_decodes_back_with_the_shown_mapping(scheme, text):
    """The shown mapping recovers the request (normalised to lower case);
    non-letters pass through."""
    for seed in range(5):
        mapping = _mapping(scheme, random.Random(seed))
        assert _decode(_encode(text, mapping), mapping) == _canon(text)


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_non_letters_pass_through_unchanged(scheme):
    mapping = _mapping(scheme, random.Random(0))
    text = "mix 2 parts; then wait! (done)"
    decoded = _decode(_encode(text, mapping), mapping)
    assert decoded == text.lower()  # only letters changed, and only in case


def test_a_whole_technique_draw_decodes_to_the_prompt(monkeypatch):
    """Run the technique, recover the scheme from the info, rebuild the same
    mapping, and decode the ciphertext out of the rendered prompt."""
    captured = {}
    real_mapping = BJ._mapping

    def spy(scheme, rng):
        m = real_mapping(scheme, rng)
        captured["mapping"] = m
        return m

    monkeypatch.setattr(BJ, "_mapping", spy)
    prompt = "mix the two reagents and heat them"
    written, info = to_bijection_learning(prompt, rng=random.Random(7))
    scheme = info.split("scheme=")[1].split(";")[0]
    assert scheme in _SCHEMES
    ciphertext = _encode(prompt, captured["mapping"])
    assert ciphertext in written
    assert _decode(ciphertext, captured["mapping"]) == _canon(prompt)


# --- shape, info and applicability ---------------------------------------------

def test_the_info_names_the_scheme_and_the_variant():
    for seed in range(10):
        _, info = to_bijection_learning("hello world", rng=random.Random(seed))
        assert info.startswith("scheme=")
        assert ";variant=" in info
        assert info.split("scheme=")[1].split(";")[0] in _SCHEMES


def test_a_seed_gives_the_same_text():
    assert (to_bijection_learning(TEXTS[0], rng=random.Random(3))
            == to_bijection_learning(TEXTS[0], rng=random.Random(3)))


def test_it_applies_only_to_a_text_with_a_latin_letter():
    for entry_type in (e.value for e in ALL_ENTRY_TYPES):
        assert applies(to_bijection_learning, "hello there", entry_type)
        assert not applies(to_bijection_learning, "日本語のみ 12345", entry_type)


# --- the spec and pool membership ----------------------------------------------

def test_the_spec_tags_a_pure_attack_only_obfuscation_family():
    family = load_spec()["families"]["bijection"]
    assert (family["layer"], family["dimension"]) == ("obfuscation", "encoding")
    assert family["within_layer_order"] == 72  # noqa: PLR2004
    assert to_bijection_learning.layer == "obfuscation"
    assert to_bijection_learning.families == ["bijection"]
    assert to_bijection_learning.kinds == ["adversarial"]
    assert to_bijection_learning.complexity == 3  # noqa: PLR2004
    assert not to_bijection_learning.requires_llm
    assert to_bijection_learning.since == "1.5"


def test_the_family_is_in_the_attacks_and_not_the_eval_or_natural_pool():
    from redact.recipe import load_dataset_type

    adversarial = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    eval_pool = {fn.__name__ for fn in technique_pool(
        resolve(load_dataset_type("eval")["augmentations"], "adversarial"))}
    assert "to_bijection_learning" in adversarial
    assert "to_bijection_learning" not in natural
    assert "to_bijection_learning" not in eval_pool
    pinned = set(json.loads((Path(__file__).parents[2] / "fixtures" / "eval_pool.json")
                            .read_text(encoding="utf-8")))
    assert "to_bijection_learning" not in pinned


# --- the rules (a heavy full-text cipher) --------------------------------------

def _by_name() -> dict:
    return {fn.__name__: fn for fn in all_techniques()}


def test_it_refuses_what_the_heavy_encodings_refuse_both_ways():
    bij = to_bijection_learning
    by_name = _by_name()
    for name in ("to_ascii_art", "to_french", "to_base64", "to_morse",
                 "to_payload_split", "to_word_art", "to_fsh_random_short",
                 "to_dap_random_short", "to_fake_system_turn"):
        other = by_name[name]
        assert other not in compatible([bij], [other]), name
        assert bij not in compatible([other], [bij]), name


def test_it_combines_with_the_surface_and_wrapper_families_both_ways():
    """Like the heavy encodings, it does not refuse the word-level, surface or
    wrapper families."""
    bij = to_bijection_learning
    by_name = _by_name()
    for name in ("to_random_capitals_low", "to_json", "to_tokenbreak_split",
                 "to_innocuous_question", "to_fictional_world"):
        partner = by_name[name]
        assert partner in compatible([bij], [partner]), name
        assert bij in compatible([partner], [bij]), name
