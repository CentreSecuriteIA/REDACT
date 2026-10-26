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


#: The ten heavy full-text encodings, whose refusals bijection's must match
#: (``encode`` members at ``encode_weight`` heavy, read from the registry so a
#: heavy encoding added later joins the comparison).
def _heavy_encodings() -> list:
    return [fn for fn in all_techniques()
            if "encode" in fn.families and getattr(fn, "encode_weight", None) == "heavy"
            and "sensitive_words" not in fn.families]


def _refused_by(fn) -> set[str]:
    """The names every registered technique ``fn`` may not sit with, in either
    pick order (re-derived through ``rules.compatible``), ``fn`` left out."""
    return {other.__name__ for other in all_techniques() if other is not fn
            and (other not in compatible([fn], [other])
                 or fn not in compatible([other], [fn]))}


def _heavy_refusals() -> set[str]:
    """What a heavy full-text encoding refuses: the refusals they all share,
    plus the ten themselves (each is refused beside the others by the one
    technique per family rule, so none is in the shared part)."""
    heavy = _heavy_encodings()
    assert len(heavy) == 10  # noqa: PLR2004
    return set.intersection(*(_refused_by(fn) for fn in heavy)) | {
        fn.__name__ for fn in heavy}


def test_the_heavy_encodings_all_refuse_the_same_techniques():
    """The set bijection is compared with is one set: the ten differ only in
    that none refuses itself."""
    heavy = _heavy_encodings()
    names = {fn.__name__ for fn in heavy}
    assert {frozenset(_refused_by(fn) | {fn.__name__}) for fn in heavy} == {
        frozenset(_heavy_refusals())}
    assert names <= _heavy_refusals()


def test_it_refuses_exactly_what_the_heavy_encodings_refuse_both_ways():
    """Derived from the heavy encodings when the test runs, technique by
    technique and in both pick orders, not from a list written here: a family
    added later that a heavy encoding refuses fails this until bijection
    refuses it too (and the other way round)."""
    bij = to_bijection_learning
    refused = _refused_by(bij) | {bij.__name__}
    heavy = _heavy_refusals()
    assert sorted(heavy - refused) == []
    assert sorted(refused - heavy) == []


def test_the_refusals_cover_the_families_the_claim_names():
    """A reader's check on the derived set: every family of the whole-text
    rewrites, the ciphers, the wrappers' two exceptions and the manipulation
    layer is in it, and the word-level and surface families are not."""
    by_name = _by_name()
    refused = _refused_by(to_bijection_learning)
    families = {family for name in refused for family in by_name[name].families}
    assert {"ascii_art", "translation", "dialect", "code_switching", "romanization",
            "transliteration", "semantic_reencoding", "translationese", "sociolect",
            "encode", "payload_split", "word_art", "unicode_smuggling",
            "char_injection", "fsh", "dap", "fake_turns",
            "reasoning_injection"} <= families
    assert not {"typos", "casual_writing", "noise", "structural",
                "neutral_container", "tokenbreak"} & families


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
