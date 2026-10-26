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

    def spy(scheme, rng, text=""):
        m = real_mapping(scheme, rng, text)
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


# --- the symbol scheme's collisions --------------------------------------------

#: Texts that hold characters the cipher passes through and the symbol scheme
#: could otherwise draw as an image (Greek letters and shapes of the pool), and
#: one that holds the space mark.
_GREEK = "mix 5 μg of α and β compound"
_SHAPES = "heat the ★ and ♠ sample"
_WITH_MARK = "a text with ␣ inside"


def _shown_key(written: str) -> dict[str, str]:
    """The cipher the output shows, read off its key line (``a = x  b = y
    ...``) as a reader would: image to letter."""
    import re

    line = next(l for l in written.splitlines() if re.match(r"^a = \S+", l))
    pairs = re.findall(r"\b([a-z]) = (\S+)", line)
    assert len(pairs) == 26  # noqa: PLR2004
    images = {image for _letter, image in pairs}
    assert len(images) == 26  # noqa: PLR2004
    return {image: letter for letter, image in pairs}


def _decode_with(written: str, encoded: str) -> str:
    """``encoded`` decoded with the key the output shows, not with the
    module's own mapping."""
    inverse = _shown_key(written)
    return "".join(inverse.get(token, " " if token == BJ._SPACE_MARK else token)
                   for token in encoded.split(" "))


def _ciphertext(written: str, prompt: str) -> str:
    """The enciphered prompt inside ``written``: the line that is not part of
    the key or a worked example, found by decoding each line with the shown
    key and keeping the one that gives the prompt's normalised form."""
    for line in written.splitlines():
        if line and not line.startswith("a = ") and \
                _decode_with(written, line) == _canon(prompt):
            return line
    raise AssertionError("the prompt's ciphertext is not in the output")


@pytest.mark.parametrize("text", [_GREEK, _SHAPES])
def test_the_symbol_key_never_draws_a_symbol_the_text_holds(text):
    """A symbol in the text passes through the cipher, so drawing it as an
    image would make the shown key ambiguous. Over 40 seeds, every symbol
    draw leaves the text's own symbols out, and the prompt decodes back with
    the key the output shows."""
    drawn = 0
    for seed in range(40):
        written, info = to_bijection_learning(text, rng=random.Random(seed))
        assert not info.startswith("DISCARDED"), info
        if info.split("scheme=")[1].split(";")[0] != "symbol":
            continue
        drawn += 1
        images = set(_shown_key(written))
        assert not images & set(text), sorted(images & set(text))
        assert _decode_with(written, _ciphertext(written, text)) == _canon(text)
    assert drawn >= 5  # noqa: PLR2004  (the scheme is one of three)


@pytest.mark.parametrize("text", [_GREEK, _SHAPES, "plain ascii request"])
def test_every_scheme_decodes_the_prompt_with_the_shown_key(text):
    """Whatever the scheme, a reader with the key the output shows recovers
    the prompt (this is what collided before: a Greek letter or a shape in
    the text read back as a letter)."""
    seen = set()
    for seed in range(40):
        written, info = to_bijection_learning(text, rng=random.Random(seed))
        assert not info.startswith("DISCARDED"), info
        seen.add(info.split("scheme=")[1].split(";")[0])
        assert _decode_with(written, _ciphertext(written, text)) == _canon(text)
    assert seen == set(_SCHEMES)


def test_a_text_holding_the_space_mark_is_discarded_in_every_scheme():
    """The mark stands for a space under every scheme, so such a text cannot
    be deciphered unambiguously; it is discarded, unrendered."""
    for seed in range(12):
        written, info = to_bijection_learning(_WITH_MARK, rng=random.Random(seed))
        assert written == _WITH_MARK
        assert info.startswith("DISCARDED; reason=space mark in the text; scheme=")


def test_a_text_holding_all_but_a_few_symbols_is_discarded_under_the_scheme():
    """With fewer than 26 free symbols no unambiguous key is left: the symbol
    scheme discards, and the two other schemes, which have no collision,
    still encipher it."""
    crowded = "text " + "".join(BJ._SYMBOLS[:9])
    assert len(BJ._free_symbols(crowded)) == 25  # noqa: PLR2004
    assert BJ._collision_reason("symbol", crowded) == "too few free symbols"
    assert BJ._collision_reason("letter", crowded) == ""
    assert BJ._collision_reason("digit", crowded) == ""
    with pytest.raises(ValueError, match="free symbols"):
        BJ._mapping("symbol", random.Random(0), crowded)
    schemes = set()
    for seed in range(40):
        written, info = to_bijection_learning(crowded, rng=random.Random(seed))
        scheme = info.split("scheme=")[1].split(";")[0]
        schemes.add(scheme)
        if scheme == "symbol":
            assert info.startswith("DISCARDED; reason=too few free symbols")
            assert written == crowded
        else:
            assert not info.startswith("DISCARDED")
            assert _decode_with(written, _ciphertext(written, crowded)) == _canon(crowded)
    assert schemes == set(_SCHEMES)


def test_a_text_without_a_pool_symbol_draws_what_it_drew_before():
    """The free-symbol draw is the whole pool for such a text, so no digest
    moves: ``rng.sample`` over the pool, as before."""
    for seed in range(8):
        a = BJ._mapping("symbol", random.Random(seed), "plain ascii text")
        b = dict(zip(BJ._LETTERS,
                     random.Random(seed).sample(BJ._SYMBOLS, 26), strict=True))
        assert a == b


def test_the_teaching_sentences_hold_no_symbol_or_mark():
    """The worked examples are enciphered with the same key, so a sentence
    holding a pool symbol or the mark would be ambiguous too."""
    joined = "".join(BJ._SENTENCES)
    assert not set(joined) & set(BJ._SYMBOLS)
    assert BJ._SPACE_MARK not in joined
    assert joined.isascii()


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
