"""The spec's rules, over the registered techniques: they bind both ways."""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.dataset.dealing import Dealer
from redact.jailbreak.hacking import get_all_hacking_functions
from redact.jailbreak.manipulation import get_all_manipulation_functions
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.requests import get_all_request_functions
from redact.jailbreak.rules import compatible, request_obfuscation_candidates
from redact.jailbreak.sampling import (
    sample_combination,
    sample_exact_combination,
)

POOL = (get_all_obfuscation_functions() + get_all_hacking_functions()
        + get_all_manipulation_functions() + get_all_request_functions())
BY_NAME = {fn.__name__: fn for fn in POOL}


def _joins(selected: str, candidate: str) -> bool:
    return BY_NAME[candidate] in compatible([BY_NAME[selected]], [BY_NAME[candidate]])


@pytest.mark.parametrize(("first", "then"), [
    ("to_fictional_world", "to_ascii_art"),
    ("to_ascii_art", "to_fictional_world"),
    ("to_fsh_random_short", "to_ascii_art"),
    ("to_ascii_art", "to_fsh_random_short"),
])
def test_a_cross_incompatibility_refuses_either_order(first, then):
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_psychopath_persona", "to_impersonate_good_person"),
    ("to_impersonate_good_person", "to_psychopath_persona"),
    ("to_invented_persona", "to_impersonate_good_person"),
    ("to_impersonate_good_person", "to_invented_persona"),
    ("to_avi", "to_impersonate_good_person"),
    ("to_impersonate_good_person", "to_avi"),
])
def test_a_persona_or_cognitive_scenario_refuses_an_impersonation_either_way(
        first, then):
    """Two identities in one prompt: the persona or scenario's, and the
    asker's claimed one."""
    assert not _joins(first, then)


@pytest.mark.parametrize("framing", ["to_fictional_world", "to_noble_goal"])
def test_a_framing_still_joins_an_impersonation(framing):
    assert _joins(framing, "to_impersonate_good_person")
    assert _joins("to_impersonate_good_person", framing)


def test_continuation_and_indirect_are_kept_apart_by_the_requests_cap_alone():
    """Both are request families, one a combination at most, so the spec
    names no rule of its own between them."""
    from redact.jailbreak.technique import load_spec

    families = load_spec()["families"]
    assert "indirect" not in families["continuation"]["cross_incompatible_with"]
    assert "continuation" not in families["indirect"]["cross_incompatible_with"]
    assert not _joins("to_continue_passage", "to_summarize")
    assert not _joins("to_summarize", "to_continue_passage")


def test_no_family_carries_a_pick_cap_of_its_own():
    """A combination takes one technique per family; the layers' caps are
    ``layer_max_picks``."""
    from redact.jailbreak.technique import load_spec

    assert not [name for name, family in load_spec()["families"].items()
                if "max_picks" in family]


@pytest.mark.parametrize(("first", "then"), [
    ("to_adversarial_suffix_emoji", "to_ascii_art"),
    ("to_ascii_art", "to_adversarial_suffix_emoji"),
    ("to_adversarial_suffix_unicode", "to_ascii_art"),
    ("to_ascii_art", "to_adversarial_suffix_unicode"),
    ("to_braille", "to_ascii_art"),
    ("to_ascii_art", "to_braille"),
    ("to_french", "to_ascii_art"),
    ("to_ascii_art", "to_swahili"),
])
def test_ascii_art_refuses_what_leaves_a_character_no_font_draws(first, then):
    """The emoji and unicode suffixes and braille, by the technique, and
    every translation: no packaged font has their characters."""
    assert not _joins(first, then)


def test_tokenbreak_split_and_ascii_art_are_never_planned_together():
    """to_ascii_art rejects what to_tokenbreak_split gives, so the spec keeps
    them apart, in either order and in either mode."""
    from redact.jailbreak.plan import assign_combination

    pair = {"to_tokenbreak_split", "to_ascii_art"}
    assert not _joins("to_tokenbreak_split", "to_ascii_art")
    assert not _joins("to_ascii_art", "to_tokenbreak_split")
    pool = [BY_NAME[name] for name in sorted(pair)]
    for seed in range(50):
        for mode in ({"exact_techniques": 2},
                     {"max_complexity": 9, "max_obfuscations": 2}):
            names = assign_combination(f"unit{seed}", pool, seed=seed, **mode)
            assert set(names) != pair, (seed, mode)


@pytest.mark.parametrize("technique", [
    "to_adversarial_suffix_punctuation", "to_adversarial_suffix_fragments",
    "to_base64", "to_morse", "to_rot47"])
def test_ascii_art_keeps_the_rest_of_their_families(technique):
    assert _joins(technique, "to_ascii_art") and _joins("to_ascii_art", technique)


@pytest.mark.parametrize("technique", [
    "to_adversarial_suffix_punctuation", "to_adversarial_suffix_fragments",
    "to_base64", "to_morse", "to_rot47"])
def test_ascii_art_draws_a_font_for_the_rest_of_their_families(technique):
    from redact.jailbreak.obfuscation.ascii_art import to_ascii_art

    for seed in range(20):
        rng = random.Random(seed)
        fn = BY_NAME[technique]
        text, _ = (fn("plain words about the weather", rng=rng)
                   if "rng" in fn.__code__.co_varnames else fn("plain words"))
        _art, info = to_ascii_art(text, rng=rng)
        assert info.startswith("font="), (technique, seed, info)


def test_a_technique_names_only_families_it_is_incompatible_with():
    from redact.jailbreak.technique import load_spec

    spec = load_spec()
    named = {family for name, entry in spec["techniques"].items()
             if not name.startswith("_")
             for family in entry.get("cross_incompatible_with", [])}
    assert named and named <= set(spec["families"])


@pytest.mark.parametrize(("first", "then"), [
    ("to_french", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_french"),
    ("to_base64", "to_dap_random_long"),
    ("to_dap_random_long", "to_base64"),
])
def test_manipulation_blocking_refuses_either_order(first, then):
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_sensitive_words_encode_base64", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_sensitive_words_encode_base64"),
    ("to_rot13", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_rot13"),
])
def test_manipulation_admits_light_and_word_level_encoding(first, then):
    assert _joins(first, then)


def test_every_pair_is_refused_or_admitted_both_ways():
    joins = {fn.__name__: {g.__name__ for g in compatible([fn], POOL)} for fn in POOL}
    one_way = [(a, b) for a in joins for b in joins[a] if a not in joins[b]]
    assert one_way == []


def test_an_exact_draw_never_holds_a_forbidden_pair():
    for seed in range(200):
        fn = sample_exact_combination(random.Random(seed), POOL, 3)
        names = [t.__name__ for t in fn.techniques]
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                assert _joins(name, other), (seed, names)


def test_a_budget_draw_never_holds_a_forbidden_pair_or_breaks_a_layer_cap():
    from collections import Counter

    from redact.jailbreak.technique import load_spec

    caps = load_spec()["layer_max_picks"]
    for seed in range(600):
        fn = sample_combination(
            random.Random(seed), POOL, max_complexity=9, max_obfuscations=3,
            allow_request_obfuscation=True,
            dealer=Dealer(key=(seed,)) if seed % 2 else None,
            sampling_probs=dict.fromkeys(
                ("hacking", "manipulation", "requests", "request_obfuscation"), 0.9))
        names = [t.__name__ for t in fn.techniques]
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                assert _joins(name, other), (seed, names)
        per_layer = Counter(t.layer for t in fn.techniques)
        over = [layer for layer, cap in caps.items() if per_layer[layer] > cap]
        assert over == [], (seed, names)


@pytest.mark.parametrize("max_complexity", [0, 1, 2, 3])
def test_a_budget_draw_stays_within_its_complexity(max_complexity):
    for seed in range(300):
        fn = sample_combination(
            random.Random(seed), POOL, max_complexity=max_complexity,
            max_obfuscations=3, allow_request_obfuscation=True,
            dealer=Dealer(key=(seed,)) if seed % 2 else None,
            sampling_probs=dict.fromkeys(
                ("hacking", "manipulation", "requests", "request_obfuscation"), 0.9))
        names = [t.__name__ for t in fn.techniques]
        assert sum(t.complexity for t in fn.techniques) <= max_complexity, (seed, names)


def test_request_obfuscation_is_light_encoding_or_translation():
    names = {fn.__name__ for fn in request_obfuscation_candidates(POOL)}
    assert {"to_rot13", "to_french"} <= names
    assert not names & {"to_base64", "to_ascii_art", "to_summarize"}
