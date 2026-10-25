"""Policy framing (hacking/policy_framing.py, spec 1.5): to_policy_puppetry
(Policy Puppetry, HiddenLayer 2025) and to_skeleton_key (Skeleton Key,
Microsoft 2024, single-turn). Two pure hacking-layer template techniques. The
frame wording is not reproduced here; the tests assert on shape, the slot,
the info, the floor, the spec tags, the pool membership and the rules (both
ways)."""

import json
import random
from pathlib import Path

import pytest

import redact.jailbreak.directives as D
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.hacking.policy_framing import (
    get_policy_framing_functions,
    to_policy_puppetry,
    to_skeleton_key,
)
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import load_spec

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, café.',
]
_FLOOR = 8
FUNCTIONS = get_policy_framing_functions()


def _pool(fn) -> list[dict]:
    return read_pools(fn.variant_pool.file)[fn.variant_pool.directive]


def test_there_are_two_techniques():
    assert [fn.__name__ for fn in FUNCTIONS] == ["to_policy_puppetry", "to_skeleton_key"]


def test_the_pools_are_ones_their_techniques_can_draw():
    check_pools(fn.variant_pool for fn in FUNCTIONS)


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_each_pool_holds_its_floor_of_variants_each_named_once(fn):
    names = [variant["name"] for variant in _pool(fn)]
    assert len(names) == len(set(names)) >= _FLOOR


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_every_variant_holds_the_prompt_once_and_is_named_in_the_info(monkeypatch, fn):
    for variant in _pool(fn):
        monkeypatch.setattr(D, "draw_variant", lambda variants, rng=None, v=variant: v)
        for text in TEXTS:
            written, info = fn(text)
            assert variant["template"].count("{prompt}") == 1
            assert written.count(text) == 1
            assert info == (f"directive={fn.variant_pool.directive}"
                            f";variant={variant['name']}")


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_a_seed_draws_its_variant_as_rng_choice_does(fn):
    for seed in range(10):
        ours, theirs = random.Random(seed), random.Random(seed)
        _, info = fn(TEXTS[0], rng=ours)
        assert info.endswith(f";variant={theirs.choice(_pool(fn))['name']}")
        assert ours.getstate() == theirs.getstate()


def test_the_spec_tags_a_pure_attack_only_hacking_family():
    family = load_spec()["families"]["policy_framing"]
    assert (family["layer"], family["dimension"]) == ("hacking", "role_framing")
    assert set(family["cross_incompatible_with"]) == {
        "fsh", "dap", "fake_turns", "impersonation"}
    for fn in FUNCTIONS:
        assert fn.layer == "hacking"
        assert fn.families == ["policy_framing"]
        assert fn.kinds == ["adversarial"]
        assert not fn.requires_llm and not getattr(fn, "needs_latin", False)
        assert fn.since == "1.5"


def test_the_family_is_in_the_attacks_and_not_the_eval_or_natural_pool():
    from redact.recipe import load_dataset_type

    names = {fn.__name__ for fn in FUNCTIONS}
    adversarial = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    eval_pool = {fn.__name__ for fn in technique_pool(
        resolve(load_dataset_type("eval")["augmentations"], "adversarial"))}
    assert names <= adversarial
    assert not names & natural
    assert not names & eval_pool
    pinned = set(json.loads((Path(__file__).parents[2] / "fixtures" / "eval_pool.json")
                            .read_text(encoding="utf-8")))
    assert not names & pinned


def _by_name() -> dict:
    return {fn.__name__: fn for fn in all_techniques()}


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_it_refuses_the_manipulation_and_impersonation_frames_both_ways(fn):
    """One social-engineering or identity frame per prompt: the family refuses
    fsh, dap, fake_turns and impersonation, in either pick order."""
    by_name = _by_name()
    for name in ("to_fsh_random_short", "to_dap_random_short",
                 "to_fake_system_turn", "to_impersonate_good_person"):
        partner = by_name[name]
        assert partner not in compatible([fn], [partner]), name
        assert fn not in compatible([partner], [fn]), name


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_one_hacking_technique_per_combination(fn):
    """The layer cap keeps it from another hacking frame, and from the other
    policy-framing technique (same family)."""
    by_name = _by_name()
    for name in ("to_fictional_world", "to_dan_persona_template"):
        other = by_name[name]
        assert other not in compatible([fn], [other]), name
    sibling = to_skeleton_key if fn is to_policy_puppetry else to_policy_puppetry
    assert sibling not in compatible([fn], [sibling])


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_it_combines_with_an_obfuscation_and_a_request_both_ways(fn):
    """No obfuscation or request rule: a policy frame tolerates an obfuscated,
    request-framed payload inside it."""
    by_name = _by_name()
    for name in ("to_base64", "to_innocuous_question"):
        partner = by_name[name]
        assert partner in compatible([fn], [partner]), name
        assert fn in compatible([partner], [fn]), name
