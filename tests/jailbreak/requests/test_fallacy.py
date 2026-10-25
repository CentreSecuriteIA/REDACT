"""Fallacy failure (requests/fallacy.py, spec 1.5, Zhou 2407.00869): one
request-layer technique that reshapes the request as a call for a
deliberately fallacious answer. The frame wording is not reproduced here; the
tests assert on shape, the slot, the info, the floor, the spec tags, the pool
membership and the rules (both ways)."""

import json
import random
from pathlib import Path

import redact.jailbreak.directives as D
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.jailbreak.requests.fallacy import get_fallacy_functions, to_fallacy_failure
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import load_spec

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, café.',
]
_FLOOR = 8
_DIRECTIVE = "fallacy_failure"


def _pool() -> list[dict]:
    return read_pools(to_fallacy_failure.variant_pool.file)[_DIRECTIVE]


def test_there_is_one_technique():
    assert [fn.__name__ for fn in get_fallacy_functions()] == ["to_fallacy_failure"]


def test_the_pool_is_one_the_technique_can_draw():
    check_pools([to_fallacy_failure.variant_pool])


def test_the_pool_holds_its_floor_of_variants_each_named_once():
    names = [variant["name"] for variant in _pool()]
    assert len(names) == len(set(names)) >= _FLOOR


def test_every_variant_holds_the_prompt_once_and_is_named_in_the_info(monkeypatch):
    for variant in _pool():
        monkeypatch.setattr(D, "draw_variant", lambda variants, rng=None, v=variant: v)
        for text in TEXTS:
            written, info = to_fallacy_failure(text)
            assert variant["template"].count("{prompt}") == 1
            assert written.count(text) == 1
            assert info == f"directive={_DIRECTIVE};variant={variant['name']}"


def test_a_seed_draws_its_variant_as_rng_choice_does():
    for seed in range(10):
        ours, theirs = random.Random(seed), random.Random(seed)
        _, info = to_fallacy_failure(TEXTS[0], rng=ours)
        assert info.endswith(f";variant={theirs.choice(_pool())['name']}")
        assert ours.getstate() == theirs.getstate()


def test_a_seed_gives_the_same_text():
    assert (to_fallacy_failure(TEXTS[1], rng=random.Random(3))
            == to_fallacy_failure(TEXTS[1], rng=random.Random(3)))


def test_the_spec_tags_a_pure_attack_only_request_family():
    families = load_spec()["families"]
    family = families["fallacy"]
    assert (family["layer"], family["dimension"]) == ("requests", "request_shape")
    assert family["cross_incompatible_with"] == []
    assert to_fallacy_failure.layer == "requests"
    assert to_fallacy_failure.families == ["fallacy"]
    assert to_fallacy_failure.kinds == ["adversarial"]
    assert to_fallacy_failure.complexity == 2  # noqa: PLR2004
    assert not to_fallacy_failure.requires_llm
    assert not getattr(to_fallacy_failure, "needs_latin", False)
    assert to_fallacy_failure.since == "1.5"


def test_the_family_is_in_the_attacks_and_not_the_eval_or_natural_pool():
    """A new family: the default attacks draw it, the natural kind does not
    (adversarial only), and it is not in the eval pool (eval.json does not
    allow the fallacy family)."""
    from redact.recipe import load_dataset_type

    adversarial = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    eval_pool = {fn.__name__ for fn in technique_pool(
        resolve(load_dataset_type("eval")["augmentations"], "adversarial"))}
    assert "to_fallacy_failure" in adversarial
    assert "to_fallacy_failure" not in natural
    assert "to_fallacy_failure" not in eval_pool
    pinned = set(json.loads((Path(__file__).parents[2] / "fixtures" / "eval_pool.json")
                            .read_text(encoding="utf-8")))
    assert "to_fallacy_failure" not in pinned


def _by_name() -> dict:
    return {fn.__name__: fn for fn in all_techniques()}


def test_one_request_per_combination_keeps_it_from_other_requests():
    """The layer cap (one request): fallacy refuses another request-layer
    technique, in either pick order, and nothing else keeps them apart."""
    by_name = _by_name()
    other_request = by_name["to_innocuous_question"]
    assert other_request not in compatible([to_fallacy_failure], [other_request])
    assert to_fallacy_failure not in compatible([other_request], [to_fallacy_failure])


def test_it_combines_with_an_obfuscation_and_a_hacking_frame_both_ways():
    """No cross rule beyond the cap: a fallacy frame sits with an obfuscation
    and with a hacking frame, whichever is picked first."""
    by_name = _by_name()
    for name in ("to_base64", "to_fictional_world"):
        partner = by_name[name]
        assert partner in compatible([to_fallacy_failure], [partner])
        assert to_fallacy_failure in compatible([partner], [to_fallacy_failure])
