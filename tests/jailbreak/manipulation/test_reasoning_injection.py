"""Reasoning injection (manipulation/reasoning_injection.py, spec 1.5, H-CoT
Kuo 2502.12893): one pure manipulation-layer technique, to_fake_reasoning,
that puts a fabricated approving reasoning block before the prompt. The frame
wording is not reproduced here; the tests assert on shape, the slot, the
info, the floor, the spec tags, the pool membership and the rules (both
ways)."""

import json
import random
from pathlib import Path

import redact.jailbreak.manipulation.reasoning_injection as RI
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.manipulation.reasoning_injection import (
    get_reasoning_injection_functions,
    to_fake_reasoning,
)
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import load_spec

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, café.',
]
_FLOOR = 8
_DIRECTIVE = "fake_reasoning"


def _pool() -> list[dict]:
    return read_pools(to_fake_reasoning.variant_pool.file)[_DIRECTIVE]


def test_there_is_one_technique():
    assert [fn.__name__ for fn in get_reasoning_injection_functions()] == [
        "to_fake_reasoning"]


def test_the_pool_is_one_the_technique_can_draw():
    check_pools([to_fake_reasoning.variant_pool])


def test_the_pool_holds_its_floor_of_variants_each_named_once():
    names = [variant["name"] for variant in _pool()]
    assert len(names) == len(set(names)) >= _FLOOR


def test_every_variant_holds_the_prompt_once_and_is_named_in_the_info(monkeypatch):
    for variant in _pool():
        monkeypatch.setattr(RI, "draw_variant", lambda variants, rng=None, v=variant: v)
        for text in TEXTS:
            written, info = to_fake_reasoning(text)
            assert variant["template"].count("{prompt}") == 1
            assert written.count(text) == 1
            # The reasoning block is injected before the prompt.
            assert written.rstrip().endswith(text)
            assert info == f"variant={variant['name']}"


def test_a_seed_draws_its_variant_as_rng_choice_does():
    for seed in range(10):
        ours, theirs = random.Random(seed), random.Random(seed)
        _, info = to_fake_reasoning(TEXTS[0], rng=ours)
        assert info == f"variant={theirs.choice(_pool())['name']}"
        assert ours.getstate() == theirs.getstate()


def test_the_spec_tags_a_pure_attack_only_manipulation_family():
    family = load_spec()["families"]["reasoning_injection"]
    assert (family["layer"], family["dimension"]) == ("manipulation", "example_priming")
    assert family["cross_incompatible_with"] == []
    assert to_fake_reasoning.layer == "manipulation"
    assert to_fake_reasoning.families == ["reasoning_injection"]
    assert to_fake_reasoning.kinds == ["adversarial"]
    assert not to_fake_reasoning.requires_llm
    assert not getattr(to_fake_reasoning, "requires_benign", False)
    assert to_fake_reasoning.since == "1.5"


def test_the_family_is_in_the_attacks_and_not_the_eval_or_natural_pool():
    from redact.recipe import load_dataset_type

    adversarial = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    eval_pool = {fn.__name__ for fn in technique_pool(
        resolve(load_dataset_type("eval")["augmentations"], "adversarial"))}
    assert "to_fake_reasoning" in adversarial
    assert "to_fake_reasoning" not in natural
    assert "to_fake_reasoning" not in eval_pool
    pinned = set(json.loads((Path(__file__).parents[2] / "fixtures" / "eval_pool.json")
                            .read_text(encoding="utf-8")))
    assert "to_fake_reasoning" not in pinned


def _by_name() -> dict:
    return {fn.__name__: fn for fn in all_techniques()}


def test_one_manipulation_per_combination_keeps_it_from_fsh_dap_fake_turns():
    by_name = _by_name()
    for name in ("to_fsh_random_short", "to_dap_random_short", "to_fake_system_turn"):
        other = by_name[name]
        assert other not in compatible([to_fake_reasoning], [other]), name
        assert to_fake_reasoning not in compatible([other], [to_fake_reasoning]), name


def test_the_manipulation_block_refuses_the_named_obfuscations_both_ways():
    """Inherited from being a manipulation family: a heavy encoding, a
    translation, ascii_art, payload_split, word_art and wording are refused,
    whichever is picked first."""
    by_name = _by_name()
    for name in ("to_base64", "to_french", "to_ascii_art", "to_payload_split",
                 "to_word_art"):
        other = by_name[name]
        assert other not in compatible([to_fake_reasoning], [other]), name
        assert to_fake_reasoning not in compatible([other], [to_fake_reasoning]), name


def test_it_combines_with_a_hacking_frame_a_request_and_a_light_obfuscation():
    """No cross rule of its own: an approval preamble sits with a persona, a
    request and a non-blocked obfuscation, both ways."""
    by_name = _by_name()
    for name in ("to_psychopath_persona", "to_innocuous_question",
                 "to_leetspeak_basic"):
        partner = by_name[name]
        assert partner in compatible([to_fake_reasoning], [partner]), name
        assert to_fake_reasoning in compatible([partner], [to_fake_reasoning]), name
