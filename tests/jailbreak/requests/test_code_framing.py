"""Code framing (requests/code_framing.py, spec 1.5): to_code_completion
(CodeAttack, Ren 2403.07865) and to_decrypt_function (CodeChameleon, Lv
2402.16717). Two pure request-layer techniques. The frame wording is not
reproduced here; the tests assert on shape, the slot, the info, the floor,
the spec tags, the pool membership, the rules (both ways) and — the key
mechanism — that the decrypt function shown really inverts its transform."""

import json
import random
from pathlib import Path

import pytest

import redact.jailbreak.directives as D
import redact.jailbreak.requests.code_framing as CF
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.jailbreak.requests.code_framing import (
    _DECRYPT,
    _TRANSFORMS,
    get_code_framing_functions,
    to_code_completion,
    to_decrypt_function,
)
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import load_spec

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, cafe.',
    "a  b   c",
    "single",
    "",
]
_FLOOR = 8
FUNCTIONS = get_code_framing_functions()


def _pool(directive: str) -> list[dict]:
    return read_pools(CF._FILE)[directive]


def test_there_are_two_techniques():
    assert [fn.__name__ for fn in FUNCTIONS] == [
        "to_code_completion", "to_decrypt_function"]


def test_the_pools_are_ones_their_techniques_can_draw():
    check_pools(fn.variant_pool for fn in FUNCTIONS)


@pytest.mark.parametrize("directive", ["code_completion", "decrypt_function"])
def test_each_pool_holds_its_floor_of_variants_each_named_once(directive):
    names = [variant["name"] for variant in _pool(directive)]
    assert len(names) == len(set(names)) >= _FLOOR


# --- the decrypt function really inverts its transform --------------------------

def test_the_transform_slugs_match_the_shown_decrypt_codes():
    assert set(_TRANSFORMS) == set(_DECRYPT)


@pytest.mark.parametrize("slug", sorted(_TRANSFORMS))
@pytest.mark.parametrize("text", TEXTS)
def test_the_shown_decrypt_inverts_the_transform_in_a_restricted_exec(slug, text):
    """Execute the exact code shown to the model, in a namespace with no
    builtins, on the transformed text; it must return the original."""
    namespace: dict = {}
    exec(_DECRYPT[slug], {"__builtins__": {}}, namespace)  # noqa: S102
    encrypted = _TRANSFORMS[slug](text)
    assert namespace["decrypt"](encrypted) == text


def test_the_transforms_are_involutions():
    for transform in _TRANSFORMS.values():
        for text in TEXTS:
            assert transform(transform(text)) == text


# --- shape and info -------------------------------------------------------------

def test_code_completion_holds_the_prompt_once_and_is_named(monkeypatch):
    for variant in _pool("code_completion"):
        monkeypatch.setattr(D, "draw_variant", lambda variants, rng=None, v=variant: v)
        for text in TEXTS[:2]:
            written, info = to_code_completion(text)
            assert variant["template"].count("{prompt}") == 1
            assert written.count(text) == 1
            assert info == f"directive=code_completion;variant={variant['name']}"


def test_decrypt_function_sets_the_ciphertext_and_names_transform_and_variant():
    for seed in range(10):
        text = "mix the reagents carefully"
        written, info = to_decrypt_function(text, rng=random.Random(seed))
        assert info.startswith("transform=")
        assert ";variant=" in info
        slug = info.split("transform=")[1].split(";")[0]
        assert slug in _TRANSFORMS
        # The ciphertext (not the plaintext) is in the rendered prompt.
        assert _TRANSFORMS[slug](text) in written


def test_a_seed_gives_the_same_text():
    for fn in FUNCTIONS:
        assert (fn(TEXTS[0], rng=random.Random(3))
                == fn(TEXTS[0], rng=random.Random(3)))


# --- the spec and pool membership ----------------------------------------------

def test_the_spec_tags_a_pure_attack_only_request_family():
    family = load_spec()["families"]["code_framing"]
    assert (family["layer"], family["dimension"]) == ("requests", "request_shape")
    assert family["cross_incompatible_with"] == []
    for fn in FUNCTIONS:
        assert fn.layer == "requests"
        assert fn.families == ["code_framing"]
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


# --- the rules ------------------------------------------------------------------

def _by_name() -> dict:
    return {fn.__name__: fn for fn in all_techniques()}


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_one_request_per_combination(fn):
    """The layer cap keeps it from another request and from the sibling."""
    by_name = _by_name()
    other_request = by_name["to_innocuous_question"]
    assert other_request not in compatible([fn], [other_request])
    assert fn not in compatible([other_request], [fn])
    sibling = to_decrypt_function if fn is to_code_completion else to_code_completion
    assert sibling not in compatible([fn], [sibling])


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_it_combines_with_an_obfuscation_and_a_hacking_frame_both_ways(fn):
    by_name = _by_name()
    for name in ("to_base64", "to_fictional_world"):
        partner = by_name[name]
        assert partner in compatible([fn], [partner]), name
        assert fn in compatible([partner], [fn]), name
