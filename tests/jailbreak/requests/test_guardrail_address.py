"""Guardrail address (requests/guardrail_address.py, spec 1.4, BELLS audit M4):
a line or block addressed to the classifier, before or after the prompt. Three
techniques, one per reference format, each a template pool of eight variants.
The reference wording is not reproduced here; the tests assert on shape, the
slot, the info, the floor, the applicability and the pools."""

import json
import random
from pathlib import Path

import pandas as pd
import pytest

import redact.jailbreak.directives as D
from redact.dataset import Manifest
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.plan import plan_run
from redact.jailbreak.pool import technique_pool
from redact.jailbreak.requests.guardrail_address import (
    EXCLUDED,
    get_guardrail_address_functions,
    is_targeted_input,
)
from redact.jailbreak.technique import applies, load_spec
from redact.types import ALL_ENTRY_TYPES, EntryType

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, café.',
]

FUNCTIONS = get_guardrail_address_functions()
_FLOOR = 8
_FILE = "requests/guardrail_templates.json"


def _pool(fn) -> list[dict]:
    ref = fn.variant_pool
    return read_pools(ref.file)[ref.directive]


def _variants(functions: list) -> list[tuple]:
    return [(fn, variant) for fn in functions for variant in _pool(fn)]


def _ids(pairs: list[tuple]) -> list[str]:
    return [f"{fn.__name__}-{variant['name']}" for fn, variant in pairs]


_ALL = _variants(FUNCTIONS)


def _written(monkeypatch, fn, variant: dict, text: str) -> tuple[str, str]:
    monkeypatch.setattr(D, "draw_variant", lambda variants, rng=None: variant)
    return fn(text)


def test_there_are_three_techniques_one_per_reference_format():
    assert [fn.__name__ for fn in FUNCTIONS] == [
        "to_guardrail_block_question", "to_guardrail_classify_benign",
        "to_guardrail_security_override"]


def test_the_pools_are_ones_their_techniques_can_draw():
    check_pools(fn.variant_pool for fn in FUNCTIONS)


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_each_pool_holds_its_floor_of_variants_each_named_once(fn):
    names = [variant["name"] for variant in _pool(fn)]
    assert len(names) == len(set(names)) == _FLOOR


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize(("fn", "variant"), _ALL, ids=_ids(_ALL))
def test_every_variant_holds_the_text_once_and_is_named_in_the_info(
        monkeypatch, fn, variant, text):
    written, info = _written(monkeypatch, fn, variant, text)
    assert variant["template"].count("{prompt}") == 1
    assert written.count(text) == 1
    assert info == f"guardrail={fn.variant_pool.directive};variant={variant['name']}"


def test_the_first_variant_of_each_is_the_reference_format(monkeypatch):
    """Format 1 and 3 put the prompt first (a note after it); format 2 puts a
    line before it and wraps the prompt in a tag."""
    first = {fn.variant_pool.directive: _pool(fn)[0] for fn in FUNCTIONS}
    text = TEXTS[0]
    block = first["block_question"]["template"].format(prompt=text)
    override = first["security_override"]["template"].format(prompt=text)
    classify = first["classify_benign"]["template"].format(prompt=text)
    assert block.startswith(text) and block.rstrip().endswith("</for-classifiers>")
    assert override.startswith(text) and len(override) > len(text)
    assert not classify.startswith(text) and text in classify


def test_a_seed_draws_its_variant_as_rng_choice_does():
    for fn in FUNCTIONS:
        for seed in range(10):
            ours, theirs = random.Random(seed), random.Random(seed)
            _, info = fn(TEXTS[0], rng=ours)
            assert info.endswith(f";variant={theirs.choice(_pool(fn))['name']}")
            assert ours.getstate() == theirs.getstate()


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_a_seed_gives_the_same_text(fn):
    assert fn(TEXTS[1], rng=random.Random(3)) == fn(TEXTS[1], rng=random.Random(3))


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_it_applies_to_every_input_but_an_absolutely_benign_one(fn):
    assert EXCLUDED == (EntryType.BENIGN,)
    assert {entry_type.value for entry_type in ALL_ENTRY_TYPES
            if applies(fn, TEXTS[0], entry_type.value)} == {
        "harmful", "dual_use_harmful", "dual_use_benign"}


def test_is_targeted_input_reads_the_entry_type_alone():
    assert is_targeted_input("anything", "harmful")
    assert not is_targeted_input("anything", "benign")


# --- the spec and the pools -----------------------------------------------------

def test_the_spec_tags_a_pure_attack_only_request_family():
    families = load_spec()["families"]
    assert "guardrail_address" in families
    family = families["guardrail_address"]
    assert (family["layer"], family["dimension"]) == ("requests", "request_shape")
    for fn in FUNCTIONS:
        assert fn.layer == "requests"
        assert fn.families == ["guardrail_address"]
        assert fn.kinds == ["adversarial"]
        assert fn.complexity == 1
        assert not fn.requires_llm and not getattr(fn, "needs_latin", False)


def test_the_family_is_in_the_attacks_and_the_eval_pool_and_not_the_natural_one():
    """Eval pool v2 admitted the family whole: the eval attacks are on
    harmful inputs, each of which it applies to."""
    from redact.recipe import load_dataset_type

    names = {fn.__name__ for fn in FUNCTIONS}
    adversarial = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    eval_pool = {fn.__name__ for fn in technique_pool(
        resolve(load_dataset_type("eval")["augmentations"], "adversarial"))}
    assert names <= adversarial
    assert not names & natural
    assert names <= eval_pool
    pinned = set(json.loads((Path(__file__).parents[2] / "fixtures" / "eval_pool.json")
                            .read_text(encoding="utf-8")))
    assert names <= pinned
    assert load_dataset_type("eval")["augmentations"]["adversarial"]["entry_types"] == [
        "harmful"]


# --- the plan draws them where they apply --------------------------------------

def test_the_plan_draws_a_guardrail_address_for_every_input_but_benign(tmp_path):
    """An attack plan over inputs of each entry type, one group each: the
    family goes to the harmful and dual-use groups, never the benign one."""
    entry_types = [entry_type.value for entry_type in ALL_ENTRY_TYPES] * 20
    inputs = pd.DataFrame({
        "prompt": [f"input {i} about a topic" for i in range(len(entry_types))],
        "category": ["Cyber"] * len(entry_types), "entry_type": entry_types})
    manifest = tmp_path / "jb.manifest.jsonl"
    # A request-only pool, so the request layer is dealt every unit.
    section = {"layers": {"deny": ["hacking", "obfuscation", "manipulation"]},
               "families": {"allow": ["guardrail_address"]}}
    plan_run(inputs, technique_pool(resolve({"adversarial": section}, "adversarial")),
             manifest_path=manifest, seed=42,
             sample_kwargs={"exact_techniques": 1}, verbose=False)
    rows = Manifest(manifest).load()
    drawn_for = {row["entry_type"] for row in rows
                 if any(name.startswith("to_guardrail_")
                        for name in row["combination"])}
    assert drawn_for == {"harmful", "dual_use_harmful", "dual_use_benign"}


def test_no_variant_names_the_prompt_slot_twice_or_an_unknown_slot():
    import string

    for fn in FUNCTIONS:
        for variant in _pool(fn):
            slots = [field for _, field, _, _ in
                     string.Formatter().parse(variant["template"]) if field]
            assert slots == ["prompt"], (fn.__name__, variant["name"])
