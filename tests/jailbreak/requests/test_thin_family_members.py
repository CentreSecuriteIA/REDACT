"""The seven members spec 1.3 added to the thin request families:
impersonation, temporal, the two distractors and asking.

Their variants are attacks, and the urgent ask is a natural variation too,
as its family; these tests compare flags and names (the prompt kept, the slot
filled, the variant named), never print a text.
"""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import draw_variant, read_pools
from redact.jailbreak.pool import all_techniques, registry, technique_pool

_PROBE = "a neutral probe about the weather"
_BY_NAME = registry(all_techniques())

#: Each new member: its family, its template file and directive, its
#: complexity, and whether its text goes before the prompt.
_MEMBERS = {
    "to_impersonate_authority": ("impersonation", "impersonation", "authority", 1,
                                 True),
    "to_impersonate_developer": ("impersonation", "impersonation", "developer", 1,
                                 True),
    "to_use_future_tense": ("temporal", "temporal", "future_tense", 3, True),
    "to_use_historical_era": ("temporal", "temporal", "historical_era", 3, True),
    "to_distractor_prefix_task": ("distractor_prefix", "distractor", "prefix_task",
                                  2, True),
    "to_distractor_suffix_task": ("distractor_suffix", "distractor", "suffix_task",
                                  2, False),
    "to_ask_with_urgency": ("asking", "asking", "ask_with_urgency", 1, True),
}


@pytest.mark.parametrize("name", _MEMBERS)
def test_a_member_writes_its_directive_s_variant_as_its_family_names_it(name):
    """``type=<directive>;variant=<name>``, the form of its family's other
    members; compared as flags, so no text is shown."""
    _family, file, directive, _complexity, before = _MEMBERS[name]
    fn = _BY_NAME[name]
    pool = read_pools(f"requests/{file}_templates.json")[directive]
    assert fn.variant_pool.directive == directive and len(pool) == 8   # noqa: PLR2004
    for seed in range(12):
        text, info = fn(_PROBE, rng=random.Random(seed))
        drawn = draw_variant(pool, random.Random(seed))["name"]
        assert info == f"type={directive};variant={drawn}"
        found = (_PROBE in text, text.find(_PROBE) > 0, "{prompt}" in text)
        assert found == (True, before, False), (name, seed)


@pytest.mark.parametrize("name", _MEMBERS)
def test_a_member_is_a_pure_technique_of_spec_1_3_that_takes_its_family_s_kinds(name):
    """Only the asking family is natural: its member draws as a variation too."""
    family, _file, _directive, complexity, _before = _MEMBERS[name]
    fn = _BY_NAME[name]
    assert (fn.families, fn.complexity, fn.requires_llm) == ([family], complexity,
                                                             False)
    kinds = ["adversarial", "natural"] if family == "asking" else ["adversarial"]
    assert (fn.kinds, fn.since) == (kinds, "1.3")


@pytest.mark.parametrize("family", ["impersonation", "temporal", "distractor_prefix",
                                    "distractor_suffix", "asking"])
def test_each_thin_family_now_has_three_members(family):
    assert len([fn for fn in all_techniques() if fn.families[0] == family]) == 3  # noqa: PLR2004


def test_the_urgent_ask_is_natural_as_its_family():
    """The owner decided (2026-10-22) that an urgent ask is a natural
    variation: the natural pool is 72, with the other asking members, and the
    other six members of spec 1.3 stay attacks."""
    natural = technique_pool(resolve(None, "natural"))
    assert _BY_NAME["to_ask_with_urgency"] in natural
    assert _BY_NAME["to_innocuous_question"] in natural
    assert not {name for name in _MEMBERS if name != "to_ask_with_urgency"} & {
        fn.__name__ for fn in natural}


def test_the_eval_type_denies_the_two_new_temporal_members():
    """Eval allows the temporal family whole, so it names them in
    techniques.deny, and its temporal row keeps to_use_past_tense alone."""
    from redact.recipe import load_dataset_type

    section = load_dataset_type("eval")["augmentations"]
    assert {"to_use_future_tense", "to_use_historical_era"} <= set(
        section["adversarial"]["techniques"]["deny"])
    pool = technique_pool(resolve(section, "adversarial"))
    drawn = {fn.__name__ for fn in pool} & set(_MEMBERS)
    assert drawn == set()
    assert [fn.__name__ for fn in pool if "temporal" in fn.families] == [
        "to_use_past_tense"]
