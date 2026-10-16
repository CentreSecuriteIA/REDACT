"""The augmentation section: its defaults, the overlay, what it checks, the rounds."""

import pytest

from redact.jailbreak import default_escalation_schedule
from redact.jailbreak.augmentation import (
    KINDS,
    apply_flat_keywords,
    defaults,
    natural_only_families,
    resolve,
    rounds_of,
)
from redact.jailbreak.pool import technique_pool

_BUDGET = {"max_complexity": 6, "max_obfuscations": 2, "include_hacking": True,
           "include_manipulation": True, "include_obfuscation": True,
           "include_requests": True}


def test_both_kinds_are_the_default_when_the_section_is_none():
    for kind in KINDS:
        assert resolve(None, kind) == resolve({}, kind) == resolve(defaults(), kind)
    assert resolve(None, "natural") == defaults()["natural"]


def test_a_kind_given_as_none_is_off():
    assert resolve({"natural": None}, "natural") is None
    assert resolve({"natural": None}, "adversarial") == resolve(None, "adversarial")


def _without(names: list, *left_out: str) -> list:
    return [name for name in names if name not in left_out]


def test_a_deny_given_alone_keeps_the_defaults_allow():
    default = defaults()["natural"]
    kind = resolve({"natural": {"families": {"deny": ["structural"]}}}, "natural")
    assert kind["families"] == {
        "allow": _without(default["families"]["allow"], "structural"),
        "deny": ["structural"]}
    assert kind["techniques"] == default["techniques"]
    pool = technique_pool(kind)
    families = {family for fn in pool for family in fn.families}
    assert "structural" not in families and "typos" in families
    assert not any(fn.layer == "hacking" for fn in pool)


def test_a_deny_given_alone_on_families_and_techniques_keeps_both_allows():
    default = defaults()["natural"]
    kind = resolve({"natural": {"families": {"deny": ["structural"]},
                                "techniques": {"deny": ["to_leetspeak_basic"]}}},
                   "natural")
    assert kind["families"]["allow"] == _without(
        default["families"]["allow"], "structural")
    # A list given replaces the default's whole: the answer_format three are back.
    assert kind["techniques"] == {
        "allow": _without(default["techniques"]["allow"], "to_leetspeak_basic"),
        "deny": ["to_leetspeak_basic"]}
    names = {fn.__name__ for fn in technique_pool(kind)}
    assert "to_leetspeak_basic" not in names and "to_prefix_injection" in names


def test_an_allow_given_alone_keeps_the_defaults_deny():
    default = defaults()["natural"]
    kind = resolve({"natural": {"techniques": {"allow": ["to_leetspeak_basic"]}}},
                   "natural")
    assert kind["techniques"] == {
        "allow": ["to_leetspeak_basic"], "deny": default["techniques"]["deny"]}
    assert kind["families"] == default["families"]
    names = {fn.__name__ for fn in technique_pool(kind)}
    assert not names & set(default["techniques"]["deny"])


def test_a_name_the_run_allows_leaves_the_defaults_deny():
    kind = resolve({"natural": {"techniques": {"allow": ["to_prefix_injection"]}}},
                   "natural")
    assert kind["techniques"] == {
        "allow": ["to_prefix_injection"],
        "deny": ["to_refusal_then_comply", "to_apology_prevention"]}
    assert "to_prefix_injection" in {fn.__name__ for fn in technique_pool(kind)}


def test_an_adversarial_deny_given_alone_keeps_allow_null():
    kind = resolve({"adversarial": {"families": {"deny": ["ascii_art"]}}},
                   "adversarial")
    assert kind["families"]["allow"] is None
    assert kind["layers"] == {"deny": []}


def test_the_natural_kind_needs_an_allow_list():
    with pytest.raises(ValueError) as raised:
        resolve({"natural": {"families": {"allow": None},
                             "techniques": {"allow": None}}}, "natural")
    assert str(raised.value) == (
        "augmentations.natural needs an allow list: write families.allow or "
        "techniques.allow, as the natural kind draws only what they name.")
    # The adversarial kind draws from every technique.
    resolve({"adversarial": {"families": {"allow": None},
                             "techniques": {"allow": None}}}, "adversarial")


def test_a_resolved_section_resolves_to_itself():
    section = {"adversarial": {"rounds": "escalation", "family_weights": {"encode": 2}},
               "natural": {"iterations": 2, "families": {"deny": ["structural"]},
                           "techniques": {"allow": ["to_prefix_injection"]}}}
    once = {kind: resolve(section, kind) for kind in KINDS}
    assert {kind: resolve(once, kind) for kind in KINDS} == once
    assert once["adversarial"]["iterations"] == len(default_escalation_schedule())


def test_the_packaged_defaults():
    adversarial, natural = defaults()["adversarial"], defaults()["natural"]
    assert adversarial["families"] == {"allow": None, "deny": ["translation"]}
    assert natural_only_families() == ["casual_writing"]
    assert resolve(None, "adversarial")["families"] == {
        "allow": None, "deny": ["translation", "casual_writing"]}
    assert "casual_writing" in natural["families"]["allow"]
    assert adversarial["rounds"] is None and adversarial["layer_rates"] == {}
    assert "translation" not in natural["families"]["allow"]
    assert natural["exact_techniques"] == natural["iterations"] == 1


# --- what a section may not say -----------------------------------------------

@pytest.mark.parametrize(("section", "kind", "message"), [
    ({"adversarials": {}}, "adversarial",
     "augmentations has unknown key(s) ['adversarials']; write ['adversarial', "
     "'natural']"),
    ({"natural": []}, "natural", "augmentations.natural must be an object or null"),
    ({"adversarial": {"max_complexty": 3}}, "adversarial",
     "augmentations.adversarial has unknown key(s) ['max_complexty']; write only"),
    ({"natural": {"rounds": "escalation"}}, "natural",
     "augmentations.natural has unknown key(s) ['rounds']"),
    ({"natural": {"layers": {"allow": ["hacking"]}}}, "natural",
     "augmentations.natural.layers has unknown key(s) ['allow']; write only ['deny']"),
    ({"natural": {"layers": {"deny": ["hackin"]}}}, "natural",
     "augmentations.natural.layers.deny names 'hackin', which is not a layer of "
     "combination_spec.json; write one of"),
    ({"natural": {"families": {"allow": ["tyops"], "deny": []}}}, "natural",
     "augmentations.natural.families.allow names 'tyops', which is not a family of "
     "combination_spec.json; write one of"),
    ({"adversarial": {"families": {"deny": ["ascii"]}}}, "adversarial",
     "augmentations.adversarial.families.deny names 'ascii', which is not a family"),
    ({"natural": {"techniques": {"allow": ["to_shout"], "deny": []}}}, "natural",
     "augmentations.natural.techniques.allow names 'to_shout', which "
     "combination_spec.json does not list"),
    ({"natural": {"families": {"allow": ["typos"], "deny": ["typos"]}}}, "natural",
     "augmentations.natural.families names ['typos'] under both allow and deny; "
     "write each under one of them"),
    ({"natural": {"techniques": {"allow": ["to_json"], "deny": ["to_json"]}}},
     "natural", "augmentations.natural.techniques names ['to_json'] under both"),
    ({"natural": {"families": {"allow": [], "deny": []}}}, "natural",
     "augmentations.natural.families.allow must be a non-empty list, got []; write "
     "null for all"),
    ({"natural": {"families": {"deny": defaults()["natural"]["families"]["allow"]}}},
     "natural", "augmentations.natural.families.deny leaves out every name of the "
     "default's allow; write families.allow as well."),
    ({"adversarial": {"family_weights": {"typos": 0}}}, "adversarial",
     "augmentations.adversarial.family_weights['typos'] must be a number above 0"),
    ({"adversarial": {"family_weights": {"typos": True}}}, "adversarial",
     "family_weights['typos'] must be a number above 0"),
    ({"adversarial": {"family_weights": {"translation": 2}}}, "adversarial",
     "augmentations.adversarial.family_weights names 'translation', which the kind "
     "does not draw from"),
    ({"adversarial": {"layer_rates": {"hacking": 1.5}}}, "adversarial",
     "augmentations.adversarial.layer_rates['hacking'] must be a number from 0 to 1"),
    ({"adversarial": {"layer_rates": {"obfuscation": 0.5}}}, "adversarial",
     "augmentations.adversarial.layer_rates names 'obfuscation'; write rates for "
     "['hacking', 'manipulation', 'requests', 'request_obfuscation']"),
    ({"adversarial": {"rounds": "escalation", "iterations": 2}}, "adversarial",
     "augmentations.adversarial: rounds gives 4 rounds and iterations says 2; the "
     "rounds set the number, so leave iterations out"),
    ({"adversarial": {"rounds": [{"max_complexity": 3}], "iterations": 3}},
     "adversarial", "rounds gives 1 rounds and iterations says 3"),
    ({"adversarial": {"rounds": "escalate"}}, "adversarial",
     "augmentations.adversarial.rounds must be null, \"escalation\" or a non-empty "
     "list of rounds"),
    ({"adversarial": {"rounds": [{"include_hacking": False}]}}, "adversarial",
     "augmentations.adversarial.rounds[0] has unknown key(s) ['include_hacking']"),
    ({"adversarial": {"rounds": [{"layer_rates": {"requests": 2}}]}}, "adversarial",
     "augmentations.adversarial.rounds[0].layer_rates['requests'] must be a number"),
    ({"adversarial": {"iterations": 0}}, "adversarial",
     "augmentations.adversarial.iterations must be a whole number of at least 1"),
    ({"natural": {"exact_techniques": 1.5}}, "natural",
     "augmentations.natural.exact_techniques must be a whole number of at least 1"),
    ({"adversarial": {"max_complexity": -1}}, "adversarial",
     "augmentations.adversarial.max_complexity must be a whole number of at least 0"),
    ({"adversarial": {"pure_only": "yes"}}, "adversarial",
     "augmentations.adversarial.pure_only must be true or false"),
    ({"adversarial": {"allow_request_obfuscation": 1}}, "adversarial",
     "augmentations.adversarial.allow_request_obfuscation must be true or false"),
    ({"natural": {"entry_types": ["bening"]}}, "natural",
     "augmentations.natural.entry_types names ['bening']; write entry types of"),
    ({"natural": {"entry_types": []}}, "natural",
     "augmentations.natural.entry_types must be a non-empty list"),
])
def test_a_section_that_breaks_a_rule_raises_naming_the_kind_and_key(
        section, kind, message):
    with pytest.raises(ValueError) as raised:
        resolve(section, kind)
    assert message in str(raised.value)


def test_a_technique_the_registry_lacks_raises(monkeypatch):
    import redact.jailbreak.augmentation as A

    monkeypatch.setattr(A, "all_techniques", list)
    with pytest.raises(ValueError, match="'to_json', which no registered technique"):
        resolve({"natural": {"techniques": {"allow": ["to_json"]}}}, "natural")


# --- the rounds ---------------------------------------------------------------

def test_a_natural_kind_draws_its_exact_count_each_round():
    kind = resolve({"natural": {"iterations": 3, "exact_techniques": 2}}, "natural")
    assert rounds_of(kind) == [{"exact_techniques": 2}] * 3


def test_an_adversarial_round_is_the_budget_by_default():
    assert rounds_of(resolve(None, "adversarial")) == [_BUDGET]
    kind = resolve({"adversarial": {"iterations": 2, "max_complexity": 4}},
                   "adversarial")
    assert rounds_of(kind) == [{**_BUDGET, "max_complexity": 4}] * 2


def test_a_denied_layer_is_left_out_of_every_round():
    kind = resolve({"adversarial": {"layers": {"deny": ["hacking"]}}}, "adversarial")
    assert rounds_of(kind) == [{**_BUDGET, "include_hacking": False}]


def test_escalation_merges_the_built_in_rounds_over_the_budget():
    kind = resolve({"adversarial": {"rounds": "escalation",
                                    "allow_request_obfuscation": True}}, "adversarial")
    budget = {**_BUDGET, "allow_request_obfuscation": True}
    assert rounds_of(kind) == [{**budget, **r} for r in default_escalation_schedule()]


def test_a_rounds_layer_rates_reach_the_sampler_as_its_sampling_probs():
    kind = resolve({"adversarial": {"rounds": [
        {"exact_techniques": 1}, {"layer_rates": {"requests": 0.2}}]}}, "adversarial")
    assert rounds_of(kind) == [{**_BUDGET, "exact_techniques": 1},
                               {**_BUDGET, "sampling_probs": {"requests": 0.2}}]


# --- the flat keywords ----------------------------------------------------------

def test_the_flat_keywords_land_in_the_adversarial_kind():
    kind = apply_flat_keywords(resolve(None, "adversarial"), {
        "include_hacking": False, "include_translation": True, "pure_only": True,
        "sampling_probs": {"hacking": 0.1}, "entry_types": ["benign"],
        "settings_per_iteration": [{"exact_techniques": 1},
                                   {"sampling_probs": {"requests": 0.3}}],
        "max_obfuscations": None,
    })
    assert kind == {
        **defaults()["adversarial"],
        "layers": {"deny": ["hacking"]},
        "families": {"allow": None, "deny": ["casual_writing"]},
        "pure_only": True, "layer_rates": {"hacking": 0.1},
        "entry_types": ["benign"], "iterations": 2,
        "rounds": [{"exact_techniques": 1}, {"layer_rates": {"requests": 0.3}}],
    }
    assert resolve({"adversarial": kind}, "adversarial") == kind


def test_flat_iterations_that_differ_from_the_flat_rounds_raise():
    rounds = [{"exact_techniques": 1}, {"exact_techniques": 2}]
    with pytest.raises(ValueError, match="iterations is 3 and settings_per_iteration "
                       "gives 2 rounds"):
        apply_flat_keywords(resolve(None, "adversarial"),
                            {"iterations": 3, "settings_per_iteration": rounds})
    kind = apply_flat_keywords(resolve(None, "adversarial"), {
        "iterations": len(rounds), "settings_per_iteration": rounds})
    assert kind["iterations"] == len(rounds)


def test_include_translation_false_denies_it_once():
    kind = apply_flat_keywords(resolve(None, "adversarial"),
                               {"include_translation": False, "include_hacking": True})
    assert kind["families"]["deny"] == ["translation", "casual_writing"]
    assert kind["layers"]["deny"] == []
