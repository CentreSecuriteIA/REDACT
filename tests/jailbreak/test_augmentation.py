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

# The packaged adversarial kind's budget round, its request obfuscation on.
_BUDGET = {"max_complexity": 6, "max_obfuscations": 2, "include_hacking": True,
           "include_manipulation": True, "include_obfuscation": True,
           "include_requests": True, "allow_request_obfuscation": True}
# The same round with it off, as the eval type has it: the key left out.
_BUDGET_OFF = {key: value for key, value in _BUDGET.items()
               if key != "allow_request_obfuscation"}


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
    kind = resolve({"natural": {"families": {"deny": ["typos"]}}}, "natural")
    assert kind["families"] == {
        "allow": _without(default["families"]["allow"], "typos"),
        "deny": ["typos"]}
    assert kind["techniques"] == default["techniques"]
    pool = technique_pool(kind)
    families = {family for fn in pool for family in fn.families}
    assert "typos" not in families and "casual_writing" in families
    assert not any(fn.layer == "hacking" for fn in pool)


def test_a_deny_given_alone_on_families_and_techniques_keeps_both_allows():
    default = defaults()["natural"]
    kind = resolve({"natural": {"families": {"deny": ["typos"]},
                                "techniques": {"deny": ["to_leetspeak_basic"]}}},
                   "natural")
    assert kind["families"]["allow"] == _without(default["families"]["allow"], "typos")
    # A list given replaces the default's whole: the answer_format three leave
    # the deny list, and the spec's kinds still keep them out of the pool.
    assert kind["techniques"] == {
        "allow": _without(default["techniques"]["allow"], "to_leetspeak_basic"),
        "deny": ["to_leetspeak_basic"]}
    names = {fn.__name__ for fn in technique_pool(kind)}
    assert "to_leetspeak_basic" not in names
    assert not names & set(default["techniques"]["deny"])
    assert "to_answer_in_markdown" in names


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
    base = {"natural": {"techniques": {"deny": ["to_leetspeak_basic", "to_rewrite"]}}}
    kind = resolve({"natural": {"techniques": {"allow": ["to_leetspeak_basic"]}}},
                   "natural", base=base)
    assert kind["techniques"] == {"allow": ["to_leetspeak_basic"],
                                  "deny": ["to_rewrite"]}
    names = {fn.__name__ for fn in technique_pool(kind)}
    assert "to_leetspeak_basic" in names and "to_rewrite" not in names


def test_an_adversarial_deny_given_alone_keeps_allow_null():
    kind = resolve({"adversarial": {"families": {"deny": ["ascii_art"]}}},
                   "adversarial")
    assert kind["families"]["allow"] is None
    assert kind["layers"] == {"deny": ["rewrite"]}


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
               "natural": {"iterations": 2, "families": {"deny": ["typos"]},
                           "techniques": {"allow": ["to_answer_as_code"]}}}
    once = {kind: resolve(section, kind) for kind in KINDS}
    assert {kind: resolve(once, kind) for kind in KINDS} == once
    assert once["adversarial"]["iterations"] == len(default_escalation_schedule())


# --- a base section under the given one ----------------------------------------

_BASE = {"adversarial": {"pure_only": True,
                         "families": {"allow": ["encode", "framing"]}},
         "natural": None}


def test_a_kind_left_out_is_the_bases():
    assert resolve(None, "adversarial", base=_BASE) == resolve(_BASE, "adversarial")
    assert resolve({}, "natural", base=_BASE) is None


def test_a_kind_given_overlays_the_bases_key_by_key():
    kind = resolve({"adversarial": {"max_complexity": 4}}, "adversarial", base=_BASE)
    assert kind == {**resolve(_BASE, "adversarial"), "max_complexity": 4}
    # Its lists overlay the base's one by one.
    kind = resolve({"adversarial": {"families": {"deny": ["framing"]}}},
                   "adversarial", base=_BASE)
    assert kind["families"] == {"allow": ["encode"],
                                "deny": ["framing", "casual_writing"]}
    assert kind["pure_only"] is True


def test_a_kind_the_base_switches_off_comes_back_on_over_the_packaged_default():
    given = {"natural": {"exact_techniques": 2}}
    assert resolve(given, "natural", base=_BASE) == resolve(given, "natural")
    assert resolve({"adversarial": None}, "adversarial", base=_BASE) is None


def test_a_section_resolved_over_a_base_resolves_to_itself_over_it():
    section = {"adversarial": {"rounds": [{"exact_techniques": 1}]}}
    once = {kind: resolve(section, kind, base=_BASE) for kind in KINDS}
    assert {kind: resolve(once, kind, base=_BASE) for kind in KINDS} == once
    assert once["adversarial"]["iterations"] == 1


@pytest.mark.parametrize("given", [{"adversarial": {"max_complexity": 3}},
                                   {"natural": {"iterations": 2}}], ids=KINDS)
def test_a_comment_inside_a_kind_is_skipped(given):
    ((kind, keys),) = given.items()
    commented = {kind: {"_comment": "why", **keys}}
    assert resolve(commented, kind) == resolve(given, kind)


def test_the_packaged_defaults():
    adversarial, natural = defaults()["adversarial"], defaults()["natural"]
    assert adversarial["families"] == {"allow": None, "deny": []}
    assert natural_only_families() == ["casual_writing"]
    assert resolve(None, "adversarial")["families"] == {
        "allow": None, "deny": ["casual_writing"]}
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
    ({"adversarial": {"pure_only": True, "family_weights": {"typos": 2}}},
     "adversarial",
     "augmentations.adversarial.family_weights names 'typos', which the kind "
     "does not draw from"),
    ({"adversarial": {"layer_rates": {"hacking": 1.5}}}, "adversarial",
     "augmentations.adversarial.layer_rates['hacking'] must be a number from 0 to 1"),
    ({"adversarial": {"layer_rates": {"obfuscation": 0.5}}}, "adversarial",
     "augmentations.adversarial.layer_rates names 'obfuscation'; write rates for "
     "['rewrite', 'hacking', 'manipulation', 'requests', 'request_obfuscation']"),
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
    ({"adversarial": {"budget": "pool"}}, "adversarial",
     "augmentations.adversarial.budget must be \"unit\", \"group\" or null, "
     "got 'pool'"),
    ({"adversarial": {"max_unit_complexity": 9}}, "adversarial",
     'augmentations.adversarial.max_unit_complexity applies only with "budget": '
     '"group"'),
    ({"adversarial": {"budget": "unit", "max_unit_complexity": 9}}, "adversarial",
     'augmentations.adversarial.max_unit_complexity applies only with "budget": '
     '"group"'),
    ({"adversarial": {"budget": "group", "max_unit_complexity": 4}}, "adversarial",
     "augmentations.adversarial.max_unit_complexity must be a whole number of at "
     "least 6, got 4"),
    ({"adversarial": {"budget": "group", "max_complexity": 8,
                      "max_unit_complexity": 7}}, "adversarial",
     "augmentations.adversarial.max_unit_complexity must be a whole number of at "
     "least 8, got 7"),
    ({"adversarial": {"budget": "group", "max_unit_complexity": 7.5}}, "adversarial",
     "augmentations.adversarial.max_unit_complexity must be a whole number of at "
     "least 6, got 7.5"),
    ({"natural": {"budget": "group"}}, "natural",
     "augmentations.natural gives ['budget'], which the natural kind ignores"),
    ({"natural": {"max_unit_complexity": 9}}, "natural",
     "augmentations.natural gives ['max_unit_complexity'], which the natural kind "
     "ignores"),
    ({"adversarial": {"budget": "group", "max_unit_complexity": 8,
                      "rounds": [{"max_complexity": 9}]}}, "adversarial",
     "augmentations.adversarial.rounds[0].max_complexity is 9, above "
     "max_unit_complexity 8"),
    ({"adversarial": {"budget": "group", "max_unit_complexity": 8,
                      "rounds": "escalation"}}, "adversarial",
     "augmentations.adversarial.rounds[3].max_complexity is 9, above "
     "max_unit_complexity 8"),
    ({"adversarial": {"rounds": [{"budget": "group"}]}}, "adversarial",
     "augmentations.adversarial.rounds[0] has unknown key(s) ['budget']"),
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


@pytest.mark.parametrize("name", [
    "to_prefix_injection",
    # A family the kind draws, less a technique the spec keeps from it.
    "to_rewrite_with_typos_insane",
])
def test_a_natural_techniques_allow_naming_one_the_kind_does_not_draw_raises(name):
    with pytest.raises(ValueError) as raised:
        resolve({"natural": {"techniques": {"allow": [name]}}}, "natural")
    assert str(raised.value) == (
        f"augmentations.natural.techniques.allow names {name!r}, which the natural "
        "kind does not draw: its kinds in combination_spec.json are "
        "['adversarial']. Take it out of techniques.allow (a config_dir copy of "
        "augmentations.json made before it left the natural kind still names "
        'it), or add "natural" to its kinds in combination_spec.json.')


@pytest.mark.parametrize(("family", "members"), [
    ("encode", "list under techniques.allow those of its techniques the natural "
               "kind draws: ['to_leetspeak_basic']."),
    ("suffixes", "list under techniques.allow those of its techniques the natural "
                 "kind draws: ['to_adversarial_suffix_punctuation', "
                 "'to_adversarial_suffix_emoji']."),
    ("structural", "the natural kind draws none of its techniques (to draw one, "
                   'add "natural" to that technique\'s kinds in '
                   "combination_spec.json)."),
])
def test_a_natural_families_allow_naming_one_the_kind_does_not_draw_names_its_members(
        family, members):
    """Not "add natural to the family's kinds": they apply to every technique
    in it that sets none of its own, so the family's other techniques would
    join the natural pool too."""
    with pytest.raises(ValueError) as raised:
        resolve({"natural": {"families": {"allow": ["typos", family]}}}, "natural")
    assert str(raised.value) == (
        f"augmentations.natural.families.allow names {family!r}, which the natural "
        "kind does not draw: its kinds in combination_spec.json are "
        "['adversarial'], and a family's kinds apply to every technique in it "
        "that sets none of its own. Take it out of families.allow (a config_dir "
        "copy of augmentations.json made before it left the natural kind still "
        f"names it); {members}")


def test_a_config_dir_copy_of_the_former_natural_kind_raises(tmp_path):
    """A copy made before the spec's kinds: its natural kind still allows
    structural, temporal and continuation."""
    import json

    from redact import paths

    packaged = paths.jailbreak_configs_dir() / "augmentations.json"
    section = json.loads(packaged.read_text(encoding="utf-8"))
    section["natural"]["families"]["allow"] = [
        "typos", "casual_writing", "structural", "answer_language", "answer_format",
        "asking", "indirect", "temporal", "continuation"]
    (tmp_path / "jailbreak").mkdir()
    (tmp_path / "jailbreak" / "augmentations.json").write_text(
        json.dumps(section), encoding="utf-8")
    with pytest.raises(ValueError, match="families.allow names 'structural', which "
                       "the natural kind does not draw"):
        resolve(None, "natural", config_dir=tmp_path)
    assert resolve(None, "adversarial", config_dir=tmp_path) == resolve(
        None, "adversarial")


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


def test_a_unit_or_null_budget_leaves_the_rounds_as_they_were():
    for budget in ("unit", None):
        kind = resolve({"adversarial": {"budget": budget}}, "adversarial")
        assert rounds_of(kind) == [_BUDGET]


def test_the_group_budget_is_written_to_the_rounds_only_when_set():
    kind = resolve({"adversarial": {"budget": "group"}}, "adversarial")
    assert rounds_of(kind) == [{**_BUDGET, "budget": "group"}]
    kind = resolve({"adversarial": {"budget": "group", "max_unit_complexity": 8}},
                   "adversarial")
    assert rounds_of(kind) == [{**_BUDGET, "budget": "group",
                                "max_unit_complexity": 8}]


def test_a_count_round_carries_the_group_budget_and_ignores_it():
    """The eval type's rounds draw a count; the keys are written to them, as
    to every round, and the sampler leaves them unread."""
    from redact.recipe import load_dataset_type

    base = load_dataset_type("eval")["augmentations"]
    kind = resolve({"adversarial": {"budget": "group"}}, "adversarial", base=base)
    group = {**_BUDGET_OFF, "budget": "group"}
    assert rounds_of(kind) == [{**group, "exact_techniques": 1},
                               {**group, "exact_techniques": 2}]


def test_a_denied_layer_is_left_out_of_every_round():
    kind = resolve({"adversarial": {"layers": {"deny": ["rewrite", "hacking"]}}},
                   "adversarial")
    assert rounds_of(kind) == [{**_BUDGET, "include_hacking": False}]


def test_escalation_merges_the_built_in_rounds_over_the_budget():
    kind = resolve({"adversarial": {"rounds": "escalation"}}, "adversarial")
    assert rounds_of(kind) == [{**_BUDGET, **r} for r in default_escalation_schedule()]


def test_the_packaged_attacks_allow_the_request_obfuscation():
    assert defaults()["adversarial"]["allow_request_obfuscation"] is True


def test_the_request_obfuscation_switched_off_leaves_the_rounds_without_it():
    """Off, a round carries no ``allow_request_obfuscation`` key at all: the
    rounds a plan made before the packaged kind turned it on."""
    kind = resolve({"adversarial": {"allow_request_obfuscation": False}},
                   "adversarial")
    assert rounds_of(kind) == [_BUDGET_OFF]


def test_the_eval_type_keeps_the_request_obfuscation_off():
    """``eval.json`` says false, so the packaged kind's true does not reach
    it: its two count rounds carry no request obfuscation key, as before."""
    from redact.recipe import load_dataset_type

    base = load_dataset_type("eval")["augmentations"]
    kind = resolve(None, "adversarial", base=base)
    assert kind["allow_request_obfuscation"] is False
    assert rounds_of(kind) == [{**_BUDGET_OFF, "exact_techniques": 1},
                               {**_BUDGET_OFF, "exact_techniques": 2}]


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
        "layers": {"deny": ["rewrite", "hacking"]},
        "families": {"allow": None, "deny": ["casual_writing"]},
        "translation": True, "pure_only": True, "layer_rates": {"hacking": 0.1},
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


# --- the translation switch ----------------------------------------------------

def _translates(section: dict, kind: str) -> bool:
    from redact.jailbreak.pool import uses_translation

    return uses_translation(technique_pool(resolve(section, kind)))


def test_translation_is_on_in_the_default_attacks_and_off_in_the_variations():
    assert defaults()["adversarial"]["translation"] is True
    assert _translates(None, "adversarial")
    assert defaults()["natural"]["translation"] is False
    assert not _translates(None, "natural")


def test_a_deny_list_of_ones_own_leaves_the_switch_as_it_is():
    assert _translates({"adversarial": {"families": {"deny": ["ascii_art"]}}},
                       "adversarial")
    assert _translates({"adversarial": {"families": {"deny": []}}}, "adversarial")
    assert not _translates({"natural": {"families": {"deny": ["typos"]}}}, "natural")


@pytest.mark.parametrize("kind", KINDS)
def test_the_switch_turns_translation_on(kind):
    from redact.jailbreak.pool import all_techniques

    pool = technique_pool(resolve({kind: {"translation": True}}, kind))
    languages = [fn for fn in all_techniques() if "translation" in fn.families]
    assert languages and all(fn in pool for fn in languages)
    # The kind's other filters still apply.
    assert set(pool) == set(technique_pool(resolve(None, kind))) | set(languages)


def test_on_the_switch_wins_over_a_family_deny_and_a_technique_deny_still_applies():
    pool = {fn.__name__ for fn in technique_pool(resolve({"adversarial": {
        "translation": True, "families": {"deny": ["translation"]},
        "techniques": {"deny": ["to_french"]}}}, "adversarial"))}
    assert "to_japanese" in pool and "to_french" not in pool


_DROPPED = "redact.jailbreak.augmentation"


@pytest.mark.parametrize(("given", "named"), [
    ({"natural": {"families": {"allow": ["typos", "translation"]}}},
     "['translation']"),
    ({"natural": {"techniques": {"allow": ["to_french"]}}}, "['to_french']"),
    ({"adversarial": {"translation": False, "families": {"allow": ["encode"]},
                      "techniques": {"allow": ["to_thai", "to_base64"]},
                      "family_weights": {"translation": 2}}},
     "['to_thai', 'family_weights.translation']"),
    ({"adversarial": {"translation": False, "family_weights": {"translation": 2}}},
     "['family_weights.translation']"),
])
def test_translation_an_allow_list_names_with_the_switch_off_is_dropped_with_a_warning(
        given, named, caplog):
    (kind,) = given
    with caplog.at_level("WARNING", logger=_DROPPED):
        resolved = resolve(given, kind)
    assert [r.getMessage() for r in caplog.records] == [
        f"augmentations.{kind}.translation is false, so the translation it names "
        f"is left out: {named}."]
    assert not _translates({kind: resolved}, kind)
    assert "translation" not in resolved["family_weights"]
    assert "translation" not in (resolved["families"]["allow"] or ())
    # Resolved again, it names none, and says nothing.
    caplog.clear()
    with caplog.at_level("WARNING", logger=_DROPPED):
        assert resolve({kind: resolved}, kind) == resolved
    assert caplog.records == []
    assert _translates({kind: {**given[kind], "translation": True}}, kind)


def test_a_list_left_empty_by_the_drop_is_no_list_beside_another():
    kind = resolve({"natural": {"techniques": {"allow": ["to_french"]}}}, "natural")
    assert kind["techniques"]["allow"] is None
    assert kind["families"]["allow"] == defaults()["natural"]["families"]["allow"]


def test_allowing_only_translation_with_the_switch_off_raises():
    with pytest.raises(ValueError) as raised:
        resolve({"adversarial": {"translation": False,
                                 "families": {"allow": ["translation"]}}},
                "adversarial")
    assert str(raised.value) == (
        "augmentations.adversarial allows only translation (['translation']) and "
        "augmentations.adversarial.translation is false; set \"translation\": "
        "true, or allow something else.")


def test_switching_translation_off_over_the_eval_type_is_one_key(caplog):
    from redact.recipe import load_dataset_type

    base = load_dataset_type("eval")["augmentations"]
    with caplog.at_level("WARNING", logger=_DROPPED):
        kind = resolve({"adversarial": {"translation": False}}, "adversarial",
                       base=base)
    assert len(caplog.records) == 1
    assert kind["family_weights"] == {}
    assert kind["techniques"]["allow"] == ["to_refusal_then_comply"]
    pool = technique_pool(kind)
    from redact.jailbreak.pool import languages
    from tests.jailbreak.test_pool import EVAL_POOL

    # The pinned eval pool less its nine languages.
    assert sorted(fn.__name__ for fn in pool) == [
        name for name in EVAL_POOL if name not in languages()]
    assert len(pool) == 28      # noqa: PLR2004
    assert not _translates({"adversarial": kind}, "adversarial")


def test_the_switch_must_be_true_or_false():
    with pytest.raises(ValueError, match="translation must be true or false"):
        resolve({"natural": {"translation": "yes"}}, "natural")


@pytest.mark.parametrize("value", [True, False])
def test_include_translation_is_the_switch(value):
    kind = apply_flat_keywords(resolve(None, "adversarial"),
                               {"include_translation": value})
    assert kind["translation"] is value
    assert kind["families"] == resolve(None, "adversarial")["families"]
    assert _translates({"adversarial": kind}, "adversarial") is value
