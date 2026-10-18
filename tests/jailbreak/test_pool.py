"""The technique pool: every registered technique, a kind's filters, the registry."""

import json
from pathlib import Path

import pytest

import redact.jailbreak.pool as JPool
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.hacking import get_all_hacking_functions
from redact.jailbreak.manipulation import get_all_manipulation_functions
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.pool import (
    all_techniques,
    registry,
    technique_pool,
    uses_translation,
)
from redact.jailbreak.requests import get_all_request_functions
from redact.jailbreak.technique import load_spec

#: The eval type's attack pool, sorted: ``fixtures/eval_pool.json``.
EVAL_POOL = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "eval_pool.json").read_text(
        encoding="utf-8"))


def _in(family: str, pool: list) -> list:
    return [fn for fn in pool if family in fn.families]


def test_all_techniques_are_the_four_layers_in_registry_order():
    assert all_techniques() == (
        get_all_obfuscation_functions() + get_all_hacking_functions()
        + get_all_manipulation_functions() + get_all_request_functions())


def _kind(**given) -> dict:
    return resolve({"adversarial": given}, "adversarial")


def test_the_default_attack_pool_is_every_technique_but_translation_and_casual():
    full = all_techniques()
    left_out = _in("translation", full) + _in("casual_writing", full)
    assert technique_pool(_kind()) == [fn for fn in full if fn not in left_out]


def _attacks(techniques: list) -> list:
    """``techniques`` less the natural-only family, which no attack draws."""
    return [fn for fn in techniques if "casual_writing" not in fn.families]


def test_switching_translation_on_makes_the_pool_every_attack_technique():
    assert technique_pool(_kind(translation=True)) == _attacks(all_techniques())


def test_a_natural_only_family_is_drawn_by_the_attacks_only_when_allowed():
    pool = technique_pool(_kind(families={"deny": ["translation", "ascii_art"]}))
    assert pool and not _in("casual_writing", pool) and not _in("ascii_art", pool)
    pool = technique_pool(_kind(families={"allow": ["casual_writing", "encode"]}))
    assert _in("casual_writing", pool)


def test_a_denied_layer_drops_that_layer_only():
    pool = technique_pool(_kind(layers={"deny": ["hacking", "requests"]},
                                translation=True))
    assert pool == _attacks(
        get_all_obfuscation_functions() + get_all_manipulation_functions())


def test_pure_only_drops_model_calls_and_benign_data():
    pool = technique_pool(_kind(pure_only=True, translation=True))
    assert pool
    assert not any(fn.requires_llm or fn.requires_benign for fn in pool)
    assert pool == _attacks([fn for fn in all_techniques()
                             if not fn.requires_llm and not fn.requires_benign])


def test_allow_lists_admit_their_families_and_techniques_then_deny_removes():
    pool = technique_pool(_kind(
        families={"allow": ["typos", "structural"], "deny": []},
        techniques={"allow": ["to_base64"], "deny": ["to_xml"]}))
    names = [fn.__name__ for fn in pool]
    assert names == [fn.__name__ for fn in all_techniques()
                     if fn.__name__ == "to_base64"
                     or ({"typos", "structural"} & set(fn.families)
                         and fn.__name__ != "to_xml")]
    assert "to_json" in names and "to_rewrite_with_typos_low" in names


def _family(*names: str) -> list[str]:
    return [fn.__name__ for fn in all_techniques() if set(names) & set(fn.families)]


#: The natural kind's default pool: what the spec marks natural, less
#: translation, which its switch keeps out.
NATURAL_POOL = sorted([
    "to_rewrite_with_typos_low", "to_rewrite_with_typos_medium",
    *_family("casual_writing", "answer_language", "asking", "indirect"),
    "to_answer_in_markdown", "to_answer_as_tutorial", "to_answer_as_code",
    "to_answer_as_structured_data",
    "to_leetspeak_basic", "to_adversarial_suffix_punctuation",
    "to_adversarial_suffix_emoji",
])


def test_the_default_variation_pool_is_the_46_natural_techniques():
    pool = technique_pool(resolve(None, "natural"))
    assert sorted(fn.__name__ for fn in pool) == NATURAL_POOL
    assert len(pool) == 46      # noqa: PLR2004
    assert {fn.families[0] for fn in pool} == {
        "typos", "casual_writing", "answer_language", "answer_format", "asking",
        "indirect", "encode", "suffixes"}


def test_the_default_attack_pool_is_123_techniques_in_21_families():
    pool = technique_pool(resolve(None, "adversarial"))
    assert len(pool) == 123     # noqa: PLR2004
    assert len({fn.families[0] for fn in pool}) == 21     # noqa: PLR2004


def test_the_natural_kind_draws_only_what_the_spec_marks_natural():
    natural = {fn.__name__ for fn in all_techniques() if "natural" in fn.kinds}
    assert natural == {*NATURAL_POOL, *_family("translation")}
    # Translation is natural too, when the kind's switch is on.
    pool = technique_pool(resolve({"natural": {"translation": True}}, "natural"))
    assert sorted(fn.__name__ for fn in pool) == sorted(natural)
    # A family allowed whole gives its natural techniques only.
    typos = resolve({"natural": {"families": {"allow": ["typos"]},
                                 "techniques": {"allow": None}}}, "natural")
    assert [fn.__name__ for fn in technique_pool(typos)] == [
        "to_rewrite_with_typos_low", "to_rewrite_with_typos_medium"]
    for name in ("to_rewrite_with_typos_high", "to_rewrite_with_typos_insane",
                 "to_answer_as_dialogue", "to_answer_as_story", "to_json",
                 "to_use_past_tense", "to_continue_passage"):
        assert name not in natural


def test_every_technique_is_adversarial_and_a_technique_s_kinds_win(monkeypatch):
    from redact.jailbreak.technique import DEFAULT_KINDS, family_kinds

    assert all("adversarial" in fn.kinds for fn in all_techniques())
    assert family_kinds("encode") == list(DEFAULT_KINDS) == ["adversarial"]
    by_name = registry(all_techniques())
    assert by_name["to_leetspeak_basic"].kinds == ["adversarial", "natural"]
    assert family_kinds("typos") == ["adversarial", "natural"]
    assert by_name["to_rewrite_with_typos_high"].kinds == ["adversarial"]
    # A kind draws only what its kinds name: a technique left out of the
    # adversarial kind would leave its pool.
    natural_only = by_name["to_lowercase"]
    monkeypatch.setattr(natural_only, "kinds", ["natural"])
    pool = technique_pool(_kind(families={"allow": ["casual_writing"]}))
    assert natural_only not in pool and pool
    assert natural_only in technique_pool(resolve(None, "natural"))


@pytest.mark.parametrize("kinds", [[], ["natural", "natural"], ["attack"], "natural"])
def test_kinds_that_are_not_a_list_of_kinds_raise_at_tagging(monkeypatch, kinds):
    import redact.jailbreak.technique as T

    spec = load_spec()
    techniques = {**spec["techniques"],
                  "to_base64": {**spec["techniques"]["to_base64"], "kinds": kinds}}
    monkeypatch.setattr(T, "load_spec", lambda: {**spec, "techniques": techniques})

    def to_base64(prompt):
        return prompt, ""

    with pytest.raises(ValueError, match="the kinds of technique 'to_base64' must be "
                       r"a non-empty list of \['adversarial', 'natural'\]"):
        T.tag([to_base64])


def test_registry_names_every_technique_it_is_given():
    techniques = all_techniques()
    by_name = registry(techniques)
    assert list(by_name) == [fn.__name__ for fn in techniques]
    assert all(by_name[fn.__name__] is fn for fn in techniques)


def _languages(pool: list) -> list[str]:
    return [fn.__name__ for fn in _in("translation", pool)]


def test_the_switch_admits_only_the_translations_an_allow_list_names():
    pool = technique_pool(_kind(translation=True, families={"allow": ["encode"]},
                                techniques={"allow": ["to_swahili", "to_thai"]}))
    assert sorted(_languages(pool)) == ["to_swahili", "to_thai"]
    assert _in("encode", pool)


def test_the_switch_admits_the_whole_family_when_no_list_names_one_of_it():
    every = _languages(all_techniques())
    pool = technique_pool(_kind(translation=True, families={"allow": ["encode"]},
                                techniques={"allow": ["to_base64"]}))
    assert _languages(pool) == every
    # A list that names the family admits the family.
    pool = technique_pool(_kind(
        translation=True, families={"allow": ["translation"]},
        techniques={"allow": ["to_thai"]}))
    assert _languages(pool) == every


def test_the_eval_pool_is_the_37_pinned_techniques_in_10_families():
    """``eval.json`` allows whole families, so a technique added to one of
    them joins the eval pool unless ``eval.json`` denies it. This fails until
    the owner decides, in ``eval.json`` and ``fixtures/eval_pool.json``."""
    from redact.recipe import load_dataset_type

    section = load_dataset_type("eval")["augmentations"]
    pool = technique_pool(resolve(section, "adversarial"))
    assert sorted(fn.__name__ for fn in pool) == EVAL_POOL
    assert len(pool) == 37      # noqa: PLR2004
    assert {fn.families[0] for fn in pool} == {
        "translation", "encode", "fsh", "dap", "framing", "structural",
        "ascii_art", "continuation", "temporal", "answer_format"}
    assert sorted(_languages(pool)) == sorted([
        "to_swahili", "to_thai", "to_khmer", "to_maori", "to_nepali", "to_zulu",
        "to_scots_gaelic", "to_bengali", "to_javanese"])
    assert sorted(fn.__name__ for fn in _in("encode", pool)) == sorted([
        "to_base64", "to_unicode_escape", "to_ascii_ordinal", "to_morse",
        "to_braille"])


def test_uses_translation():
    assert uses_translation(technique_pool(_kind(translation=True)))
    assert not uses_translation(technique_pool(_kind()))
    assert not uses_translation(technique_pool(_kind(families={"deny": []})))
    assert not uses_translation(technique_pool(
        _kind(translation=True, layers={"deny": ["obfuscation"]})))


def test_the_registered_techniques_match_the_spec():
    JPool._check_against_spec(all_techniques())


@pytest.mark.parametrize(("drop_from", "missing"), [
    ("spec", "registered but not in the spec: to_base64"),
    ("registry", "in the spec but not registered: to_base64"),
])
def test_a_name_the_spec_and_the_registry_do_not_share_raises(
        monkeypatch, drop_from, missing):
    spec = load_spec()
    techniques = all_techniques()
    if drop_from == "spec":
        trimmed = {k: v for k, v in spec["techniques"].items() if k != "to_base64"}
        monkeypatch.setattr(JPool, "load_spec", lambda: {**spec, "techniques": trimmed})
    else:
        techniques = [fn for fn in techniques if fn.__name__ != "to_base64"]

    with pytest.raises(ValueError, match=missing):
        JPool._check_against_spec(techniques)
