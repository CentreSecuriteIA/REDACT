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
from redact.jailbreak.rewrite import get_all_rewrite_functions
from redact.jailbreak.technique import load_spec

#: The eval type's attack pool, sorted: ``fixtures/eval_pool.json``.
EVAL_POOL = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "eval_pool.json").read_text(
        encoding="utf-8"))
#: The natural kind's default pool, sorted: ``fixtures/natural_pool.json``.
NATURAL_POOL = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "natural_pool.json").read_text(
        encoding="utf-8"))


def _in(family: str, pool: list) -> list:
    return [fn for fn in pool if family in fn.families]


def test_all_techniques_are_the_five_layers_in_registry_order():
    assert all_techniques() == (
        get_all_obfuscation_functions() + get_all_hacking_functions()
        + get_all_manipulation_functions() + get_all_request_functions()
        + get_all_rewrite_functions())


def _kind(**given) -> dict:
    return resolve({"adversarial": given}, "adversarial")


def _natural_only(techniques: list) -> list:
    """The techniques of ``techniques`` no attack may draw: their kinds are
    ``["natural"]`` (small talk, the benign roles)."""
    return [fn for fn in techniques if "adversarial" not in fn.kinds]


def test_the_default_attack_pool_is_every_active_technique_but_casual():
    """Translation is on in the packaged attacks, served by the local
    translator. A retired technique is drawn only when named, a natural-only
    one never, and the rewrite layer, off by default, only when a section
    allows it."""
    full = all_techniques()
    left_out = (_in("casual_writing", full) + [fn for fn in full if fn.deprecated]
                + _natural_only(full) + _rewrites(full))
    assert technique_pool(_kind()) == [fn for fn in full if fn not in left_out]


def test_the_packaged_natural_kind_keeps_translation_off():
    from redact.jailbreak.augmentation import resolve

    assert resolve(None, "adversarial")["translation"] is True
    assert resolve(None, "natural")["translation"] is False


def _rewrites(techniques: list) -> list:
    """The techniques of the rewrite layer, which the spec has off by default."""
    return [fn for fn in techniques if fn.layer == "rewrite"]


def _attacks(techniques: list) -> list:
    """``techniques`` less casual writing, which no attack draws unless
    allowed, the natural-only families, which no attack draws, the retired
    techniques, which no list here names, and the rewrite layer, off by
    default."""
    return [fn for fn in techniques
            if "casual_writing" not in fn.families and not fn.deprecated
            and "adversarial" in fn.kinds and fn.layer != "rewrite"]


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
        translation=False,
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


#: The noise of low and medium density, which the natural kind draws since
#: spec 1.2; the high density is the attacks' alone.
_MILD_NOISE = [f"to_{noise}_{density}" for noise in (
    "random_capitals", "scrambled_letters", "keyboard_typos", "emoji_sprinkle")
    for density in ("low", "medium")]


def test_the_default_variation_pool_is_the_72_pinned_natural_techniques():
    """A technique added to a family the spec marks natural takes its
    family's kinds, so it joins the natural pool unless its entry says
    ``["adversarial"]``. This fails until it is decided, in the spec and
    ``fixtures/natural_pool.json``. Spec 1.2 added six answer languages and
    the natural-only small talk (2) and benign roles (3). Spec 1.3 added one:
    ``to_ask_with_urgency``, which was decided to be natural; its three
    answer formats say ``["adversarial"]``."""
    pool = technique_pool(resolve(None, "natural"))
    assert sorted(fn.__name__ for fn in pool) == NATURAL_POOL
    assert len(pool) == 72      # noqa: PLR2004
    # Family by family: two typos, seven whole families, the asking family
    # whole, four answer formats, the mild noise, and three techniques of
    # attack families.
    listed = sorted([
        "to_rewrite_with_typos_low", "to_rewrite_with_typos_medium",
        *_family("casual_writing", "neutral_container", "answer_language",
                 "indirect", "smalltalk_prefix", "smalltalk_suffix",
                 "benign_role"),
        *_family("asking"),
        "to_answer_in_markdown", "to_answer_as_tutorial", "to_answer_as_code",
        "to_answer_as_structured_data", *_MILD_NOISE,
        "to_leetspeak_basic", "to_adversarial_suffix_punctuation",
        "to_adversarial_suffix_emoji"])
    assert listed == NATURAL_POOL
    assert {fn.families[0] for fn in pool} == {
        "typos", "casual_writing", "noise", "neutral_container",
        "answer_language", "answer_format", "asking", "indirect",
        "smalltalk_prefix", "smalltalk_suffix", "benign_role", "encode",
        "suffixes"}


def test_the_default_attack_pool_is_215_techniques_in_29_families():
    """Every active technique but casual writing and the natural-only
    families: the 23 translations included, now the packaged adversarial
    kind has its translation switch on (the four spec 1.3 ones are Hmong,
    Chinese, Hindi and Amharic); the seven spec 1.1 retired left out,
    the fifteen spec 1.2 encodings, the twelve spec 1.2 noise techniques, the
    six spec 1.2 neutral containers, the six spec 1.2 answer languages, the
    four spec 1.2 many-shot fsh and dap variants, the eleven spec 1.3
    persona templates, the three spec 1.3 answer formats, the three spec 1.3
    fake turns, the seven spec 1.3 members of the thin request families and
    the two spec 1.3 payload splits, the spec 1.3 word art and the three
    spec 1.3 persona archetypes and the two spec 1.3 wording techniques in,
    and spec 1.4's letter spacing. Small talk and the benign roles leave it
    as it was."""
    pool = technique_pool(resolve(None, "adversarial"))
    assert len(pool) == 215     # noqa: PLR2004
    assert len({fn.families[0] for fn in pool}) == 29     # noqa: PLR2004


def test_the_rewrite_layer_is_off_by_default_and_layers_allow_adds_its_16_techniques():
    """The spec has the layer off by default, so the default attack pool is
    the 215 and no other pool holds a rewrite technique; a section that allows
    the layer gains the 16 of its three families, and nothing else leaves or
    joins."""
    default = technique_pool(resolve(None, "adversarial"))
    allowed = technique_pool(resolve(
        {"adversarial": {"layers": {"allow": ["rewrite"]}}}, "adversarial"))
    gained = [fn for fn in allowed if fn not in default]
    assert (len(default), len(allowed), len(gained)) == (215, 231, 16)   # noqa: PLR2004
    assert gained == _rewrites(all_techniques())
    assert {fn.families[0] for fn in gained} == {
        "persuasion", "tense_rewrite", "model_distractors"}
    assert all(fn in allowed for fn in default)
    assert not _rewrites(technique_pool(resolve(None, "natural")))


def test_the_natural_kind_and_the_eval_type_draw_no_rewrite_technique_though_allowed():
    """Every rewrite technique says ``["adversarial"]``, so allowing the layer
    in the natural kind would turn nothing on, and raises; ``eval.json``
    names families and techniques, none of the rewrite layer."""
    from redact.recipe import load_dataset_type

    with pytest.raises(ValueError, match="so it turns nothing on"):
        resolve({"natural": {"layers": {"allow": ["rewrite"]}}}, "natural")
    section = load_dataset_type("eval")["augmentations"]
    allowed = resolve({"adversarial": {"layers": {"allow": ["rewrite"]}}},
                      "adversarial", base=section)
    assert sorted(fn.__name__ for fn in technique_pool(allowed)) == EVAL_POOL


def test_the_natural_kind_draws_only_what_the_spec_marks_natural():
    natural = {fn.__name__ for fn in all_techniques() if "natural" in fn.kinds}
    assert natural == {*NATURAL_POOL, *_family("translation")}
    # Translation is natural too, when the kind's switch is on.
    pool = technique_pool(resolve({"natural": {"translation": True}}, "natural"))
    assert sorted(fn.__name__ for fn in pool) == sorted(natural)
    assert (len(NATURAL_POOL), len(pool)) == (72, 95)     # noqa: PLR2004
    # A family allowed whole gives its natural techniques only.
    typos = resolve({"natural": {"families": {"allow": ["typos"]},
                                 "techniques": {"allow": None}}}, "natural")
    assert [fn.__name__ for fn in technique_pool(typos)] == [
        "to_rewrite_with_typos_low", "to_rewrite_with_typos_medium"]
    for name in ("to_rewrite_with_typos_high", "to_rewrite_with_typos_insane",
                 "to_answer_as_dialogue", "to_answer_as_story", "to_json",
                 "to_use_past_tense", "to_continue_passage",
                 "to_random_capitals_high", "to_scrambled_letters_high",
                 "to_keyboard_typos_high", "to_emoji_sprinkle_high",
                 "to_dual_answer", "to_tagged_prefix", "to_answer_in_encoding"):
        assert name not in natural
    assert "to_ask_with_urgency" in natural


def test_only_the_natural_only_families_are_not_adversarial(monkeypatch):
    """And a technique's own kinds win over its family's."""
    from redact.jailbreak.technique import DEFAULT_KINDS, family_kinds

    assert {fn.families[0] for fn in _natural_only(all_techniques())} == {
        "smalltalk_prefix", "smalltalk_suffix", "benign_role"}
    assert family_kinds("benign_role") == ["natural"]
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


@pytest.mark.parametrize("family", ["smalltalk_prefix", "smalltalk_suffix",
                                    "benign_role"])
def test_an_attack_allow_list_naming_a_natural_only_family_raises(family):
    with pytest.raises(ValueError, match=f"names '{family}', which the "
                       "adversarial kind does not draw"):
        _kind(families={"allow": [family]})


def _tag_base64(monkeypatch, spec_version="1.0", **entry):
    """A ``to_base64`` tagged from a spec at ``spec_version`` whose entry for
    it has ``entry`` over its own."""
    import redact.jailbreak.technique as T

    spec = load_spec()
    techniques = {**spec["techniques"],
                  "to_base64": {**spec["techniques"]["to_base64"], **entry}}
    monkeypatch.setattr(T, "load_spec", lambda: {
        **spec, "version": spec_version, "techniques": techniques})

    def to_base64(prompt):
        return prompt, ""

    T.tag([to_base64])
    return to_base64


@pytest.mark.parametrize("kinds", [[], ["natural", "natural"], ["attack"], "natural"])
def test_kinds_that_are_not_a_list_of_kinds_raise_at_tagging(monkeypatch, kinds):
    with pytest.raises(ValueError, match="the kinds of technique 'to_base64' must be "
                       r"a non-empty list of \['adversarial', 'natural'\]"):
        _tag_base64(monkeypatch, kinds=kinds)


def test_needs_latin_is_false_unless_the_entry_says_true(monkeypatch):
    assert _tag_base64(monkeypatch).needs_latin is False
    assert _tag_base64(monkeypatch, needs_latin=True).needs_latin is True


@pytest.mark.parametrize("needs_latin", ["true", 1, None])
def test_a_needs_latin_that_is_not_true_or_false_raises_at_tagging(
        monkeypatch, needs_latin):
    with pytest.raises(ValueError,
                       match="'to_base64': needs_latin must be true or false"):
        _tag_base64(monkeypatch, needs_latin=needs_latin)


@pytest.mark.parametrize(("entry", "message"), [
    ({"families": []}, "families must be a non-empty list of families of "
                       "combination_spec.json, got []"),
    ({"families": ["encodez"]}, "families must be a non-empty list of families "
                                "of combination_spec.json, got ['encodez']"),
    ({"complexity": 4}, "complexity must be a whole number from 0 to 3, got 4"),
    ({"complexity": "high"}, "complexity must be a whole number from 0 to 3, got "
                             "'high'"),
    ({"complexity": True}, "complexity must be a whole number from 0 to 3, got True"),
    ({"requires_llm": "yes"}, "requires_llm must be true or false, got 'yes'"),
    ({"requires_benign": 1}, "requires_benign must be true or false, got 1"),
    ({"encode_weight": "medium"}, 'encode_weight must be "light", "heavy" or '
                                  "absent, got 'medium'"),
    ({"max_tokens": True}, "max_tokens must be a whole number of at least 1, got "
                           "True"),
    ({"complexty": 2}, "has unknown key(s) ['complexty'] in combination_spec.json"),
], ids=["no family", "unknown family", "complexity 4", "complexity text",
        "complexity bool", "requires_llm text", "requires_benign number",
        "encode_weight", "max_tokens bool", "unknown key"])
def test_a_spec_entry_that_breaks_a_rule_raises_at_tagging(monkeypatch, entry, message):
    """At import, not when a plan is made or a family is misread: a family
    that does not exist used to put the technique in the obfuscation layer,
    and a complexity of 7 to leave it undrawn."""
    with pytest.raises(ValueError) as raised:
        _tag_base64(monkeypatch, **entry)
    assert str(raised.value).startswith("Technique 'to_base64'")
    assert message in str(raised.value)


def test_check_and_calls_come_from_the_first_family_unless_the_entry_says():
    by_name = registry(all_techniques())
    # A family's: the sensitive-word encodings take sensitive_words', their
    # first family, not encode's (which has none).
    encoded = by_name["to_sensitive_words_encode_base64"]
    assert (encoded.check, encoded.calls) == ("word_list", [1, 1])
    assert by_name["to_swahili"].check == "translate"
    # The entry's own: the synonyms are one call per marked word.
    assert by_name["to_synonym_substitution"].calls == [1, 11]
    # A pure technique, and fsh's that picks its pairs without a check.
    assert (by_name["to_base64"].check, by_name["to_base64"].calls) == (None, None)
    picks = by_name["to_fsh_selected_short"]
    assert (picks.check, picks.calls) == (None, [1, 1])
    assert by_name["to_fsh_random_short"].calls is None


@pytest.mark.parametrize("calls, tagged", [(1, [1, 1]), ([2, 4], [2, 4])])
def test_calls_is_a_whole_number_or_a_low_high_pair(monkeypatch, calls, tagged):
    assert _tag_base64(monkeypatch, calls=calls).calls == tagged


@pytest.mark.parametrize("calls", [0, [3, 2], [1], [1, 2, 3], "2", True, [1.0, 2]])
def test_calls_that_are_not_whole_numbers_raise_at_tagging(monkeypatch, calls):
    with pytest.raises(ValueError, match="'to_base64': calls must be a whole number"):
        _tag_base64(monkeypatch, calls=calls)


def test_a_check_not_among_the_checks_raises_at_tagging(monkeypatch):
    assert _tag_base64(monkeypatch, check="word_list").check == "word_list"
    assert _tag_base64(monkeypatch, check="preserve").check == "preserve"
    with pytest.raises(ValueError, match="'to_base64': check must be one of "
                       r"\['typos', 'scenario', 'word_list', 'translate', "
                       r"'preserve'\]"):
        _tag_base64(monkeypatch, check="typo")


def test_a_technique_is_dated_by_the_versions_that_added_and_retired_it(
        monkeypatch):
    dated = _tag_base64(monkeypatch, since="1.1", deprecated="1.12")
    assert (dated.since, dated.deprecated) == ("1.1", "1.12")
    # An entry without them: one the library had before the spec dated them.
    undated = _tag_base64(monkeypatch)
    assert (undated.since, undated.deprecated) == ("1.0", None)


@pytest.mark.parametrize("key", ["since", "deprecated"])
@pytest.mark.parametrize("version", [1.1, "v1.1", "1.1.", "1..1", "", None])
def test_a_since_or_deprecated_that_is_not_dotted_whole_numbers_raises_at_tagging(
        monkeypatch, key, version):
    with pytest.raises(ValueError, match=f"the {key} of technique 'to_base64' must "
                       "be a version of dotted whole numbers"):
        _tag_base64(monkeypatch, **{key: version})


def test_a_spec_version_that_is_not_dotted_whole_numbers_raises_at_tagging(
        monkeypatch):
    with pytest.raises(ValueError, match="the version of the spec must be"):
        _tag_base64(monkeypatch, spec_version="1.1-beta")


def test_versions_compare_as_whole_numbers():
    from redact.jailbreak.technique import version_key

    assert version_key("1.10") > version_key("1.9") > version_key("1.1")
    assert version_key("2") > version_key("1.99")
    assert version_key("1.0.1") > version_key("1.0")


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


def test_the_eval_pool_is_the_36_pinned_techniques_in_10_families():
    """``eval.json`` allows whole families, so a technique added to one of
    them joins the eval pool unless ``eval.json`` denies it. This fails until
    it is decided, in ``eval.json`` and ``fixtures/eval_pool.json``."""
    from redact.recipe import load_dataset_type

    section = load_dataset_type("eval")["augmentations"]
    pool = technique_pool(resolve(section, "adversarial"))
    assert sorted(fn.__name__ for fn in pool) == EVAL_POOL
    assert len(pool) == 36      # noqa: PLR2004
    assert {fn.families[0] for fn in pool} == {
        "translation", "encode", "fsh", "dap", "framing", "structural",
        "ascii_art", "continuation", "temporal", "answer_format"}
    assert sorted(_languages(pool)) == sorted([
        "to_swahili", "to_thai", "to_khmer", "to_nepali", "to_zulu",
        "to_scots_gaelic", "to_bengali", "to_javanese"])
    assert sorted(fn.__name__ for fn in _in("encode", pool)) == sorted([
        "to_base64", "to_unicode_escape", "to_ascii_ordinal", "to_morse",
        "to_braille"])


#: The eval techniques that call a model with no check: they ask it only
#: which benign subcategory's pairs to show, and keep none of its text.
PICKS_PAIRS_ONLY = {
    f"to_{family}_selected_{size}"
    for family in ("fsh", "dap") for size in ("short", "long", "many")}


def test_every_eval_technique_that_has_a_model_write_the_prompt_is_checked():
    by_name = registry(all_techniques())
    unchecked = [name for name in EVAL_POOL
                 if by_name[name].requires_llm and by_name[name].check is None
                 and name not in PICKS_PAIRS_ONLY]
    assert unchecked == []


def test_the_eval_pool_holds_no_many_shot_variant():
    """``eval.json`` denies the four, so fsh and dap keep their eight short
    and long variants there."""
    from redact.recipe import load_dataset_type

    section = load_dataset_type("eval")["augmentations"]
    pool = technique_pool(resolve(section, "adversarial"))
    assert sorted(fn.__name__ for fn in _in("fsh", pool) + _in("dap", pool)) == sorted(
        f"to_{family}_{form}_{size}" for family in ("fsh", "dap")
        for form in ("random", "selected") for size in ("short", "long"))
    assert {"to_fsh_random_many", "to_fsh_selected_many", "to_dap_random_many",
            "to_dap_selected_many"} <= set(section["adversarial"]["techniques"]["deny"])


def test_uses_translation():
    assert uses_translation(technique_pool(_kind(translation=True)))
    assert uses_translation(technique_pool(_kind()))
    assert not uses_translation(technique_pool(_kind(translation=False)))
    assert not uses_translation(technique_pool(
        _kind(translation=False, families={"deny": []})))
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


# --- a row's families and layers ------------------------------------------------

def _registry() -> dict:
    return JPool.registry(JPool.all_techniques())


def test_chain_tags_give_each_technique_its_first_family_and_layer_in_order():
    """``to_sensitive_words_encode_base64`` is a ``sensitive_words`` and an
    ``encode`` technique: its first family is the one its row names."""
    chain = "to_hypothetical_framing+to_sensitive_words_encode_base64+to_rot13"
    assert JPool.chain_tags(chain, _registry()) == (
        "cognitive+sensitive_words+encode", "hacking+obfuscation+obfuscation")


def test_chain_tags_read_a_retired_technique_as_any_other():
    registry = _registry()
    assert registry["to_politician_persona"].deprecated == "1.1"
    assert JPool.chain_tags("to_politician_persona", registry) == (
        "persona", "hacking")


@pytest.mark.parametrize("chain, expected", [
    ("to_not_a_technique", ("?to_not_a_technique", "?")),
    ("to_rot13+to_not_a_technique+to_summarize",
     ("encode+?to_not_a_technique+indirect", "obfuscation+?+requests")),
    ("", ("", "")),
    ("identity", ("", "")),
])
def test_chain_tags_mark_an_unregistered_name_and_never_raise(chain, expected):
    assert JPool.chain_tags(chain, _registry()) == expected
