"""The spec's rules, over the registered techniques: they bind both ways."""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.dataset.dealing import Dealer
from redact.jailbreak.hacking import get_all_hacking_functions
from redact.jailbreak.manipulation import get_all_manipulation_functions
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.requests import get_all_request_functions
from redact.jailbreak.rewrite import get_all_rewrite_functions
from redact.jailbreak.rules import compatible, request_obfuscation_candidates
from redact.jailbreak.sampling import (
    sample_combination,
    sample_exact_combination,
)

POOL = (get_all_obfuscation_functions() + get_all_hacking_functions()
        + get_all_manipulation_functions() + get_all_request_functions()
        + get_all_rewrite_functions())
BY_NAME = {fn.__name__: fn for fn in POOL}


def _joins(selected: str, candidate: str) -> bool:
    return BY_NAME[candidate] in compatible([BY_NAME[selected]], [BY_NAME[candidate]])


@pytest.mark.parametrize(("first", "then"), [
    ("to_fictional_world", "to_ascii_art"),
    ("to_ascii_art", "to_fictional_world"),
    ("to_fsh_random_short", "to_ascii_art"),
    ("to_ascii_art", "to_fsh_random_short"),
])
def test_a_cross_incompatibility_refuses_either_order(first, then):
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_psychopath_persona", "to_impersonate_good_person"),
    ("to_impersonate_good_person", "to_psychopath_persona"),
    ("to_invented_persona", "to_impersonate_good_person"),
    ("to_impersonate_good_person", "to_invented_persona"),
    ("to_avi", "to_impersonate_good_person"),
    ("to_impersonate_good_person", "to_avi"),
])
def test_a_persona_or_cognitive_scenario_refuses_an_impersonation_either_way(
        first, then):
    """Two identities in one prompt: the persona or scenario's, and the
    asker's claimed one."""
    assert not _joins(first, then)


@pytest.mark.parametrize("persona", ["to_dan_persona_template",
                                     "to_liberated_model_persona_template"])
@pytest.mark.parametrize("other", [
    "to_fsh_random_short", "to_dap_selected_long", "to_ascii_art",
    "to_impersonate_good_person"])
def test_a_persona_template_refuses_manipulation_ascii_art_and_impersonation(
        persona, other):
    """fsh and dap set their own scene, ascii_art cannot draw so long a
    frame, and an impersonation would be a second identity."""
    assert not _joins(persona, other)
    assert not _joins(other, persona)


@pytest.mark.parametrize("answer", ["to_refusal_then_comply", "to_prefix_injection",
                                    "to_dual_answer", "to_tagged_prefix"])
def test_a_persona_template_refuses_the_answer_formats_its_mechanics_ask_for(answer):
    """The mechanics already ask for a dual answer or a set opening: the four
    answer formats list the family, so every persona template refuses them."""
    assert not _joins("to_aim_persona_template", answer)
    assert not _joins(answer, "to_aim_persona_template")


@pytest.mark.parametrize("other", [
    "to_base64", "to_rot13", "to_json", "to_answer_as_code", "to_summarize",
    "to_use_past_tense"])
def test_a_persona_template_joins_an_encoding_a_wrapper_or_a_request(other):
    assert _joins("to_dan_persona_template", other)
    assert _joins(other, "to_dan_persona_template")


@pytest.mark.parametrize("encoded", [
    "to_base64", "to_rot13", "to_morse", "to_hex", "to_sensitive_words_encode_base64",
    "to_sensitive_words_encode_leetspeak_basic"])
def test_an_answer_in_an_encoding_refuses_an_encoded_prompt_either_way(encoded):
    """A decode-and-reply-encoded request on top of an encoded prompt: the
    ten sensitive-word encodings are encode members too."""
    assert not _joins("to_answer_in_encoding", encoded)
    assert not _joins(encoded, "to_answer_in_encoding")


@pytest.mark.parametrize("other", [
    "to_sensitive_words_split", "to_sensitive_words_variables",
    "to_synonym_substitution", "to_french", "to_json", "to_dan_persona_template"])
def test_an_answer_in_an_encoding_joins_what_encodes_nothing(other):
    """It names encode alone: the sensitive-word steps that encode no word,
    a translation, a wrapper and a persona template still join it."""
    assert _joins("to_answer_in_encoding", other)
    assert _joins(other, "to_answer_in_encoding")


def test_a_persona_template_is_the_one_hacking_technique_of_its_combination():
    for other in ("to_fictional_world", "to_psychopath_persona", "to_avi",
                  "to_stan_persona_template"):
        assert not _joins("to_dan_persona_template", other)


#: What the two frame families refuse, as families, written out from the
#: spec's rules: the layer's one pick, the cross lists, and for fake turns the
#: manipulation block. Every family not named here must still join them, so a
#: rule added by mistake shows as a pair that no longer joins.
_REFUSED_BY = {
    "to_dan_persona_template": {
        "persona_template", "persona", "cognitive", "framing",
        "fsh", "dap", "fake_turns", "ascii_art", "impersonation"},
    "to_fake_history": {
        "fake_turns", "fsh", "dap", "framing", "persona", "cognitive",
        "persona_template", "continuation", "indirect", "structural",
        "neutral_container", "ascii_art", "translation", "payload_split",
        "word_art", "wording", "unicode_smuggling"},
}
#: The four answer formats a persona template's own rules refuse; no other
#: technique of their family is refused.
_ANSWER_FORMATS_OF_THE_MECHANICS = {
    "to_refusal_then_comply", "to_prefix_injection", "to_dual_answer",
    "to_tagged_prefix"}


def _one_technique_of_each_family() -> dict[str, str]:
    """The first technique of each family, but a heavy encoding (which
    fake turns refuse) and the four answer formats above."""
    first: dict[str, str] = {}
    for fn in POOL:
        if (getattr(fn, "encode_weight", None) == "heavy"
                or fn.__name__ in _ANSWER_FORMATS_OF_THE_MECHANICS):
            continue
        first.setdefault(fn.families[0], fn.__name__)
    return first


_JOINING = [(frame, name) for frame, refused in _REFUSED_BY.items()
            for family, name in _one_technique_of_each_family().items()
            if family not in refused]


@pytest.mark.parametrize(("frame", "other"), _JOINING,
                         ids=[f"{frame}-{name}" for frame, name in _JOINING])
def test_a_frame_joins_a_technique_of_every_family_it_has_no_rule_against(frame, other):
    assert _joins(frame, other)
    assert _joins(other, frame)


def test_every_family_has_a_technique_to_try_and_every_refused_name_is_a_family():
    families = {fn.families[0] for fn in POOL}
    assert set(_one_technique_of_each_family()) == families
    assert all(refused <= families for refused in _REFUSED_BY.values())


@pytest.mark.parametrize("identity", ["to_impersonate_authority",
                                      "to_impersonate_developer"])
@pytest.mark.parametrize("scene", ["to_psychopath_persona", "to_avi",
                                   "to_dan_persona_template"])
def test_the_new_impersonations_are_refused_by_a_persona_or_scenario_either_way(
        identity, scene):
    """The family-level rule covers the spec 1.3 members with no entry of
    their own."""
    assert not _joins(identity, scene)
    assert not _joins(scene, identity)


@pytest.mark.parametrize("framing", ["to_fictional_world", "to_noble_goal"])
def test_a_framing_still_joins_an_impersonation(framing):
    assert _joins(framing, "to_impersonate_good_person")
    assert _joins("to_impersonate_good_person", framing)


def test_continuation_and_indirect_are_kept_apart_by_the_requests_cap_alone():
    """Both are request families, one a combination at most, so the spec
    names no rule of its own between them."""
    from redact.jailbreak.technique import load_spec

    families = load_spec()["families"]
    assert "indirect" not in families["continuation"]["cross_incompatible_with"]
    assert "continuation" not in families["indirect"]["cross_incompatible_with"]
    assert not _joins("to_continue_passage", "to_summarize")
    assert not _joins("to_summarize", "to_continue_passage")


def test_no_family_carries_a_pick_cap_of_its_own():
    """A combination takes one technique per family; the layers' caps are
    ``layer_max_picks``."""
    from redact.jailbreak.technique import load_spec

    assert not [name for name, family in load_spec()["families"].items()
                if "max_picks" in family]


@pytest.mark.parametrize(("first", "then"), [
    ("to_adversarial_suffix_emoji", "to_ascii_art"),
    ("to_ascii_art", "to_adversarial_suffix_emoji"),
    ("to_adversarial_suffix_unicode", "to_ascii_art"),
    ("to_ascii_art", "to_adversarial_suffix_unicode"),
    ("to_braille", "to_ascii_art"),
    ("to_ascii_art", "to_braille"),
    ("to_french", "to_ascii_art"),
    ("to_ascii_art", "to_swahili"),
])
def test_ascii_art_refuses_what_leaves_a_character_no_font_draws(first, then):
    """The emoji and unicode suffixes and braille, by the technique, and
    every translation: no packaged font has their characters."""
    assert not _joins(first, then)


def test_tokenbreak_split_and_ascii_art_are_never_planned_together():
    """to_ascii_art rejects what to_tokenbreak_split gives, so the spec keeps
    them apart, in either order and in either mode."""
    from redact.jailbreak.plan import assign_combination

    pair = {"to_tokenbreak_split", "to_ascii_art"}
    assert not _joins("to_tokenbreak_split", "to_ascii_art")
    assert not _joins("to_ascii_art", "to_tokenbreak_split")
    pool = [BY_NAME[name] for name in sorted(pair)]
    for seed in range(50):
        for mode in ({"exact_techniques": 2},
                     {"max_complexity": 9, "max_obfuscations": 2}):
            names = assign_combination(f"unit{seed}", pool, seed=seed, **mode)
            assert set(names) != pair, (seed, mode)


@pytest.mark.parametrize("technique", [
    "to_adversarial_suffix_punctuation", "to_adversarial_suffix_fragments",
    "to_rot47", "to_separator"])
def test_ascii_art_keeps_the_rest_of_their_families(technique):
    assert _joins(technique, "to_ascii_art") and _joins("to_ascii_art", technique)


_HEAVY_FULL_TEXT = [
    "to_base64", "to_unicode_escape", "to_ascii_ordinal", "to_morse", "to_braille",
    "to_hex", "to_binary", "to_html_entities", "to_base32", "to_base85"]


@pytest.mark.parametrize("technique", _HEAVY_FULL_TEXT)
def test_ascii_art_refuses_the_heavy_full_text_encodings_either_way(technique):
    """A code for every character lengthens the text, so its drawing often
    runs past the cap (unicode escapes are rejected from about 140
    characters of input), and both disguise the whole text: one is enough."""
    assert not _joins(technique, "to_ascii_art")
    assert not _joins("to_ascii_art", technique)


def test_every_heavy_full_text_encoding_the_spec_has_is_kept_from_ascii_art():
    """The ten of ``_HEAVY_FULL_TEXT`` and any heavy ``encode`` technique added
    later: a heavy member of ``encode`` alone (not the sensitive-word forms,
    which encode only marked words) names ``ascii_art`` in its own entry."""
    heavy = {fn.__name__ for fn in POOL
             if fn.families == ["encode"] and fn.encode_weight == "heavy"}
    assert heavy == set(_HEAVY_FULL_TEXT)
    assert [name for name in sorted(heavy) if _joins(name, "to_ascii_art")] == []


@pytest.mark.parametrize("technique", [
    "to_sensitive_words_encode_base64", "to_sensitive_words_encode_unicode",
    "to_sensitive_words_encode_ascii", "to_sensitive_words_encode_rot13",
    "to_sensitive_words_encode_separator"])
def test_ascii_art_keeps_the_sensitive_word_encodings_heavy_and_light(technique):
    """They encode only the words a model marks, so the text stays short
    and mostly plain: the heavy ones are not the heavy full-text encodings."""
    assert _joins(technique, "to_ascii_art") and _joins("to_ascii_art", technique)


@pytest.mark.parametrize("technique", [
    "to_rot13", "to_rot18", "to_leetspeak_basic", "to_reversed_characters",
    "to_url_encoding", "to_caesar_cipher", "to_atbash"])
def test_ascii_art_keeps_the_light_encodings_that_write_ascii(technique):
    assert _joins(technique, "to_ascii_art") and _joins("to_ascii_art", technique)


@pytest.mark.parametrize("technique", [
    "to_lookalike_letters", "to_zero_width_noise", "to_fullwidth",
    "to_math_styled"])
def test_ascii_art_refuses_the_non_ascii_encodings_either_way(technique):
    """Their output is non-ASCII (look-alike and math letters, zero-width
    noise, fullwidth forms), which no font draws."""
    assert not _joins(technique, "to_ascii_art")
    assert not _joins("to_ascii_art", technique)


@pytest.mark.parametrize("tokenbreak", [
    "to_tokenbreak_prepend", "to_tokenbreak_split", "to_tokenbreak_delimiter"])
def test_zero_width_noise_and_tokenbreak_are_never_planned_together(tokenbreak):
    """Both work on the characters between letters; the spec keeps them apart."""
    assert not _joins("to_zero_width_noise", tokenbreak)
    assert not _joins(tokenbreak, "to_zero_width_noise")


@pytest.mark.parametrize(("first", "then"), [
    ("to_keyboard_typos_low", "to_rewrite_with_typos_low"),
    ("to_rewrite_with_typos_high", "to_scrambled_letters_medium"),
    ("to_random_capitals_high", "to_lowercase"),
    ("to_uppercase", "to_random_capitals_low"),
    ("to_emoji_sprinkle_medium", "to_dropped_spaces"),
    ("to_abbreviations", "to_emoji_sprinkle_low"),
])
def test_noise_refuses_the_typos_and_casual_writing_either_way(first, then):
    """Three kinds of surface noise on one text: one is enough."""
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_emoji_sprinkle_low", "to_ascii_art"),
    ("to_ascii_art", "to_emoji_sprinkle_medium"),
    ("to_emoji_sprinkle_high", "to_ascii_art"),
])
def test_ascii_art_refuses_the_emoji_sprinkle_either_way(first, then):
    """No font has a glyph for an emoji."""
    assert not _joins(first, then)


@pytest.mark.parametrize("technique", [
    "to_random_capitals_high", "to_scrambled_letters_high", "to_keyboard_typos_high"])
def test_ascii_art_keeps_the_noise_of_latin_letters(technique):
    assert _joins(technique, "to_ascii_art") and _joins("to_ascii_art", technique)


_CONTAINERS = ["to_code_comment", "to_yaml_document", "to_csv_row",
               "to_email_message", "to_transcript_line", "to_log_entry"]


@pytest.mark.parametrize("container", _CONTAINERS)
@pytest.mark.parametrize("other", ["to_json", "to_xml", "to_markdown", "to_ascii_art"])
def test_a_neutral_container_refuses_structural_and_ascii_art_either_way(
        container, other):
    """One document around the prompt is enough, and a container's lines
    are too long for a font to draw."""
    assert not _joins(container, other)
    assert not _joins(other, container)


@pytest.mark.parametrize("other", ["to_base64", "to_french", "to_summarize"])
def test_a_neutral_container_joins_an_encoding_a_translation_or_a_request(other):
    assert _joins("to_csv_row", other) and _joins(other, "to_csv_row")


@pytest.mark.parametrize(("first", "then"), [
    ("to_hex", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_hex"),
    ("to_base32", "to_dap_random_long"),
    ("to_dap_random_long", "to_base85"),
])
def test_the_new_heavy_encodings_are_blocked_from_manipulation(first, then):
    """Heavy full-text encodings leave fsh/dap's benign pairs unreadable."""
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_caesar_cipher", "to_thai"),
    ("to_thai", "to_caesar_cipher"),
    ("to_atbash", "to_japanese"),
    ("to_japanese", "to_atbash"),
])
def test_a_new_latin_only_cipher_refuses_a_translation_into_another_script(
        first, then):
    """After the translation the cipher would change nothing; before it, the
    translator would get a cipher."""
    assert not _joins(first, then)


@pytest.mark.parametrize("technique", [
    "to_adversarial_suffix_punctuation", "to_adversarial_suffix_fragments",
    "to_rot47", "to_separator"])
def test_ascii_art_draws_a_font_for_the_rest_of_their_families(technique):
    from redact.jailbreak.obfuscation.ascii_art import to_ascii_art

    for seed in range(20):
        rng = random.Random(seed)
        fn = BY_NAME[technique]
        text, _ = (fn("plain words about the weather", rng=rng)
                   if "rng" in fn.__code__.co_varnames else fn("plain words"))
        _art, info = to_ascii_art(text, rng=rng)
        assert info.startswith("font="), (technique, seed, info)


_MANY_SHOT = ["to_fsh_random_many", "to_fsh_selected_many",
              "to_dap_random_many", "to_dap_selected_many"]


@pytest.mark.parametrize("many", _MANY_SHOT)
@pytest.mark.parametrize(
    "typos", ["to_rewrite_with_typos_low", "to_rewrite_with_typos_insane"])
def test_a_many_shot_variant_refuses_the_typos_either_way(many, typos):
    """A model rewriting so long a prompt would not return it whole."""
    assert not _joins(many, typos)
    assert not _joins(typos, many)


@pytest.mark.parametrize("many", _MANY_SHOT)
@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_french", "to_base64", "to_hex",
    "to_fictional_world", "to_psychopath_persona", "to_persona_roleplay",
    "to_continue_passage", "to_summarize", "to_fsh_random_short"])
def test_a_many_shot_variant_keeps_the_manipulation_layers_rules(many, other):
    """The blocked obfuscations, the hacking and request families fsh and dap
    refuse, and a second manipulation technique."""
    assert not _joins(many, other)
    assert not _joins(other, many)


@pytest.mark.parametrize("many", _MANY_SHOT)
@pytest.mark.parametrize("other", ["to_rot13", "to_sensitive_words_encode_base64",
                                   "to_refusal_then_comply"])
def test_a_many_shot_variant_joins_what_fsh_and_dap_join(many, other):
    assert _joins(many, other) and _joins(other, many)


def test_a_technique_names_only_families_it_is_incompatible_with():
    from redact.jailbreak.technique import load_spec

    spec = load_spec()
    named = {family for name, entry in spec["techniques"].items()
             if not name.startswith("_")
             for family in entry.get("cross_incompatible_with", [])}
    assert named and named <= set(spec["families"])


def test_a_family_names_only_families_the_spec_has():
    from redact.jailbreak.technique import load_spec

    families = load_spec()["families"]
    named = {other for entry in families.values()
             for other in entry["cross_incompatible_with"]}
    assert named and named <= set(families)


_FAKE_TURNS = ["to_fake_system_turn", "to_fake_assistant_turn", "to_fake_history"]


@pytest.mark.parametrize("fake", _FAKE_TURNS)
@pytest.mark.parametrize("other", [
    "to_continue_passage", "to_summarize", "to_fictional_world",
    "to_psychopath_persona", "to_avi", "to_dan_persona_template",
    "to_json", "to_markdown", "to_csv_row", "to_email_message"])
def test_a_fake_turn_refuses_what_fsh_and_dap_refuse_and_a_wrapper_either_way(
        fake, other):
    """The fsh/dap list (continuation, indirect, framing, persona, cognitive),
    the persona templates, and the structural and neutral containers: an
    invented turn was a conversation, not a document to wrap. That reason
    went with the old order (the manipulation layer applies last, so the
    turn stays outside a wrapper); the refusal is kept as it was, and this
    test holds it."""
    assert not _joins(fake, other)
    assert not _joins(other, fake)


@pytest.mark.parametrize("fake", _FAKE_TURNS)
@pytest.mark.parametrize("other", [
    "to_french", "to_thai", "to_ascii_art", "to_base64", "to_morse", "to_hex"])
def test_a_fake_turn_takes_the_manipulation_layer_s_obfuscation_block(fake, other):
    """manipulation_blocked_obfuscation reads the manipulation layer, so the
    fake turns are kept from translation, ascii_art and the heavy
    full-text encodings, as fsh and dap are."""
    assert not _joins(fake, other)
    assert not _joins(other, fake)


@pytest.mark.parametrize("fake", _FAKE_TURNS)
@pytest.mark.parametrize("other", ["to_fsh_random_short", "to_dap_selected_long",
                                   "to_fake_system_turn"])
def test_a_fake_turn_is_the_one_manipulation_of_its_combination(fake, other):
    assert not _joins(fake, other)


@pytest.mark.parametrize("fake", _FAKE_TURNS)
@pytest.mark.parametrize("other", [
    "to_rot13", "to_sensitive_words_encode_base64", "to_rewrite_with_typos_low",
    "to_answer_as_code", "to_refusal_then_comply", "to_impersonate_good_person"])
def test_a_fake_turn_joins_a_light_or_word_level_encoding_typos_or_a_request(
        fake, other):
    assert _joins(fake, other) and _joins(other, fake)


_PAYLOAD_SPLITS = ["to_payload_split", "to_payload_split_semantic"]


@pytest.mark.parametrize("split", _PAYLOAD_SPLITS)
@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_base64", "to_rot13", "to_sensitive_words_encode_base64",
    "to_fake_system_turn", "to_fake_history", "to_payload_split",
    "to_payload_split_semantic"])
def test_a_payload_split_refuses_ascii_art_encode_and_fake_turns_either_way(
        split, other):
    """Parts a font or a cipher would break apart, an invented turn already
    a conversation, and one cut of the prompt is enough (a family holds one
    technique)."""
    assert not _joins(split, other)
    assert not _joins(other, split)


@pytest.mark.parametrize("split", _PAYLOAD_SPLITS)
@pytest.mark.parametrize("other", [
    "to_fsh_random_short", "to_fsh_selected_long", "to_dap_random_long",
    "to_dap_selected_short"])
def test_a_payload_split_takes_the_manipulation_layer_s_obfuscation_block(
        split, other):
    """manipulation_blocked_obfuscation names it: the benign pairs of fsh and
    dap would sit between the parts and the request to join them."""
    assert not _joins(split, other)
    assert not _joins(other, split)


@pytest.mark.parametrize("split", _PAYLOAD_SPLITS)
@pytest.mark.parametrize("other", [
    "to_rewrite_with_typos_low", "to_tokenbreak_split",
    "to_sensitive_words_split", "to_json", "to_csv_row", "to_summarize",
    "to_psychopath_persona", "to_fictional_world"])
def test_a_payload_split_joins_typos_a_marking_step_a_wrapper_a_request_or_a_scene(
        split, other):
    assert _joins(split, other) and _joins(other, split)


@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_payload_split", "to_payload_split_semantic", "to_base64",
    "to_rot13", "to_sensitive_words_split", "to_sensitive_words_encode_base64",
    "to_tokenbreak_split", "to_json", "to_code_comment", "to_fake_system_turn"])
def test_word_art_refuses_what_draws_or_encodes_the_text_either_way(other):
    """A font on the whole text, another split, an encoding, another word-level
    step on the word, a wrapper and an invented turn: one disguise of the
    word is enough, and the art is long."""
    assert not _joins("to_word_art", other)
    assert not _joins(other, "to_word_art")


@pytest.mark.parametrize("other", [
    "to_fsh_random_short", "to_fsh_selected_long", "to_dap_random_long",
    "to_dap_selected_short"])
def test_word_art_takes_the_manipulation_layer_s_obfuscation_block(other):
    """manipulation_blocked_obfuscation names it: the art is long and the benign
    pairs of fsh and dap would sit between it and the request."""
    assert not _joins("to_word_art", other)
    assert not _joins(other, "to_word_art")


@pytest.mark.parametrize("other", [
    "to_rewrite_with_typos_low", "to_adversarial_suffix_punctuation",
    "to_random_capitals_low", "to_lowercase", "to_summarize",
    "to_psychopath_persona", "to_fictional_world"])
def test_word_art_joins_typos_noise_a_suffix_a_request_or_a_scene(other):
    assert _joins("to_word_art", other) and _joins(other, "to_word_art")


@pytest.mark.parametrize("translation", ["to_french", "to_thai", "to_hindi"])
@pytest.mark.parametrize("marking", [
    "to_sensitive_words_split", "to_sensitive_words_encode_base64",
    "to_sensitive_words_variables", "to_synonym_substitution",
    "to_tokenbreak_split", "to_tokenbreak_prepend", "to_word_art",
    "to_pseudonym_substitution", "to_word_game"])
def test_a_translation_refuses_a_step_that_has_a_model_mark_words_either_way(
        translation, marking):
    """The marking call finds a word in 62 of 100 harmful English inputs but in
    3 to 36 of their translations, so a heavy encoding after one is discarded
    and a light one changes nothing: the spec keeps the families apart,
    whichever is picked first."""
    assert not _joins(translation, marking)
    assert not _joins(marking, translation)


@pytest.mark.parametrize("translation", ["to_french", "to_thai", "to_hindi"])
@pytest.mark.parametrize("typos", [
    "to_rewrite_with_typos_low", "to_rewrite_with_typos_medium",
    "to_rewrite_with_typos_high", "to_rewrite_with_typos_insane"])
def test_a_translation_refuses_typos_either_way(translation, typos):
    """A translated prompt is already hard to read, for the model that writes
    text and for the model that answers: a rewrite with typos on top is one
    change too many, whichever is picked first."""
    assert not _joins(translation, typos)
    assert not _joins(typos, translation)


@pytest.mark.parametrize("translation", ["to_french", "to_thai", "to_hindi"])
@pytest.mark.parametrize("casual", [
    "to_lowercase", "to_no_sentence_capitals", "to_dropped_spaces",
    "to_doubled_punctuation", "to_abbreviations", "to_stripped_accents"])
def test_a_translation_refuses_casual_writing_either_way(translation, casual):
    """Casing, spacing, punctuation, abbreviations and accents are one more
    change to a text already hard to read."""
    assert not _joins(translation, casual)
    assert not _joins(casual, translation)


@pytest.mark.parametrize("translation", ["to_french", "to_thai", "to_hindi"])
@pytest.mark.parametrize("noise", [
    "to_random_capitals_low", "to_scrambled_letters_medium",
    "to_keyboard_typos_high", "to_emoji_sprinkle_low"])
def test_a_translation_refuses_noise_either_way(translation, noise):
    """Random capitals, scrambled letters, keyboard typos and emoji are one
    more change to a text already hard to read, whatever its script."""
    assert not _joins(translation, noise)
    assert not _joins(noise, translation)


@pytest.mark.parametrize("translation", ["to_french", "to_thai", "to_hindi"])
@pytest.mark.parametrize("split", _PAYLOAD_SPLITS)
def test_a_translation_refuses_a_payload_split_either_way(translation, split):
    """The prompt cut in parts named by letters is one more change to a text
    already hard to read."""
    assert not _joins(translation, split)
    assert not _joins(split, translation)


@pytest.mark.parametrize("translation", ["to_french", "to_thai", "to_hindi"])
@pytest.mark.parametrize("encoding", [
    "to_rot13", "to_leetspeak_basic", "to_separator", "to_hex", "to_base64",
    "to_unicode_escape", "to_morse", "to_braille",
    "to_sensitive_words_encode_rot13", "to_sensitive_words_encode_base64"])
def test_a_translation_refuses_an_encoding_either_way(translation, encoding):
    """Both rewrite the whole text, so one is enough: every member of the
    ``encode`` family, light or heavy, the ten sensitive-word encodings
    included, and into any script, whichever is picked first."""
    assert not _joins(translation, encoding)
    assert not _joins(encoding, translation)


@pytest.mark.parametrize("other", [
    "to_adversarial_suffix_punctuation", "to_json", "to_csv_row", "to_summarize",
    "to_psychopath_persona", "to_fictional_world"])
def test_a_translation_keeps_the_families_it_has_no_rule_against(other):
    assert _joins("to_french", other) and _joins(other, "to_french")


@pytest.mark.parametrize("wording", ["to_pseudonym_substitution", "to_word_game"])
@pytest.mark.parametrize("other", [
    "to_sensitive_words_split", "to_sensitive_words_variables",
    "to_sensitive_words_encode_base64", "to_synonym_substitution",
    "to_tokenbreak_split", "to_tokenbreak_delimiter", "to_word_art", "to_ascii_art"])
def test_wording_refuses_the_other_word_level_steps_and_ascii_art_either_way(
        wording, other):
    """The other steps that put the marked words in other terms, a font on the
    text, and (above) a translation: one disguise of the words is enough, and
    the key or the puzzles are long."""
    assert not _joins(wording, other)
    assert not _joins(other, wording)


_SENSITIVE_WORDS = [
    "to_sensitive_words_split", "to_sensitive_words_variables",
    "to_synonym_substitution", "to_sensitive_words_encode_rot13",
    "to_sensitive_words_encode_base64"]
_TOKENBREAKS = [
    "to_tokenbreak_prepend", "to_tokenbreak_split", "to_tokenbreak_delimiter"]


@pytest.mark.parametrize("sensitive", _SENSITIVE_WORDS)
@pytest.mark.parametrize("tokenbreak", _TOKENBREAKS)
def test_sensitive_words_and_tokenbreak_refuse_each_other_either_way(
        sensitive, tokenbreak):
    """Each has the model mark the text's words, so a second marking call would
    read a text the first changed: one marking step per text, as for wording
    and word art."""
    assert not _joins(sensitive, tokenbreak)
    assert not _joins(tokenbreak, sensitive)


@pytest.mark.parametrize("marking", _SENSITIVE_WORDS + _TOKENBREAKS)
@pytest.mark.parametrize("other", [
    "to_rewrite_with_typos_low", "to_lowercase", "to_adversarial_suffix_punctuation",
    "to_json", "to_csv_row", "to_summarize", "to_psychopath_persona",
    "to_fictional_world"])
def test_a_word_marking_step_keeps_the_families_it_has_no_rule_against(marking, other):
    assert _joins(marking, other) and _joins(other, marking)


@pytest.mark.parametrize("wording", ["to_pseudonym_substitution", "to_word_game"])
@pytest.mark.parametrize("other", [
    "to_fsh_random_short", "to_fsh_selected_long", "to_dap_random_long",
    "to_dap_selected_short", "to_fake_system_turn"])
def test_wording_takes_the_manipulation_layer_s_obfuscation_block(wording, other):
    """manipulation_blocked_obfuscation names the family: the key or the
    puzzles and the benign pairs of fsh and dap would sit between the reader
    and the request."""
    assert not _joins(wording, other)
    assert not _joins(other, wording)


@pytest.mark.parametrize("wording", ["to_pseudonym_substitution", "to_word_game"])
@pytest.mark.parametrize("other", [
    "to_rewrite_with_typos_low", "to_random_capitals_low", "to_lowercase",
    "to_adversarial_suffix_punctuation", "to_base64", "to_rot13", "to_payload_split",
    "to_json", "to_csv_row", "to_summarize", "to_psychopath_persona",
    "to_fictional_world"])
def test_wording_joins_typos_noise_a_suffix_an_encoding_a_wrapper_a_request_or_a_scene(
        wording, other):
    assert _joins(wording, other) and _joins(other, wording)


def test_wording_is_one_technique_of_its_family_in_a_combination():
    assert not _joins("to_pseudonym_substitution", "to_word_game")
    assert not _joins("to_word_game", "to_pseudonym_substitution")


@pytest.mark.parametrize(("first", "then"), [
    ("to_french", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_french"),
    ("to_base64", "to_dap_random_long"),
    ("to_dap_random_long", "to_base64"),
])
def test_manipulation_blocking_refuses_either_order(first, then):
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_sensitive_words_encode_base64", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_sensitive_words_encode_base64"),
    ("to_rot13", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_rot13"),
])
def test_manipulation_admits_light_and_word_level_encoding(first, then):
    assert _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_rot13", "to_thai"),
    ("to_thai", "to_rot13"),
    ("to_leetspeak_advanced", "to_russian"),
    ("to_russian", "to_leetspeak_advanced"),
    ("to_sensitive_words_encode_rot47", "to_japanese"),
    ("to_japanese", "to_sensitive_words_encode_rot47"),
])
def test_a_latin_only_encoding_refuses_a_translation_into_another_script(first, then):
    """After it, the encoding would change nothing, yet count in
    ``num_techniques``; before it, the translator would get a cipher."""
    assert not _joins(first, then)


def test_each_latin_only_technique_is_refused_beside_a_non_latin_translation():
    """``needs_latin`` marks what would change nothing in a text of another
    script, and no rule reads it: the family rules do the work, a translation
    refusing ``encode`` (the Latin-only ciphers and their sensitive-word
    forms) and ``noise`` (the random capitals, scrambled letters and keyboard
    typos). A new Latin-only technique in a family a translation does not
    refuse fails here, until the spec keeps the two apart."""
    from redact.jailbreak.technique import LATIN

    latin_only = [fn for fn in POOL if fn.needs_latin]
    into_another_script = [fn for fn in POOL
                           if "translation" in fn.families and fn.script != LATIN]
    assert latin_only and into_another_script
    joined = [(a.__name__, b.__name__) for a in latin_only for b in into_another_script
              if b in compatible([a], [b]) or a in compatible([b], [a])]
    assert joined == []


_PERSUASIONS = ["to_persuade_evidence_based", "to_persuade_scarcity"]
_TENSE_REWRITES = ["to_rewrite_in_past_tense", "to_rewrite_in_future_tense"]
_MODEL_DISTRACTORS = ["to_model_distractor_prefix", "to_model_distractor_suffix"]


@pytest.mark.parametrize("persuasion", _PERSUASIONS)
@pytest.mark.parametrize("other", [
    "to_impersonate_good_person", "to_impersonate_authority", "to_ascii_art"])
def test_a_persuasion_refuses_an_impersonation_and_ascii_art_either_way(
        persuasion, other):
    """Two identities in one prompt, and a font on a rewrite the model wrote
    at length."""
    assert not _joins(persuasion, other)
    assert not _joins(other, persuasion)


@pytest.mark.parametrize("tense", _TENSE_REWRITES)
@pytest.mark.parametrize("other", [
    "to_use_past_tense", "to_use_future_tense", "to_use_historical_era"])
def test_a_tense_rewrite_refuses_the_temporal_requests_either_way(tense, other):
    """The request would be set in time twice."""
    assert not _joins(tense, other)
    assert not _joins(other, tense)


@pytest.mark.parametrize("distractor", _MODEL_DISTRACTORS)
@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_distractor_prefix_related", "to_distractor_suffix_task"])
def test_a_model_distractor_refuses_ascii_art_and_the_template_distractors_either_way(
        distractor, other):
    assert not _joins(distractor, other)
    assert not _joins(other, distractor)


@pytest.mark.parametrize("first", [*_PERSUASIONS, *_TENSE_REWRITES,
                                   *_MODEL_DISTRACTORS])
@pytest.mark.parametrize("second", [*_PERSUASIONS, *_TENSE_REWRITES,
                                    *_MODEL_DISTRACTORS])
def test_a_combination_holds_one_technique_of_the_rewrite_layer(first, second):
    assert not _joins(first, second)


@pytest.mark.parametrize("rewrite", [_PERSUASIONS[0], _TENSE_REWRITES[0],
                                     _MODEL_DISTRACTORS[0]])
@pytest.mark.parametrize("other", [
    "to_base64", "to_french", "to_json", "to_answer_as_code", "to_fictional_world",
    "to_psychopath_persona", "to_fsh_random_short", "to_summarize"])
def test_a_rewrite_joins_the_other_layers_it_has_no_rule_against(rewrite, other):
    assert _joins(rewrite, other) and _joins(other, rewrite)


@pytest.mark.parametrize(("first", "second"), [
    ("translation", "encode"),
    ("sensitive_words", "tokenbreak"),
], ids=["translation-encode", "sensitive_words-tokenbreak"])
def test_no_planned_unit_holds_two_families_the_spec_keeps_apart(first, second):
    """The default attacks, dealt, in a budget round and in a count round of
    two: a unit never holds a technique of each. A technique counts under all
    its families (a sensitive-word encoding is also ``encode``)."""
    from tests.jailbreak import simulate

    def units(rounds):
        rows, pool = simulate.plan_kind(
            {"rounds": rounds} if rounds else {}, simulate.inputs(900, groups=3))
        by = {fn.__name__: fn for fn in pool}
        return [{family for name in row["combination"] for family in by[name].families}
                for row in rows]

    for rounds in (None, [{"exact_techniques": 2}]):
        held = units(rounds)
        assert any(first in families for families in held)
        assert [families for families in held if {first, second} <= families] == []


def test_the_heavy_full_text_encodings_never_meet_ascii_art_in_a_planned_unit():
    """The rule is on each encoding's own entry, not its family's."""
    from tests.jailbreak import simulate

    rows, pool = simulate.plan_kind(
        {"rounds": [{"exact_techniques": 2}]}, simulate.inputs(2000, groups=3))
    assert {"to_ascii_art"} <= {name for row in rows for name in row["combination"]}
    held = [row["combination"] for row in rows
            if "to_ascii_art" in row["combination"]
            and set(_HEAVY_FULL_TEXT) & set(row["combination"])]
    assert held == []


def test_every_pair_is_refused_or_admitted_both_ways():
    joins = {fn.__name__: {g.__name__ for g in compatible([fn], POOL)} for fn in POOL}
    one_way = [(a, b) for a in joins for b in joins[a] if a not in joins[b]]
    assert one_way == []


def test_an_exact_draw_never_holds_a_forbidden_pair():
    for seed in range(200):
        fn = sample_exact_combination(random.Random(seed), POOL, 3)
        names = [t.__name__ for t in fn.techniques]
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                assert _joins(name, other), (seed, names)


def test_a_budget_draw_never_holds_a_forbidden_pair_or_breaks_a_layer_cap():
    from collections import Counter

    from redact.jailbreak.technique import load_spec

    caps = load_spec()["layer_max_picks"]
    for seed in range(600):
        fn = sample_combination(
            random.Random(seed), POOL, max_complexity=9, max_obfuscations=3,
            allow_request_obfuscation=True,
            dealer=Dealer(key=(seed,)) if seed % 2 else None,
            sampling_probs=dict.fromkeys(
                ("hacking", "manipulation", "requests", "request_obfuscation"), 0.9))
        names = [t.__name__ for t in fn.techniques]
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                assert _joins(name, other), (seed, names)
        per_layer = Counter(t.layer for t in fn.techniques)
        over = [layer for layer, cap in caps.items() if per_layer[layer] > cap]
        assert over == [], (seed, names)


@pytest.mark.parametrize("max_complexity", [0, 1, 2, 3])
def test_a_budget_draw_stays_within_its_complexity(max_complexity):
    for seed in range(300):
        fn = sample_combination(
            random.Random(seed), POOL, max_complexity=max_complexity,
            max_obfuscations=3, allow_request_obfuscation=True,
            dealer=Dealer(key=(seed,)) if seed % 2 else None,
            sampling_probs=dict.fromkeys(
                ("hacking", "manipulation", "requests", "request_obfuscation"), 0.9))
        names = [t.__name__ for t in fn.techniques]
        assert sum(t.complexity for t in fn.techniques) <= max_complexity, (seed, names)


@pytest.mark.parametrize("max_obfuscations", [0, 1, 2, 3])
def test_a_budget_draw_counts_its_request_obfuscation_against_max_obfuscations(
        max_obfuscations):
    for seed in range(300):
        fn = sample_combination(
            random.Random(seed), POOL, max_complexity=12,
            max_obfuscations=max_obfuscations, allow_request_obfuscation=True,
            dealer=Dealer(key=(seed,)) if seed % 2 else None,
            sampling_probs=dict.fromkeys(
                ("hacking", "manipulation", "requests", "request_obfuscation"), 0.9))
        names = [t.__name__ for t in fn.techniques]
        obfuscations = sum(t.layer == "obfuscation" for t in fn.techniques)
        assert obfuscations <= max_obfuscations, (seed, names)
        held = getattr(fn, "request_obfuscation", None)
        if held is not None:
            # Right after the request it obfuscates; only the manipulation
            # layer, which inserts the finished question, comes after it.
            at = names.index(held.__name__)
            assert fn.techniques[at - 1].layer == "requests", (seed, names)
            assert {t.layer for t in fn.techniques[at + 1:]} <= {"manipulation"}, (
                seed, names)


def test_request_obfuscation_is_light_encoding_or_translation():
    names = {fn.__name__ for fn in request_obfuscation_candidates(POOL)}
    assert {"to_rot13", "to_french"} <= names
    assert not names & {"to_base64", "to_ascii_art", "to_summarize"}


# --- a translation drawn as the request obfuscation ------------------------------
# It applies right after the request, to the finished request (a manipulation
# technique, the only one that may follow it, never meets a translation), so it
# would translate a suffix, a wrapper's keys or a container's lines: the spec's
# request_obfuscation_refuses keeps them out of its combination. A translation
# drawn in phase 4 runs before those steps and meets none of these rules.

_AFTER_A_TRANSLATION = [
    "to_adversarial_suffix_punctuation", "to_adversarial_suffix_unicode",
    "to_json", "to_xml", "to_markdown", "to_csv_row", "to_code_comment",
    "to_email_message"]


def test_a_request_obfuscation_that_translates_keeps_out_what_it_would_translate():
    from redact.jailbreak.rules import after_request_obfuscation

    left = {fn.__name__ for fn in after_request_obfuscation(BY_NAME["to_french"], POOL)}
    assert not left & set(_AFTER_A_TRANSLATION)
    assert {fn.__name__ for fn in POOL} - left == {
        fn.__name__ for fn in POOL
        if {"suffixes", "structural", "neutral_container"} & set(fn.families)}


@pytest.mark.parametrize("pick", ["to_rot13", "to_sensitive_words_encode_rot13"])
def test_a_request_obfuscation_that_is_not_a_translation_keeps_nothing_out(pick):
    from redact.jailbreak.rules import after_request_obfuscation

    assert after_request_obfuscation(BY_NAME[pick], POOL) == POOL


_SMALL_POOL = [BY_NAME[name] for name in [
    "to_french", "to_summarize", "to_fictional_world",
    "to_adversarial_suffix_punctuation", "to_json", "to_csv_row"]]


def _drawn(request_obfuscation: bool, seeds=range(200)):
    """The combinations of a small pool where a scene and a request are sure
    and the request obfuscation is allowed or not, drawn without and with a
    dealer."""
    for seed in seeds:
        yield sample_combination(
            random.Random(seed), _SMALL_POOL, max_complexity=12, max_obfuscations=3,
            allow_request_obfuscation=request_obfuscation,
            sampling_probs=dict.fromkeys(
                ("hacking", "requests", "request_obfuscation"), 1.0),
            include_manipulation=False,
            dealer=Dealer(key=(seed,)) if seed % 2 else None)


def _translates_last(fn) -> bool:
    return "translation" in getattr(fn.request_obfuscation, "families", ())


def test_a_translation_drawn_last_comes_without_a_suffix_wrapper_or_container():
    translating = [fn for fn in _drawn(True) if _translates_last(fn)]
    assert len(translating) > 20     # noqa: PLR2004
    for fn in translating:
        names = {t.__name__ for t in fn.techniques}
        assert not names & set(_AFTER_A_TRANSLATION), names


def test_a_translation_drawn_last_still_joins_a_scene_and_a_request():
    names = {frozenset(t.__name__ for t in fn.techniques)
             for fn in _drawn(True) if _translates_last(fn)}
    assert names == {frozenset({"to_french", "to_summarize", "to_fictional_world"})}


def test_a_translation_drawn_in_phase_4_still_joins_a_suffix_wrapper_container():
    held = {t.__name__ for fn in _drawn(False)
            if "to_french" in {t.__name__ for t in fn.techniques}
            for t in fn.techniques}
    assert {"to_adversarial_suffix_punctuation", "to_json", "to_csv_row"} <= held


_TRANSLITERATIONS = ["to_cyrillic_transliteration", "to_greek_transliteration",
                     "to_devanagari_transliteration", "to_arabic_transliteration"]


@pytest.mark.parametrize("translit", _TRANSLITERATIONS)
@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_base64", "to_rot13", "to_french", "to_russian",
    "to_rewrite_with_typos_low", "to_lowercase",
    "to_emoji_sprinkle_low", "to_payload_split"])
def test_a_transliteration_refuses_the_whole_text_obfuscations_either_way(
        translit, other):
    """A transliteration is a whole-text script rewrite at the front: it
    refuses the same obfuscation families a translation does (and translation
    itself), each way."""
    assert not _joins(translit, other)
    assert not _joins(other, translit)


@pytest.mark.parametrize("first", _TRANSLITERATIONS)
@pytest.mark.parametrize("second", _TRANSLITERATIONS)
def test_a_combination_holds_one_transliteration(first, second):
    """One per family, so two transliterations never co-occur."""
    assert not _joins(first, second)


@pytest.mark.parametrize("translit", _TRANSLITERATIONS)
@pytest.mark.parametrize("other", [
    "to_json", "to_summarize", "to_fsh_random_short", "to_fictional_world"])
def test_a_transliteration_joins_a_wrapper_a_request_or_the_hacking_layers(
        translit, other):
    """It changes the letters, not the frame, so a structural wrapper, a
    request template, a hacking frame or a few-shot example may sit with it
    (it is not in the manipulation block)."""
    assert _joins(translit, other)
    assert _joins(other, translit)


@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_base64", "to_french", "to_russian",
    "to_rewrite_with_typos_low", "to_lowercase", "to_payload_split",
    "to_cyrillic_transliteration"])
def test_code_switching_refuses_the_whole_text_obfuscations_either_way(other):
    """Code-switching is a whole-text language rewrite at the front: it
    refuses the obfuscation families a translation does, translation, and the
    other language-mix families, each way."""
    assert not _joins("to_code_switch_spanish", other)
    assert not _joins(other, "to_code_switch_spanish")


@pytest.mark.parametrize("second", ["to_code_switch_french", "to_code_switch_arabic"])
def test_a_combination_holds_one_code_switch(second):
    assert not _joins("to_code_switch_spanish", second)


@pytest.mark.parametrize("other", ["to_json", "to_summarize", "to_fsh_random_short"])
def test_code_switching_joins_a_wrapper_a_request_or_a_few_shot(other):
    """It keeps English words, so a wrapper, a request template or a few-shot
    frame may sit with it (not in the manipulation block)."""
    assert _joins("to_code_switch_spanish", other)
    assert _joins(other, "to_code_switch_spanish")


@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_base64", "to_french", "to_russian", "to_lowercase",
    "to_cyrillic_transliteration", "to_code_switch_spanish"])
def test_romanization_refuses_the_whole_text_obfuscations_either_way(other):
    """Romanization is a whole-text language rewrite at the front: it refuses
    the obfuscation families a translation does, translation, and the other
    language-mix families, each way."""
    assert not _joins("to_romanize_hindi", other)
    assert not _joins(other, "to_romanize_hindi")


def test_a_combination_holds_one_romanization():
    assert not _joins("to_romanize_hindi", "to_romanize_arabic")


@pytest.mark.parametrize("other", ["to_json", "to_summarize", "to_fsh_random_short"])
def test_romanization_joins_a_wrapper_a_request_or_a_few_shot(other):
    assert _joins("to_romanize_hindi", other)
    assert _joins(other, "to_romanize_hindi")


@pytest.mark.parametrize("other", [
    "to_ascii_art", "to_base64", "to_french", "to_russian", "to_lowercase",
    "to_cyrillic_transliteration", "to_code_switch_spanish", "to_romanize_hindi"])
def test_dialect_refuses_the_whole_text_obfuscations_either_way(other):
    """A dialect is a translation into a regional variant: it refuses the
    obfuscation families a translation does, translation, and the other
    language-mix families, each way."""
    assert not _joins("to_dialect_mexican_spanish", other)
    assert not _joins(other, "to_dialect_mexican_spanish")


def test_a_combination_holds_one_dialect():
    assert not _joins("to_dialect_mexican_spanish", "to_dialect_canadian_french")


@pytest.mark.parametrize("other", ["to_json", "to_summarize", "to_fsh_random_short"])
def test_a_dialect_joins_a_wrapper_a_request_or_a_few_shot(other):
    """Unlike translation, a dialect is not in the manipulation block, so fsh
    and dap may sit with it (a deliberate deviation; see the blockers note)."""
    assert _joins("to_dialect_mexican_spanish", other)
    assert _joins(other, "to_dialect_mexican_spanish")
