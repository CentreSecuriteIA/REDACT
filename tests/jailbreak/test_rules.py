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
    "to_base64", "to_morse", "to_rot47"])
def test_ascii_art_keeps_the_rest_of_their_families(technique):
    assert _joins(technique, "to_ascii_art") and _joins("to_ascii_art", technique)


@pytest.mark.parametrize("technique", [
    "to_lookalike_letters", "to_zero_width_noise", "to_fullwidth",
    "to_math_styled"])
def test_ascii_art_refuses_the_non_ascii_encodings_either_way(technique):
    """Their output is non-ASCII (look-alike and math letters, zero-width
    noise, fullwidth forms), which no font draws."""
    assert not _joins(technique, "to_ascii_art")
    assert not _joins("to_ascii_art", technique)


@pytest.mark.parametrize("technique", [
    "to_hex", "to_binary", "to_base32", "to_base85", "to_caesar_cipher",
    "to_atbash", "to_reversed_characters", "to_url_encoding", "to_html_entities"])
def test_ascii_art_keeps_the_ascii_output_encodings(technique):
    """These write ASCII, so a font can draw them."""
    assert _joins(technique, "to_ascii_art") and _joins("to_ascii_art", technique)


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


@pytest.mark.parametrize(("first", "then"), [
    ("to_random_capitals_low", "to_russian"),
    ("to_greek", "to_scrambled_letters_medium"),
    ("to_keyboard_typos_high", "to_thai"),
    ("to_japanese", "to_keyboard_typos_low"),
])
def test_the_latin_letter_noise_refuses_a_translation_into_another_script(
        first, then):
    assert not _joins(first, then)


@pytest.mark.parametrize("translation", ["to_thai", "to_french"])
def test_the_emoji_sprinkle_joins_a_translation_into_any_script(translation):
    assert _joins("to_emoji_sprinkle_low", translation)
    assert _joins(translation, "to_emoji_sprinkle_low")


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
    "to_base64", "to_morse", "to_rot47"])
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
    invented turn is a conversation, not a document to wrap."""
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
    "to_french", "to_thai", "to_rewrite_with_typos_low", "to_tokenbreak_split",
    "to_sensitive_words_split", "to_json", "to_csv_row", "to_summarize",
    "to_psychopath_persona", "to_fictional_world"])
def test_a_payload_split_joins_a_translation_noise_a_wrapper_a_request_or_a_scene(
        split, other):
    assert _joins(split, other) and _joins(other, split)


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


@pytest.mark.parametrize(("first", "then"), [
    ("to_rot13", "to_french"),
    ("to_vietnamese", "to_leetspeak_basic"),
    ("to_base64", "to_thai"),
    ("to_bengali", "to_separator"),
    ("to_morse", "to_greek"),
])
def test_a_latin_script_translation_or_another_encoding_still_joins(first, then):
    assert _joins(first, then) and _joins(then, first)


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
        last = getattr(fn, "request_obfuscation", None)
        if last is not None:
            assert names[-1] == last.__name__, (seed, names)


def test_request_obfuscation_is_light_encoding_or_translation():
    names = {fn.__name__ for fn in request_obfuscation_candidates(POOL)}
    assert {"to_rot13", "to_french"} <= names
    assert not names & {"to_base64", "to_ascii_art", "to_summarize"}
