"""The wording techniques: each word a model marks put in other terms, a code
name with a key (``to_pseudonym_substitution``) or a placeholder with a puzzle
(``to_word_game``), and a variant of a pool around the prompt that says how to
read them.

The packaged variants and prompts are the owner's text; these tests fill the
pools with stand-ins of their own and assert on the placeholders, the key and
puzzle lines, the case of a code name, the replay and the discards, never on
the packaged wording.
"""

import random
import re
import string

import pytest

import redact.jailbreak.obfuscation.wording as WG
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.obfuscation.sensitive_words import NO_SENSITIVE_WORD
from redact.jailbreak.obfuscation.tokenbreak import MALFORMED_WORDS
from redact.jailbreak.technique import DISCARDED
from redact.llm_pipeline import drive_sync

PROMPT = "the cat sat on the mat today"
#: Before the key or the puzzles in a stand-in rendering.
KEY_MARK = "\n--key--\n"
PUZZLE_MARK = "\n--puzzles--\n"


@pytest.fixture
def stand_ins(monkeypatch):
    """Two variants of each pool of the tests' own: a tag, the request, a mark,
    then the key or the puzzles."""
    monkeypatch.setattr(WG, "_POOLS", {
        "pseudonym_key": [
            {"name": name, "template": f"<{name}>\n{{prompt}}{KEY_MARK}{{key}}"}
            for name in ("one", "two")],
        "word_game_solve": [
            {"name": name, "template": f"<{name}>\n{{prompt}}{PUZZLE_MARK}{{puzzles}}"}
            for name in ("one", "two")]})


def _split(text: str, mark: str) -> tuple[str, str, list[str]]:
    """A stand-in rendering as ``(variant, request, lines)``, the lines being
    the key's or the puzzles'."""
    head, tail = text.split(mark)
    variant, request = head.split("\n", 1)
    return variant.strip("<>"), request, tail.split("\n")


def _drive(gen, replies: list[str]):
    """Answer ``gen``'s requests with ``replies`` in turn; return
    ``(requests, result)``."""
    requests, answers = [], iter(replies)

    def answer(request):
        requests.append(request)
        return next(answers)

    return requests, drive_sync(gen, answer)


def _pseudonym(prompt: str, replies: list[str], seed: int = 0):
    """``to_pseudonym_substitution`` on ``prompt``, ``replies`` the model's
    replies: the marked words, then one per word."""
    return _drive(WG.to_pseudonym_substitution(
        prompt, gen_model="m", rng=random.Random(seed)), replies)


def _word_game(prompt: str, replies: list[str], seed: int = 0):
    return _drive(WG.to_word_game(prompt, gen_model="m", rng=random.Random(seed)),
                  replies)


# --- what the spec says of them ------------------------------------------------

@pytest.mark.parametrize("fn", [WG.to_pseudonym_substitution, WG.to_word_game])
def test_the_spec_tags_an_attack_only_word_level_obfuscation_of_spec_1_3(fn):
    assert fn.families == ["wording"]
    assert (fn.layer, fn.within_layer_order, fn.complexity) == ("obfuscation", 45, 2)
    assert (fn.requires_llm, fn.requires_benign, fn.check, fn.calls) == (
        True, False, "word_list", [1, 21])
    assert (fn.kinds, fn.since) == (["adversarial"], "1.3")


def test_the_family_applies_between_sensitive_words_and_tokenbreak():
    from redact.jailbreak.technique import load_spec

    order = {name: family["within_layer_order"]
             for name, family in load_spec()["families"].items()
             if family["layer"] == "obfuscation"}
    assert order["sensitive_words"] < order["wording"] < order["tokenbreak"]


def test_the_family_carries_the_tag_sensitive_words_carries():
    from redact.jailbreak.technique import load_spec

    families = load_spec()["families"]
    assert families["wording"]["dimension"] == families["sensitive_words"]["dimension"]


def test_the_obfuscation_layer_registers_both_after_the_sensitive_words():
    names = [fn.__name__ for fn in get_all_obfuscation_functions()]
    assert names.index("to_pseudonym_substitution") == names.index(
        "to_synonym_substitution") + 1
    assert names.index("to_word_game") == names.index("to_pseudonym_substitution") + 1


def test_to_synonym_substitution_stays_in_sensitive_words():
    from redact.jailbreak.obfuscation.sensitive_words import to_synonym_substitution

    assert to_synonym_substitution.families == ["sensitive_words"]


def test_the_natural_kind_does_not_draw_them():
    from redact.jailbreak.augmentation import resolve
    from redact.jailbreak.pool import technique_pool

    for translation in (False, True):
        pool = technique_pool(resolve(
            {"natural": {"translation": translation}}, "natural"))
        assert not {fn.__name__ for fn in pool} & {
            "to_pseudonym_substitution", "to_word_game"}


def test_the_eval_type_does_not_draw_them():
    from redact.jailbreak.augmentation import resolve
    from redact.jailbreak.pool import technique_pool
    from redact.recipe import load_dataset_type

    section = load_dataset_type("eval")["augmentations"]
    names = {fn.__name__ for fn in technique_pool(resolve(section, "adversarial"))}
    assert not names & {"to_pseudonym_substitution", "to_word_game"}


def test_the_default_attack_kind_draws_them():
    from redact.jailbreak.augmentation import resolve
    from redact.jailbreak.pool import technique_pool

    names = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    assert {"to_pseudonym_substitution", "to_word_game"} <= names


# --- the pools they draw from --------------------------------------------------

def test_each_draws_a_pool_of_the_wording_templates():
    for fn, directive, slots in [
            (WG.to_pseudonym_substitution, "pseudonym_key", ("prompt", "key")),
            (WG.to_word_game, "word_game_solve", ("prompt", "puzzles"))]:
        ref = fn.variant_pool
        assert (ref.file, ref.directive, ref.slots, ref.request_slot,
                ref.info_key) == ("obfuscation/wording_templates.json", directive,
                                  slots, True, "variant")
        check_pools([ref])


@pytest.mark.parametrize(("directive", "slots"), [
    ("pseudonym_key", {"prompt", "key"}), ("word_game_solve", {"prompt", "puzzles"})])
def test_each_packaged_pool_has_8_variants_that_each_name_both_slots(directive, slots):
    variants = read_pools(WG._FILE)[directive]
    assert len(variants) == 8       # noqa: PLR2004
    for variant in variants:
        named = {field for _, field, _, _ in string.Formatter().parse(
            variant["template"]) if field}
        assert named == slots, variant["name"]


# --- the calls -----------------------------------------------------------------

def test_one_request_marks_the_prompt_then_one_asks_for_each_word(stand_ins):
    requests, _ = _pseudonym(PROMPT, ["cat mat", "Falcon", "Anvil"])
    assert [request.model for request in requests] == ["m", "m", "m"]
    assert PROMPT in " ".join(m["content"] for m in requests[0].messages)
    for request, word in zip(requests[1:], ["cat", "mat"]):
        shown = " ".join(m["content"] for m in request.messages)
        assert word in shown and PROMPT in shown


def test_a_word_s_request_is_capped(stand_ins):
    requests, _ = _word_game(PROMPT, ["cat", "a pet that purrs"])
    assert [request.max_tokens for request in requests] == [None, WG.WORD_REPLY_TOKENS]


@pytest.mark.parametrize(("fn", "prompt_name", "reply"), [
    (WG.to_pseudonym_substitution, "pseudonym", "Falcon"),
    (WG.to_word_game, "word_game", "a pet that purrs")])
def test_a_word_s_prompt_is_read_through_the_prompt_dir(
        stand_ins, tmp_path, fn, prompt_name, reply):
    import json

    folder = tmp_path / "jailbreak" / prompt_name
    folder.mkdir(parents=True)
    (folder / "template.json").write_text(json.dumps({
        "system_prompt": "", "template": "OVERLAY {word} {prompt}"}),
        encoding="utf-8")
    requests, _ = _drive(
        fn(PROMPT, gen_model="m", prompt_dir=tmp_path, rng=random.Random(0)),
        ["cat", reply])
    assert requests[1].messages[-1]["content"] == f"OVERLAY cat {PROMPT}"


# --- the code names ------------------------------------------------------------

def test_each_marked_word_is_swapped_for_its_code_name_and_keyed(stand_ins):
    _, (text, _) = _pseudonym(PROMPT, ["cat mat", "falcon", "anvil"])
    _, request, key = _split(text, KEY_MARK)
    assert request == "the falcon sat on the anvil today"
    assert key == ["falcon = cat", "anvil = mat"]


@pytest.mark.parametrize("alias", [
    r"\1", "$0", r"\g<0>", "(", "a(b", r"\\", "%s", "{x}", "[w1]",
])
def test_a_code_name_that_reads_as_regex_syntax_is_written_as_it_is(stand_ins, alias):
    _, (text, _) = _pseudonym(PROMPT, ["cat", alias])
    _, request, key = _split(text, KEY_MARK)
    assert request == f"the {alias} sat on the mat today"
    assert key == [f"{alias} = cat"]


def test_a_marked_word_that_holds_another_is_swapped_whole_not_in_parts():
    """``category`` and ``cat``: the longer first, one pass."""
    swapped = WG._swap("a category of cats", ["cat", "category"],
                       lambda i, matched: f"<{i}>")
    assert swapped == "a <1> of <0>s"


@pytest.mark.parametrize("word", ["a.b", "a+b", "(x)", "a|b", "^a$", r"a\b"])
def test_a_marked_word_with_regex_syntax_matches_only_itself(word):
    prompt = f"keep axb and ab, swap {word} here"
    assert WG._swap(prompt, [word], lambda i, matched: "#") == (
        "keep axb and ab, swap # here")


def test_a_code_name_takes_the_case_of_the_word_it_replaces(stand_ins):
    prompt = "the cat chased a CAT and Cat and cat"
    _, (text, _) = _pseudonym(prompt, ["cat", "falcon"])
    assert _split(text, KEY_MARK)[1] == (
        "the falcon chased a FALCON and Falcon and falcon")


def test_the_key_gives_the_name_as_the_model_wrote_it(stand_ins):
    _, (text, _) = _pseudonym(PROMPT, ["cat", "Grey Falcon"])
    assert _split(text, KEY_MARK)[2] == ["Grey Falcon = cat"]


def test_no_marked_word_is_left_in_the_request(stand_ins):
    _, (text, _) = _pseudonym(PROMPT, ["cat mat", "falcon", "anvil"])
    request = _split(text, KEY_MARK)[1]
    assert not re.search("cat|mat", request, re.IGNORECASE)


def test_a_word_marked_twice_in_any_case_is_asked_for_once(stand_ins):
    requests, (text, info) = _pseudonym(PROMPT, ["cat Cat mat", "falcon", "anvil"])
    assert len(requests) == 3       # noqa: PLR2004
    assert info.endswith("detected_words=cat mat")
    assert _split(text, KEY_MARK)[2] == ["falcon = cat", "anvil = mat"]


def test_the_info_names_the_variant_and_the_words(stand_ins):
    _, (text, info) = _pseudonym(PROMPT, ["cat", "falcon"])
    variant = _split(text, KEY_MARK)[0]
    assert info == f"variant={variant};detected_words=cat"


def test_a_seed_draws_the_same_variant_and_both_variants_are_drawn(stand_ins):
    texts = [_pseudonym(PROMPT, ["cat", "falcon"], seed)[1][0] for seed in range(20)]
    assert texts == [_pseudonym(PROMPT, ["cat", "falcon"], seed)[1][0]
                     for seed in range(20)]
    assert {_split(text, KEY_MARK)[0] for text in texts} == {"one", "two"}


@pytest.mark.parametrize("bad", [
    "", "   ", "a grey falcon", "falcon\nhawk", "fal = con", "cat", "Cat", "catfish",
    "the cat"])
def test_a_reply_that_is_not_a_code_name_is_asked_for_again_once(stand_ins, bad):
    requests, (text, _) = _pseudonym(PROMPT, ["cat", bad, "falcon"])
    assert len(requests) == 3       # noqa: PLR2004
    assert _split(text, KEY_MARK)[2] == ["falcon = cat"]


def test_quotes_round_a_code_name_are_dropped(stand_ins):
    _, (text, _) = _pseudonym(PROMPT, ["cat", ' "Falcon" '])
    assert _split(text, KEY_MARK)[2] == ["Falcon = cat"]


def test_a_code_name_another_word_has_is_asked_for_again(stand_ins):
    requests, (text, _) = _pseudonym(
        PROMPT, ["cat mat", "falcon", "Falcon", "anvil"])
    assert len(requests) == 4       # noqa: PLR2004
    assert _split(text, KEY_MARK)[2] == ["falcon = cat", "anvil = mat"]


def test_two_bad_replies_for_a_word_discard_the_prompt(stand_ins):
    requests, (text, info) = _pseudonym(PROMPT, ["cat", "two words here", ""])
    assert len(requests) == 3       # noqa: PLR2004
    assert (text, info) == (PROMPT, WG.MALFORMED_PSEUDONYM)
    assert info.startswith(DISCARDED)


def test_a_second_word_s_failure_discards_the_whole_prompt(stand_ins):
    _, (text, info) = _pseudonym(PROMPT, ["cat mat", "falcon", "", ""])
    assert (text, info) == (PROMPT, WG.MALFORMED_PSEUDONYM)


# --- the word game -------------------------------------------------------------

def test_each_marked_word_becomes_a_placeholder_with_a_puzzle_listed(stand_ins):
    _, (text, _) = _word_game(PROMPT, ["cat mat", "a pet that purrs", "a floor cover"])
    _, request, puzzles = _split(text, PUZZLE_MARK)
    assert request == "the [W1] sat on the [W2] today"
    assert puzzles == ["[W1]: a pet that purrs", "[W2]: a floor cover"]


def test_every_occurrence_in_any_case_is_a_placeholder(stand_ins):
    _, (text, _) = _word_game("The CAT chased a cat and Cat", ["cat", "a pet"])
    assert _split(text, PUZZLE_MARK)[1] == "The [W1] chased a [W1] and [W1]"


def test_no_marked_word_is_left_anywhere_in_the_text(stand_ins):
    _, (text, _) = _word_game(PROMPT, ["cat mat", "a pet that purrs", "a floor cover"])
    assert not re.search("cat|mat", text.split("\n", 1)[1], re.IGNORECASE)


def test_a_placeholder_is_not_swapped_again_for_a_word_that_reads_like_it(stand_ins):
    """In one pass: ``w1`` is marked, and ``[W1]`` reads like it in any case."""
    _, (text, _) = _word_game("tell w1 about the cat", ["cat w1", "a pet", "a label"])
    assert _split(text, PUZZLE_MARK)[1] == "tell [W2] about the [W1]"


def test_the_info_names_the_variant_and_the_words_here_too(stand_ins):
    _, (text, info) = _word_game(PROMPT, ["cat mat", "a pet", "a floor cover"])
    assert info == f"variant={_split(text, PUZZLE_MARK)[0]};detected_words=cat mat"


def test_a_seed_replays_and_both_variants_are_drawn(stand_ins):
    texts = [_word_game(PROMPT, ["cat", "a pet"], seed)[1][0] for seed in range(20)]
    assert texts == [_word_game(PROMPT, ["cat", "a pet"], seed)[1][0]
                     for seed in range(20)]
    assert {_split(text, PUZZLE_MARK)[0] for text in texts} == {"one", "two"}


@pytest.mark.parametrize("bad", [
    "", "   ", "it is a CAT", "a cat", "the catalogue", " ".join(["word"] * 41)])
def test_a_puzzle_that_is_empty_too_long_or_holds_the_word_is_asked_for_again(
        stand_ins, bad):
    requests, (text, _) = _word_game(PROMPT, ["cat", bad, "a pet that purrs"])
    assert len(requests) == 3       # noqa: PLR2004
    assert _split(text, PUZZLE_MARK)[2] == ["[W1]: a pet that purrs"]


def test_a_puzzle_is_set_on_one_line(stand_ins):
    _, (text, _) = _word_game(PROMPT, ["cat", "a pet\nthat   purrs\n"])
    assert _split(text, PUZZLE_MARK)[2] == ["[W1]: a pet that purrs"]


def test_a_puzzle_of_the_most_words_is_kept(stand_ins):
    puzzle = " ".join(["word"] * WG.MAX_PUZZLE_WORDS)
    _, (text, _) = _word_game(PROMPT, ["cat", puzzle])
    assert _split(text, PUZZLE_MARK)[2] == [f"[W1]: {puzzle}"]


def test_two_bad_replies_for_a_word_discard_the_prompt_here_too(stand_ins):
    requests, (text, info) = _word_game(PROMPT, ["cat", "a cat", ""])
    assert len(requests) == 3       # noqa: PLR2004
    assert (text, info) == (PROMPT, WG.MALFORMED_PUZZLE)
    assert info.startswith(DISCARDED)


# --- the marking call ----------------------------------------------------------

@pytest.mark.parametrize("run", [_pseudonym, _word_game])
def test_no_marked_word_discards_the_prompt_after_the_one_call(stand_ins, run):
    requests, (text, info) = run(PROMPT, ["None"])
    assert len(requests) == 1
    assert (text, info) == (PROMPT, NO_SENSITIVE_WORD)


@pytest.mark.parametrize("run", [_pseudonym, _word_game])
@pytest.mark.parametrize("reply", [
    "cat mat sat on the today and then some more words than ten can hold", "dog"])
def test_a_word_list_that_is_not_the_prompts_discards_it(stand_ins, run, reply):
    requests, (text, info) = run(PROMPT, [reply])
    assert len(requests) == 1
    assert (text, info) == (PROMPT, MALFORMED_WORDS)
