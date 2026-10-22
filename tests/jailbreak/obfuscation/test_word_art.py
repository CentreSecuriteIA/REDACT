"""The word-art technique: the first word a model marks drawn as ASCII art,
``[MASK]`` in its place in the prompt, and a variant of the pool asking the
reader to read the art and put the word back.

The packaged variants are the owner's text; these tests fill the pool with
stand-ins of their own and assert on the art, the mask, the font, the info
and the verdicts, never on the packaged wording.
"""

import random
import re
import string

import pytest

import redact.jailbreak.obfuscation.word_art as WA
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.obfuscation.ascii_art import to_ascii_art
from redact.jailbreak.obfuscation.sensitive_words import NO_SENSITIVE_WORD
from redact.jailbreak.obfuscation.tokenbreak import MALFORMED_WORDS
from redact.jailbreak.technique import DISCARDED
from redact.llm_pipeline import drive_sync

PROMPT = "the cat sat on the mat today"
#: Between the art and the request in a stand-in rendering.
DIVIDER = "\n--\n"


@pytest.fixture
def stand_ins(monkeypatch):
    """Two variants of the tests' own: the art, a divider, the request."""
    monkeypatch.setattr(WA, "_POOLS", {"decode": [
        {"name": "one", "template": "<one>\n{art}" + DIVIDER + "{prompt}"},
        {"name": "two", "template": "<two>\n{art}" + DIVIDER + "{prompt}"}]})


def _split(text: str) -> tuple[str, str, str]:
    """A stand-in rendering as ``(variant, art, request)``."""
    head, request = text.split(DIVIDER)
    variant, art = head.split("\n", 1)
    return variant.strip("<>"), art, request


def _drive(gen, replies: list[str]):
    """Answer ``gen``'s requests with ``replies`` in turn; return
    ``(requests, result)``."""
    requests, answers = [], iter(replies)

    def answer(request):
        requests.append(request)
        return next(answers)

    return requests, drive_sync(gen, answer)


def _word_art(prompt: str, marked: str, seed: int = 0):
    """``to_word_art`` on ``prompt`` with ``marked`` the model's reply."""
    return _drive(WA.to_word_art(prompt, gen_model="m", rng=random.Random(seed)),
                  [marked])


# --- what the spec says of it --------------------------------------------------

def test_the_spec_tags_an_attack_only_word_level_obfuscation_of_spec_1_3():
    fn = WA.to_word_art
    assert fn.families == ["word_art"]
    assert (fn.layer, fn.within_layer_order, fn.complexity) == ("obfuscation", 55, 3)
    assert (fn.requires_llm, fn.requires_benign, fn.check, fn.calls) == (
        True, False, "word_list", [1, 1])
    assert (fn.kinds, fn.since) == (["adversarial"], "1.3")


def test_the_obfuscation_layer_registers_it():
    assert "to_word_art" in [fn.__name__ for fn in get_all_obfuscation_functions()]


def test_the_eval_type_does_not_draw_it():
    from redact.jailbreak.augmentation import resolve
    from redact.jailbreak.pool import technique_pool
    from redact.recipe import load_dataset_type

    section = load_dataset_type("eval")["augmentations"]
    names = {fn.__name__ for fn in technique_pool(resolve(section, "adversarial"))}
    assert "to_word_art" not in names


# --- the pool it draws from ----------------------------------------------------

def test_it_draws_the_decode_pool_whose_templates_name_the_art_and_the_request():
    ref = WA.to_word_art.variant_pool
    assert (ref.file, ref.directive, ref.slots, ref.request_slot, ref.info_key) == (
        "obfuscation/word_art_templates.json", "decode", ("art", "prompt"), True,
        "variant")
    check_pools([ref])


def test_the_packaged_pool_has_8_variants_that_each_name_both_slots():
    variants = read_pools(WA._FILE)["decode"]
    assert len(variants) == 8       # noqa: PLR2004
    for variant in variants:
        slots = {field for _, field, _, _ in string.Formatter().parse(
            variant["template"]) if field}
        assert slots == {"art", "prompt"}, variant["name"]


# --- the marking call ----------------------------------------------------------

def test_one_request_asks_the_generation_model_to_mark_the_prompt(stand_ins):
    requests, _ = _word_art(PROMPT, "mat")
    (request,) = requests
    assert request.model == "m"
    assert any(PROMPT in message["content"] for message in request.messages)


# --- the art, the mask and the request ------------------------------------------

def test_the_marked_word_is_masked_in_the_request(stand_ins):
    _, (text, _) = _word_art(PROMPT, "mat")
    assert _split(text)[2] == "the cat sat on the [MASK] today"


def test_every_occurrence_in_any_case_is_masked_so_the_word_fills_back_in(stand_ins):
    prompt = "the cat chased a cat"
    _, (text, _) = _word_art(prompt, "cat")
    request = _split(text)[2]
    assert request == "the [MASK] chased a [MASK]"
    assert request.replace(WA.MASK, "cat") == prompt
    _, (text, _) = _word_art("The CAT chased a cat", "cat")
    assert _split(text)[2] == "The [MASK] chased a [MASK]"


def test_the_art_is_the_word_in_the_font_ascii_art_draws_for_the_seed(stand_ins):
    for seed in range(10):
        art, drawn = to_ascii_art("mat", rng=random.Random(seed))
        _, (text, info) = _word_art(PROMPT, "mat", seed)
        assert _split(text)[1] == art.rstrip()
        assert drawn in info.split(";")


def test_the_art_has_no_trailing_blank_lines(stand_ins):
    _, (text, _) = _word_art(PROMPT, "mat")
    art = _split(text)[1]
    assert art == art.rstrip()


def test_only_the_first_marked_word_is_drawn(stand_ins):
    _, (text, info) = _word_art(PROMPT, "mat cat")
    assert _split(text)[2] == "the cat sat on the [MASK] today"
    assert info.endswith("detected_words=mat")


def test_the_info_names_the_variant_the_font_and_the_word(stand_ins):
    variants, fonts = set(), set()
    for seed in range(40):
        _, (text, info) = _word_art(PROMPT, "mat", seed)
        variant, font, word = re.fullmatch(
            r"variant=(\w+);font=(\w+);detected_words=(\w+)", info).groups()
        assert (word, _split(text)[0]) == ("mat", variant)
        variants.add(variant)
        fonts.add(font)
    assert variants == {"one", "two"}
    assert len(fonts) > 1


def test_a_replayed_seed_gives_the_same_text(stand_ins):
    first = _word_art(PROMPT, "mat", 5)[1]
    assert _word_art(PROMPT, "mat", 5)[1] == first


# --- what discards the prompt ----------------------------------------------------

def test_a_reply_that_marks_no_word_discards_the_prompt(stand_ins):
    requests, (text, info) = _word_art(PROMPT, "none")
    assert (text, info) == (PROMPT, NO_SENSITIVE_WORD)
    assert info.startswith(DISCARDED)
    assert len(requests) == 1


@pytest.mark.parametrize("reply", [
    "dog",                                                      # not in the prompt
    "the cat sat on the mat today and the cat sat on the mat",  # too many words
    "sat 5x7",                                                  # not a word
], ids=["not in the prompt", "too many", "not a word"])
def test_a_reply_that_is_not_a_short_list_of_the_prompt_s_words_discards_it(
        stand_ins, reply):
    _, (text, info) = _word_art(PROMPT, reply)
    assert (text, info) == (PROMPT, MALFORMED_WORDS)


def test_a_word_no_font_draws_is_discarded_with_ascii_art_s_reason(stand_ins):
    prompt = "the 猫 sat"
    _, (text, info) = _word_art(prompt, "猫")
    assert text == prompt
    assert info == to_ascii_art("猫")[1]
    assert info.startswith(f"{DISCARDED}; reason=")
