"""The casual writing family: casing, spacing, punctuation, abbreviations, accents."""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.obfuscation.casual_writing import (
    get_casual_writing_functions,
    to_abbreviations,
    to_doubled_punctuation,
    to_doubled_spaces,
    to_dropped_punctuation,
    to_dropped_spaces,
    to_lowercase,
    to_no_sentence_capitals,
    to_stripped_accents,
    to_uppercase,
)
from redact.jailbreak.pool import technique_pool
from redact.jailbreak.technique import applies
from tests.jailbreak import simulate

TEXT = "Can you tell me about the café? It costs 3.5 euros. Thanks, Anna!"


@pytest.mark.parametrize(("fn", "expected"), [
    (to_lowercase, "can you tell me about the café? it costs 3.5 euros. thanks, anna!"),
    (to_uppercase, "CAN YOU TELL ME ABOUT THE CAFÉ? IT COSTS 3.5 EUROS. THANKS, ANNA!"),
    (to_no_sentence_capitals,
     "can you tell me about the café? it costs 3.5 euros. thanks, Anna!"),
    (to_dropped_punctuation,
     "Can you tell me about the café It costs 3.5 euros Thanks Anna"),
    (to_doubled_punctuation,
     "Can you tell me about the café?? It costs 3.5 euros.. Thanks, Anna!!"),
    (to_abbreviations, "Can u tell me abt the café? It costs 3.5 euros. thx, Anna!"),
    (to_stripped_accents,
     "Can you tell me about the cafe? It costs 3.5 euros. Thanks, Anna!"),
])
def test_a_transform_writes_the_prompt_as_a_person_might(fn, expected):
    assert fn(TEXT) == (expected, "")


@pytest.mark.parametrize(("fn", "spaces"), [(to_dropped_spaces, 10),
                                            (to_doubled_spaces, 14)])
def test_a_share_of_the_spaces_changes_by_the_rng(fn, spaces):
    text, info = fn(TEXT, rng=random.Random(1))
    assert info == "" and text.count(" ") == spaces      # 2 of TEXT's 12
    assert text.replace(" ", "") == TEXT.replace(" ", "")
    assert fn(TEXT, rng=random.Random(1)) == (text, info)


def test_a_sentence_starting_with_an_accented_capital_is_lowered():
    assert to_no_sentence_capitals("Élan vital. Über alles.") == (
        "élan vital. über alles.", "")


@pytest.mark.parametrize(("text", "expected"), [
    ("don't", "dont"), ("don’t", "dont"), ("“Yes” – «no» — ‘ok’", "Yes  no  ok"),
])
def test_curly_quotes_and_dashes_are_dropped_as_the_straight_ones(text, expected):
    assert to_dropped_punctuation(text) == (expected, "")


@pytest.mark.parametrize("text", [
    "がぎぐ パン", "ที่นี่ ต้ม", "नमस्ते", "ក្រុម", "বন্ধু", "plain ascii",
])
def test_the_marks_of_other_scripts_stay(text):
    assert to_stripped_accents(text) == (text, "")


def test_accents_leave_latin_letters():
    assert to_stripped_accents("café naïve Việt") == ("cafe naive Viet", "")


# Neutral texts with and without a word of the abbreviation map or an
# accented Latin letter, in Latin and other scripts, composed and decomposed.
_SAMPLES = [
    TEXT, "", "plain ascii", "Please send the message tomorrow.",
    "THANKS, YOU ARE KIND", "yourself", "a pupil's notes", "café naïve Việt",
    "café", "Über alles", "Привет, как дела?", "がぎぐ パン", "नमस्ते",
]


@pytest.mark.parametrize("fn", [to_abbreviations, to_stripped_accents])
def test_a_technique_applies_exactly_to_the_texts_it_changes(fn):
    entry_types = ("harmful", "dual_use_harmful", "dual_use_benign", "benign")
    answers = {(text, entry_type): applies(fn, text, entry_type)
               for text in _SAMPLES for entry_type in entry_types}
    assert answers == {(text, entry_type): fn(text)[0] != text
                       for text, entry_type in answers}
    assert set(answers.values()) == {True, False}     # the samples hold both


@pytest.mark.parametrize("text", [
    "\u0301", "\u0301How do I pick a lock?", "\u0323\u0301e", "\u0e31ก",
    "\u0301e\u0301", " \u0301café", "\u0301\u0301",
])
def test_a_text_starting_with_a_mark_keeps_that_mark(text):
    """A mark with no letter before it has none to strip."""
    stripped, _ = to_stripped_accents(text)
    assert stripped.startswith(text[0])
    assert applies(to_stripped_accents, text, "harmful") == (stripped != text)


def test_no_mix_of_letters_and_marks_makes_a_technique_or_its_predicate_raise():
    """Random texts of Latin letters, accented letters, marks of several
    scripts, other scripts, spaces and punctuation, many starting with a mark."""
    alphabet = ["a", "e", "n", "é", "ñ", "ǿ", "я", "か", "क", "ก", " ", ".", "3",
                "\u0300", "\u0301", "\u0323", "\u0308", "\u0e31", "\u093f"]
    rng = random.Random(0)
    for _ in range(3000):
        text = "".join(rng.choices(alphabet, k=rng.randint(0, 10)))
        for fn in (to_abbreviations, to_stripped_accents):
            assert applies(fn, text, "harmful") == (fn(text)[0] != text)


def test_a_natural_plan_draws_neither_for_a_text_it_would_leave_unchanged():
    """The simulated inputs hold no accented letter and the word "about"."""
    combinations, _ = simulate.deal_kind({}, 300, kind="natural", groups=3)
    drawn = {name for names in combinations for name in names}
    assert "to_stripped_accents" not in drawn and "to_abbreviations" in drawn


def test_a_prompt_without_spaces_keeps_them():
    assert to_dropped_spaces("one", rng=random.Random(0)) == ("one", "")


def test_the_family_is_tagged_pure_and_cheap():
    for fn in get_casual_writing_functions():
        assert fn.families == ["casual_writing"] and fn.layer == "obfuscation"
        assert fn.complexity in (0, 1) and not fn.requires_llm


def test_the_family_varies_the_natural_kind_and_not_the_attacks():
    family = set(get_casual_writing_functions())
    assert family <= set(technique_pool(resolve(None, "natural")))
    assert not family & set(technique_pool(resolve(None, "adversarial")))
