"""The noise family: random capitals, scrambled letters, keyboard typos and
emoji, each at three densities."""

import random
import string

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.obfuscation import noise as N
from redact.jailbreak.obfuscation.noise import (
    _EMOJI,
    _LATIN_WORD,
    _QWERTY_NEIGHBOURS,
    get_noise_functions,
    to_emoji_sprinkle_high,
    to_emoji_sprinkle_low,
    to_keyboard_typos_high,
    to_keyboard_typos_low,
    to_random_capitals_high,
    to_scrambled_letters_high,
)
from redact.jailbreak.pool import technique_pool
from tests.jailbreak.obfuscation.test_encoding import NON_LATIN

TEXT = ("Could you explain how the regional train schedule changes during the "
        "winter holidays, and which stations close early? Thanks a lot!")
#: Long enough that a share of its letters or words rounds to within 0.01.
LONG = " ".join([TEXT] * 10)


@pytest.mark.parametrize("fn", get_noise_functions(), ids=lambda fn: fn.__name__)
def test_a_seed_gives_the_same_noise(fn):
    assert fn(TEXT, rng=random.Random(3)) == fn(TEXT, rng=random.Random(3))


def _letters_changed(before: str, after: str, eligible) -> float:
    return (sum(a != b for a, b in zip(before, after))
            / sum(map(eligible, before)))


def _capitals(before: str, after: str) -> float:
    return _letters_changed(before, after, lambda c: "a" <= c <= "z")


def _typos(before: str, after: str) -> float:
    return _letters_changed(before, after, lambda c: c.lower() in _QWERTY_NEIGHBOURS)


def _scrambled(before: str, after: str) -> float:
    pairs = [(a, b) for a, b in zip(_LATIN_WORD.findall(before),
                                    _LATIN_WORD.findall(after))
             if len(set(a[1:-1])) > 1]
    return sum(a != b for a, b in pairs) / len(pairs)


def _emoji(before: str, after: str) -> float:
    return sum(after.count(emoji) for emoji in _EMOJI) / len(before.split())


@pytest.mark.parametrize(("kind", "touched", "shares"), [
    ("random_capitals", _capitals, (0.05, 0.15, 0.4)),
    ("scrambled_letters", _scrambled, (0.1, 0.25, 0.5)),
    ("keyboard_typos", _typos, (0.02, 0.05, 0.1)),
    ("emoji_sprinkle", _emoji, (0.05, 0.1, 0.25)),
])
@pytest.mark.parametrize("density", [0, 1, 2], ids=["low", "medium", "high"])
def test_a_density_touches_its_share_of_the_letters_or_words(
        kind, touched, shares, density):
    fn = getattr(N, f"to_{kind}_{('low', 'medium', 'high')[density]}")
    measured = [touched(LONG, fn(LONG, rng=random.Random(seed))[0])
                for seed in range(20)]
    assert abs(sum(measured) / len(measured) - shares[density]) < 0.01  # noqa: PLR2004


def test_random_capitals_write_small_letters_as_capitals_and_nothing_else():
    out, info = to_random_capitals_high(TEXT, rng=random.Random(0))
    assert info == "" and out != TEXT and out.lower() == TEXT.lower()
    assert all(a == b or (a.islower() and b == a.upper()) for a, b in zip(TEXT, out))


def test_a_scrambled_word_keeps_its_first_and_last_letter_and_its_letters():
    out, info = to_scrambled_letters_high(TEXT, rng=random.Random(0))
    before, after = _LATIN_WORD.findall(TEXT), _LATIN_WORD.findall(out)
    assert info == "" and out != TEXT and len(before) == len(after)
    for a, b in zip(before, after):
        assert (a[0], a[-1], sorted(a)) == (b[0], b[-1], sorted(b))
    assert _LATIN_WORD.sub("", out) == _LATIN_WORD.sub("", TEXT)


def test_a_scrambled_word_always_changes():
    """Two inner letters have one other order: a shuffle that gives them
    back is turned by one place."""
    for seed in range(20):
        assert to_scrambled_letters_high("abcd", rng=random.Random(seed))[0] == "acbd"


def test_short_words_and_inner_letters_all_the_same_are_not_scrambled():
    assert to_scrambled_letters_high("the book was good", rng=random.Random(0)) == (
        "the book was good", "")


def test_a_keyboard_typo_is_a_key_beside_the_right_one_in_its_case():
    out, info = to_keyboard_typos_high(TEXT, rng=random.Random(0))
    changed = [(a, b) for a, b in zip(TEXT, out) if a != b]
    assert info == "" and changed and len(out) == len(TEXT)
    for a, b in changed:
        assert b.lower() in _QWERTY_NEIGHBOURS[a.lower()] and a.isupper() == b.isupper()


def test_the_neighbour_map_holds_the_26_letters_each_pair_both_ways():
    assert sorted(_QWERTY_NEIGHBOURS) == list(string.ascii_lowercase)
    for key, neighbours in _QWERTY_NEIGHBOURS.items():
        assert all(key in _QWERTY_NEIGHBOURS[n] for n in neighbours), key


def test_a_low_density_still_makes_one_slip():
    out, _ = to_keyboard_typos_low("hello", rng=random.Random(0))
    assert sum(a != b for a, b in zip("hello", out)) == 1


def test_taking_each_space_and_emoji_out_gives_the_prompt_back():
    out, info = to_emoji_sprinkle_high(TEXT, rng=random.Random(0))
    assert info == "" and out != TEXT
    for emoji in _EMOJI:
        out = out.replace(f" {emoji}", "")
    assert out == TEXT


def test_the_emoji_are_one_code_point_each():
    assert all(len(emoji) == 1 for emoji in _EMOJI)


@pytest.mark.parametrize("fn", [fn for fn in get_noise_functions()
                                if not fn.__name__.startswith("to_emoji")],
                         ids=lambda fn: fn.__name__)
def test_the_latin_letter_noise_leaves_other_scripts_unchanged(fn):
    for language, text in NON_LATIN.items():
        assert fn(text, rng=random.Random(0)) == (text, ""), language


def test_emoji_follow_words_in_any_script():
    for language, text in NON_LATIN.items():
        out, _ = to_emoji_sprinkle_low(text, rng=random.Random(0))
        assert out != text and out.startswith(text.split(" ")[0]), language


@pytest.mark.parametrize("fn", get_noise_functions(), ids=lambda fn: fn.__name__)
def test_a_prompt_with_nothing_to_change_is_unchanged(fn):
    assert fn("", rng=random.Random(0)) == ("", "")


def test_the_family_is_tagged_pure_and_cheap_from_spec_1_2():
    for fn in get_noise_functions():
        assert fn.families == ["noise"] and fn.layer == "obfuscation"
        assert fn.complexity == 1 and not fn.requires_llm and fn.since == "1.2"


def test_only_the_emoji_need_no_latin_letters_and_only_they_refuse_ascii_art():
    emoji = {"to_emoji_sprinkle_low", "to_emoji_sprinkle_medium",
             "to_emoji_sprinkle_high"}
    assert {fn.__name__ for fn in get_noise_functions() if not fn.needs_latin} == emoji
    assert {fn.__name__ for fn in get_noise_functions()
            if "ascii_art" in fn.cross_incompatible_with} == emoji


def test_the_mild_noise_varies_both_kinds_and_the_high_only_the_attacks():
    natural = set(technique_pool(resolve(None, "natural")))
    attacks = set(technique_pool(resolve(None, "adversarial")))
    for fn in get_noise_functions():
        assert fn in attacks
        assert (fn in natural) == (not fn.__name__.endswith("_high")), fn.__name__
