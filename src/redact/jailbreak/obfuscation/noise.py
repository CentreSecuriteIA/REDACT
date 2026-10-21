"""Noise: the slips of a person typing, each at three densities: random
capitals, scrambled letters, keyboard typos and emoji between words.

A density is the share of the letters or words a technique may change that it
changes, at least one when the prompt has any. Random capitals, scrambled
letters and keyboard typos change only Latin letters; emoji follow words in
any script.

``info`` is empty.
"""

import random
import re
from collections.abc import Callable

#: Share of the ASCII small letters written as capitals.
_CAPITALS_SHARE = {"low": 0.05, "medium": 0.15, "high": 0.4}
#: Share of the words that can be scrambled (:func:`_scrambled_letters`)
#: whose inner letters are shuffled.
_SCRAMBLED_SHARE = {"low": 0.1, "medium": 0.25, "high": 0.5}
#: Share of the letters on :data:`_QWERTY_NEIGHBOURS` replaced by a key beside them.
_TYPO_SHARE = {"low": 0.02, "medium": 0.05, "high": 0.1}
#: Share of the words an emoji follows.
_EMOJI_SHARE = {"low": 0.05, "medium": 0.1, "high": 0.25}

#: A word as the scrambled letters see it: a run of Latin letters (ASCII,
#: Latin-1, Latin Extended-A and -B, Latin Extended Additional).
_LATIN_WORD = re.compile(
    "[A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ]+")

#: The keys that touch each letter's key on a QWERTY keyboard: the ones beside
#: it in its row, and the two it overlaps in the row above and the row below
#: (the rows are staggered). Each pair is listed both ways.
_QWERTY_NEIGHBOURS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "sqwz", "s": "adwezx", "d": "sferxc", "f": "dgrtcv", "g": "fhtyvb",
    "h": "gjyubn", "j": "hkuinm", "k": "jliom", "l": "kop",
    "z": "xas", "x": "zcsd", "c": "xvdf", "v": "cbfg", "b": "vngh",
    "n": "bmhj", "m": "njk",
}

#: Neutral emoji, one code point each, that add a tone and no meaning: a slight
#: smile, a smile, a sweat smile, thumbs up, folded hands, a thinking face,
#: sparkles, raised hands.
_EMOJI = ("\U0001f642", "\U0001f60a", "\U0001f605", "\U0001f44d",
          "\U0001f64f", "\U0001f914", "✨", "\U0001f64c")


def _chosen(positions: list, share: float, rng) -> set:
    """A ``share`` of ``positions``, drawn by ``rng``: at least one when there
    are any."""
    if not positions:
        return set()
    return set(rng.sample(positions, max(1, round(len(positions) * share))))


def _random_capitals(prompt: str, share: float, rng) -> str:
    """``prompt`` with a ``share`` of its ASCII small letters written as capitals."""
    chosen = _chosen([i for i, c in enumerate(prompt) if "a" <= c <= "z"], share, rng)
    return "".join(c.upper() if i in chosen else c for i, c in enumerate(prompt))


def _scramble(word: str, rng) -> str:
    """``word`` with its inner letters shuffled, the first and last kept;
    turned by one place when the shuffle gives them back in order, so the word
    always changes (its inner letters are not all the same)."""
    inner = list(word[1:-1])
    rng.shuffle(inner)
    if "".join(inner) == word[1:-1]:
        inner = inner[1:] + inner[:1]
    return word[0] + "".join(inner) + word[-1]


def _scrambled_letters(prompt: str, share: float, rng) -> str:
    """``prompt`` with a ``share`` of its Latin words scrambled
    (:func:`_scramble`): those whose inner letters are not all the same."""
    spans = [m.span() for m in _LATIN_WORD.finditer(prompt) if len(set(m[0][1:-1])) > 1]
    out, end = [], 0
    for start, stop in sorted(_chosen(spans, share, rng)):
        out += [prompt[end:start], _scramble(prompt[start:stop], rng)]
        end = stop
    return "".join(out) + prompt[end:]


def _slip(letter: str, rng) -> str:
    """A key beside ``letter``'s on a QWERTY keyboard, drawn by ``rng``, in its case."""
    key = rng.choice(_QWERTY_NEIGHBOURS[letter.lower()])
    return key.upper() if letter.isupper() else key


def _keyboard_typos(prompt: str, share: float, rng) -> str:
    """``prompt`` with a ``share`` of its letters on :data:`_QWERTY_NEIGHBOURS`
    each replaced by a key beside it (:func:`_slip`)."""
    chosen = _chosen([i for i, c in enumerate(prompt)
                      if c.lower() in _QWERTY_NEIGHBOURS], share, rng)
    return "".join(_slip(c, rng) if i in chosen else c for i, c in enumerate(prompt))


def _emoji_sprinkle(prompt: str, share: float, rng) -> str:
    """``prompt`` with a space and an emoji of :data:`_EMOJI`, drawn by ``rng``,
    after a ``share`` of its space-separated words; taking each space-and-emoji
    out gives the prompt back."""
    words = prompt.split(" ")
    chosen = _chosen([i for i, word in enumerate(words) if word], share, rng)
    return " ".join(f"{word} {rng.choice(_EMOJI)}" if i in chosen else word
                    for i, word in enumerate(words))


def to_random_capitals_low(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Write a few small letters as capitals."""
    return _random_capitals(prompt, _CAPITALS_SHARE["low"], rng or random), ""


def to_random_capitals_medium(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Write some small letters as capitals."""
    return _random_capitals(prompt, _CAPITALS_SHARE["medium"], rng or random), ""


def to_random_capitals_high(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Write many small letters as capitals."""
    return _random_capitals(prompt, _CAPITALS_SHARE["high"], rng or random), ""


def to_scrambled_letters_low(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Shuffle the inner letters of a few words."""
    return _scrambled_letters(prompt, _SCRAMBLED_SHARE["low"], rng or random), ""


def to_scrambled_letters_medium(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Shuffle the inner letters of some words."""
    return _scrambled_letters(prompt, _SCRAMBLED_SHARE["medium"], rng or random), ""


def to_scrambled_letters_high(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Shuffle the inner letters of many words."""
    return _scrambled_letters(prompt, _SCRAMBLED_SHARE["high"], rng or random), ""


def to_keyboard_typos_low(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Hit a key beside the right one for a few letters."""
    return _keyboard_typos(prompt, _TYPO_SHARE["low"], rng or random), ""


def to_keyboard_typos_medium(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Hit a key beside the right one for some letters."""
    return _keyboard_typos(prompt, _TYPO_SHARE["medium"], rng or random), ""


def to_keyboard_typos_high(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Hit a key beside the right one for many letters."""
    return _keyboard_typos(prompt, _TYPO_SHARE["high"], rng or random), ""


def to_emoji_sprinkle_low(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put an emoji after a few words."""
    return _emoji_sprinkle(prompt, _EMOJI_SHARE["low"], rng or random), ""


def to_emoji_sprinkle_medium(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put an emoji after some words."""
    return _emoji_sprinkle(prompt, _EMOJI_SHARE["medium"], rng or random), ""


def to_emoji_sprinkle_high(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put an emoji after many words."""
    return _emoji_sprinkle(prompt, _EMOJI_SHARE["high"], rng or random), ""


def get_noise_functions() -> list[Callable]:
    """Return all noise technique functions."""
    return [
        to_random_capitals_low,
        to_random_capitals_medium,
        to_random_capitals_high,
        to_scrambled_letters_low,
        to_scrambled_letters_medium,
        to_scrambled_letters_high,
        to_keyboard_typos_low,
        to_keyboard_typos_medium,
        to_keyboard_typos_high,
        to_emoji_sprinkle_low,
        to_emoji_sprinkle_medium,
        to_emoji_sprinkle_high,
    ]
