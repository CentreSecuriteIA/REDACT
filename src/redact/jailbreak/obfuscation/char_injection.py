"""Character injection and reordering: a text a reader still reads as the
prompt, with characters reordered, flipped, deleted or marked so a classifier
reading the raw code points meets something else (Boucher 2106.09898 bad
characters, Hackett 2504.11168 char injections).

Four kinds, all pure:

- :func:`to_bidi_override` stores the prompt reversed inside a
  right-to-left override (``U+202E`` ... ``U+202C``), so the Unicode
  bidirectional algorithm renders it in the original order
  (:func:`render_override` is a reference renderer of the override, used by
  the tests). The raw code points are the prompt reversed.
- :func:`to_upside_down` maps each character to its upside-down look-alike and
  reverses the string, so it reads as flipped text; a reader turns it the
  right way up, a classifier reads look-alike code points. A character with no
  look-alike passes through (still reversed).
- ``to_deletion_chars_{low,medium,high}`` inject, at three densities, a junk
  letter followed by a backspace control character (``U+0008``) between
  characters: a terminal that honours the backspace deletes the junk, so a
  reader sees the prompt, while the raw text carries the junk
  (:func:`strip_deletions` removes the injected pairs).
- ``to_zalgo_{low,medium,high}`` stack combining diacritical marks
  (``U+0300``-``U+036F``) on each visible character at three densities; the
  base characters are unchanged, so :func:`strip_zalgo` removes the marks and
  gives the prompt back.

These reorder or inject across the whole text, so the family refuses every
other obfuscation family (one would corrupt or be corrupted by the injection)
but still joins the hacking, request and manipulation layers, which frame or
wrap the finished text around it. ``info`` is ``""``.
"""

import random
import unicodedata
from collections.abc import Callable

#: The right-to-left override and the pop that closes it.
_RLO, _LRO, _PDF = "‮", "‭", "‬"
#: The backspace control character a terminal renders by deleting the
#: preceding character.
_BACKSPACE = "\u0008"
#: The combining diacritical marks zalgo stacks: the block U+0300-U+036F, less
#: the few code points with combining class 0 (the grapheme joiner U+034F),
#: so each one visibly stacks on its base character.
_COMBINING = [chr(point) for point in range(0x0300, 0x0370)
              if unicodedata.combining(chr(point))]
#: The junk letters a deletion injects before each backspace.
_JUNK = "abcdefghijklmnopqrstuvwxyz"

#: Injected deletion pairs per gap, and combining marks per character, by
#: density.
_DELETION_RATE = {"low": 0.2, "medium": 0.4, "high": 0.7}
_ZALGO_MARKS = {"low": (1, 2), "medium": (3, 5), "high": (6, 10)}

#: The upside-down look-alike of each character; applied then the string is
#: reversed. Every value is distinct, so the map inverts (the tests check it).
_FLIP: dict[str, str] = {
    "a": "ɐ", "b": "q", "c": "ɔ", "d": "p", "e": "ǝ",
    "f": "ɟ", "g": "ƃ", "h": "ɥ", "i": "ᴉ", "j": "ɾ",
    "k": "ʞ", "l": "ʅ", "m": "ɯ", "n": "u", "o": "o",
    "p": "d", "q": "b", "r": "ɹ", "s": "s", "t": "ʇ",
    "u": "n", "v": "ʌ", "w": "ʍ", "x": "x", "y": "ʎ", "z": "z",
    "A": "∀", "B": "ᙠ", "C": "Ɔ", "D": "ᗡ", "E": "Ǝ",
    "F": "Ⅎ", "G": "⅁", "H": "H", "I": "I", "J": "ſ",
    "K": "⋊", "L": "⅂", "M": "W", "N": "N", "O": "O",
    "P": "Ԁ", "Q": "Ό", "R": "ᴚ", "S": "S", "T": "⊥",
    "U": "∩", "V": "Λ", "W": "M", "X": "X", "Y": "⅄", "Z": "Z",
    "0": "0", "1": "Ɩ", "2": "ᄅ", "3": "Ɛ", "4": "ㄣ",
    "5": "Ϛ", "6": "9", "7": "ㄥ", "8": "8", "9": "6",
    ".": "˙", ",": "'", "?": "¿", "!": "¡", '"': "„",
    "'": ",", "(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{",
    "<": ">", ">": "<", "&": "⅋", "_": "‾",
}


def render_override(text: str) -> str:
    """The visual order of ``text`` under a simplified Unicode bidirectional
    override: a right-to-left override (``U+202E``) reverses the run up to its
    pop (``U+202C``), a left-to-right override (``U+202D``) keeps it. Only the
    override form :func:`to_bidi_override` writes is modelled; the control
    characters themselves are not shown.

    Reversing a right-to-left run and reversing it back are inverse, so
    ``render_override`` of a ``to_bidi_override`` output is the exact prompt.
    """
    base: list[str] = []
    stack: list[tuple[str, list[str]]] = [("L", base)]
    for char in text:
        if char == _RLO:
            stack.append(("R", []))
        elif char == _LRO:
            stack.append(("L", []))
        elif char == _PDF:
            if len(stack) > 1:
                direction, buffer = stack.pop()
                stack[-1][1].extend(reversed(buffer) if direction == "R" else buffer)
        else:
            stack[-1][1].append(char)
    while len(stack) > 1:
        direction, buffer = stack.pop()
        stack[-1][1].extend(reversed(buffer) if direction == "R" else buffer)
    return "".join(base)


def strip_deletions(text: str) -> str:
    """``text`` without the junk-then-backspace pairs a deletion injected: each
    backspace and the one character before it are removed, as a terminal would
    render them."""
    out: list[str] = []
    for char in text:
        if char == _BACKSPACE:
            if out:
                out.pop()
        else:
            out.append(char)
    return "".join(out)


def strip_zalgo(text: str) -> str:
    """``text`` without the combining marks (``U+0300``-``U+036F``) zalgo
    stacked."""
    return "".join(char for char in text if not 0x0300 <= ord(char) <= 0x036F)  # noqa: PLR2004


def to_bidi_override(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Store the prompt reversed inside a right-to-left override, so it renders
    in order while the raw code points are reversed."""
    return _RLO + prompt[::-1] + _PDF, ""


def to_upside_down(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Flip each character to its upside-down look-alike and reverse the
    string, so it reads as flipped text."""
    return "".join(_FLIP.get(char, char) for char in prompt)[::-1], ""


def _make_deletion(level: str) -> Callable:
    rate = _DELETION_RATE[level]

    def fn(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
        rng = rng or random
        out: list[str] = []
        for char in prompt:
            out.append(char)
            if rng.random() < rate:
                out.append(rng.choice(_JUNK))
                out.append(_BACKSPACE)
        return "".join(out), ""

    fn.__name__ = fn.__qualname__ = f"to_deletion_chars_{level}"
    fn.__doc__ = f"Inject deletion pairs at {level} density."
    return fn


def _make_zalgo(level: str) -> Callable:
    low, high = _ZALGO_MARKS[level]

    def fn(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
        rng = rng or random
        out: list[str] = []
        for char in prompt:
            out.append(char)
            if not char.isspace():
                for _ in range(rng.randint(low, high)):
                    out.append(rng.choice(_COMBINING))
        return "".join(out), ""

    fn.__name__ = fn.__qualname__ = f"to_zalgo_{level}"
    fn.__doc__ = f"Stack {level}-density combining marks on characters."
    return fn


to_deletion_chars_low = _make_deletion("low")
to_deletion_chars_medium = _make_deletion("medium")
to_deletion_chars_high = _make_deletion("high")
to_zalgo_low = _make_zalgo("low")
to_zalgo_medium = _make_zalgo("medium")
to_zalgo_high = _make_zalgo("high")


def get_char_injection_functions() -> list[Callable]:
    """Return all character-injection technique functions."""
    return [
        to_bidi_override,
        to_upside_down,
        to_deletion_chars_low,
        to_deletion_chars_medium,
        to_deletion_chars_high,
        to_zalgo_low,
        to_zalgo_medium,
        to_zalgo_high,
    ]
