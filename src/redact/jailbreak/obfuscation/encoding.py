"""Encoding techniques: the prompt as a cipher, a character code, a byte codec,
Morse or Braille, reversed, or in look-alike, fullwidth or math-styled letters.

``info`` is empty, except ``sep=<separator>`` from ``to_separator``,
``shift=<n>`` from ``to_caesar_cipher`` and ``style=<name>`` from
``to_math_styled`` (traceability metadata, not part of the encoded text).
"""

import base64
import codecs
import random
import unicodedata
import urllib.parse
from collections.abc import Callable

# ROT47 operates on the printable-ASCII range '!' (33) through '~' (126).
_ROT47_PRINTABLE_MIN = 33
_ROT47_PRINTABLE_MAX = 126

# The last code point a four-digit escape can name.
_BMP_MAX = 0xFFFF

# Joined to the character before them: the zero-width joiner and the two
# emoji variation selectors (as is every combining mark).
_ZWJ = "\u200d"
_JOINED = {_ZWJ, "\ufe0e", "\ufe0f"}


def to_base64(prompt: str) -> tuple[str, str]:
    """Encode string to Base64."""
    return base64.b64encode(prompt.encode()).decode(), ""


def to_rot13(prompt: str) -> tuple[str, str]:
    """Encode string using ROT13 cipher (letters only)."""
    return codecs.encode(prompt, "rot_13"), ""


def to_rot18(prompt: str) -> tuple[str, str]:
    """Encode string using ROT18: ROT13 for letters + ROT5 for ASCII digits."""
    result = []
    for c in prompt:
        if c.isalpha():
            result.append(codecs.encode(c, "rot_13"))
        elif "0" <= c <= "9":
            result.append(str((int(c) + 5) % 10))
        else:
            result.append(c)
    return "".join(result), ""


def to_rot47(prompt: str) -> tuple[str, str]:
    """Encode string using ROT47: rotate all printable ASCII (33-126) by 47."""
    result = []
    for c in prompt:
        o = ord(c)
        if _ROT47_PRINTABLE_MIN <= o <= _ROT47_PRINTABLE_MAX:
            result.append(chr((o - _ROT47_PRINTABLE_MIN + 47) % 94 + _ROT47_PRINTABLE_MIN))
        else:
            result.append(c)
    return "".join(result), ""


def to_unicode_escape(prompt: str) -> tuple[str, str]:
    """Encode each character as a Unicode escape sequence (\\uXXXX, or
    \\UXXXXXXXX above the Basic Multilingual Plane)."""
    return "".join(
        f"\\U{ord(c):08x}" if ord(c) > _BMP_MAX else f"\\u{ord(c):04x}"
        for c in prompt), ""


def to_ascii_ordinal(prompt: str) -> tuple[str, str]:
    """Encode each character as its decimal ASCII ordinal, space-separated."""
    return " ".join(str(ord(c)) for c in prompt), ""


SEPARATORS = ["-", "_", ".", "|", "/", "*", "~"]


def _clusters(text: str) -> list[str]:
    """``text`` cut into the characters a reader sees: a combining mark, a
    zero-width joiner or a variation selector stays with the character before
    it, and so does the character after a joiner."""
    clusters: list[str] = []
    for c in text:
        joined = (unicodedata.category(c).startswith("M") or c in _JOINED
                  or (clusters and clusters[-1].endswith(_ZWJ)))
        if clusters and joined:
            clusters[-1] += c
        else:
            clusters.append(c)
    return clusters


def to_separator(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Insert a random separator character between every character a reader sees."""
    sep = (rng or random).choice(SEPARATORS)
    return sep.join(_clusters(prompt)), f"sep={sep}"


_LEET_MAP_BASIC = {
    "a": "4", "A": "4",
    "e": "3", "E": "3",
    "i": "1", "I": "1",
    "o": "0", "O": "0",
    "s": "5", "S": "5",
    "t": "7", "T": "7",
}

_LEET_MAP_INTERMEDIATE = {
    **_LEET_MAP_BASIC,
    "b": "8", "B": "8",
    "g": "9", "G": "9",
    "l": "|", "L": "|",
    "z": "2", "Z": "2",
    "c": "(", "C": "(",
    "r": "|2", "R": "|2",
}

_LEET_MAP_ADVANCED = {
    **_LEET_MAP_INTERMEDIATE,
    "h": "|-|", "H": "|-|",
    "n": "|\\|", "N": "|\\|",
    "u": "|_|", "U": "|_|",
    "w": "\\/\\/", "W": "\\/\\/",
    "x": "><", "X": "><",
    "f": "|=", "F": "|=",
    "k": "|<", "K": "|<",
    "m": "|\\/|", "M": "|\\/|",
    "v": "\\/", "V": "\\/",
}


def to_leetspeak_basic(prompt: str) -> tuple[str, str]:
    """Leetspeak with 6-character substitution map (vowels + s/t)."""
    return "".join(_LEET_MAP_BASIC.get(c, c) for c in prompt), ""


def to_leetspeak_intermediate(prompt: str) -> tuple[str, str]:
    """Leetspeak with extended map adding b, g, l, z, c, r substitutions."""
    return "".join(_LEET_MAP_INTERMEDIATE.get(c, c) for c in prompt), ""


def to_leetspeak_advanced(prompt: str) -> tuple[str, str]:
    """Leetspeak with full map including multi-character substitutions."""
    return "".join(_LEET_MAP_ADVANCED.get(c, c) for c in prompt), ""


_MORSE_MAP = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    " ": "/",
}


def to_morse(prompt: str) -> tuple[str, str]:
    """Convert string to Morse code."""
    return " ".join(_MORSE_MAP.get(c.upper(), c) for c in prompt), ""


_BRAILLE_MAP = {
    "a": "\u2801", "b": "\u2803", "c": "\u2809", "d": "\u2819",
    "e": "\u2811", "f": "\u280b", "g": "\u281b", "h": "\u2813",
    "i": "\u280a", "j": "\u281a", "k": "\u2805", "l": "\u2807",
    "m": "\u280d", "n": "\u281d", "o": "\u2815", "p": "\u280f",
    "q": "\u281f", "r": "\u2817", "s": "\u280e", "t": "\u281e",
    "u": "\u2825", "v": "\u2827", "w": "\u283a", "x": "\u282d",
    "y": "\u283d", "z": "\u2835",
    " ": " ",
}


def to_braille(prompt: str) -> tuple[str, str]:
    """Convert string to Braille."""
    return "".join(_BRAILLE_MAP.get(c.lower(), c) for c in prompt), ""


def to_reversed_characters(prompt: str) -> tuple[str, str]:
    """Reverse the characters a reader sees, so a base letter keeps the
    combining marks that follow it."""
    return "".join(reversed(_clusters(prompt))), ""


def to_reversed_words(prompt: str) -> tuple[str, str]:
    """Reverse the characters a reader sees within each space-separated word,
    the word order kept."""
    return " ".join("".join(reversed(_clusters(word)))
                    for word in prompt.split(" ")), ""


def to_reversed_word_order(prompt: str) -> tuple[str, str]:
    """Reverse the order of the space-separated words, each word kept as it is."""
    return " ".join(reversed(prompt.split(" "))), ""


#: Latin letters and the visually identical Cyrillic letter each maps to, an
#: injective map so a reader (and a test) can map it back. ``to_lookalike_letters``
#: replaces a fraction of a prompt's mapped letters and keeps the rest.
_LOOKALIKE_MAP = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "J": "Ј",
    "K": "К", "M": "М", "O": "О", "P": "Р", "S": "Ѕ", "T": "Т", "X": "Х",
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о", "p": "р",
    "s": "ѕ", "x": "х", "y": "у",
}

#: The fractions of eligible characters ``to_lookalike_letters`` and
#: ``to_zero_width_noise`` draw from.
_NOISE_DENSITIES = (0.25, 0.5, 0.75, 1.0)


def to_lookalike_letters(
        prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Replace a fraction of the Latin letters that have a look-alike with the
    visually identical Cyrillic letter (:data:`_LOOKALIKE_MAP`), the fraction
    drawn from :data:`_NOISE_DENSITIES` and the letters by ``rng``; at least one
    is replaced, the rest kept."""
    rng = rng or random
    positions = [i for i, c in enumerate(prompt) if c in _LOOKALIKE_MAP]
    if not positions:
        return prompt, ""
    density = rng.choice(_NOISE_DENSITIES)
    chosen = set(rng.sample(positions, max(1, round(density * len(positions)))))
    return "".join(_LOOKALIKE_MAP[c] if i in chosen else c
                   for i, c in enumerate(prompt)), ""


#: The zero-width characters ``to_zero_width_noise`` scatters: the zero-width
#: space, non-joiner and joiner.
_ZERO_WIDTH = (chr(0x200B), chr(0x200C), chr(0x200D))


def to_zero_width_noise(
        prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Insert zero-width characters between the characters a reader sees, at a
    density drawn from :data:`_NOISE_DENSITIES` by ``rng`` (at least one gap
    filled); stripping the zero-width characters recovers the text."""
    rng = rng or random
    clusters = _clusters(prompt)
    if len(clusters) < 2:   # noqa: PLR2004
        return prompt, ""
    gaps = range(1, len(clusters))
    chosen = set(rng.sample(list(gaps), max(1, round(rng.choice(_NOISE_DENSITIES)
                                                      * len(gaps)))))
    out = []
    for i, cluster in enumerate(clusters):
        if i in chosen:
            out.append(rng.choice(_ZERO_WIDTH))
        out.append(cluster)
    return "".join(out), ""


def to_hex(prompt: str) -> tuple[str, str]:
    """Encode each UTF-8 byte as two hex digits, space-separated."""
    return " ".join(f"{b:02x}" for b in prompt.encode()), ""


def to_binary(prompt: str) -> tuple[str, str]:
    """Encode each UTF-8 byte as eight bits, space-separated."""
    return " ".join(f"{b:08b}" for b in prompt.encode()), ""


def to_url_encoding(prompt: str) -> tuple[str, str]:
    """Percent-encode the prompt, as a URL encodes its unsafe characters."""
    return urllib.parse.quote(prompt), ""


def to_html_entities(prompt: str) -> tuple[str, str]:
    """Encode each character as a numeric HTML entity (``&#<code>;``)."""
    return "".join(f"&#{ord(c)};" for c in prompt), ""


def to_base32(prompt: str) -> tuple[str, str]:
    """Encode the UTF-8 bytes as Base32."""
    return base64.b32encode(prompt.encode()).decode(), ""


def to_base85(prompt: str) -> tuple[str, str]:
    """Encode the UTF-8 bytes as Base85."""
    return base64.b85encode(prompt.encode()).decode(), ""


#: The shifts ``to_caesar_cipher`` draws from: 13 is left out, it is ``to_rot13``.
_CAESAR_SHIFTS = (3, 5, 7, 11, 17, 19, 23)


def to_caesar_cipher(
        prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Shift every Latin letter by an amount drawn from :data:`_CAESAR_SHIFTS`
    by ``rng`` (never 13, which is :func:`to_rot13`). The shift is reported in
    the info for traceability, not stated in the encoded text."""
    shift = (rng or random).choice(_CAESAR_SHIFTS)
    out = []
    for c in prompt:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - 97 + shift) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - 65 + shift) % 26 + 65))
        else:
            out.append(c)
    return "".join(out), f"shift={shift}"


def to_atbash(prompt: str) -> tuple[str, str]:
    """Mirror each Latin letter in the alphabet (a<->z, b<->y); applying it
    twice recovers the text."""
    out = []
    for c in prompt:
        if "a" <= c <= "z":
            out.append(chr(219 - ord(c)))   # ord('a') + ord('z')
        elif "A" <= c <= "Z":
            out.append(chr(155 - ord(c)))   # ord('A') + ord('Z')
        else:
            out.append(c)
    return "".join(out), ""


#: ASCII to its fullwidth forms: the fullwidth block (U+FF01 on) follows the
#: ASCII printable block (U+0021 on) at this offset.
_FULLWIDTH_OFFSET = 0xFF01 - 0x21


def to_fullwidth(prompt: str) -> tuple[str, str]:
    """Replace the Latin letters and ASCII digits with their fullwidth forms."""
    return "".join(
        chr(ord(c) + _FULLWIDTH_OFFSET)
        if c.isascii() and (c.isalpha() or c.isdigit()) else c
        for c in prompt), ""


#: Mathematical Alphanumeric Symbols styles with no reserved holes across A-Z
#: and a-z (the styles whose letters are all one contiguous run): the code
#: points of each style's ``A`` and ``a``. ``to_math_styled`` draws one.
_MATH_STYLES = {
    "bold": (0x1D400, 0x1D41A),
    "bold_italic": (0x1D468, 0x1D482),
    "bold_script": (0x1D4D0, 0x1D4EA),
    "bold_fraktur": (0x1D56C, 0x1D586),
    "sans_serif": (0x1D5A0, 0x1D5BA),
    "sans_serif_bold": (0x1D5D4, 0x1D5EE),
    "sans_serif_italic": (0x1D608, 0x1D622),
    "sans_serif_bold_italic": (0x1D63C, 0x1D656),
    "monospace": (0x1D670, 0x1D68A),
}


def to_math_styled(
        prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Replace the Latin letters with one Mathematical Alphanumeric style,
    drawn from :data:`_MATH_STYLES` by ``rng``. The style is reported in the
    info, so a reader can map it back."""
    style = (rng or random).choice(list(_MATH_STYLES))
    upper, lower = _MATH_STYLES[style]
    out = []
    for c in prompt:
        if "A" <= c <= "Z":
            out.append(chr(upper + ord(c) - 65))
        elif "a" <= c <= "z":
            out.append(chr(lower + ord(c) - 97))
        else:
            out.append(c)
    return "".join(out), f"style={style}"


def get_encoding_functions() -> list[Callable]:
    """Return all encoding/cyphering technique functions."""
    return [
        to_base64,
        to_rot13,
        to_rot18,
        to_rot47,
        to_unicode_escape,
        to_ascii_ordinal,
        to_separator,
        to_leetspeak_basic,
        to_leetspeak_intermediate,
        to_leetspeak_advanced,
        to_morse,
        to_braille,
        to_reversed_characters,
        to_reversed_words,
        to_reversed_word_order,
        to_lookalike_letters,
        to_zero_width_noise,
        to_hex,
        to_binary,
        to_url_encoding,
        to_html_entities,
        to_base32,
        to_base85,
        to_caesar_cipher,
        to_atbash,
        to_fullwidth,
        to_math_styled,
    ]
