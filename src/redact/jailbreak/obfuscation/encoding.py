"""Encoding/cyphering obfuscation: base64, rot13/18/47, leetspeak, morse, braille,
unicode escape, ascii ordinal, separator.

All functions are pure transforms: str -> (str, str). No LLM dependency.
Ported from reference obfuscation.py lines 388-436.
"""

import codecs
import random


def to_base64(prompt: str) -> tuple[str, str]:
    """Encode string to Base64."""
    return codecs.encode(prompt.encode(), "base64").decode(), ""


def to_rot13(prompt: str) -> tuple[str, str]:
    """Encode string using ROT13 cipher (letters only)."""
    return codecs.encode(prompt, "rot_13"), ""


def to_rot18(prompt: str) -> tuple[str, str]:
    """Encode string using ROT18: ROT13 for letters + ROT5 for digits."""
    result = []
    for c in prompt:
        if c.isalpha():
            result.append(codecs.encode(c, "rot_13"))
        elif c.isdigit():
            result.append(str((int(c) + 5) % 10))
        else:
            result.append(c)
    return "".join(result), ""


def to_rot47(prompt: str) -> tuple[str, str]:
    """Encode string using ROT47: rotate all printable ASCII (33-126) by 47."""
    result = []
    for c in prompt:
        o = ord(c)
        if 33 <= o <= 126:
            result.append(chr((o - 33 + 47) % 94 + 33))
        else:
            result.append(c)
    return "".join(result), ""


def to_unicode_escape(prompt: str) -> tuple[str, str]:
    """Encode each character as a Unicode escape sequence (\\uXXXX)."""
    return "".join(f"\\u{ord(c):04x}" for c in prompt), ""


def to_ascii_ordinal(prompt: str) -> tuple[str, str]:
    """Encode each character as its decimal ASCII ordinal, space-separated."""
    return " ".join(str(ord(c)) for c in prompt), ""


_SEPARATORS = ["-", "_", ".", "|", "/", "*", "~"]


def to_separator(prompt: str) -> tuple[str, str]:
    """Insert a random separator character between every character."""
    sep = random.choice(_SEPARATORS)
    return sep.join(prompt), f"sep={sep}"


# ---------------------------------------------------------------------------
# Leetspeak — three levels of substitution depth
# ---------------------------------------------------------------------------

LEET_MAP_BASIC = {
    "a": "4", "A": "4",
    "e": "3", "E": "3",
    "i": "1", "I": "1",
    "o": "0", "O": "0",
    "s": "5", "S": "5",
    "t": "7", "T": "7",
}

LEET_MAP_INTERMEDIATE = {
    **LEET_MAP_BASIC,
    "b": "8", "B": "8",
    "g": "9", "G": "9",
    "l": "|", "L": "|",
    "z": "2", "Z": "2",
    "c": "(", "C": "(",
    "r": "|2", "R": "|2",
}

LEET_MAP_ADVANCED = {
    **LEET_MAP_INTERMEDIATE,
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
    return "".join(LEET_MAP_BASIC.get(c, c) for c in prompt), ""


def to_leetspeak_intermediate(prompt: str) -> tuple[str, str]:
    """Leetspeak with extended map adding b, g, l, z, c, r substitutions."""
    return "".join(LEET_MAP_INTERMEDIATE.get(c, c) for c in prompt), ""


def to_leetspeak_advanced(prompt: str) -> tuple[str, str]:
    """Leetspeak with full map including multi-character substitutions."""
    return "".join(LEET_MAP_ADVANCED.get(c, c) for c in prompt), ""


# Keep original name as alias for backwards compatibility
to_leetspeak = to_leetspeak_basic


MORSE_MAP = {
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
    return " ".join(MORSE_MAP.get(c.upper(), c) for c in prompt), ""


BRAILLE_MAP = {
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
    return "".join(BRAILLE_MAP.get(c.lower(), c) for c in prompt), ""


def get_encoding_functions() -> list:
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
    ]
