"""Encoding/cyphering obfuscation: base64, rot13, leetspeak, morse, braille.

All functions are pure transforms: str -> (str, str). No LLM dependency.
Ported from reference obfuscation.py lines 388-436.
"""

import codecs


def to_base64(prompt: str) -> tuple[str, str]:
    """Encode string to Base64."""
    return codecs.encode(prompt.encode(), "base64").decode(), ""


def to_rot13(prompt: str) -> tuple[str, str]:
    """Encode string using ROT13 cipher."""
    return codecs.encode(prompt, "rot_13"), ""


LEET_MAP = {
    "a": "4", "A": "4",
    "e": "3", "E": "3",
    "i": "1", "I": "1",
    "o": "0", "O": "0",
    "s": "5", "S": "5",
    "t": "7", "T": "7",
}


def to_leetspeak(prompt: str) -> tuple[str, str]:
    """Convert string to leetspeak."""
    return "".join(LEET_MAP.get(c, c) for c in prompt), ""


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
    return [to_base64, to_rot13, to_leetspeak, to_morse, to_braille]
