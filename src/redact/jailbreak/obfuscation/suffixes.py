"""Adversarial suffix generation: punctuation, fragments, unicode, emoji.

All functions are pure transforms: str -> (str, str). No LLM dependency.
Ported from reference obfuscation.py lines 850-941.
"""

import random


def to_adversarial_suffix_punctuation(prompt: str) -> tuple[str, str]:
    """Add punctuation-based adversarial suffix."""
    punctuation_types = ["!", "_", "=", "*", "/", '"', "~", "|", "bracket"]
    length = random.randint(8, 40)

    suffix = ""
    i = 0
    while i < length:
        punct_type = random.choice(punctuation_types)
        repeat_count = random.randint(1, min(5, length - i))

        if punct_type == "bracket":
            bracket_type = random.choice(["{", "[", "("])
            close_bracket = (
                "}" if bracket_type == "{" else
                "]" if bracket_type == "[" else ")"
            )
            max_pairs = min(repeat_count, (length - i) // 2)
            if max_pairs > 0:
                suffix += bracket_type * max_pairs + close_bracket * max_pairs
                i += max_pairs * 2
            else:
                i += 1
        else:
            suffix += punct_type * repeat_count
            i += repeat_count

    return prompt + " " + suffix, suffix


def to_adversarial_suffix_fragments(prompt: str) -> tuple[str, str]:
    """Add word fragment-based adversarial suffix."""
    words = [
        "inst", "direct", "inverse", "desc", "steps",
        "style", "output", "format", "theorem", "logic",
    ]
    length = random.randint(2, 10)
    suffix = " ".join(random.choice(words) for _ in range(length))
    return prompt + " " + suffix, suffix


def to_adversarial_suffix_unicode(prompt: str) -> tuple[str, str]:
    """Add Unicode symbol-based adversarial suffix."""
    symbols = ["\u2200", "\u2203", "\u2211", "\u21d2", "\u222b", "\u03bb", "\u03a9", "\u03b4"]
    length = random.randint(3, 10)
    suffix = " ".join(random.choice(symbols) for _ in range(length))
    return prompt + " " + suffix, suffix


def to_adversarial_suffix_emoji(prompt: str) -> tuple[str, str]:
    """Add emoji-based adversarial suffix."""
    emojis = ["\U0001f513", "\u26a0\ufe0f", "\u2705", "\U0001f680",
              "\U0001f525", "\U0001f916", "\U0001f4ac", "\U0001f4af"]
    length = random.randint(5, 20)

    suffix = ""
    i = 0
    while i < length:
        emoji = random.choice(emojis)
        repeat_count = random.randint(1, min(5, length - i))
        suffix += emoji * repeat_count
        i += repeat_count

    return prompt + " " + suffix, suffix


def get_suffix_functions() -> list:
    """Return all adversarial suffix technique functions."""
    return [
        to_adversarial_suffix_punctuation,
        to_adversarial_suffix_fragments,
        to_adversarial_suffix_unicode,
        to_adversarial_suffix_emoji,
    ]
