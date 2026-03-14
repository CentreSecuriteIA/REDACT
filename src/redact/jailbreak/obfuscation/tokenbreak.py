"""TokenBreak obfuscation: prepend, split, delimiter.

Requires LLM for extract_harmful() step. Pure word-level obfuscation
functions (tokenbreak_prepend, tokenbreak_split, tokenbreak_delimiter)
are standalone.

Ported from reference obfuscation.py lines 600-843.
"""

import re
import random
from pathlib import Path

from Redact_Library.LLMs.calls import generate_sample
from Redact_Library.LLMs.prompts import load_prompt, build_messages
from Redact_Library.LLMs.base import LLMBackend
from Redact_Library.LLMs.wrappers import RateLimiter


# ---------------------------------------------------------------------------
# Pure word-level transforms (no LLM)
# ---------------------------------------------------------------------------


def match_case(original: str, replacement: str) -> str:
    """Apply the case pattern of original to replacement."""
    if not original:
        return replacement
    if original.isupper():
        return replacement.upper()
    elif original.islower():
        return replacement.lower()
    elif original[0].isupper():
        return replacement.capitalize()
    else:
        return replacement


_PREPEND_TOKENS = [
    "q", "z", "x", "v", "Q", "Z", "X", "V",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "!", "@", "#", "_", "~",
]


def tokenbreak_prepend(word: str) -> tuple[str, str]:
    """Prepend a random token to the word."""
    token = random.choice(_PREPEND_TOKENS)
    return token + word, token


_INVISIBLE_CHAR_NAMES = {
    "\u200B": "ZWSP",
    "\u200C": "ZWNJ",
    "\u200D": "ZWJ",
    "\u2060": "WJ",
    "\uFEFF": "BOM",
}

_CHAR_TOKENS = [
    "q", "z", "x", "v", "Q", "Z", "X", "V",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "!", "@", "#", "_", "~",
]

_INVISIBLE_CHARS = [
    "\u200B",  # Zero-Width Space
    "\u200C",  # Zero-Width Non-Joiner (ZWNJ)
    "\u200D",  # Zero-Width Joiner (ZWJ)
    "\u2060",  # Word Joiner
    "\uFEFF",  # Byte Order Mark (BOM)
]

_VARIATION_SELECTORS = [
    "\uFE00", "\uFE01", "\uFE02", "\uFE03", "\uFE04", "\uFE05",
    "\uFE06", "\uFE07", "\uFE08", "\uFE09", "\uFE0A", "\uFE0B",
    "\uFE0C", "\uFE0D", "\uFE0E", "\uFE0F",
]


def tokenbreak_split(word: str) -> tuple[str, str]:
    """Split word with a random token in the middle."""
    token_type = random.choice(["char", "invisible_char", "variation_selector"])

    if token_type == "char":
        token = random.choice(_CHAR_TOKENS)
        token_repr = token
    elif token_type == "invisible_char":
        token = random.choice(_INVISIBLE_CHARS)
        token_repr = _INVISIBLE_CHAR_NAMES.get(token, "invis")
    else:
        token = random.choice(_VARIATION_SELECTORS)
        token_repr = f"VS{ord(token) - 0xFE00:02d}"

    if len(word) <= 1:
        return word + token, token_repr
    position = random.randint(1, len(word) - 1)
    obfuscated_word = word[:position] + token + word[position:]
    return obfuscated_word, token_repr


_STANDARD_DELIMITERS = ['"', '"""', "'", "'''", "bracket"]
_STRUCTURAL_DELIMITERS = [
    "<context>", "<system_instruction>",
    "<admin_note>", "<translation_task>",
]
_SEPARATOR_DELIMITERS = [".", "-", "_", "~", "/", "\\", "|"]


def tokenbreak_delimiter(word: str) -> tuple[str, str]:
    """Wrap or split word with delimiters."""
    delimiter_type = random.choice(["standard", "structural", "separator"])

    if delimiter_type == "standard":
        delimiter = random.choice(_STANDARD_DELIMITERS)
        if delimiter == "bracket":
            brackets = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
            open_b, close_b = random.choice(brackets)
            return open_b + word + close_b, f"{open_b}{close_b}"
        return delimiter + word + delimiter, delimiter
    elif delimiter_type == "structural":
        delimiter = random.choice(_STRUCTURAL_DELIMITERS)
        return delimiter + word + delimiter, delimiter
    else:
        delimiter = random.choice(_SEPARATOR_DELIMITERS)
        return delimiter.join(list(word)), delimiter


def tokenbreak_obfuscate_word(word: str, attack: str) -> tuple[str, str]:
    """Obfuscate a single word using specified attack type.

    Args:
        word: Word to obfuscate.
        attack: Attack type ('prepend', 'split', or 'delimiter').

    Returns:
        (obfuscated_word, token_repr)
    """
    if attack == "prepend":
        return tokenbreak_prepend(word)
    elif attack == "split":
        return tokenbreak_split(word)
    elif attack == "delimiter":
        return tokenbreak_delimiter(word)
    else:
        raise ValueError(f"Unknown attack type: {attack}")


# ---------------------------------------------------------------------------
# LLM-dependent functions
# ---------------------------------------------------------------------------

_TOKENBREAK_EMPTY_RESPONSES = {"none", "n/a", "no harmful words", ""}


def extract_harmful(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> str:
    """Extract harmful words from prompt using LLM.

    Loads prompt from Prompts/Jailbreak/extract_harmful/template.json.

    Returns:
        Space-separated harmful words, or "" if none detected.
    """
    config = load_prompt("Jailbreak", "extract_harmful", prompt_dir)
    messages = build_messages(config, prompt=prompt)
    result = generate_sample(backend, model, messages, rate_limiter)
    result = result.strip()
    if result.lower() == "none":
        return ""
    # Guard: if the response looks like prose (>10 words), discard it
    if len(result.split()) > 10:
        return ""
    return result


def _apply_tokenbreak(
    prompt: str,
    attack: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Shared logic for all tokenbreak variants.

    Extracts harmful words via LLM, obfuscates each one with the given
    attack type, and returns the modified prompt.
    """
    raw = extract_harmful(prompt, backend, model, rate_limiter, prompt_dir)
    harmful_words = [
        w for w in raw.split()
        if w.lower() not in _TOKENBREAK_EMPTY_RESPONSES
    ]

    if not harmful_words:
        # No harmful words detected — obfuscate every word in the full prompt
        def _obfuscate_token(m: re.Match) -> str:
            return tokenbreak_obfuscate_word(m.group(), attack)[0]
        obfuscated = re.sub(r"\S+", _obfuscate_token, prompt)
        return obfuscated, "detected_words="

    token_infos: list[str] = []
    result = prompt
    for word in harmful_words:
        obfuscated_word, token_repr = tokenbreak_obfuscate_word(word, attack)
        token_infos.append(token_repr)
        result = re.sub(
            re.escape(word),
            lambda m, ow=obfuscated_word: match_case(m.group(), ow),
            result,
            flags=re.IGNORECASE,
        )
    return result, f"detected_words={' '.join(harmful_words)}; tokens={','.join(token_infos)}"


def to_tokenbreak_prepend(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """TokenBreak: prepend a random token before each harmful word."""
    return _apply_tokenbreak(prompt, "prepend", backend, model, rate_limiter)


def to_tokenbreak_split(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """TokenBreak: insert a character inside each harmful word."""
    return _apply_tokenbreak(prompt, "split", backend, model, rate_limiter)


def to_tokenbreak_delimiter(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """TokenBreak: wrap each harmful word with delimiters."""
    return _apply_tokenbreak(prompt, "delimiter", backend, model, rate_limiter)


def get_tokenbreak_functions() -> list:
    """Return all tokenbreak technique functions."""
    return [to_tokenbreak_prepend, to_tokenbreak_split, to_tokenbreak_delimiter]
