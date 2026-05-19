"""TokenBreak obfuscation: prepend, split, delimiter, sensitive-word encoding.

Two tiers:

  Tier 1 — Structural tokenbreak (existing):
    to_tokenbreak_prepend, to_tokenbreak_split, to_tokenbreak_delimiter
    Extract harmful words via LLM, then apply structural word-level obfuscation
    with randomly selected tokens/delimiters (choice recorded in additional_info).

  Tier 2 — Sensitive-word encoding and character obfuscation:
    to_sensitive_words_encode_*  (10 variants, one per encoding from encoding.py)
    to_sensitive_words_split/star/hyphen/underscore/variables  (5 char-level variants)
    Same LLM extraction step; each function applies a fixed named transform so
    no random selection is needed at the function level.

    Note: sensitive_words_hyphen and sensitive_words_underscore are the fixed-
    separator counterparts of tokenbreak_delimiter's random separator mode.
    sensitive_words_split (space) and sensitive_words_star (*) have no
    equivalent in Tier 1.

Ported from reference obfuscation.py lines 600-843.
"""

import re
import random
from pathlib import Path

from redact.llms.calls import generate_sample
from redact.llms.prompts import load_prompt, build_messages
from redact.llms.base import LLMBackend
from redact.llms.wrappers import RateLimiter
from redact.jailbreak.obfuscation.encoding import (
    to_base64,
    to_rot13,
    to_rot18,
    to_rot47,
    to_unicode_escape,
    to_ascii_ordinal,
    to_leetspeak_basic,
    to_leetspeak_intermediate,
    to_leetspeak_advanced,
    _SEPARATORS,
)


# ---------------------------------------------------------------------------
# Pure word-level transforms (no LLM) — Tier 1 helpers
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
    """Obfuscate a single word using specified attack type."""
    if attack == "prepend":
        return tokenbreak_prepend(word)
    elif attack == "split":
        return tokenbreak_split(word)
    elif attack == "delimiter":
        return tokenbreak_delimiter(word)
    else:
        raise ValueError(f"Unknown attack type: {attack}")


# ---------------------------------------------------------------------------
# LLM extraction — shared by both tiers
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
    config = load_prompt("jailbreak", "extract_harmful", prompt_dir)
    messages = build_messages(config, prompt=prompt)
    result = generate_sample(backend, model, messages, rate_limiter)
    result = result.strip()
    if result.lower() == "none":
        return ""
    if len(result.split()) > 10:
        return ""
    return result


def _get_harmful_words(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> list[str]:
    """Run extract_harmful and return a clean list of detected words."""
    raw = extract_harmful(prompt, backend, model, rate_limiter, prompt_dir)
    return [w for w in raw.split() if w.lower() not in _TOKENBREAK_EMPTY_RESPONSES]


# ---------------------------------------------------------------------------
# Tier 1 — Structural tokenbreak
# ---------------------------------------------------------------------------


def _apply_tokenbreak(
    prompt: str,
    attack: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Extract harmful words, apply structural tokenbreak attack to each."""
    harmful_words = _get_harmful_words(prompt, backend, model, rate_limiter, prompt_dir)

    if not harmful_words:
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
    return result, f"detected_words={' '.join(harmful_words)};tokens={','.join(token_infos)}"


def to_tokenbreak_prepend(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """TokenBreak: prepend a random token before each harmful word."""
    return _apply_tokenbreak(prompt, "prepend", backend, model, rate_limiter)


def to_tokenbreak_split(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """TokenBreak: insert a random character inside each harmful word."""
    return _apply_tokenbreak(prompt, "split", backend, model, rate_limiter)


def to_tokenbreak_delimiter(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """TokenBreak: wrap each harmful word with randomly chosen delimiters."""
    return _apply_tokenbreak(prompt, "delimiter", backend, model, rate_limiter)


# ---------------------------------------------------------------------------
# Tier 2a — Sensitive-word encoding (fixed encoding per function)
# ---------------------------------------------------------------------------


def _apply_sensitive_words_encode(
    prompt: str,
    encode_fn,
    encode_name: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Apply a specific encoding function to each detected harmful word.

    If no harmful words are detected, encodes the entire prompt as fallback.
    encode_fn must follow the (str) -> (str, str) signature from encoding.py.
    """
    harmful_words = _get_harmful_words(prompt, backend, model, rate_limiter, prompt_dir)

    if not harmful_words:
        encoded, _ = encode_fn(prompt)
        return encoded, f"encoding={encode_name};detected_words="

    result = prompt
    for word in harmful_words:
        encoded_word, _ = encode_fn(word)
        result = re.sub(re.escape(word), encoded_word, result, flags=re.IGNORECASE)
    return result, f"encoding={encode_name};detected_words={' '.join(harmful_words)}"


def to_sensitive_words_encode_base64(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Encode each harmful word with Base64."""
    return _apply_sensitive_words_encode(prompt, to_base64, "base64", backend, model, rate_limiter)


def to_sensitive_words_encode_rot13(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Encode each harmful word with ROT13."""
    return _apply_sensitive_words_encode(prompt, to_rot13, "rot13", backend, model, rate_limiter)


def to_sensitive_words_encode_rot18(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Encode each harmful word with ROT18 (ROT13 + ROT5)."""
    return _apply_sensitive_words_encode(prompt, to_rot18, "rot18", backend, model, rate_limiter)


def to_sensitive_words_encode_rot47(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Encode each harmful word with ROT47."""
    return _apply_sensitive_words_encode(prompt, to_rot47, "rot47", backend, model, rate_limiter)


def to_sensitive_words_encode_unicode(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Encode each harmful word as Unicode escape sequences."""
    return _apply_sensitive_words_encode(prompt, to_unicode_escape, "unicode", backend, model, rate_limiter)


def to_sensitive_words_encode_ascii(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Encode each harmful word as space-separated ASCII ordinals."""
    return _apply_sensitive_words_encode(prompt, to_ascii_ordinal, "ascii", backend, model, rate_limiter)


def to_sensitive_words_encode_separator(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Separate each character of harmful words with a randomly chosen separator.

    Separator is chosen once per call and applied consistently to all detected
    words. The chosen separator is recorded in additional_info.
    """
    harmful_words = _get_harmful_words(prompt, backend, model, rate_limiter)
    sep = random.choice(_SEPARATORS)

    if not harmful_words:
        encoded = sep.join(prompt)
        return encoded, f"encoding=separator;sep={sep};detected_words="

    result = prompt
    for word in harmful_words:
        result = re.sub(re.escape(word), sep.join(word), result, flags=re.IGNORECASE)
    return result, f"encoding=separator;sep={sep};detected_words={' '.join(harmful_words)}"


def to_sensitive_words_encode_leetspeak_basic(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Apply basic leetspeak substitution to each harmful word."""
    return _apply_sensitive_words_encode(prompt, to_leetspeak_basic, "leetspeak_basic", backend, model, rate_limiter)


def to_sensitive_words_encode_leetspeak_intermediate(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Apply intermediate leetspeak substitution to each harmful word."""
    return _apply_sensitive_words_encode(prompt, to_leetspeak_intermediate, "leetspeak_intermediate", backend, model, rate_limiter)


def to_sensitive_words_encode_leetspeak_advanced(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Apply advanced leetspeak substitution to each harmful word."""
    return _apply_sensitive_words_encode(prompt, to_leetspeak_advanced, "leetspeak_advanced", backend, model, rate_limiter)


# ---------------------------------------------------------------------------
# Tier 2b — Sensitive-word character-level obfuscation (fixed separators)
# ---------------------------------------------------------------------------


def _apply_sensitive_words_char(
    prompt: str,
    separator: str,
    sep_name: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Separate each character of each harmful word with a fixed separator.

    If no harmful words are detected, applies to every word in the prompt.
    """
    harmful_words = _get_harmful_words(prompt, backend, model, rate_limiter, prompt_dir)

    if not harmful_words:
        obfuscated = re.sub(r"\S+", lambda m: separator.join(m.group()), prompt)
        return obfuscated, f"char_sep={sep_name};detected_words="

    result = prompt
    for word in harmful_words:
        result = re.sub(re.escape(word), separator.join(word), result, flags=re.IGNORECASE)
    return result, f"char_sep={sep_name};detected_words={' '.join(harmful_words)}"


def to_sensitive_words_split(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Separate each character of harmful words with spaces: bomb -> b o m b."""
    return _apply_sensitive_words_char(prompt, " ", "split", backend, model, rate_limiter)


def to_sensitive_words_star(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Separate each character of harmful words with stars: bomb -> b*o*m*b."""
    return _apply_sensitive_words_char(prompt, "*", "star", backend, model, rate_limiter)


def to_sensitive_words_hyphen(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Separate each character of harmful words with hyphens: bomb -> b-o-m-b.

    Fixed-separator counterpart of tokenbreak_delimiter's random separator mode.
    """
    return _apply_sensitive_words_char(prompt, "-", "hyphen", backend, model, rate_limiter)


def to_sensitive_words_underscore(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Separate each character of harmful words with underscores: bomb -> b_o_m_b.

    Fixed-separator counterpart of tokenbreak_delimiter's random separator mode.
    """
    return _apply_sensitive_words_char(prompt, "_", "underscore", backend, model, rate_limiter)


def to_sensitive_words_variables(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Replace harmful words with placeholder variable names.

    Each detected harmful word is substituted with VAR_1, VAR_2, etc.
    A definition clause is appended at the end of the prompt:
    "where VAR_1 = {word1}, VAR_2 = {word2}..."

    This indirection separates the harmful term from its use in the sentence
    while keeping the meaning recoverable from context.

    If no harmful words are detected, the prompt is returned unchanged.
    """
    harmful_words = _get_harmful_words(prompt, backend, model, rate_limiter)

    if not harmful_words:
        return prompt, "char_sep=variables;detected_words="

    result = prompt
    definitions: list[str] = []
    for i, word in enumerate(harmful_words, start=1):
        var_name = f"VAR_{i}"
        result = re.sub(re.escape(word), var_name, result, flags=re.IGNORECASE)
        definitions.append(f"{var_name} = {word}")

    result += "\n\nwhere " + ", ".join(definitions)
    return result, f"char_sep=variables;detected_words={' '.join(harmful_words)}"


# ---------------------------------------------------------------------------
# Tier 2c — Synonym substitution (LLM-generated benign synonyms)
# ---------------------------------------------------------------------------


def _get_synonym(
    word: str,
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> str:
    """Ask the LLM for a creative benign synonym for word in context of prompt.

    Returns the raw LLM response stripped of whitespace. Caller handles
    fallback if the response is empty or looks like prose.
    """
    config = load_prompt("jailbreak", "synonym_substitution", prompt_dir)
    messages = build_messages(config, word=word, prompt=prompt)
    result = generate_sample(backend, model, messages, rate_limiter)
    return result.strip()


def to_synonym_substitution(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Replace each harmful word with a creative LLM-generated benign synonym.

    Steps:
      1. Detect harmful words via extract_harmful() (one LLM call).
      2. For each detected word, call the LLM to generate one imaginative
         benign synonym that preserves contextual meaning (one call per word).
      3. Substitute each word in the prompt; record the mapping in additional_info.

    If no harmful words are detected, the prompt is returned unchanged.
    If synonym generation produces a multi-word response (>4 words) it is
    discarded and the original word is kept for that slot.

    Returns:
        (modified_prompt, additional_info)
        additional_info format: "substitutions=word1->syn1;word2->syn2"
    """
    harmful_words = _get_harmful_words(prompt, backend, model, rate_limiter, prompt_dir)

    if not harmful_words:
        return prompt, "substitutions="

    result = prompt
    substitution_pairs: list[str] = []

    for word in harmful_words:
        synonym = _get_synonym(word, prompt, backend, model, rate_limiter, prompt_dir)

        # Discard if the model returned prose instead of a word/phrase
        if not synonym or len(synonym.split()) > 4:
            synonym = word  # keep original as fallback

        result = re.sub(
            re.escape(word),
            lambda m, s=synonym: match_case(m.group(), s),
            result,
            flags=re.IGNORECASE,
        )
        substitution_pairs.append(f"{word}->{synonym}")

    return result, f"substitutions={';'.join(substitution_pairs)}"


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def get_tokenbreak_functions() -> list:
    """Return Tier 1 structural tokenbreak functions."""
    return [to_tokenbreak_prepend, to_tokenbreak_split, to_tokenbreak_delimiter]


def get_sensitive_words_functions() -> list:
    """Return all Tier 2 sensitive-word encoding, obfuscation, and synonym functions."""
    return [
        to_sensitive_words_encode_base64,
        to_sensitive_words_encode_rot13,
        to_sensitive_words_encode_rot18,
        to_sensitive_words_encode_rot47,
        to_sensitive_words_encode_unicode,
        to_sensitive_words_encode_ascii,
        to_sensitive_words_encode_separator,
        to_sensitive_words_encode_leetspeak_basic,
        to_sensitive_words_encode_leetspeak_intermediate,
        to_sensitive_words_encode_leetspeak_advanced,
        to_sensitive_words_split,
        to_sensitive_words_star,
        to_sensitive_words_hyphen,
        to_sensitive_words_underscore,
        to_sensitive_words_variables,
        to_synonym_substitution,
    ]
