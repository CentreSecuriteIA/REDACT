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

import random
import re
from pathlib import Path

from redact.jailbreak.dynamic_functions import bind_functions
from redact.jailbreak.obfuscation.encoding import (
    _SEPARATORS,
    to_ascii_ordinal,
    to_base64,
    to_leetspeak_advanced,
    to_leetspeak_basic,
    to_leetspeak_intermediate,
    to_rot13,
    to_rot18,
    to_rot47,
    to_unicode_escape,
)
from redact.jailbreak.protocol import LLMRequest, TechniqueGen
from redact.llms.base import LLMBackend
from redact.llms.calls import generate_sample
from redact.llms.prompts import build_messages, load_prompt
from redact.llms.wrappers import RateLimiter

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
# Above this many words, treat the extraction response as a non-answer
# (the model rambled instead of returning a short word list).
_MAX_HARMFUL_WORDS_RESPONSE_LEN = 10


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
    if len(result.split()) > _MAX_HARMFUL_WORDS_RESPONSE_LEN:
        return ""
    return result


def _get_harmful_words(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> list[str]:
    """Run extract_harmful and return a clean list of detected words (sync)."""
    raw = extract_harmful(prompt, backend, model, rate_limiter, prompt_dir)
    return [w for w in raw.split() if w.lower() not in _TOKENBREAK_EMPTY_RESPONSES]


def _harmful_words_gen(
    prompt: str, *, gen_model: str | None = None, prompt_dir=None, **kwargs
) -> TechniqueGen:
    """Yield the extract-harmful request; return the cleaned harmful-word list.

    Shared first round for every tokenbreak/sensitive-word technique generator.
    """
    config = load_prompt("jailbreak", "extract_harmful", prompt_dir)
    messages = build_messages(config, prompt=prompt)
    raw = (yield LLMRequest(gen_model, messages)).strip()
    if raw.lower() == "none" or len(raw.split()) > _MAX_HARMFUL_WORDS_RESPONSE_LEN:
        return []
    return [w for w in raw.split() if w.lower() not in _TOKENBREAK_EMPTY_RESPONSES]


# ---------------------------------------------------------------------------
# Pure finishers — apply a transform given the already-detected harmful words
# ---------------------------------------------------------------------------


def _tokenbreak_finish(prompt: str, attack: str, harmful_words: list[str]) -> tuple[str, str]:
    """Structural tokenbreak applied to each harmful word (or every word)."""
    if not harmful_words:
        def _obfuscate_token(m: re.Match) -> str:
            return tokenbreak_obfuscate_word(m.group(), attack)[0]
        return re.sub(r"\S+", _obfuscate_token, prompt), "detected_words="

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


def _sensitive_encode_finish(
    prompt: str, encode_fn, encode_name: str, harmful_words: list[str]
) -> tuple[str, str]:
    """Apply ``encode_fn`` to each harmful word (or the whole prompt as fallback)."""
    if not harmful_words:
        encoded, _ = encode_fn(prompt)
        return encoded, f"encoding={encode_name};detected_words="
    result = prompt
    for word in harmful_words:
        encoded_word, _ = encode_fn(word)
        result = re.sub(re.escape(word), encoded_word, result, flags=re.IGNORECASE)
    return result, f"encoding={encode_name};detected_words={' '.join(harmful_words)}"


def _sensitive_separator_finish(prompt: str, harmful_words: list[str]) -> tuple[str, str]:
    """Split each harmful word's characters with one randomly chosen separator."""
    sep = random.choice(_SEPARATORS)
    if not harmful_words:
        return sep.join(prompt), f"encoding=separator;sep={sep};detected_words="
    result = prompt
    for word in harmful_words:
        result = re.sub(re.escape(word), sep.join(word), result, flags=re.IGNORECASE)
    return result, f"encoding=separator;sep={sep};detected_words={' '.join(harmful_words)}"


def _sensitive_char_finish(
    prompt: str, separator: str, sep_name: str, harmful_words: list[str]
) -> tuple[str, str]:
    """Split each harmful word's characters with a fixed separator."""
    if not harmful_words:
        obfuscated = re.sub(r"\S+", lambda m: separator.join(m.group()), prompt)
        return obfuscated, f"char_sep={sep_name};detected_words="
    result = prompt
    for word in harmful_words:
        result = re.sub(re.escape(word), separator.join(word), result, flags=re.IGNORECASE)
    return result, f"char_sep={sep_name};detected_words={' '.join(harmful_words)}"


def _sensitive_variables_finish(prompt: str, harmful_words: list[str]) -> tuple[str, str]:
    """Replace harmful words with VAR_n placeholders + a definitions clause."""
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
# Tier 1 — Structural tokenbreak (technique generators)
# ---------------------------------------------------------------------------


def to_tokenbreak_prepend(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: prepend a random token before each harmful word."""
    words = yield from _harmful_words_gen(prompt, **kwargs)
    return _tokenbreak_finish(prompt, "prepend", words)


def to_tokenbreak_split(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: insert a random character inside each harmful word."""
    words = yield from _harmful_words_gen(prompt, **kwargs)
    return _tokenbreak_finish(prompt, "split", words)


def to_tokenbreak_delimiter(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: wrap each harmful word with randomly chosen delimiters."""
    words = yield from _harmful_words_gen(prompt, **kwargs)
    return _tokenbreak_finish(prompt, "delimiter", words)


# ---------------------------------------------------------------------------
# Tier 2a — Sensitive-word encoding (technique generators, fixed encoding each)
# ---------------------------------------------------------------------------

# (function name, encode_fn, encode_name, docstring) — encode_fn is None for
# the special separator variant handled by _sensitive_separator_finish.
_ENCODE_SPECS = [
    ("to_sensitive_words_encode_base64", to_base64, "base64", "Encode each harmful word with Base64."),
    ("to_sensitive_words_encode_rot13", to_rot13, "rot13", "Encode each harmful word with ROT13."),
    ("to_sensitive_words_encode_rot18", to_rot18, "rot18", "Encode each harmful word with ROT18."),
    ("to_sensitive_words_encode_rot47", to_rot47, "rot47", "Encode each harmful word with ROT47."),
    ("to_sensitive_words_encode_unicode", to_unicode_escape, "unicode", "Encode each harmful word as Unicode escapes."),
    ("to_sensitive_words_encode_ascii", to_ascii_ordinal, "ascii", "Encode each harmful word as ASCII ordinals."),
    ("to_sensitive_words_encode_leetspeak_basic", to_leetspeak_basic, "leetspeak_basic", "Basic leetspeak on each harmful word."),
    ("to_sensitive_words_encode_leetspeak_intermediate", to_leetspeak_intermediate, "leetspeak_intermediate", "Intermediate leetspeak on each harmful word."),
    ("to_sensitive_words_encode_leetspeak_advanced", to_leetspeak_advanced, "leetspeak_advanced", "Advanced leetspeak on each harmful word."),
]


def _make_encode_fn(fname: str, encode_fn, encode_name: str, doc: str):
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        words = yield from _harmful_words_gen(prompt, **kwargs)
        return _sensitive_encode_finish(prompt, encode_fn, encode_name, words)
    fn.__name__ = fname
    fn.__qualname__ = fname
    fn.__doc__ = doc
    return fn


bind_functions(globals(), [
    _make_encode_fn(_fname, _encode_fn, _encode_name, _doc)
    for _fname, _encode_fn, _encode_name, _doc in _ENCODE_SPECS
])


def to_sensitive_words_encode_separator(prompt: str, **kwargs) -> TechniqueGen:
    """Separate each character of harmful words with a randomly chosen separator."""
    words = yield from _harmful_words_gen(prompt, **kwargs)
    return _sensitive_separator_finish(prompt, words)


# ---------------------------------------------------------------------------
# Tier 2b — Sensitive-word character-level obfuscation (technique generators)
# ---------------------------------------------------------------------------

_CHAR_SPECS = [
    ("to_sensitive_words_split", " ", "split", "Separate harmful-word chars with spaces: bomb -> b o m b."),
    ("to_sensitive_words_star", "*", "star", "Separate harmful-word chars with stars: bomb -> b*o*m*b."),
    ("to_sensitive_words_hyphen", "-", "hyphen", "Separate harmful-word chars with hyphens: bomb -> b-o-m-b."),
    ("to_sensitive_words_underscore", "_", "underscore", "Separate harmful-word chars with underscores: bomb -> b_o_m_b."),
]


def _make_char_fn(fname: str, separator: str, sep_name: str, doc: str):
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        words = yield from _harmful_words_gen(prompt, **kwargs)
        return _sensitive_char_finish(prompt, separator, sep_name, words)
    fn.__name__ = fname
    fn.__qualname__ = fname
    fn.__doc__ = doc
    return fn


bind_functions(globals(), [
    _make_char_fn(_fname, _sep, _sep_name, _doc)
    for _fname, _sep, _sep_name, _doc in _CHAR_SPECS
])


def to_sensitive_words_variables(prompt: str, **kwargs) -> TechniqueGen:
    """Replace harmful words with VAR_n placeholders + a definitions clause."""
    words = yield from _harmful_words_gen(prompt, **kwargs)
    return _sensitive_variables_finish(prompt, words)


# ---------------------------------------------------------------------------
# Tier 2c — Synonym substitution (extract round + one synonym round per word)
# ---------------------------------------------------------------------------

# A "synonym" longer than this many words is treated as a non-answer (the
# original word is kept instead) — a single creative synonym, not a phrase.
_MAX_SYNONYM_WORDS = 4


def to_synonym_substitution(prompt: str, *, prompt_dir=None, **kwargs) -> TechniqueGen:
    """Replace each harmful word with a creative LLM-generated benign synonym.

    Yields the harmful-word extraction request, then one synonym request per
    detected word. A multi-word (>4) synonym response is discarded (original
    kept). Returns ``(modified_prompt, "substitutions=w1->s1;...")``.
    """
    gen_model = kwargs.get("gen_model")
    words = yield from _harmful_words_gen(prompt, prompt_dir=prompt_dir, **kwargs)
    if not words:
        return prompt, "substitutions="

    config = load_prompt("jailbreak", "synonym_substitution", prompt_dir)
    result = prompt
    substitution_pairs: list[str] = []
    for word in words:
        messages = build_messages(config, word=word, prompt=prompt)
        synonym = (yield LLMRequest(gen_model, messages)).strip()
        if not synonym or len(synonym.split()) > _MAX_SYNONYM_WORDS:
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
