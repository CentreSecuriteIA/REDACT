"""Tokenbreak and sensitive-word techniques over the words a model marks harmful.

``info`` lists the detected words with the token, encoding, separator or
substitutions applied to them.
"""

import random
import re
from collections.abc import Callable
from functools import partial

from redact.jailbreak.obfuscation.encoding import (
    SEPARATORS,
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
from redact.llms.prompting import build_messages, load_prompt


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


_CHAR_TOKENS = [
    "q", "z", "x", "v", "Q", "Z", "X", "V",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "!", "@", "#", "_", "~",
]


def tokenbreak_prepend(word: str) -> tuple[str, str]:
    """Prepend a random token to the word."""
    token = random.choice(_CHAR_TOKENS)
    return token + word, token


# Zero-width space, non-joiner, joiner, word joiner, byte order mark.
_INVISIBLE_CHAR_NAMES = {
    "\u200B": "ZWSP",
    "\u200C": "ZWNJ",
    "\u200D": "ZWJ",
    "\u2060": "WJ",
    "\uFEFF": "BOM",
}
_INVISIBLE_CHARS = list(_INVISIBLE_CHAR_NAMES)

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


_ATTACKS = {
    "prepend": tokenbreak_prepend,
    "split": tokenbreak_split,
    "delimiter": tokenbreak_delimiter,
}


def tokenbreak_obfuscate_word(word: str, attack: str) -> tuple[str, str]:
    """Obfuscate a single word using specified attack type."""
    if attack not in _ATTACKS:
        raise ValueError(f"Unknown attack type: {attack}")
    return _ATTACKS[attack](word)


_TOKENBREAK_EMPTY_RESPONSES = {"none", "n/a", "no harmful words", ""}
# Above this many words, treat the extraction response as a non-answer
# (the model rambled instead of returning a short word list).
_MAX_HARMFUL_WORDS_RESPONSE_LEN = 10


def _parse_harmful_words(reply: str) -> list[str]:
    """The words of an extraction reply: none for ``none``, or for a reply
    longer than a short word list."""
    raw = reply.strip()
    if raw.lower() == "none" or len(raw.split()) > _MAX_HARMFUL_WORDS_RESPONSE_LEN:
        return []
    return [w for w in raw.split() if w.lower() not in _TOKENBREAK_EMPTY_RESPONSES]


def _harmful_words_gen(
    prompt: str, *, gen_model: str | None = None, prompt_dir=None, **kwargs
) -> TechniqueGen:
    """Yield the extract-harmful request; return the cleaned harmful-word list.

    Shared first round for every tokenbreak/sensitive-word technique generator.
    """
    config = load_prompt("jailbreak", "extract_harmful", prompt_dir)
    messages = build_messages(config, prompt=prompt)
    return _parse_harmful_words((yield LLMRequest(gen_model, messages)))


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
        # A function, so re does not read escapes (\u, \1) in the encoded word.
        result = re.sub(
            re.escape(word), lambda _m, e=encoded_word: e, result, flags=re.IGNORECASE
        )
    return result, f"encoding={encode_name};detected_words={' '.join(harmful_words)}"


def _sensitive_separator_finish(prompt: str, harmful_words: list[str]) -> tuple[str, str]:
    """Split each harmful word's characters with one randomly chosen separator."""
    sep = random.choice(SEPARATORS)
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


# (function name, encode_fn, encode_name, docstring)
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


def _make_word_fn(name: str, finish: Callable, doc: str) -> Callable:
    """The technique that finds the prompt's words, then applies ``finish``."""

    def fn(prompt: str, **kwargs) -> TechniqueGen:
        words = yield from _harmful_words_gen(prompt, **kwargs)
        return finish(prompt, harmful_words=words)

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = doc
    return fn


# One technique per encoding, named to_sensitive_words_encode_<encoding>.
_ENCODED = [
    _make_word_fn(name, partial(_sensitive_encode_finish, encode_fn=encode_fn,
                                encode_name=encode_name), doc)
    for name, encode_fn, encode_name, doc in _ENCODE_SPECS
]


def to_sensitive_words_encode_separator(prompt: str, **kwargs) -> TechniqueGen:
    """Separate each character of harmful words with a randomly chosen separator."""
    words = yield from _harmful_words_gen(prompt, **kwargs)
    return _sensitive_separator_finish(prompt, words)


_CHAR_SPECS = [
    ("to_sensitive_words_split", " ", "split", "Separate harmful-word chars with spaces: bomb -> b o m b."),
    ("to_sensitive_words_star", "*", "star", "Separate harmful-word chars with stars: bomb -> b*o*m*b."),
    ("to_sensitive_words_hyphen", "-", "hyphen", "Separate harmful-word chars with hyphens: bomb -> b-o-m-b."),
    ("to_sensitive_words_underscore", "_", "underscore", "Separate harmful-word chars with underscores: bomb -> b_o_m_b."),
]


# One technique per separator, named to_sensitive_words_<separator name>.
_SEPARATED = [
    _make_word_fn(name, partial(_sensitive_char_finish, separator=separator,
                                sep_name=sep_name), doc)
    for name, separator, sep_name, doc in _CHAR_SPECS
]
globals().update({fn.__name__: fn for fn in _ENCODED + _SEPARATED})


def to_sensitive_words_variables(prompt: str, **kwargs) -> TechniqueGen:
    """Replace harmful words with VAR_n placeholders + a definitions clause."""
    words = yield from _harmful_words_gen(prompt, **kwargs)
    return _sensitive_variables_finish(prompt, words)


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


def get_tokenbreak_functions() -> list[Callable]:
    """Return the tokenbreak technique functions."""
    return [to_tokenbreak_prepend, to_tokenbreak_split, to_tokenbreak_delimiter]


def get_sensitive_words_functions() -> list[Callable]:
    """Return the sensitive-word technique functions."""
    # The separator encoding sits between the six encodings and the three
    # leetspeak levels.
    return [
        *_ENCODED[:6],
        to_sensitive_words_encode_separator,
        *_ENCODED[6:],
        *_SEPARATED,
        to_sensitive_words_variables,
        to_synonym_substitution,
    ]
