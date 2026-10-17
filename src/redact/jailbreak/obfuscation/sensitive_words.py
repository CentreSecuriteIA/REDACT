"""Sensitive-word techniques over the words a model marks harmful.

``info`` lists the detected words with the encoding, separator or
substitutions applied to them.
"""

import random
import re
from collections.abc import Callable
from functools import partial

from redact.jailbreak.obfuscation.encoding import (
    SEPARATORS,
    _clusters,
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
from redact.jailbreak.technique import TechniqueGen
from redact.llm_pipeline import LLMRequest
from redact.llms.prompting import build_messages, load_prompt

from .tokenbreak import harmful_words_gen, match_case


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


def _sensitive_separator_finish(
    prompt: str, harmful_words: list[str], rng: random.Random | None = None,
) -> tuple[str, str]:
    """Split each harmful word's characters with one randomly chosen separator."""
    sep = (rng or random).choice(SEPARATORS)
    if not harmful_words:
        return (sep.join(_clusters(prompt)),
                f"encoding=separator;sep={sep};detected_words=")
    result = prompt
    for word in harmful_words:
        result = re.sub(re.escape(word), sep.join(_clusters(word)), result,
                        flags=re.IGNORECASE)
    return (
        result,
        f"encoding=separator;sep={sep};detected_words={' '.join(harmful_words)}",
    )


def _sensitive_char_finish(
    prompt: str, separator: str, sep_name: str, harmful_words: list[str]
) -> tuple[str, str]:
    """Split each harmful word's characters with a fixed separator."""
    if not harmful_words:
        obfuscated = re.sub(r"\S+", lambda m: separator.join(_clusters(m.group())),
                            prompt)
        return obfuscated, f"char_sep={sep_name};detected_words="
    result = prompt
    for word in harmful_words:
        result = re.sub(
            re.escape(word), separator.join(_clusters(word)), result,
            flags=re.IGNORECASE)
    return result, f"char_sep={sep_name};detected_words={' '.join(harmful_words)}"


def _sensitive_variables_finish(
    prompt: str, harmful_words: list[str]
) -> tuple[str, str]:
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


# (function name, encode_fn, encode_name, docstring)
_ENCODE_SPECS = [
    (
        "to_sensitive_words_encode_base64",
        to_base64,
        "base64",
        "Encode each harmful word with Base64.",
    ),
    (
        "to_sensitive_words_encode_rot13",
        to_rot13,
        "rot13",
        "Encode each harmful word with ROT13.",
    ),
    (
        "to_sensitive_words_encode_rot18",
        to_rot18,
        "rot18",
        "Encode each harmful word with ROT18.",
    ),
    (
        "to_sensitive_words_encode_rot47",
        to_rot47,
        "rot47",
        "Encode each harmful word with ROT47.",
    ),
    (
        "to_sensitive_words_encode_unicode",
        to_unicode_escape,
        "unicode",
        "Encode each harmful word as Unicode escapes.",
    ),
    (
        "to_sensitive_words_encode_ascii",
        to_ascii_ordinal,
        "ascii",
        "Encode each harmful word as ASCII ordinals.",
    ),
    (
        "to_sensitive_words_encode_leetspeak_basic",
        to_leetspeak_basic,
        "leetspeak_basic",
        "Basic leetspeak on each harmful word.",
    ),
    (
        "to_sensitive_words_encode_leetspeak_intermediate",
        to_leetspeak_intermediate,
        "leetspeak_intermediate",
        "Intermediate leetspeak on each harmful word.",
    ),
    (
        "to_sensitive_words_encode_leetspeak_advanced",
        to_leetspeak_advanced,
        "leetspeak_advanced",
        "Advanced leetspeak on each harmful word.",
    ),
]


def _make_word_fn(name: str, finish: Callable, doc: str) -> Callable:
    """The technique that finds the prompt's words, then applies ``finish``."""

    def fn(prompt: str, **kwargs) -> TechniqueGen:
        words = yield from harmful_words_gen(prompt, **kwargs)
        return finish(prompt, harmful_words=words)

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = doc
    return fn


# One technique per encoding, named to_sensitive_words_encode_<encoding>.
_ENCODED = [
    _make_word_fn(
        name,
        partial(_sensitive_encode_finish, encode_fn=encode_fn, encode_name=encode_name),
        doc,
    )
    for name, encode_fn, encode_name, doc in _ENCODE_SPECS
]


def to_sensitive_words_encode_separator(prompt: str, **kwargs) -> TechniqueGen:
    """Separate each character of harmful words with a randomly chosen separator."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    return _sensitive_separator_finish(prompt, words, kwargs.get("rng"))


_CHAR_SPECS = [
    (
        "to_sensitive_words_split",
        " ",
        "split",
        "Separate harmful-word chars with spaces: bomb -> b o m b.",
    ),
    (
        "to_sensitive_words_star",
        "*",
        "star",
        "Separate harmful-word chars with stars: bomb -> b*o*m*b.",
    ),
    (
        "to_sensitive_words_hyphen",
        "-",
        "hyphen",
        "Separate harmful-word chars with hyphens: bomb -> b-o-m-b.",
    ),
    (
        "to_sensitive_words_underscore",
        "_",
        "underscore",
        "Separate harmful-word chars with underscores: bomb -> b_o_m_b.",
    ),
]


# One technique per separator, named to_sensitive_words_<separator name>.
_SEPARATED = [
    _make_word_fn(
        name,
        partial(_sensitive_char_finish, separator=separator, sep_name=sep_name),
        doc,
    )
    for name, separator, sep_name, doc in _CHAR_SPECS
]


globals().update({fn.__name__: fn for fn in _ENCODED + _SEPARATED})


def to_sensitive_words_variables(prompt: str, **kwargs) -> TechniqueGen:
    """Replace harmful words with VAR_n placeholders + a definitions clause."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    return _sensitive_variables_finish(prompt, words)


# A "synonym" longer than this many words is treated as a non-answer (the
# original word is kept instead) — a single creative synonym, not a phrase.
_MAX_SYNONYM_WORDS = 4


def to_synonym_substitution(prompt: str, **kwargs) -> TechniqueGen:
    """Replace each harmful word with a creative LLM-generated benign synonym.

    Yields the harmful-word extraction request, then one synonym request per
    detected word. A multi-word (>4) synonym response is discarded (original
    kept). Returns ``(modified_prompt, "substitutions=w1->s1,...")``.
    """
    gen_model = kwargs.get("gen_model")
    prompt_dir = kwargs.get("prompt_dir")
    words = yield from harmful_words_gen(prompt, **kwargs)
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
    return result, f"substitutions={','.join(substitution_pairs)}"


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
