"""Obfuscation-based jailbreak techniques.

Eight technique families, each in its own module:
- encoding: base64, rot13/18/47, leetspeak, unicode, ordinal, separator, morse, braille (pure, no LLM)
- translation: 20 languages across resource tiers (translation-role model)
- structural: json, xml, markdown wrapping (pure, no LLM)
- ascii_art: pyfiglet-based text art (pure, no LLM)
- tokenbreak: prepend, split, delimiter (needs LLM for extract_harmful)
- sensitive_words: encoding + char obfuscation applied to LLM-detected harmful words
- typos: LLM-rewritten prompts with 4 typo density levels
- suffixes: punctuation, fragments, unicode, emoji (pure, no LLM)
"""

from collections.abc import Callable

from .ascii_art import get_ascii_art_functions
from .encoding import get_encoding_functions
from .structural import get_structural_functions
from .suffixes import get_suffix_functions
from .tokenbreak import get_sensitive_words_functions, get_tokenbreak_functions
from .translation import get_translation_functions
from .typos import get_typos_functions


def get_all_obfuscation_functions() -> list[Callable]:
    """Return flat list of all obfuscation technique functions."""
    return (
        get_encoding_functions()
        + get_translation_functions()
        + get_structural_functions()
        + get_ascii_art_functions()
        + get_tokenbreak_functions()
        + get_sensitive_words_functions()
        + get_typos_functions()
        + get_suffix_functions()
    )
