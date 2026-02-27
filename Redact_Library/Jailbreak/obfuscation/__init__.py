"""Obfuscation-based jailbreak techniques.

Six technique families, each in its own module:
- encoding: base64, rot13, leetspeak, morse, braille (pure, no LLM)
- translation: 6 low-resource languages (delegates to LLMs/translator.py)
- structural: json, xml, markdown wrapping (pure, no LLM)
- ascii_art: pyfiglet-based text art (pure, no LLM)
- tokenbreak: prepend, split, delimiter (needs LLM for extract_harmful)
- suffixes: punctuation, fragments, unicode, emoji (pure, no LLM)
"""

from .encoding import get_encoding_functions
from .translation import get_translation_functions
from .structural import get_structural_functions
from .ascii_art import get_ascii_art_functions
from .tokenbreak import get_tokenbreak_functions
from .suffixes import get_suffix_functions


def get_obfuscation_types() -> list[str]:
    """Return list of obfuscation type names."""
    return list(get_type_to_getter().keys())


def get_type_to_getter() -> dict[str, callable]:
    """Return mapping of obfuscation type name to its getter function."""
    return {
        "encoding_cyphering": get_encoding_functions,
        "low_resource_language": get_translation_functions,
        "structural_obfuscation": get_structural_functions,
        "ascii_art": get_ascii_art_functions,
        "tokenbreak": get_tokenbreak_functions,
        "adversarial_suffixes": get_suffix_functions,
    }


def get_all_obfuscation_functions() -> list:
    """Return flat list of all obfuscation technique functions."""
    funcs = []
    for getter in get_type_to_getter().values():
        funcs.extend(getter())
    return funcs
