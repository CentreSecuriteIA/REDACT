"""Obfuscation techniques, one module per spec family."""

from collections.abc import Callable

from .ascii_art import get_ascii_art_functions
from .casual_writing import get_casual_writing_functions
from .char_injection import get_char_injection_functions
from .code_switching import get_code_switching_functions
from .dialect import get_dialect_functions
from .encoding import get_encoding_functions
from .neutral_container import get_neutral_container_functions
from .noise import get_noise_functions
from .payload_split import get_payload_split_functions
from .romanization import get_romanization_functions
from .sensitive_words import get_sensitive_words_functions
from .special_tokens import get_special_tokens_functions
from .structural import get_structural_functions
from .suffixes import get_suffix_functions
from .tokenbreak import get_tokenbreak_functions
from .translation import get_translation_functions
from .transliteration import get_transliteration_functions
from .typos import get_typos_functions
from .unicode_smuggling import get_unicode_smuggling_functions
from .word_art import get_word_art_functions
from .wording import get_wording_functions


def get_all_obfuscation_functions() -> list[Callable]:
    """Return flat list of all obfuscation technique functions."""
    return (
        get_encoding_functions()
        + get_translation_functions()
        + get_structural_functions()
        + get_ascii_art_functions()
        + get_tokenbreak_functions()
        + get_sensitive_words_functions()
        + get_wording_functions()
        + get_typos_functions()
        + get_suffix_functions()
        + get_casual_writing_functions()
        + get_noise_functions()
        + get_neutral_container_functions()
        + get_payload_split_functions()
        + get_word_art_functions()
        + get_transliteration_functions()
        + get_code_switching_functions()
        + get_romanization_functions()
        + get_dialect_functions()
        + get_unicode_smuggling_functions()
        + get_char_injection_functions()
        + get_special_tokens_functions()
    )
