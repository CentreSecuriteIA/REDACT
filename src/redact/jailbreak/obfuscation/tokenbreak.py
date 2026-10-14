"""Tokenbreak techniques over the words a model marks harmful.

``info`` lists the detected words with the tokens applied to them.
"""

import random
import re
from collections.abc import Callable

from redact.jailbreak.technique import TechniqueGen
from redact.llm_pipeline import LLMRequest
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


def harmful_words_gen(
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


def to_tokenbreak_prepend(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: prepend a random token before each harmful word."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    return _tokenbreak_finish(prompt, "prepend", words)


def to_tokenbreak_split(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: insert a random character inside each harmful word."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    return _tokenbreak_finish(prompt, "split", words)


def to_tokenbreak_delimiter(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: wrap each harmful word with randomly chosen delimiters."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    return _tokenbreak_finish(prompt, "delimiter", words)


def get_tokenbreak_functions() -> list[Callable]:
    """Return the tokenbreak technique functions."""
    return [to_tokenbreak_prepend, to_tokenbreak_split, to_tokenbreak_delimiter]
