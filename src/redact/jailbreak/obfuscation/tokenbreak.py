"""Tokenbreak techniques over the words a model marks harmful.

``info`` lists the detected words with the tokens applied to them. A reply
that is not a short list of the prompt's words discards the prompt. A split
or a separator cuts a word between the characters a reader sees, never
between a letter and its marks, since a word of Bengali, Thai, Khmer or
Nepali holds marks and a joiner that belong to its letter.
"""

import random
import re
import unicodedata
from collections.abc import Callable

from redact.jailbreak.obfuscation.encoding import _clusters
from redact.jailbreak.technique import DISCARDED, TechniqueGen
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


def tokenbreak_prepend(word: str, rng: random.Random | None = None) -> tuple[str, str]:
    """Prepend a random token to the word."""
    token = (rng or random).choice(_CHAR_TOKENS)
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


def tokenbreak_split(word: str, rng: random.Random | None = None) -> tuple[str, str]:
    """Split word with a random token in the middle, between two of the
    characters a reader sees (a letter keeps its marks)."""
    rng = rng or random
    token_type = rng.choice(["char", "invisible_char", "variation_selector"])

    if token_type == "char":
        token = rng.choice(_CHAR_TOKENS)
        token_repr = token
    elif token_type == "invisible_char":
        token = rng.choice(_INVISIBLE_CHARS)
        token_repr = _INVISIBLE_CHAR_NAMES.get(token, "invis")
    else:
        token = rng.choice(_VARIATION_SELECTORS)
        token_repr = f"VS{ord(token) - 0xFE00:02d}"

    clusters = _clusters(word)
    if len(clusters) <= 1:
        return word + token, token_repr
    position = rng.randint(1, len(clusters) - 1)
    obfuscated_word = "".join(clusters[:position]) + token + "".join(clusters[position:])
    return obfuscated_word, token_repr


_STANDARD_DELIMITERS = ['"', '"""', "'", "'''", "bracket"]
_STRUCTURAL_DELIMITERS = [
    "<context>", "<system_instruction>",
    "<admin_note>", "<translation_task>",
]
_SEPARATOR_DELIMITERS = [".", "-", "_", "~", "/", "\\", "|"]


def tokenbreak_delimiter(
    word: str, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Wrap word with delimiters, or split it with a separator between the
    characters a reader sees."""
    rng = rng or random
    delimiter_type = rng.choice(["standard", "structural", "separator"])

    if delimiter_type == "standard":
        delimiter = rng.choice(_STANDARD_DELIMITERS)
        if delimiter == "bracket":
            brackets = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
            open_b, close_b = rng.choice(brackets)
            return open_b + word + close_b, f"{open_b}{close_b}"
        return delimiter + word + delimiter, delimiter
    elif delimiter_type == "structural":
        delimiter = rng.choice(_STRUCTURAL_DELIMITERS)
        return delimiter + word + delimiter, delimiter
    else:
        delimiter = rng.choice(_SEPARATOR_DELIMITERS)
        return delimiter.join(_clusters(word)), delimiter


_ATTACKS = {
    "prepend": tokenbreak_prepend,
    "split": tokenbreak_split,
    "delimiter": tokenbreak_delimiter,
}


def tokenbreak_obfuscate_word(
    word: str, attack: str, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Obfuscate a single word using specified attack type."""
    if attack not in _ATTACKS:
        raise ValueError(f"Unknown attack type: {attack}")
    return _ATTACKS[attack](word, rng)


_TOKENBREAK_EMPTY_RESPONSES = {"none", "n/a", "no harmful words", ""}
# Above this many words, treat the extraction response as a non-answer
# (the model rambled instead of returning a short word list).
_MAX_HARMFUL_WORDS_RESPONSE_LEN = 10
# A list marker before a word: a bullet, or a number with "." or ")".
_LIST_MARKER = re.compile(r"[-*•]|\d+[.)]")
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
# Around a word of a reply: punctuation, brackets, markdown and quotes,
# straight and curly, and an ellipsis.
_SURROUNDING = ".,;:!?\"'()[]*“”‘’„«»…"

#: The info of a technique whose word reply was malformed.
MALFORMED_WORDS = f"{DISCARDED}; reason=malformed word list"


def _is_mark(char: str) -> bool:
    """Whether ``char`` is a combining mark (Unicode category M)."""
    return unicodedata.category(char).startswith("M")


def _is_format(char: str) -> bool:
    """Whether ``char`` is a format character (Unicode category Cf): a zero-width
    space, non-joiner or joiner, which Persian, Khmer or Bengali words hold."""
    return unicodedata.category(char) == "Cf"


def _is_word(token: str) -> bool:
    """Whether ``token`` is a word: a letter, then letters, digits,
    apostrophes, hyphens, combining marks and format characters.

    ``\\w`` matches no combining mark, yet most Bengali, Nepali, Thai or Khmer
    words hold one (a vowel sign, a virama), nor a zero-width joiner or
    non-joiner, which Bengali and Persian words hold, so these are taken out
    before the match; a token may not start with one.
    """
    if not token or _is_mark(token[0]) or _is_format(token[0]):
        return False
    letters = "".join(c for c in token if not (_is_mark(c) or _is_format(c)))
    return _WORD.fullmatch(letters) is not None


def _is_lone_letter(word: str) -> bool:
    """Whether ``word`` is one letter (with its marks) that is not an
    ideograph, such as ``a`` or ``я``: a technique would change that letter
    wherever it stands in the prompt, inside other words too. A Chinese
    character is a word by itself, and stays."""
    letters = [c for c in word if not (_is_mark(c) or _is_format(c))]
    return (len(letters) == 1
            and not unicodedata.name(letters[0], "").startswith("CJK UNIFIED IDEOGRAPH"))


def _parse_harmful_words(reply: str, prompt: str) -> list[str] | None:
    """The words of an extraction reply: none for a ``none``-like reply, else
    each token with list markers dropped and surrounding punctuation stripped.

    A word named twice, in any case, counts once. Returns ``None`` for a
    malformed reply: more than ten words, a token that is not a word
    (:func:`_is_word`), a word of one letter that is no ideograph
    (:func:`_is_lone_letter`), or a word ``prompt`` does not hold (in any
    case).
    """
    raw = reply.strip()
    if raw.lower().strip(_SURROUNDING) in _TOKENBREAK_EMPTY_RESPONSES:
        return []
    words = [token.strip(_SURROUNDING) for token in raw.split()
             if not _LIST_MARKER.fullmatch(token)]
    if not words or len(words) > _MAX_HARMFUL_WORDS_RESPONSE_LEN:
        return None
    text = prompt.lower()
    if not all(_is_word(word) and not _is_lone_letter(word) and word.lower() in text
               for word in words):
        return None
    first_spelling: dict[str, str] = {}
    for word in words:
        first_spelling.setdefault(word.lower(), word)
    return list(first_spelling.values())


def harmful_words_gen(
    prompt: str, *, gen_model: str | None = None, prompt_dir=None, **kwargs
) -> TechniqueGen:
    """Yield the extract-harmful request; return the prompt's harmful words,
    or ``None`` for a malformed reply (:func:`_parse_harmful_words`).

    Shared first round for every tokenbreak/sensitive-word technique generator.
    """
    config = load_prompt("jailbreak", "extract_harmful", prompt_dir)
    messages = build_messages(config, prompt=prompt)
    return _parse_harmful_words((yield LLMRequest(gen_model, messages)), prompt)


def _tokenbreak_finish(
    prompt: str, attack: str, harmful_words: list[str], rng: random.Random | None,
) -> tuple[str, str]:
    """Structural tokenbreak applied to each harmful word (or every word)."""
    if not harmful_words:
        def _obfuscate_token(m: re.Match) -> str:
            return tokenbreak_obfuscate_word(m.group(), attack, rng)[0]
        return re.sub(r"\S+", _obfuscate_token, prompt), "detected_words="

    token_infos: list[str] = []
    result = prompt
    for word in harmful_words:
        obfuscated_word, token_repr = tokenbreak_obfuscate_word(word, attack, rng)
        token_infos.append(token_repr)
        result = re.sub(
            re.escape(word),
            lambda m, ow=obfuscated_word: match_case(m.group(), ow),
            result,
            flags=re.IGNORECASE,
        )
    return result, f"detected_words={' '.join(harmful_words)};tokens={','.join(token_infos)}"


def _tokenbreak_gen(
    prompt: str, attack: str, *, rng: random.Random | None = None, **kwargs,
) -> TechniqueGen:
    """Find the prompt's harmful words, then apply ``attack`` to each."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    if words is None:
        return prompt, MALFORMED_WORDS
    return _tokenbreak_finish(prompt, attack, words, rng)


def to_tokenbreak_prepend(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: prepend a random token before each harmful word."""
    return (yield from _tokenbreak_gen(prompt, "prepend", **kwargs))


def to_tokenbreak_split(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: insert a random character inside each harmful word."""
    return (yield from _tokenbreak_gen(prompt, "split", **kwargs))


def to_tokenbreak_delimiter(prompt: str, **kwargs) -> TechniqueGen:
    """TokenBreak: wrap each harmful word with randomly chosen delimiters."""
    return (yield from _tokenbreak_gen(prompt, "delimiter", **kwargs))


def get_tokenbreak_functions() -> list[Callable]:
    """Return the tokenbreak technique functions."""
    return [to_tokenbreak_prepend, to_tokenbreak_split, to_tokenbreak_delimiter]
