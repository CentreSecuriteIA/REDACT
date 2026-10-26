"""Bijection learning (Huang 2410.01294): a per-unit random letter cipher
taught to the model in context, then an encoded request.

``to_bijection_learning`` draws a bijection over the 26 ASCII letters from the
unit's rng — one of three schemes: letter to letter, letter to a distinct
two-digit number, or letter to a distinct symbol — and shows it through worked
examples (benign sentences from ``obfuscation/bijection_sentences.json``, each
with its encoding) before the encoded prompt. It is a pure transform (no model
call).

The cipher is a full-text, character-level encoding, so it refuses exactly what
the heavy full-text encodings refuse (see the spec family ``_comment``): a
translation and the other whole-text language rewrites, the meaning
re-encodings, ``ascii_art``, the ``encode`` family, ``payload_split``,
``word_art``, the whole-prompt hidings and reorderings, the manipulation
families and ``to_answer_in_encoding``. A test re-derives that set from the ten
heavy encodings when it runs and requires this family's to match it exactly, so
neither side can gain a refusal the other lacks. Letters are normalised to lower
case before mapping (the cipher is over 26 symbols), so decoding recovers the
request in lower case; every non-letter passes through, and the encoding is
emitted as space-separated tokens so a reader (and the test) can decode it
unambiguously. It is drawn only for a prompt that holds a Latin letter
(:func:`_has_latin`): a text of another script has nothing for the cipher to
act on.

A character of the prompt that is not a letter passes through the cipher, so
under the symbol scheme the key's images are drawn from the symbols the prompt
does not hold (:func:`_free_symbols`); a prompt holding the space mark, or more
than eight of the pool's symbols, cannot be enciphered unambiguously and is
discarded before anything is rendered (:func:`_collision_reason`).

``info`` is ``scheme=<letter|digit|symbol>;variant=<variant>``, or
``DISCARDED; reason=...; scheme=<scheme>``.
"""

import random
import string
from collections.abc import Callable

from ..directives import draw_variant, load_templates, read_pools, uses_pool
from ..technique import DISCARDED, applicable_when

_FILE = "obfuscation/bijection_learning_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)

#: The benign sentences the teaching examples are drawn from.
_SENTENCES: list[str] = load_templates("obfuscation", "bijection_sentences.json")[
    "sentences"]

#: The 26 lower-case ASCII letters the cipher maps.
_LETTERS = string.ascii_lowercase

#: The distinct non-ASCII symbols the symbol scheme draws from (34, so up to
#: eight of them may be left out for a text that holds them). A symbol in the
#: text itself passes through the cipher, and a reader decoding with the shown
#: key would read it as the letter it stands for, so the images are drawn from
#: the symbols the text does not hold (:func:`_free_symbols`); a text holding
#: more than eight of them, or the space mark, is discarded.
_SYMBOLS = list("αβγδεζηθικλμνξοπρστυφχψω☀☁☂☃★☆♠♣♥♦")

#: The placeholder a space is encoded as (non-ASCII, distinct from every
#: symbol image), so spaces in the source do not clash with the token
#: separator.
_SPACE_MARK = "␣"

#: How many worked examples the teaching section shows.
_EXAMPLES = 5

#: The cipher schemes: how each letter's image is built.
_SCHEMES = ("letter", "digit", "symbol")


def _has_latin(text: str) -> bool:
    """Whether ``text`` holds an ASCII letter: the only characters the cipher
    maps, so it is drawn only for a text that has one."""
    return any(c.isascii() and c.isalpha() for c in text)


def _free_symbols(text: str) -> list[str]:
    """The symbols of :data:`_SYMBOLS` ``text`` does not hold, in pool order.

    A symbol the text holds passes through the cipher unchanged, so drawing it
    as a letter's image would make the shown key ambiguous: the reader would
    decode that character as the letter. The whole pool for a text with none
    of them, which is the usual case, so the draw is the one made before this
    check existed."""
    return [symbol for symbol in _SYMBOLS if symbol not in text]


def _collision_reason(scheme: str, text: str) -> str:
    """Why ``text`` cannot be enciphered under ``scheme``, or ``""``.

    The space mark stands for a space in every scheme, so a text holding it
    is ambiguous whatever the scheme; the symbol scheme also needs 26 symbols
    the text does not hold. The letter and digit schemes have no collision: a
    letter of the text is always mapped, and every other character becomes a
    one-character token, which no two-digit image equals."""
    if _SPACE_MARK in text:
        return "space mark in the text"
    if scheme == "symbol" and len(_free_symbols(text)) < len(_LETTERS):
        return "too few free symbols"
    return ""


def _mapping(scheme: str, rng: random.Random, text: str = "") -> dict[str, str]:
    """A bijection from the 26 lower-case letters to their images under
    ``scheme``, drawn from ``rng``: a permutation of the letters, the numbers
    ``00``..``25`` shuffled, or 26 shuffled symbols that ``text`` does not
    hold (:func:`_free_symbols`). Every image is distinct and, for the symbol
    scheme, absent from ``text``, so the mapping is invertible and the shown
    key decodes the text unambiguously.

    Raises:
        ValueError: The symbol scheme and ``text`` holds more than eight of
            the pool's symbols (:func:`_collision_reason` is the caller's
            check).
    """
    if scheme == "letter":
        images = rng.sample(_LETTERS, len(_LETTERS))
    elif scheme == "digit":
        images = rng.sample([f"{n:02d}" for n in range(26)], 26)
    else:
        free = _free_symbols(text)
        if len(free) < len(_LETTERS):
            raise ValueError(
                f"the text holds all but {len(free)} of the symbol scheme's "
                "images, so no key over 26 free symbols is left")
        images = rng.sample(free, len(_LETTERS))
    return dict(zip(_LETTERS, images, strict=True))


def _canon(text: str) -> str:
    """``text`` with its ASCII letters lower-cased (the cipher's normalised
    form); every other character is left as it is."""
    return "".join(c.lower() if (c.isascii() and c.isalpha()) else c for c in text)


def _encode(text: str, mapping: dict[str, str]) -> str:
    """``text`` enciphered: each ASCII letter mapped (lower-cased first), each
    space shown as :data:`_SPACE_MARK`, every other character kept, the
    per-character tokens joined by a single space so the result decodes
    unambiguously."""
    tokens = []
    for c in text:
        if c.isascii() and c.isalpha():
            tokens.append(mapping[c.lower()])
        elif c == " ":
            tokens.append(_SPACE_MARK)
        else:
            tokens.append(c)
    return " ".join(tokens)


def _decode(encoded: str, mapping: dict[str, str]) -> str:
    """The inverse of :func:`_encode` with the same ``mapping``: recovers the
    normalised (lower-cased) source. A token that is an image maps back to its
    letter, :data:`_SPACE_MARK` to a space, any other token to itself."""
    inverse = {image: letter for letter, image in mapping.items()}
    out = []
    for token in encoded.split(" "):
        if token in inverse:
            out.append(inverse[token])
        elif token == _SPACE_MARK:
            out.append(" ")
        else:
            out.append(token)
    return "".join(out)


def _examples_block(mapping: dict[str, str], rng: random.Random) -> str:
    """The key line (``a = <image>  b = <image> ...``) and
    :data:`_EXAMPLES` worked plaintext/ciphertext pairs drawn from
    :data:`_SENTENCES`; no English of its own (the framing template holds the
    instructions)."""
    key = "  ".join(f"{letter} = {mapping[letter]}" for letter in _LETTERS)
    chosen = rng.sample(_SENTENCES, min(_EXAMPLES, len(_SENTENCES)))
    worked = "\n\n".join(f"{sentence}\n{_encode(sentence, mapping)}"
                         for sentence in chosen)
    return f"{key}\n\n{worked}"


@applicable_when(lambda text, entry_type: _has_latin(text))
@uses_pool(_FILE, "bijection_learning", slots=("prompt", "examples"))
def to_bijection_learning(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Teach a random letter cipher by example, then show the encoded prompt.

    A prompt the drawn scheme cannot encipher unambiguously is discarded
    before anything is rendered (:func:`_collision_reason`): one holding the
    space mark, or, under the symbol scheme, more than eight of the symbols
    the key draws its images from, since those characters pass through the
    cipher and a reader would decode them as letters.
    """
    rng = rng or random
    scheme = rng.choice(_SCHEMES)
    reason = _collision_reason(scheme, prompt)
    if reason:
        return prompt, f"{DISCARDED}; reason={reason}; scheme={scheme}"
    mapping = _mapping(scheme, rng, prompt)
    examples = _examples_block(mapping, rng)
    variant = draw_variant(_TEMPLATES["bijection_learning"], rng)
    text = variant["template"].format_map(
        {"prompt": _encode(prompt, mapping), "examples": examples})
    return text, f"scheme={scheme};variant={variant['name']}"


def get_bijection_functions() -> list[Callable]:
    """Return the bijection-learning technique function."""
    return [to_bijection_learning]
