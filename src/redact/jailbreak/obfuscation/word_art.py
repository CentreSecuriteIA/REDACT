"""Word-art technique: one word of the prompt a model marks, drawn as ASCII art
in a pyfiglet font, and the prompt with ``[MASK]`` in its place, with a
variant of the pool ``decode`` of ``obfuscation/word_art_templates.json``
around both that asks the reader to read the art and put the word back.

The marking call and its strict word-list parse are those of the other
word-level techniques (:func:`~.tokenbreak.harmful_words_gen`); the first
marked word is the one drawn, and every occurrence of it, in any case, is
masked, so the word filled back in gives the prompt again. The font is
``to_ascii_art``'s, drawn from those with a glyph for every character of the
word within its cap, so a word no font draws is discarded with the reason
``to_ascii_art`` gives. A reply that is not a short list of the prompt's
words, or marks no word, discards the prompt.

``info`` is ``variant=<variant>;font=<font>;detected_words=<word>``.
"""

import random
import re
from collections.abc import Callable

from redact.jailbreak.directives import draw_variant, read_pools, uses_pool
from redact.jailbreak.obfuscation.ascii_art import to_ascii_art
from redact.jailbreak.obfuscation.sensitive_words import NO_SENSITIVE_WORD
from redact.jailbreak.obfuscation.tokenbreak import MALFORMED_WORDS, harmful_words_gen
from redact.jailbreak.technique import DISCARDED, TechniqueGen

_FILE = "obfuscation/word_art_templates.json"
_POOLS: dict[str, list[dict]] = read_pools(_FILE)

#: What stands in the prompt for the word drawn as art.
MASK = "[MASK]"


def _word_art_gen(
    prompt: str, *, rng: random.Random | None = None, **kwargs,
) -> TechniqueGen:
    """Mark the prompt's first harmful word, draw it in a font and mask it in
    the prompt, in a variant of the pool."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    if words is None:
        return prompt, MALFORMED_WORDS
    if not words:
        return prompt, NO_SENSITIVE_WORD
    word = words[0]
    art, info = to_ascii_art(word, rng=rng)
    if info.startswith(DISCARDED):
        return prompt, info
    masked = re.sub(re.escape(word), MASK, prompt, flags=re.IGNORECASE)
    variant = draw_variant(_POOLS["decode"], rng)
    text = variant["template"].format_map({"art": art.rstrip(), "prompt": masked})
    return text, f"variant={variant['name']};{info};detected_words={word}"


@uses_pool(_FILE, "decode", slots=("art", "prompt"))
def to_word_art(prompt: str, **kwargs) -> TechniqueGen:
    """Draw the prompt's first harmful word as ASCII art, put ``[MASK]`` in its
    place and ask for the art to be read and the word put back."""
    return (yield from _word_art_gen(prompt, **kwargs))


def get_word_art_functions() -> list[Callable]:
    """Return the word-art technique."""
    return [to_word_art]
