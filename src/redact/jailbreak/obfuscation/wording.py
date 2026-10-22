"""Wording techniques over the words a model marks harmful: each marked word is
put in other terms, and a variant of a pool of
``obfuscation/wording_templates.json`` around the prompt says how to read them.

``to_pseudonym_substitution`` has the model give each marked word a code name
(prompt ``jailbreak/pseudonym``, one call per word), swaps it in, case-matched,
and adds the key, one ``<alias> = <word>`` line per word, in a variant of the
pool ``pseudonym_key``. ``to_word_game`` replaces each marked word with a
placeholder, ``[W1]``, ``[W2]``, ..., and has the model write a puzzle whose
answer is the word (prompt ``jailbreak/word_game``, one call per word); a
variant of the pool ``word_game_solve`` lists them, one ``[Wn]: <puzzle>`` line
per placeholder, and asks for them to be solved first. Every occurrence of a
marked word, in any case, is replaced, in one pass, so no replacement is
replaced again.

The marking call and its strict word-list parse are those of the other
word-level techniques (:func:`~.tokenbreak.harmful_words_gen`). A reply that is
not a short list of the prompt's words, or marks no word, discards the prompt.
A word's reply is asked for again once when it is not a code name of one or two
words (on one line, without ``=``, holding no marked word, not an earlier code
name), or when it is not a puzzle (empty, longer than :data:`MAX_PUZZLE_WORDS`
or holding the word itself); a second such reply discards the prompt.

``info`` is ``variant=<variant>;detected_words=<words>``.
"""

import random
import re
from collections.abc import Callable, Generator
from functools import partial

from redact.jailbreak.directives import draw_variant, read_pools, uses_pool
from redact.jailbreak.obfuscation.sensitive_words import NO_SENSITIVE_WORD
from redact.jailbreak.obfuscation.tokenbreak import (
    MALFORMED_WORDS,
    harmful_words_gen,
    match_case,
)
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import LLMRequest, extracted
from redact.llms.prompting import build_messages, load_prompt

_FILE = "obfuscation/wording_templates.json"
_POOLS: dict[str, list[dict]] = read_pools(_FILE)

#: The most words of a code name: a name, not a phrase.
MAX_ALIAS_WORDS = 2
#: The most words of a puzzle: a clue, not an essay.
MAX_PUZZLE_WORDS = 40
#: The reply limit of a word's call, in tokens: a code name or a puzzle is short.
WORD_REPLY_TOKENS = 128
#: Replies asked for each word: the first, and one more.
WORD_ATTEMPTS = 2
#: What the key of a code name separates the name from its word with.
KEY_SEPARATOR = " = "

#: The info of a prompt whose code names, or puzzles, no reply gave.
MALFORMED_PSEUDONYM = f"{DISCARDED}; reason=malformed pseudonym"
MALFORMED_PUZZLE = f"{DISCARDED}; reason=malformed puzzle"

_QUOTES = "\"'`"


def _distinct(words: list[str]) -> list[str]:
    """``words`` without a repeat, in any case, the first of each kept."""
    seen: set[str] = set()
    kept = []
    for word in words:
        if word.casefold() not in seen:
            seen.add(word.casefold())
            kept.append(word)
    return kept


def _swap(
    prompt: str, words: list[str], replacement: Callable[[int, str], str],
) -> str:
    """``prompt`` with every occurrence of each of ``words``, in any case,
    replaced by ``replacement(i, matched)``, ``i`` the word's place and
    ``matched`` the text it matched. One pass, the longest word first where
    two start together, so a replacement is never replaced again."""
    pattern = re.compile(
        "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True)),
        re.IGNORECASE)

    def replace(match: re.Match) -> str:
        i = next(i for i, word in enumerate(words)
                 if re.fullmatch(re.escape(word), match.group(), re.IGNORECASE))
        return replacement(i, match.group())

    return pattern.sub(replace, prompt)


def _alias_of(
    words: list[str], word: str, reply: str, earlier: list[str],
) -> list[str]:
    """``[alias]`` for a reply that is a code name for ``word``: one line of
    one or two words, quotes around it dropped, without ``=``, holding no
    marked word and equal to no earlier alias, in any case; ``[]`` for any
    other reply, which :func:`~redact.llm_pipeline.extracted` asks for again."""
    alias = reply.strip().strip(_QUOTES).strip()
    folded = alias.casefold()
    if (not alias or len(alias.splitlines()) != 1
            or len(alias.split()) > MAX_ALIAS_WORDS or "=" in alias
            or any(marked.casefold() in folded for marked in words)
            or folded in {text.casefold() for text in earlier}):
        return []
    return [alias]


def _puzzle_of(word: str, reply: str, earlier: list[str]) -> list[str]:
    """``[puzzle]`` for a reply that is a puzzle for ``word``: not empty, at
    most :data:`MAX_PUZZLE_WORDS` words, set on one line, and without the
    word itself, in any case; ``[]`` for any other reply."""
    puzzle = " ".join(reply.split())
    if (not puzzle or len(puzzle.split()) > MAX_PUZZLE_WORDS
            or word.casefold() in puzzle.casefold()):
        return []
    return [puzzle]


def _replies_gen(
    words: list[str], prompt: str, config: dict, accept: Callable,
    *, gen_model: str | None,
) -> Generator[LLMRequest, str, list[str] | None]:
    """One reply per marked word: ``config``'s prompt shown the word and the
    ``prompt``, asked for up to :data:`WORD_ATTEMPTS` times until
    ``accept(word, reply, earlier)`` gives it (``earlier`` the replies kept so
    far). ``None`` when a word has no reply."""
    replies: list[str] = []
    for word in words:
        messages = build_messages(config, word=word, prompt=prompt)
        kept, _attempts = yield from extracted(
            gen_model, lambda: messages,
            lambda reply: accept(word, reply, replies),
            max_tokens=WORD_REPLY_TOKENS, max_attempts=WORD_ATTEMPTS)
        if not kept:
            return None
        replies.append(kept[0])
    return replies


def _pseudonym_gen(
    prompt: str, *, rng: random.Random | None = None, **kwargs,
) -> TechniqueGen:
    """Mark the prompt's harmful words, ask for a code name for each, swap
    them in and add the key, in a variant of the pool."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    if words is None:
        return prompt, MALFORMED_WORDS
    words = _distinct(words)
    if not words:
        return prompt, NO_SENSITIVE_WORD
    config = load_prompt("jailbreak", "pseudonym", kwargs.get("prompt_dir"))
    aliases = yield from _replies_gen(
        words, prompt, config, partial(_alias_of, words),
        gen_model=kwargs.get("gen_model"))
    if aliases is None:
        return prompt, MALFORMED_PSEUDONYM
    swapped = _swap(prompt, words,
                    lambda i, matched: match_case(matched, aliases[i]))
    key = "\n".join(f"{alias}{KEY_SEPARATOR}{word}"
                    for alias, word in zip(aliases, words))
    variant = draw_variant(_POOLS["pseudonym_key"], rng)
    text = variant["template"].format_map({"prompt": swapped, "key": key})
    return text, f"variant={variant['name']};detected_words={' '.join(words)}"


def _placeholder(i: int) -> str:
    """What stands in the prompt for the ``i``-th marked word."""
    return f"[W{i + 1}]"


def _word_game_gen(
    prompt: str, *, rng: random.Random | None = None, **kwargs,
) -> TechniqueGen:
    """Mark the prompt's harmful words, ask for a puzzle for each, put a
    placeholder in the prompt for each and list the puzzles, in a variant of
    the pool."""
    words = yield from harmful_words_gen(prompt, **kwargs)
    if words is None:
        return prompt, MALFORMED_WORDS
    words = _distinct(words)
    if not words:
        return prompt, NO_SENSITIVE_WORD
    config = load_prompt("jailbreak", "word_game", kwargs.get("prompt_dir"))
    puzzles = yield from _replies_gen(
        words, prompt, config, _puzzle_of, gen_model=kwargs.get("gen_model"))
    if puzzles is None:
        return prompt, MALFORMED_PUZZLE
    swapped = _swap(prompt, words, lambda i, _matched: _placeholder(i))
    lines = "\n".join(f"{_placeholder(i)}: {puzzle}"
                      for i, puzzle in enumerate(puzzles))
    variant = draw_variant(_POOLS["word_game_solve"], rng)
    text = variant["template"].format_map({"prompt": swapped, "puzzles": lines})
    return text, f"variant={variant['name']};detected_words={' '.join(words)}"


@uses_pool(_FILE, "pseudonym_key", slots=("prompt", "key"))
def to_pseudonym_substitution(prompt: str, **kwargs) -> TechniqueGen:
    """Replace each harmful word with a code name a model gives it, and add a
    key that says which name stands for which word."""
    return (yield from _pseudonym_gen(prompt, **kwargs))


@uses_pool(_FILE, "word_game_solve", slots=("prompt", "puzzles"))
def to_word_game(prompt: str, **kwargs) -> TechniqueGen:
    """Replace each harmful word with a placeholder, and give a puzzle for each
    that a reader is asked to solve first."""
    return (yield from _word_game_gen(prompt, **kwargs))


def get_wording_functions() -> list[Callable]:
    """Return the wording techniques: a code name, and a word game."""
    return [to_pseudonym_substitution, to_word_game]
