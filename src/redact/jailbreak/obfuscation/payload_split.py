"""Payload-split techniques: the prompt cut into two to four parts, each named
by a letter, with a variant of the pool ``reassembly`` of
``obfuscation/payload_split_templates.json`` around them that asks the reader
to join the parts in order.

``to_payload_split`` cuts at word boundaries, into parts of nearly equal word
count (the whitespace of the prompt becomes single spaces).
``to_payload_split_semantic`` has a model cut the prompt where its meaning
allows, and keeps the parts only when they give the prompt back, compared
without whitespace: a check in code, no check model. A reply with the wrong
count of parts is asked again, as for any numbered list; parts that differ
from the prompt discard it. Both draw the count from ``rng``, at most the
prompt's words, and discard a prompt of fewer than two words.

Each part is written in double quotes with a quote or a backslash in it
escaped, as a JSON string is (``json.dumps``), so a prompt holding quotes
leaves the quoting balanced; a part without either reads as it did.

``info`` is ``variant=<variant>;parts=<count>``.
"""

import json
import random
import string
from collections.abc import Callable
from functools import partial

from redact.jailbreak.directives import draw_variant, read_pools, uses_pool
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import extracted
from redact.llms.prompting import build_messages, extract_and_clean, load_prompt

_FILE = "obfuscation/payload_split_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)

#: The fewest and the most parts a prompt is cut into.
MIN_PARTS = 2
MAX_PARTS = 4

#: The info of a prompt of fewer words than ``MIN_PARTS``.
TOO_FEW_WORDS = f"{DISCARDED}; reason=too few words to split"
#: The info of a reply that no attempt gave as a list of the asked parts.
MALFORMED_PARTS = f"{DISCARDED}; reason=malformed parts"
#: The info of parts that, joined, are not the prompt.
PARTS_DIFFER = f"{DISCARDED}; reason=parts differ from the prompt"


def _parts_count(words: list[str], rng: random.Random) -> int:
    """How many parts to cut a prompt of ``words`` into: drawn from ``rng``
    between :data:`MIN_PARTS` and :data:`MAX_PARTS`, at most the number of
    words; 0 for a prompt of fewer than :data:`MIN_PARTS` words."""
    count = rng.randint(MIN_PARTS, MAX_PARTS)
    return min(count, len(words)) if len(words) >= MIN_PARTS else 0


def _cut(words: list[str], count: int) -> list[str]:
    """``words`` in ``count`` runs, the first ones one word longer when they
    do not divide evenly, each joined by single spaces."""
    size, longer = divmod(len(words), count)
    parts, start = [], 0
    for i in range(count):
        end = start + size + (i < longer)
        parts.append(" ".join(words[start:end]))
        start = end
    return parts


def _reassembly(parts: list[str], rng: random.Random | None) -> tuple[str, str]:
    """``parts`` named A, B, ... in a variant of the pool drawn by ``rng``
    (:func:`~redact.jailbreak.directives.draw_variant`): the text, and its info."""
    names = string.ascii_uppercase[:len(parts)]
    lines = "\n".join(f"{name}: {json.dumps(part, ensure_ascii=False)}"
                      for name, part in zip(names, parts))
    variant = draw_variant(_TEMPLATES["reassembly"], rng)
    text = variant["template"].format_map({"parts": lines, "names": ", ".join(names)})
    return text, f"variant={variant['name']};parts={len(parts)}"


@uses_pool(_FILE, "reassembly", slots=("parts", "names"), request_slot=False)
def to_payload_split(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Cut the prompt into two to four parts at word boundaries, name each by
    a letter and ask for them to be joined."""
    rng = rng or random
    words = prompt.split()
    count = _parts_count(words, rng)
    if not count:
        return prompt, TOO_FEW_WORDS
    return _reassembly(_cut(words, count), rng)


def _semantic_gen(
    prompt: str, *, gen_model: str | None = None, prompt_dir=None,
    rng: random.Random | None = None, **kwargs,
) -> TechniqueGen:
    """Ask the model for the prompt in a drawn count of parts, until a reply is
    a numbered list of that many (up to 3 replies), then keep them only if they
    give the prompt back."""
    rng = rng or random
    count = _parts_count(prompt.split(), rng)
    if not count:
        return prompt, TOO_FEW_WORDS
    messages = build_messages(
        load_prompt("jailbreak", "payload_split_semantic", prompt_dir),
        count=count, prompt=prompt)
    parts, _attempts = yield from extracted(
        gen_model, lambda: messages,
        partial(extract_and_clean, expected_count=count, strip_markdown=False))
    if not parts:
        return prompt, MALFORMED_PARTS
    if "".join(" ".join(parts).split()) != "".join(prompt.split()):
        return prompt, PARTS_DIFFER
    return _reassembly(parts, rng)


@uses_pool(_FILE, "reassembly", slots=("parts", "names"), request_slot=False)
def to_payload_split_semantic(prompt: str, **kwargs) -> TechniqueGen:
    """Have a model cut the prompt into two to four parts where its meaning
    allows, name each by a letter and ask for them to be joined."""
    return (yield from _semantic_gen(prompt, **kwargs))


def get_payload_split_functions() -> list[Callable]:
    """Return the payload-split techniques: a cut at word boundaries, and a
    cut by a model."""
    return [to_payload_split, to_payload_split_semantic]
