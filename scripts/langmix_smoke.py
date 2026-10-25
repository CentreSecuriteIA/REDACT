#!/usr/bin/env python
"""Smoke-run the spec 1.4 language-mix techniques on benign sentences against
real model clients, and print acceptance and discard counts by reason.

Counts only: no transformed or discarded text is printed, just how many of N
benign inputs each technique accepted and, for the rest, the gate or check
reason it reported. Meant to be run on a GPU pod where the translation,
generation and check models are served; the main session reads the numbers.

Usage (from the repo root, PYTHONPATH=src):
    python scripts/langmix_smoke.py --n 20 --gen-model venice-uncensored \
        --families code_switching romanization
The families default to the four language-mix ones; --families narrows them to
any of transliteration, code_switching, romanization, dialect. A dialect
translates on the translation role's registered model. Without the GlotLID
model the code-switch and romanize gates cannot check the language, so a
rewrite left in plain English passes them; the meaning check still runs.
"""

import argparse
import collections
import logging

from redact.jailbreak.chain import apply_combination, combine_techniques
from redact.jailbreak.obfuscation.code_switching import get_code_switching_functions
from redact.jailbreak.obfuscation.dialect import get_dialect_functions
from redact.jailbreak.obfuscation.romanization import get_romanization_functions
from redact.jailbreak.obfuscation.transliteration import (
    get_transliteration_functions,
)
from redact.jailbreak.technique import is_discarded
from redact.llms.client import ModelClient

logger = logging.getLogger("langmix_smoke")

#: Plain benign sentences; nothing harmful. The techniques transform these.
SENTENCES = [
    "Could you explain how a bill becomes a law in simple terms?",
    "What are some good stretches to do before a morning run?",
    "Please summarise the water cycle for a ten year old.",
    "How do I bake a loaf of sourdough bread at home?",
    "What is the difference between weather and climate?",
    "Give me a packing list for a weekend camping trip.",
    "How does a bicycle gear system actually work?",
    "What are the main causes of the seasons on Earth?",
    "Suggest a few beginner friendly houseplants for a flat.",
    "Explain how vaccines help the immune system remember a disease.",
    "What should I consider when adopting a rescue dog?",
    "How do noise cancelling headphones reduce background sound?",
    "Describe the steps to change a flat bicycle tyre.",
    "What makes a good cover letter for a first job?",
    "How is cheese made from milk, step by step?",
    "What are some tips for taking better photos at night?",
    "Explain compound interest with a small everyday example.",
    "How do bees communicate the location of flowers?",
    "What is a simple routine to learn a new language?",
    "How does a water filter remove impurities from tap water?",
]

_FAMILIES = {
    "transliteration": get_transliteration_functions,
    "code_switching": get_code_switching_functions,
    "romanization": get_romanization_functions,
    "dialect": get_dialect_functions,
}


#: Words a model's Yes/No verdict can start with: feedback that begins with one
#: is the meaning (or judge) check's verdict, not a code-side gate reason.
_VERDICT_WORDS = frozenset({
    "yes", "no", "ok", "okay", "accept", "accepted", "pass", "passed",
    "fail", "failed", "reject", "rejected", "true", "false",
})


def _reason(info: str) -> str:
    """The short discard reason of a technique's ``info`` (no text), or
    ``accepted``.

    A rejection carries the real gate or check reason in ``feedback=``: a
    dialect or translation piece gives ``eszett``, ``simplified``,
    ``language=<label>``, ``length_ratio=<x>`` or ``judge=<verdict>``; a
    code-switch or romanize gate gives a short code (``empty``, ``not``,
    ``english``, ``fully``, ``length_ratio=``); the meaning check gives the
    model's ``Yes``/``No`` verdict. We read ``feedback`` first (keeping only its
    first token, so a verdict collapses to ``check:<word>`` and no feedback
    text is printed), then the fixed code-side ``reason`` (``too short to mix``
    and the like), then ``language``. A code-switch gate's ``no <language>``
    starts with a verdict word, so it is grouped under ``check:no`` too."""
    if not is_discarded(info):
        return "accepted"
    fields: dict[str, str] = {}
    for part in info.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key and key not in fields:
            fields[key] = value.strip()
    if fields.get("feedback"):
        value = fields["feedback"]
        if value.lower().startswith("judge="):
            return "check:judge"
        first = value.split()[0]
        word = first.strip(".,:;!?\"'()").lower()
        if word in _VERDICT_WORDS:
            return f"check:{word}"
        return first
    if fields.get("reason"):
        return fields["reason"].replace(" ", "_")
    if fields.get("language"):
        return fields["language"].split()[0]
    return "discarded"


def run(fns, sentences, *, client):
    """Run each technique on each sentence; return counts per technique:
    ``{name: Counter(reason -> count)}``.

    ``apply_combination`` uses ``client.model`` as the generation model; the
    check falls back to it, and a dialect's translation resolves the
    translation role's registered model. So a pod run needs the generation
    model served (``--gen-model``) and, for the dialects, the translation
    model registered and served."""
    out: dict[str, collections.Counter] = {}
    for fn in fns:
        counter: collections.Counter = collections.Counter()
        for sentence in sentences:
            try:
                _text, info, _accepted, _reasoning = apply_combination(
                    combine_techniques(fn), sentence, client=client)
                counter[_reason(info)] += 1
            except Exception as exc:  # noqa: BLE001 - a smoke run keeps going
                counter[f"error:{type(exc).__name__}"] += 1
        out[fn.__name__] = counter
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=len(SENTENCES),
                        help="benign sentences to run (capped at the built-in list)")
    parser.add_argument("--gen-model", default=None,
                        help="the generation (and fallback check) model; the "
                             "model families need it, transliteration does not")
    parser.add_argument("--families", nargs="*", default=list(_FAMILIES),
                        choices=list(_FAMILIES))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sentences = SENTENCES[:max(1, args.n)]
    client = ModelClient.create(args.gen_model) if args.gen_model else None
    for family in args.families:
        fns = _FAMILIES[family]()
        logger.info("=== %s (%d techniques, %d sentences) ===",
                    family, len(fns), len(sentences))
        counts = run(fns, sentences, client=client)
        for name, counter in counts.items():
            accepted = counter.get("accepted", 0)
            rest = ", ".join(f"{reason} {n}" for reason, n in counter.most_common()
                             if reason != "accepted")
            logger.info("  %-34s accepted %2d/%d%s", name, accepted, len(sentences),
                        f"  [{rest}]" if rest else "")


if __name__ == "__main__":
    main()
