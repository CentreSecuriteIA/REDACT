#!/usr/bin/env python
"""Smoke-run the spec 1.5 literature families (semantic re-encoding,
translationese, sociolect) on benign sentences against real model clients, and
print acceptance and discard counts by reason.

Counts only: no transformed or discarded text is printed, just how many of N
benign inputs each technique accepted and, for the rest, the gate or check
reason it reported. Meant to be run on a GPU pod where the generation,
translation and check models are served; the main session reads the numbers.

Usage (from the repo root, PYTHONPATH=src):
    python scripts/literature_smoke.py --n 20 --gen-model venice-uncensored \
        --families semantic_reencoding sociolect
The families default to the three spec 1.5 ones; --families narrows them.
semantic_reencoding and sociolect run on the generation model (--gen-model),
the check falling back to it; translationese runs on the translation role's
registered model (a GPU translator, or deepseek-v3.2 by name) and judges with
the generation model.
"""

import argparse
import collections
import logging

from redact.jailbreak.chain import apply_combination, combine_techniques
from redact.jailbreak.obfuscation.semantic_reencoding import (
    get_semantic_reencoding_functions,
)
from redact.jailbreak.obfuscation.sociolect import get_sociolect_functions
from redact.jailbreak.obfuscation.translationese import get_translationese_functions
from redact.jailbreak.technique import is_discarded
from redact.llms.client import ModelClient

logger = logging.getLogger("literature_smoke")

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
    "semantic_reencoding": get_semantic_reencoding_functions,
    "translationese": get_translationese_functions,
    "sociolect": get_sociolect_functions,
}

#: The fixed code-side gate reasons these families write into ``feedback=``,
#: each mapped to the short label this script prints. A feedback that starts
#: with one of these prefixes is a gate reason (code, not model text); any
#: other feedback is a check model's verdict. Longest prefixes first, so
#: "acrostic count" is not read as "acrostic mismatch".
_GATE_LABELS = (
    ("acrostic count", "acrostic_count"),
    ("acrostic mismatch", "acrostic_mismatch"),
    ("no math", "no_math_notation"),
    ("too few", "too_few_lines"),
    ("not verse", "not_verse"),
    ("length_ratio", "length_ratio"),
    ("unchanged", "unchanged"),
    ("empty", "empty"),
    ("forward:", "forward_leg"),
    ("back:", "back_leg"),
    ("judge=", "check_judge"),
    ("to pivot", "leg_to_pivot"),
    ("to english", "leg_to_english"),
)

#: Words a model's Yes/No verdict can start with.
_VERDICT_WORDS = frozenset({
    "yes", "no", "ok", "okay", "accept", "accepted", "pass", "passed",
    "fail", "failed", "reject", "rejected", "true", "false",
})


def _reason(info: str) -> str:
    """The short discard reason of a technique's ``info`` (no text), or
    ``accepted``.

    A rejection carries its real gate or check reason in ``feedback=``: a
    fixed code-side reason (``no math notation``, ``acrostic mismatch``,
    ``length_ratio=...``, a translationese leg's ``forward:``/``back:`` or its
    ``judge=``) or the check model's ``Yes``/``No`` verdict. We read
    ``feedback`` first: a gate reason becomes its label from
    :data:`_GATE_LABELS` and a check verdict becomes ``check:<word>``, so no
    model text is ever printed; then we fall back to the fixed ``reason`` field
    and to ``pivot``/``language``.
    """
    if not is_discarded(info):
        return "accepted"
    fields: dict[str, str] = {}
    for part in info.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key and key not in fields:
            fields[key] = value.strip()
    value = fields.get("feedback", "")
    if value:
        low = value.lower()
        for prefix, label in _GATE_LABELS:
            if low.startswith(prefix):
                return label
        word = value.split()[0].strip(".,:;!?\"'()").lower()
        return f"check:{word}" if word in _VERDICT_WORDS else "check:other"
    if fields.get("reason"):
        return fields["reason"].replace(" ", "_")
    return fields.get("pivot") or fields.get("language") or "discarded"


def run(fns, sentences, *, client):
    """Run each technique on each sentence; return ``{name: Counter(reason)}``."""
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
                        help="the generation (and fallback check, and judge) model")
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
