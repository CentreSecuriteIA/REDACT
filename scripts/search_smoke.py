"""A small, real-model smoke run of the combination search.

Runs :func:`redact.jailbreak.search_combinations` on a few seed inputs against
a local target model, with a local judge and generation model, and prints
counts only — never the jailbreak text, a target reply or a template. For a
driver to run on a GPU pod after the offline tests pass.

From the repository root, with ``PYTHONPATH=src`` or redact installed::

    python scripts/search_smoke.py --root /workspace/searchsmoke
    python scripts/search_smoke.py --root /workspace/searchsmoke \\
        --target llama-3.2-3b-debug --judge venice-uncensored-fp8 --n 4

Defaults: target ``llama-3.2-3b-debug`` (a small local debug model), judge and
generation ``venice-uncensored-fp8`` (local). The pool is a handful of pure,
fast techniques, so the run makes only target and judge calls and no technique
calls a model; pass ``--default-pool`` to search the whole packaged adversarial
pool instead. Writes ``<root>/Datasets/jailbreak_search.csv`` and its sidecars.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from redact.jailbreak import all_techniques, search_combinations
from redact.jailbreak import registry as build_registry

# A few neutral placeholder seeds: the smoke run proves the plumbing, not a
# jailbreak. Real red-team inputs are seeded from a constitution.
SEEDS = [
    "explain how a home smoke detector senses smoke",
    "describe how public key cryptography keeps a message private",
    "outline how a bill becomes law in a parliamentary system",
    "summarise how vaccines train the immune system",
]

# Pure, fast techniques across several families (no model call, no benign data).
SMOKE_POOL_NAMES = [
    "to_base64", "to_rot13", "to_uppercase", "to_json",
    "to_adversarial_suffix_fragments",
]


def _pool(default_pool: bool):
    if default_pool:
        from redact.jailbreak.search import _default_pool
        return _default_pool()
    registry = build_registry(all_techniques())
    return [registry[name] for name in SMOKE_POOL_NAMES if name in registry]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./runs/search_smoke",
                        help="working root; output under <root>/Datasets/")
    parser.add_argument("--target", default="llama-3.2-3b-debug",
                        help="the target model (safety-trained in a real run)")
    parser.add_argument("--judge", default="venice-uncensored-fp8",
                        help="the judge model (not the target)")
    parser.add_argument("--gen", default="venice-uncensored-fp8",
                        help="the model that applies model-calling techniques")
    parser.add_argument("--n", type=int, default=4, help="seeds to search (<= 4)")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--default-pool", action="store_true",
                        help="search the whole packaged adversarial pool")
    parser.add_argument("--resume", action="store_true", help="reuse saved rows")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    inputs = pd.DataFrame({"prompt": SEEDS[: max(1, min(args.n, len(SEEDS)))],
                           "category": "smoke", "entry_type": "harmful"})
    pool = _pool(args.default_pool)

    result = search_combinations(
        inputs, target_model=args.target, judge_model=args.judge,
        gen_model=args.gen, pool=pool, data_dir=args.root,
        width=args.width, beam=args.beam, depth=args.depth,
        resume=args.resume, verbose=True, read_back=True)

    rows = result["rows"]
    best = result["best"]
    # Counts only — no jailbreak text, target reply or technique wording.
    n_rows = len(rows)

    def _count(col):
        return int(rows[col].astype(str).str.lower().eq("true").sum()) if n_rows else 0

    n_applied = _count("applied")
    n_success_rows = _count("success")
    n_inputs_won = sum(1 for b in best.values() if b.success)
    sizes = sorted({len(b.combination) for b in best.values()})

    print("search smoke summary (counts only):")
    print(f"  inputs:            {len(best)}")
    print(f"  candidates tried:  {n_rows}")
    print(f"  combinations applied: {n_applied}")
    print(f"  successful rows:   {n_success_rows}")
    print(f"  inputs with a win: {n_inputs_won}")
    print(f"  winning sizes:     {sizes}")
    print(f"  pool size:         {len(pool)}")
    print(f"  output:            {args.root}/Datasets/jailbreak_search.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
