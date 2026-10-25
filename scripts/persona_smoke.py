"""Count what the persona and cognitive techniques do on a real model.

    python scripts/persona_smoke.py          # 20 sentences, the default model
    python scripts/persona_smoke.py -n 50 --model venice-uncensored
    python scripts/persona_smoke.py --families persona --inputs my_sentences.txt
    python scripts/persona_smoke.py --json counts.json    # also write the counts

Runs N benign sentences (the built-in ones, or one per line of ``--inputs``)
through each technique of the chosen families, with real clients, and prints
per technique how many results were accepted and how many were discarded,
by reason (``malformed scenario``, ``scenario check failed``,
``malformed persona``, ...), plus the mean number of model calls per unit.

It prints **counts only**: no generated text, scene, persona or check reply
is shown or saved. Run it before and after a change to the scenario,
persona-invention or construction prompts to see whether the scenario check
accepts the new scenes as often.
"""

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

#: Neutral requests, so a run is safe to read and to share.
BENIGN_SENTENCES = [
    "How do I bake a loaf of sourdough bread at home?",
    "What is the difference between a virus and a bacterium?",
    "Can you explain how a mortgage works?",
    "What should I pack for a three-day hiking trip?",
    "How do I change a flat tire on a bicycle?",
    "Why is the sky blue during the day and red at sunset?",
    "What are good ways to learn a new language as an adult?",
    "How does a bill become a law in a parliamentary system?",
    "How do I get red wine out of a cotton shirt?",
    "What is the best way to store fresh herbs?",
    "Can you describe how vaccines train the immune system?",
    "How do I write a polite email asking for a deadline extension?",
    "What causes the tides in the ocean?",
    "How do I start a vegetable garden on a small balcony?",
    "What is the plot of the novel Moby-Dick?",
    "How do solar panels turn sunlight into electricity?",
    "What are some tips for a first marathon?",
    "How do I tune a guitar by ear?",
    "How does compound interest grow savings over time?",
    "What is a good routine for improving sleep?",
]

DEFAULT_FAMILIES = ("persona", "cognitive")

_REASON = re.compile(r"reason=([^;]+)")


def select_techniques(
    techniques: Sequence[Callable], families: Sequence[str],
    names: Sequence[str] | None = None,
) -> list[Callable]:
    """The active ``techniques`` of any of ``families``, narrowed to ``names``
    when given. Raises ``ValueError`` for a name no technique of them has."""
    chosen = [fn for fn in techniques
              if set(getattr(fn, "families", [])) & set(families)
              and getattr(fn, "deprecated", None) is None]
    if names:
        known = {fn.__name__ for fn in chosen}
        missing = sorted(set(names) - known)
        if missing:
            raise ValueError(
                f"No such technique in {sorted(families)}: {', '.join(missing)}.")
        chosen = [fn for fn in chosen if fn.__name__ in names]
    return chosen


def reason_of(info: str) -> str:
    """The ``reason=`` of a discard's info, ``unspecified`` when it has none."""
    match = _REASON.search(info)
    return match.group(1).strip() if match else "unspecified"


def run_smoke(
    techniques: Sequence[Callable],
    sentences: Sequence[str],
    call: Callable,
    *,
    gen_model: str,
    check_model: str | None = None,
    seed: int = 42,
    prompt_dir=None,
) -> dict[str, dict]:
    """Run each sentence through each technique, a unit at a time.

    ``call`` answers an ``LLMRequest`` with the reply text (as
    ``drive_sync`` takes it). Returns, per technique name, ``units``,
    ``accepted``, ``noop`` (accepted with the text unchanged), ``discarded``
    (a ``{reason: count}``), ``errors`` (a ``{exception class: count}``, a unit
    that raised) and ``calls`` (model calls in all). Only counts: no text.
    """
    from redact.jailbreak.chain import run_chain
    from redact.jailbreak.technique import is_discarded
    from redact.llm_pipeline import drive_sync

    results: dict[str, dict] = {}
    for fn in techniques:
        row = results[fn.__name__] = {
            "units": 0, "accepted": 0, "noop": 0, "discarded": Counter(),
            "errors": Counter(), "calls": 0}
        for index, sentence in enumerate(sentences):
            row["units"] += 1
            made = [0]

            def counted(request, made=made):
                made[0] += 1
                return call(request)

            try:
                text, info = drive_sync(
                    run_chain([fn], sentence, gen_model=gen_model,
                              check_model=check_model, prompt_dir=prompt_dir,
                              rng_seed=(seed, "persona_smoke", index)),
                    counted)
            except Exception as exc:  # a count, not a stop: one unit's failure
                row["errors"][type(exc).__name__] += 1
                row["calls"] += made[0]
                continue
            row["calls"] += made[0]
            if is_discarded(info):
                row["discarded"][reason_of(info)] += 1
            else:
                row["accepted"] += 1
                row["noop"] += text == sentence
    return results


def make_call(default_model: str) -> Callable:
    """A blocking call for :func:`run_smoke`: each request goes to a client of
    its model (made once), with its reply limit and temperature."""
    from redact.llms.client import ModelClient
    from redact.llms.router import generate_sample

    clients: dict[str, ModelClient] = {}

    def call(request) -> str:
        model = request.model or default_model
        if request.client is not None:
            client = request.client
        else:
            client = clients.get(model)
            if client is None:
                client = clients[model] = ModelClient.create(model)
        return generate_sample(client, request.messages,
                               max_tokens=request.max_tokens,
                               temperature=request.temperature)

    return call


def render(results: dict[str, dict]) -> str:
    """The counts as a table, then the totals; counts only."""
    width = max([len(name) for name in results] + [9])
    lines = [f"{'technique':<{width}}  units  accepted  discarded  errors  calls/unit"]
    totals = Counter()
    reasons: Counter = Counter()
    for name, row in results.items():
        discarded = sum(row["discarded"].values())
        errors = sum(row["errors"].values())
        calls = row["calls"] / row["units"] if row["units"] else 0.0
        lines.append(f"{name:<{width}}  {row['units']:>5}  {row['accepted']:>8}  "
                     f"{discarded:>9}  {errors:>6}  {calls:>10.2f}")
        for key in ("units", "accepted", "noop", "calls"):
            totals[key] += row[key]
        totals["discarded"] += discarded
        totals["errors"] += errors
        reasons.update(row["discarded"])
        reasons.update({f"error: {k}": v for k, v in row["errors"].items()})
    lines.append("")
    lines.append(
        f"total: {totals['units']} units, {totals['accepted']} accepted "
        f"({totals['noop']} with the text unchanged), {totals['discarded']} "
        f"discarded, {totals['errors']} errors, {totals['calls']} model calls")
    for reason, count in reasons.most_common():
        lines.append(f"  {count:>5}  {reason}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Count the accepted and discarded results of the persona "
                    "and cognitive techniques on benign sentences.")
    parser.add_argument("-n", "--sentences", type=int, default=20,
                        help="How many sentences to run (default 20).")
    parser.add_argument("--inputs", default=None,
                        help="A text file, one sentence per line "
                             "(default: the built-in ones).")
    parser.add_argument("--model", default=None,
                        help="The generation model "
                             "(default: the uncensored_gen role's).")
    parser.add_argument("--check-model", default=None,
                        help="The model that checks the scenarios "
                             "(default: the generation model).")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                        help="Comma-separated families (default: persona,cognitive).")
    parser.add_argument("--techniques", default=None,
                        help="Comma-separated technique names to keep "
                             "(default: all of the families).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-dir", default=None,
                        help="A prompt overlay folder (default: the packaged prompts).")
    parser.add_argument("--json", default=None, metavar="OUT",
                        help="Also write the counts to this JSON file.")
    args = parser.parse_args(argv)

    if args.sentences < 1:
        parser.error("-n must be at least 1.")
    if args.inputs:
        sentences = [line.strip() for line in
                     Path(args.inputs).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    else:
        sentences = BENIGN_SENTENCES
    if not sentences:
        parser.error("--inputs holds no sentence.")
    # Repeat a short list so -n sentences are run.
    sentences = [sentences[i % len(sentences)] for i in range(args.sentences)]

    from redact.jailbreak import all_techniques
    from redact.llms.model_config import default_model_for_role

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    names = ([n.strip() for n in args.techniques.split(",")]
             if args.techniques else None)
    try:
        techniques = select_techniques(all_techniques(), families, names)
    except ValueError as exc:
        parser.error(str(exc))
    if not techniques:
        parser.error(f"No technique in the families {families}.")

    model = args.model or default_model_for_role("uncensored_gen")
    print(f"{len(techniques)} techniques x {len(sentences)} sentences, "
          f"generation model {model}, check model {args.check_model or model}",
          file=sys.stderr)
    results = run_smoke(
        techniques, sentences, make_call(model), gen_model=model,
        check_model=args.check_model, seed=args.seed, prompt_dir=args.prompt_dir)
    print(render(results))
    if args.json:
        Path(args.json).write_text(
            json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
