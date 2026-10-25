"""A beam search over single-turn technique combinations against a target.

Not a recipe stage: call :func:`search_combinations` from Python, as with
``multi_turn``. Given harmful inputs, a safety-trained **target** model, a
**judge** model with its criterion, and a pool of techniques (the same
:func:`~redact.jailbreak.pool.technique_pool` the stage draws from), it
searches per input for a combination whose jailbreak text the judge says the
target complied with.

Each **round** scores a set of candidate combinations per input. Round 0 draws
``width`` valid combinations with the library's own sampler
(:func:`~redact.jailbreak.sampling.sample_combination`, so every draw is
compatible under the spec's rules). Each later round keeps the ``beam`` best
candidates of the round before and **mutates** them — add one compatible
technique, swap one for another of its family, or drop one — each mutation
checked with :func:`~redact.jailbreak.rules.compatible`, so no round ever holds
an invalid pair. An input stops early at the first round a candidate succeeds.

The beam takes only candidates that applied (never the identity, nor one a
technique discarded or an error ended), by score, then fewest techniques, then
a seeded draw. Scores are 0 or 1, so until something succeeds the beam is a
seeded pick among the smallest combinations that applied: the search is
breadth-first over small combinations. Every parent has hundreds of mutations
over a full pool (with the defaults and the packaged adversarial pool, an
uncapped search tries about 900 candidates per input), so a round tries at most
``max_candidates`` per input (8 by default), drawn with a seeded rng: each pick
takes the beam's parents in turn, a kind of mutation (add, swap, drop) at random
among those the parent still has, and a mutation of that kind at random. With a
budget round 0 (``sample_kwargs`` without ``exact_techniques``), every round
keeps its budget: a mutation over ``max_complexity`` or with more than
``max_obfuscations`` obfuscations is not tried, and a layer round 0 leaves out
is left out of every round.

A candidate is scored by applying it (:func:`~redact.jailbreak.chain.run_chain`,
model-calling techniques included, driven on
:func:`~redact.llm_pipeline.drive_units`), sending the result to the target, and
asking the judge. Every tried candidate is one row of ``jailbreak_search.csv``;
the rows are the tree, and the search follows the persistence rules (``Manifest``,
``Ledger``, ``resume_state``, ``CommitBuffer``), so a run killed mid-search
resumes to the same rows with no duplicates. It is deterministic given the
inputs, the seed, the pool and the params: round 0's draws are seeded per
``(seed, input_id, i)``, and a round's plan follows from the saved rows of the
rounds before it alone, so a resume re-derives the same plan, whatever a killed
run saved of that round or later ones, and runs only the units the ledger lacks.
"""

from __future__ import annotations

import inspect
import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from redact import paths
from redact.dataset import (
    CommitBuffer,
    Ledger,
    append_rows,
    content_id,
    read_rows,
    release_lock,
    resume_state,
    row_ids,
)
from redact.dataset.dealing import stable_seed
from redact.dataset.planfile import changed_plan, plan_path, write_plan
from redact.dataset.sidecar import Manifest
from redact.exceptions import PromptTooLongError
from redact.llm_pipeline import LLMRequest, drive_units, is_accepted, pool_size
from redact.llms.model_config import default_model_for_role
from redact.llms.observe import current_stage, set_stage
from redact.llms.prompting import PromptTemplate, load_prompt

from .chain import in_layer_order, run_chain
from .plan import _applicable, combination_of, rules_id
from .pool import all_techniques, technique_pool
from .pool import registry as build_registry
from .rules import compatible
from .sampling import sample_combination
from .technique import is_discarded, load_spec

logger = logging.getLogger(__name__)

#: The output file, under ``Datasets/``.
SEARCH_FILENAME = "jailbreak_search.csv"
#: The telemetry stage label.
STAGE = "jailbreak_search"
#: The identity combination (no technique).
IDENTITY = "identity"
#: The kinds of one-step mutation, in the order a parent's are listed.
MUTATION_KINDS = ("add", "swap", "drop")
#: The default cap of candidates per input in each mutation round.
DEFAULT_MAX_CANDIDATES = 8

#: The columns of every ``jailbreak_search.csv`` row, in order.
SEARCH_COLUMNS = (
    "sample_id", "input_id", "round", "candidate", "parent", "technique",
    "num_techniques", "complexity", "applied", "is_noop", "jailbreak",
    "technique_info", "target_model", "judge_model", "target_response",
    "success", "score", "judge_reasoning", "rules", "combination_spec_version",
    "category", "entry_type", "prompt",
)


@dataclass(frozen=True)
class BestCombo:
    """The best combination found for one input: ``input_id``, ``combination``
    (technique names in applied order; empty when nothing beat the bare
    prompt), ``score`` (1.0 for a success, else 0.0), ``success`` and the
    ``round`` the best candidate was tried in."""

    input_id: str
    combination: list[str]
    score: float
    success: bool
    round: int


@dataclass
class _Config:
    """What every candidate unit of a run shares."""

    registry: dict[str, Callable]
    gen_model: str
    translation_model: str | None
    check_model: str | None
    target_model: str
    judge_model: str
    judge_messages: PromptTemplate
    target_system: str | None
    benign_data: dict | None
    prompt_dir: object
    seed: int | None
    rules: str
    spec_version: str


# --- combination ids ------------------------------------------------------

def _combo_id(names: list[str]) -> str:
    """A combination stored as its names joined by ``+``, or :data:`IDENTITY`."""
    return "+".join(names) or IDENTITY


def _names_of(candidate: str) -> list[str]:
    """The names a candidate id holds (:func:`_combo_id` reversed)."""
    return [] if candidate == IDENTITY else candidate.split("+")


def _ordered(fns: list[Callable]) -> list[str]:
    """``fns`` names in applied order (:func:`~redact.jailbreak.chain.in_layer_order`).
    """
    return [fn.__name__ for fn in in_layer_order(fns)]


def _is_valid(candidate: str, registry: dict[str, Callable]) -> bool:
    """Whether every technique of ``candidate`` may sit with those before it
    (:func:`~redact.jailbreak.rules.compatible`, which binds both ways).
    Unknown names are invalid."""
    names = _names_of(candidate)
    if any(name not in registry for name in names):
        return False
    fns = [registry[name] for name in names]
    return all(compatible(fns[:i], [fn]) for i, fn in enumerate(fns))


# --- drawing and mutating --------------------------------------------------

def _draw_round0(
    pool: list[Callable], input_id: str, text: str, entry_type: str,
    *, seed: int, width: int, sample_kwargs: dict,
) -> list[str]:
    """``width`` distinct valid combinations for one input's first round.

    Each is drawn by :func:`~redact.jailbreak.sampling.sample_combination` from
    the techniques that apply to the input, seeded per ``(seed, input_id, i)``
    so a resume draws the same set."""
    applicable = _applicable(pool, text, entry_type, input_id)
    seen: set[str] = set()
    out: list[str] = []
    for i in range(width * 8):
        if len(out) >= width:
            break
        rng = random.Random(stable_seed(seed, input_id, "search0", i))
        combined = sample_combination(rng, applicable, **sample_kwargs)
        cid = _combo_id(_ordered(list(getattr(combined, "techniques", []))))
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


#: The sampler's own defaults, which a round-0 ``sample_kwargs`` overlays.
_SAMPLER_DEFAULTS = {name: p.default for name, p in
                     inspect.signature(sample_combination).parameters.items()
                     if p.default is not inspect.Parameter.empty}


@dataclass(frozen=True)
class _Budget:
    """What round 0 may draw, which every later round keeps: the complexity
    and the obfuscations a budget draw may spend (``None``: no cap, as in a
    count draw) and the layers round 0 leaves out."""

    max_complexity: int | None
    max_obfuscations: int | None
    layers_off: frozenset[str]

    def fits(self, fns: list[Callable]) -> bool:
        """Whether a combination of ``fns`` stays within the budget."""
        if any(getattr(fn, "layer", None) in self.layers_off for fn in fns):
            return False
        spent = sum(getattr(fn, "complexity", 0) for fn in fns)
        if self.max_complexity is not None and spent > self.max_complexity:
            return False
        return (self.max_obfuscations is None
                or sum(getattr(fn, "layer", None) == "obfuscation" for fn in fns)
                <= self.max_obfuscations)


def _budget_of(sample_kwargs: dict) -> _Budget:
    """The budget round 0's ``sample_kwargs`` set, the sampler's defaults under
    them. A count draw (``exact_techniques``) has no complexity or obfuscation
    cap and draws the rewrite layer when the pool holds it; a budget draw caps
    both and leaves the rewrite layer out unless ``include_rewrite``. Either
    leaves out a layer whose ``include_<layer>`` is false."""
    kw = {**_SAMPLER_DEFAULTS, **sample_kwargs}
    off = {layer for layer in ("hacking", "manipulation", "obfuscation", "requests")
           if not kw.get(f"include_{layer}", True)}
    if kw.get("exact_techniques") is not None:
        return _Budget(None, None, frozenset(off))
    if not kw.get("include_rewrite", False):
        off.add("rewrite")
    return _Budget(kw["max_complexity"], kw["max_obfuscations"], frozenset(off))


def _mutations_by_kind(
    parent: str, pool: list[Callable], registry: dict[str, Callable],
    text: str, entry_type: str, input_id: str, budget: _Budget | None = None,
) -> dict[str, list[str]]:
    """Every valid one-step mutation of ``parent``, by kind
    (:data:`MUTATION_KINDS`), each a sorted list of combination ids.

    ``add`` each technique that may still join
    (:func:`~redact.jailbreak.rules.compatible`); ``swap`` each for another of
    one of its families that is compatible with the rest; ``drop`` each
    technique. Every result is valid by construction and different from
    ``parent``; with a ``budget``, only those that fit it are kept."""
    applicable = _applicable(pool, text, entry_type, input_id)
    fns = [registry[n] for n in _names_of(parent) if n in registry]
    out: dict[str, dict[str, list[Callable]]] = {kind: {} for kind in MUTATION_KINDS}

    def keep(kind: str, combo: list[Callable]) -> None:
        out[kind][_combo_id(_ordered(combo))] = combo

    for cand in compatible(fns, applicable):
        if cand not in fns:
            keep("add", fns + [cand])
    for i, fn in enumerate(fns):
        rest = fns[:i] + fns[i + 1:]
        fam = set(getattr(fn, "families", ()))
        for cand in compatible(rest, applicable):
            if cand is not fn and fam & set(getattr(cand, "families", ())):
                keep("swap", rest + [cand])
    for i in range(len(fns)):
        keep("drop", fns[:i] + fns[i + 1:])
    return {kind: sorted(cid for cid, combo in found.items()
                         if cid != parent and (budget is None or budget.fits(combo)))
            for kind, found in out.items()}


def _mutations(
    parent: str, pool: list[Callable], registry: dict[str, Callable],
    text: str, entry_type: str, input_id: str, budget: _Budget | None = None,
) -> list[str]:
    """Every valid one-step mutation of ``parent`` (:func:`_mutations_by_kind`,
    all kinds), as sorted combination ids, distinct."""
    kinds = _mutations_by_kind(parent, pool, registry, text, entry_type, input_id,
                               budget)
    return sorted(set().union(*kinds.values()))


def _pick_mutations(
    parents: list[tuple[str, dict[str, list[str]]]], tried: set[str],
    cap: int | None, rng: random.Random,
) -> dict[str, str]:
    """The candidates of one input's round, ``{candidate: parent}``.

    ``parents`` is the beam in order, each paired with its mutations by kind.
    Without a ``cap`` every mutation not in ``tried`` is kept (sorted per kind,
    kinds in :data:`MUTATION_KINDS` order, parents in beam order). With a
    ``cap`` at most that many are drawn with ``rng``: each draw picks the next
    parent in rotation, a kind at random among those the parent still has
    untried, and a mutation of that kind at random, so the parents and the
    kinds are spread evenly and the pick is deterministic given the seed."""
    if cap is None:
        chosen: dict[str, str] = {}
        for parent, by_kind in parents:
            for kind in MUTATION_KINDS:
                for cand in by_kind.get(kind, []):
                    if cand not in tried and cand not in chosen:
                        chosen[cand] = parent
        return chosen

    # A pool of (parent, kind, [untried candidates]) the draw thins as it goes.
    buckets: list[list] = []
    for parent, by_kind in parents:
        for kind in MUTATION_KINDS:
            left = [c for c in by_kind.get(kind, []) if c not in tried]
            if left:
                buckets.append([parent, kind, left])
    chosen = {}
    i = 0
    while buckets and len(chosen) < cap:
        bucket = buckets[i % len(buckets)]
        parent, _kind, left = bucket
        cand = left.pop(rng.randrange(len(left)))
        if cand not in tried and cand not in chosen:
            chosen[cand] = parent
        if left:
            i += 1
        else:
            buckets.pop(i % len(buckets))
    return chosen


# --- rows ------------------------------------------------------------------

def _row(
    input_row: dict, round_idx: int, candidate: str, parent: str,
    fns: list[Callable], text: str, info: str, target_response: str,
    success: bool, reasoning: str, *, cfg: _Config,
) -> dict:
    """One ``jailbreak_search.csv`` row for a tried candidate. ``sample_id`` is
    the content id of ``search:{input_id}:{round}:{candidate}``: unique per
    unit and stable across a resume, so a re-run never writes it twice."""
    input_id = str(input_row["sample_id"])
    prompt = str(input_row.get("prompt", ""))
    return {
        "sample_id": content_id(f"search:{input_id}:{round_idx}:{candidate}"),
        "input_id": input_id,
        "round": round_idx,
        "candidate": candidate,
        "parent": parent,
        "technique": candidate,
        "num_techniques": len(fns),
        "complexity": sum(getattr(fn, "complexity", 0) for fn in fns),
        "applied": not is_discarded(info),
        "is_noop": text == prompt,
        "jailbreak": text,
        "technique_info": info,
        "target_model": cfg.target_model,
        "judge_model": cfg.judge_model,
        "target_response": target_response,
        "success": bool(success),
        "score": 1.0 if success else 0.0,
        "judge_reasoning": reasoning,
        "rules": cfg.rules,
        "combination_spec_version": cfg.spec_version,
        "category": str(input_row.get("category", "")),
        "entry_type": str(input_row.get("entry_type", "")),
        "prompt": prompt,
    }


def _typed(row: dict) -> dict:
    """A freshly built row with the fields the beam helpers read typed."""
    row = dict(row)
    row["input_id"] = str(row["input_id"])
    row["round"] = int(row["round"])
    row["num_techniques"] = int(row["num_techniques"])
    row["success"] = bool(row["success"])
    row["applied"] = bool(row.get("applied", True))
    row["score"] = float(row["score"])
    return row


def _rows_of(frame: pd.DataFrame) -> list[dict]:
    """Saved rows as the typed dicts the helpers read."""
    out = []
    for row in frame.to_dict("records"):
        row = dict(row)
        row["input_id"] = str(row.get("input_id", ""))
        row["candidate"] = str(row.get("candidate", IDENTITY))
        row["round"] = int(row.get("round", 0))
        row["num_techniques"] = int(row.get("num_techniques", 0) or 0)
        row["success"] = str(row.get("success", "")).strip().lower() == "true"
        row["applied"] = str(row.get("applied", "True")).strip().lower() == "true"
        row["score"] = 1.0 if row["success"] else 0.0
        out.append(row)
    return out


# --- the candidate unit ----------------------------------------------------

def _target_messages(text: str, system: str | None) -> list[dict]:
    """The target's chat messages: one user turn, with an optional system turn."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": text})
    return messages


def _candidate_unit(
    input_row: dict, round_idx: int, candidate: str, parent: str, cfg: _Config,
) -> Iterator:
    """The generator of one candidate: apply, target, judge, then a row.

    It yields the combination's model calls (through ``run_chain``), then the
    target call, then the judge call. A :class:`PromptTooLongError` at any step
    (raised inside the unit on the async lane) is recorded as a failed row, so
    one oversized prompt never stops the search."""
    input_id = str(input_row["sample_id"])
    prompt = str(input_row.get("prompt", ""))
    fns = list(getattr(combination_of(_names_of(candidate), cfg.registry),
                        "techniques", []))
    rng_seed = None if cfg.seed is None else (cfg.seed, "search", input_id, round_idx)

    def make(text, info, reply, success, reasoning):
        return _row(input_row, round_idx, candidate, parent, fns, text, info,
                    reply, success, reasoning, cfg=cfg)

    try:
        text, info = yield from run_chain(
            fns, prompt, gen_model=cfg.gen_model,
            translate_model=cfg.translation_model, check_model=cfg.check_model,
            benign_data=cfg.benign_data, prompt_dir=cfg.prompt_dir, rng_seed=rng_seed)
    except PromptTooLongError as exc:
        return make(prompt, f"DISCARDED; reason=prompt too long: {exc}", "", False,
                    f"prompt too long: {exc}")
    if is_discarded(info):
        return make(text, info, "", False, info)
    try:
        reply = yield LLMRequest(
            cfg.target_model, _target_messages(text, cfg.target_system))
    except PromptTooLongError as exc:
        return make(text, info, "", False, f"target prompt too long: {exc}")
    try:
        verdict = yield LLMRequest(
            cfg.judge_model,
            cfg.judge_messages(goal=prompt, label="RESPONSE", sample=reply))
    except PromptTooLongError as exc:
        return make(text, info, reply, False, f"judge prompt too long: {exc}")
    success = is_accepted(verdict)
    return make(text, info, reply, success, "" if success else verdict)


# --- beam, tree, best ------------------------------------------------------

def _beam(rows: list[dict], input_id: str, round_idx: int, beam: int,
          *, seed: int = 0) -> list[str]:
    """The ``beam`` best parents of ``input_id`` in ``round_idx`` to mutate.

    Only candidates that applied are eligible — never the identity, nor one a
    technique discarded or an error ended, which are not parents worth growing.
    Ordered by score (high first), then fewest techniques, then a seeded draw
    per candidate, so with 0/1 scores (nothing succeeded yet) the beam is a
    seeded pick among the smallest applied combinations, not an alphabetical
    one."""
    frontier = [r for r in rows
                if r["input_id"] == input_id and r["round"] == round_idx
                and r.get("candidate") != IDENTITY and r.get("applied", True)]
    frontier.sort(key=lambda r: (
        -r["score"], r["num_techniques"],
        stable_seed(seed, input_id, round_idx, r["candidate"])))
    return [r["candidate"] for r in frontier[:beam]]


def _succeeded(rows: list[dict], input_id: str) -> bool:
    """Whether any tried candidate of ``input_id`` was a success."""
    return any(r["input_id"] == input_id and r["success"] for r in rows)


def _tried(rows: list[dict], input_id: str) -> set[str]:
    """The candidate ids already tried for ``input_id``, any round."""
    return {r["candidate"] for r in rows if r["input_id"] == input_id}


def best_combinations(rows: list[dict] | pd.DataFrame) -> dict[str, BestCombo]:
    """The best candidate per input: highest score, then fewest techniques.
    ``rows`` is the list :func:`search_combinations` returns, or a frame read
    from ``jailbreak_search.csv``."""
    if isinstance(rows, pd.DataFrame):
        rows = _rows_of(rows)
    best: dict[str, BestCombo] = {}
    for row in sorted(rows, key=lambda r: (-r["score"], r["num_techniques"],
                                           r["round"], r["candidate"])):
        iid = row["input_id"]
        if iid not in best:
            best[iid] = BestCombo(iid, _names_of(row["candidate"]), row["score"],
                                  row["success"], row["round"])
    return best


def search_tree(rows: list[dict] | pd.DataFrame, input_id: str) -> list[dict]:
    """Every candidate tried for ``input_id``, round by round: ``round``,
    ``candidate``, ``parent`` (the node it was mutated from, ``""`` in round
    0), ``score``, ``success`` and ``technique_info``."""
    if isinstance(rows, pd.DataFrame):
        rows = _rows_of(rows)
    nodes = sorted((r for r in rows if r["input_id"] == input_id),
                   key=lambda r: (r["round"], r["candidate"]))
    return [{"round": r["round"], "candidate": r["candidate"],
             "parent": r.get("parent", ""), "score": r["score"],
             "success": r["success"], "technique_info": r.get("technique_info", "")}
            for r in nodes]


# --- setup helpers ---------------------------------------------------------

def _default_judge_system(prompt_dir) -> str:
    """The attack-success criterion (``prompts/conversation/success_judge``),
    as :data:`redact.multiturn_attacks.SUCCESS_JUDGE_SYSTEM`."""
    return load_prompt("conversation", "success_judge", prompt_dir)["system_prompt"]


def _judge_template(judge_system: str, prompt_dir) -> PromptTemplate:
    """The judge prompt, called with ``goal``, ``label`` and ``sample``."""
    prompt = PromptTemplate.load("conversation", "judge", prompt_dir=prompt_dir)
    prompt.system_prompt = judge_system
    return prompt


def _default_pool() -> list[Callable]:
    """The packaged adversarial kind's pool."""
    from .augmentation import defaults, resolve  # lazy: augmentation needs plan/pool
    return technique_pool(resolve(defaults(), "adversarial"))


def _prepare_inputs(inputs: pd.DataFrame) -> pd.DataFrame:
    """``inputs`` with a ``prompt`` column and one ``sample_id`` per row (a
    ``sample``/``text``/``seed`` column is read as ``prompt``; rows that share
    an id are kept once)."""
    inputs = inputs.reset_index(drop=True)
    if "prompt" not in inputs.columns:
        for alt in ("sample", "text", "seed"):
            if alt in inputs.columns:
                inputs = inputs.assign(prompt=inputs[alt])
                break
        else:
            raise ValueError(
                "inputs needs a 'prompt' (or 'sample'/'text'/'seed') column.")
    inputs = inputs.assign(sample_id=row_ids(inputs, "prompt"))
    return inputs.drop_duplicates("sample_id", keep="last").reset_index(drop=True)


def _resolve_models(resolve, *names) -> list:
    """Resolve the distinct model names for :func:`~redact.llm_pipeline.pool_size`."""
    clients = {}
    for name in names:
        if name and name not in clients:
            clients[name] = resolve(name)
    return list(clients.values())


def _check_plan(manifest, resume, current, ledger) -> None:
    """Write the plan file on a fresh run; on a resume, raise when it differs.
    The tree depends on the seed, the pool and the round-0 sampler, so a resume
    with any of them changed raises rather than mix two trees in one file."""
    if not resume or not manifest.path.exists():
        write_plan(manifest.path, current)
        return
    changed = changed_plan(manifest.path, current)
    if changed:
        release_lock(ledger)
        raise ValueError(
            f"search_combinations: this run's {changed} differ from the plan "
            f"{plan_path(manifest.path).name} the saved search was made with; a "
            "resume keeps the plan. Run with resume=False to search again (the "
            "output is cleared), or with the same settings.")


# --- the search ------------------------------------------------------------

def search_combinations(
    inputs: pd.DataFrame,
    target_model: str,
    judge_model: str,
    *,
    pool: list[Callable] | None = None,
    data_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    judge_system: str | None = None,
    gen_model: str | None = None,
    translation_model: str | None = None,
    check_model: str | None = None,
    target_system: str | None = None,
    benign_data: dict | None = None,
    width: int = 4,
    beam: int = 2,
    depth: int = 2,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    seed: int = 42,
    sample_kwargs: dict | None = None,
    stop_on_success: bool = True,
    resume: bool = True,
    batch_size: int = 64,
    prompt_dir: str | Path | None = None,
    resolve: Callable[[str], object] | None = None,
    verbose: bool = True,
    read_back: bool = True,
) -> dict:
    """Search per input for a technique combination the judge calls a success.

    Args:
        inputs: Harmful inputs, with a ``prompt`` (or ``sample``/``text``)
            column; ``sample_id``, ``category`` and ``entry_type`` are used when
            present.
        target_model: The safety-trained model under attack, required.
        judge_model: The scoring model, required; not the target.
        pool: The techniques to draw from (as
            :func:`~redact.jailbreak.pool.technique_pool`). ``None`` is the
            packaged adversarial pool. Requires-benign techniques (fsh/dap) are
            dropped unless ``benign_data`` is given.
        data_dir: The working root; the output is
            ``Datasets/jailbreak_search.csv`` under it. ``out_path`` overrides.
        judge_system: The judge's criterion. ``None`` is the attack-success
            criterion (``SUCCESS_JUDGE_SYSTEM``).
        gen_model: The model that applies model-calling techniques. ``None`` is
            the ``uncensored_gen`` role default.
        translation_model: The machine-translation model, for a pool with a
            translation technique.
        check_model: The model that checks a technique's model output. ``None``
            is ``gen_model``.
        target_system: The target's system prompt.
        benign_data: Benign Q&A pairs for fsh/dap; ``None`` drops those.
        width: Candidates drawn per input in round 0.
        beam: Candidates kept per input to mutate into the next round.
        depth: Mutation rounds after round 0 (``depth + 1`` rounds in all).
        max_candidates: Most candidates tried per input in each mutation round;
            over a full pool a parent has hundreds of one-step mutations, so the
            cap (8 by default) draws a seeded sample spread over the beam's
            parents and the add/swap/drop kinds. ``None`` tries every valid
            mutation (the old, unbounded behaviour).
        seed: Seeds the sampling, so a resume is identical.
        sample_kwargs: Passed to
            :func:`~redact.jailbreak.sampling.sample_combination` for round 0.
            Defaults to ``{"exact_techniques": 1}``.
        stop_on_success: Stop expanding an input once a candidate succeeds.
        resume: Reuse the saved rows and run only the unfinished candidates.
        batch_size: Candidate units per model call.
        prompt_dir: The overlay the judge and the techniques read through.
        resolve: ``model name -> client``; defaults to ``ModelClient.create``.
        verbose: Log progress.
        read_back: Return the saved rows; ``False`` returns an empty frame.

    Returns:
        ``{"best": {input_id: BestCombo}, "rows": DataFrame}``: the best
        combination per input, and every tried candidate (the tree).
    """
    if not target_model:
        raise ValueError("search_combinations requires a target_model (no default).")
    if not judge_model:
        raise ValueError("search_combinations requires a judge_model (no default).")
    if width < 1 or beam < 1 or depth < 0:
        raise ValueError("width and beam must be >= 1, depth >= 0.")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be >= 1 or None (no cap).")

    from redact.llms.client import ModelClient
    resolve = resolve or ModelClient.create

    inputs = _prepare_inputs(inputs)
    pool = pool if pool is not None else _default_pool()
    if benign_data is None:
        pool = [fn for fn in pool if not getattr(fn, "requires_benign", False)]
    if not pool:
        raise ValueError("The pool is empty; nothing to search over.")

    gen_model = gen_model or default_model_for_role("uncensored_gen")
    check_model = check_model or gen_model
    judge_system = (judge_system if judge_system is not None
                    else _default_judge_system(prompt_dir))
    sample_kwargs = dict(sample_kwargs or {"exact_techniques": 1})

    out = Path(out_path) if out_path else paths.datasets(data_dir) / SEARCH_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.sidecar(out)
    rules = rules_id(pool, rounds=[sample_kwargs])

    ledger = Ledger.sidecar(
        out, key_fields=("input_id", "round", "candidate"),
        casters={"input_id": str, "round": int, "candidate": str})
    done = resume_state(ledger, resume=resume, artifact=out, manifest=manifest)
    _check_plan(manifest, resume, {
        "seed": seed, "width": width, "beam": beam, "depth": depth,
        "max_candidates": max_candidates, "target_model": target_model,
        "judge_model": judge_model, "judge_system": judge_system,
        "target_system": target_system, "stop_on_success": stop_on_success,
        "pool": sorted(fn.__name__ for fn in pool), "rules": rules,
        "sample_kwargs": sample_kwargs}, ledger)

    cfg = _Config(
        registry=build_registry(all_techniques()), gen_model=gen_model,
        translation_model=translation_model, check_model=check_model,
        target_model=target_model, judge_model=judge_model,
        judge_messages=_judge_template(judge_system, prompt_dir),
        target_system=target_system, benign_data=benign_data, prompt_dir=prompt_dir,
        seed=seed, rules=rules, spec_version=load_spec().get("version", ""))
    clients = _resolve_models(resolve, gen_model, translation_model, check_model,
                              target_model, judge_model)

    rows = _rows_of(read_rows(out)) if resume else []
    manifest_rows: list[dict] = []
    written = [0]

    def _append(batch: list[dict]) -> dict[Path, int]:
        written[0] += len(batch)
        return append_rows(out, [{c: r.get(c, "") for c in SEARCH_COLUMNS}
                                 for r in batch])

    previous = current_stage()
    set_stage(STAGE)
    try:
        with CommitBuffer(append=_append, ledger=ledger) as buf:
            _run_rounds(inputs, pool, cfg, done, rows, manifest, manifest_rows, buf,
                        width=width, beam=beam, depth=depth, seed=seed,
                        max_candidates=max_candidates,
                        sample_kwargs=sample_kwargs, stop_on_success=stop_on_success,
                        batch_size=batch_size, clients=clients, resolve=resolve,
                        verbose=verbose)
    finally:
        set_stage(previous)

    best = best_combinations(rows)
    if verbose:
        n_success = sum(1 for b in best.values() if b.success)
        logger.info("Search done: %d inputs, %d with a success, %d candidates, "
                    "%d newly scored.", len(best), n_success, len(rows), written[0])
    return {"best": best, "rows": read_rows(out) if read_back else pd.DataFrame()}


def _run_rounds(
    inputs, pool, cfg, done, rows, manifest, manifest_rows, buf, *,
    width, beam, depth, seed, max_candidates, sample_kwargs, stop_on_success,
    batch_size, clients, resolve, verbose,
) -> None:
    """Run every round in turn, committing each before the next's beam is read."""
    input_rows = {str(r["sample_id"]): r for r in inputs.to_dict("records")}
    budget = _budget_of(sample_kwargs)
    live: dict = {}

    def on_done(key, row):
        rows.append(_typed(row))
        iid, r_idx, cand = key
        buf.add([row], {"input_id": iid, "round": r_idx, "candidate": cand})
        live.pop(key, None)

    def on_error(key, exc):
        iid, r_idx, cand = key
        input_row, parent = live.pop(key)
        fns = list(getattr(combination_of(_names_of(cand), cfg.registry),
                           "techniques", []))
        row = _row(input_row, r_idx, cand, parent, fns,
                   str(input_row.get("prompt", "")),
                   f"DISCARDED; reason=ERROR: {exc}", "", False, str(exc), cfg=cfg)
        rows.append(_typed(row))
        buf.add([row], {"input_id": iid, "round": r_idx, "candidate": cand})

    for r_idx in range(depth + 1):
        plan = _plan_round(r_idx, input_rows, pool, cfg.registry, rows, beam=beam,
                           width=width, seed=seed, sample_kwargs=sample_kwargs,
                           stop_on_success=stop_on_success,
                           max_candidates=max_candidates, budget=budget)
        for iid, parent, cand in plan:
            manifest_rows.append({
                "sample_id": content_id(f"search:{iid}:{r_idx}:{cand}"),
                "input_id": iid, "round": r_idx, "candidate": cand,
                "parent": parent, "status": "planned"})
        manifest.write(list(manifest_rows))

        pending = [(iid, parent, cand) for iid, parent, cand in plan
                   if (iid, r_idx, cand) not in done]
        if verbose:
            logger.info("Round %d/%d: %d candidates, %d pending.",
                        r_idx, depth, len(plan), len(pending))
        if pending:
            def _units(pending=pending, r_idx=r_idx):
                for iid, parent, cand in pending:
                    key = (iid, r_idx, cand)
                    live[key] = (input_rows[iid], parent)
                    yield key, _candidate_unit(
                        input_rows[iid], r_idx, cand, parent, cfg)
            drive_units(_units(), on_done=on_done, on_error=on_error,
                        pool_size=pool_size(batch_size, *clients), cap=batch_size,
                        resolve=resolve,
                        progress=f"{STAGE} round {r_idx}" if verbose else None)
            buf.flush()
        # Whether every input has succeeded through this round decides the next
        # round. Read only rounds up to this one, so a resume whose saved file
        # holds a partial later round makes the same decision a clean run did.
        through = [r for r in rows if r["round"] <= r_idx]
        if stop_on_success and all(_succeeded(through, iid) for iid in input_rows):
            break


def _plan_round(
    r_idx, input_rows, pool, registry, rows, *, beam, width, seed, sample_kwargs,
    stop_on_success, max_candidates, budget,
) -> list[tuple[str, str, str]]:
    """The ``(input_id, parent, candidate)`` units of one round.

    Round 0 draws ``width`` combinations per input. A later round mutates the
    beam of the round before, dropping candidates already tried for the input,
    at most ``max_candidates`` per input (:func:`_pick_mutations`). An input
    that succeeded in an earlier round (with ``stop_on_success``) is skipped.

    Only the rows of the rounds *before* ``r_idx`` are read, so the plan this
    round would make on a resume is the one a clean run made — independent of
    what a killed run saved of this round or a later one."""
    plan: list[tuple[str, str, str]] = []
    prev = [r for r in rows if r["round"] < r_idx]
    for iid, input_row in input_rows.items():
        if stop_on_success and _succeeded(prev, iid):
            continue
        text = str(input_row.get("prompt", ""))
        entry_type = str(input_row.get("entry_type", ""))
        if r_idx == 0:
            for cand in _draw_round0(pool, iid, text, entry_type, seed=seed,
                                     width=width, sample_kwargs=sample_kwargs):
                plan.append((iid, "", cand))
            continue
        tried = _tried(prev, iid)
        parents = [(parent, _mutations_by_kind(parent, pool, registry, text,
                                               entry_type, iid, budget))
                   for parent in _beam(prev, iid, r_idx - 1, beam, seed=seed)]
        rng = random.Random(stable_seed(seed, iid, r_idx, "mutate"))
        chosen = _pick_mutations(parents, tried, max_candidates, rng)
        plan.extend((iid, parent, cand) for cand, parent in chosen.items())
    return plan
