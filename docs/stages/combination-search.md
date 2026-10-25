# Combination search

!!! warning "Not a recipe stage"
    `redact.jailbreak.search_combinations` is called from Python, like the
    `multi_turn` functions. It is not part of a recipe run and does not run in
    the pipeline. It reuses the recipe stages' persistence (`Manifest`,
    `Ledger`, `resume_state`, `CommitBuffer`), so it resumes like them.

Front and core `search_combinations` (`jailbreak/search.py`), exported from
`redact` and `redact.jailbreak`.

## Purpose

Given harmful inputs, a safety-trained **target** model and a **judge**, search
per input for a *combination of techniques* whose jailbreak text the judge says
the target complied with. It is the reference library's "search over technique
combinations", built on the project's own sampler, rules and persistence: every
candidate is a combination the library's rules allow, applied the way the
jailbreak stage applies one, and scored by a judge the way `multiturn_attacks`
scores an attack.

```python
import pandas as pd
from redact import search_combinations
from redact.jailbreak import technique_pool
from redact.jailbreak.augmentation import defaults, resolve

inputs = pd.DataFrame({"prompt": [...], "category": "...", "entry_type": "harmful"})
pool = technique_pool(resolve(defaults(), "adversarial"))   # or any pool you build

result = search_combinations(
    inputs,
    target_model="a-safety-trained-model",   # required, no default
    judge_model="venice-uncensored-fp8",     # required; not the target
    pool=pool, data_dir="./runs/search",
    width=4, beam=2, depth=2,
)
best = result["best"]        # {input_id: BestCombo}
rows = result["rows"]        # every candidate tried, the search tree
```

## How the search runs

A **round** scores a set of candidate combinations per input.

- **Round 0** draws `width` valid combinations per input with the library's own
  sampler (`sample_combination`), from the techniques that *apply* to that
  input (`technique.applies`). Every draw is compatible under the spec's rules
  by construction. The default round-0 draw is `{"exact_techniques": 1}` — one
  technique each — so the search starts simple and grows by mutation; pass
  `sample_kwargs` to draw budget combinations instead.
- **Each later round** keeps the `beam` best candidates of the round before (the
  frontier) and **mutates** each: **add** one compatible technique, **swap** one
  for another of its family, or **drop** one. Every mutation is checked with
  `rules.compatible`, so no round ever holds a pair the rules forbid. A
  combination already tried for an input is not tried again.
- **Scoring** a candidate: apply it with `run_chain` (model-calling techniques
  included, driven on `drive_units` like every stage), send the result to the
  target, and ask the judge. The judge's reply is read with `is_accepted`
  (`Yes`/`No`), as the conversation judge is.
- **Early stop**: an input stops expanding at the first round a candidate
  succeeds (`stop_on_success`, the default).

The frontier takes only candidates that **applied** — never the identity, nor
one a technique discarded or an error ended — ranked by score (a success is 1.0,
else 0.0), then by fewest techniques, then by a seeded draw per candidate. Scores
are 0 or 1, so until a candidate succeeds the beam is a seeded pick among the
smallest applied combinations: the search is **breadth-first over small
combinations** rather than a ranking of equal zeros.

### The candidate cap

A parent has one mutation per compatible add, swap and drop. Over a full pool
that is hundreds: with the defaults (`width` 4, `beam` 2, `depth` 2) and the
packaged adversarial pool an uncapped search tries about **900 candidates per
input** (round 0's 4, then ~433, then ~472), each a target call and a judge call
plus the techniques' own model calls. So each mutation round tries at most
`max_candidates` candidates per input (**8** by default), drawn with the seeded
rng: each draw takes the beam's parents in rotation, a kind (add/swap/drop) at
random among those the parent still has untried, and a mutation of that kind at
random, so the parents and the kinds are spread evenly. `max_candidates=None`
tries every valid mutation (the old, unbounded behaviour).

With a **budget** round 0 (a `sample_kwargs` without `exact_techniques`) every
round keeps that budget: a mutation over `max_complexity`, or with more than
`max_obfuscations` obfuscations, is not tried, and a layer round 0 was told to
leave out stays out of every round.

## Parameters

| Name | Default | What it is |
|---|---|---|
| `inputs` | — | a frame with a `prompt` (or `sample`/`text`) column; `sample_id`, `category`, `entry_type` are used when present |
| `target_model` | — (required) | the safety-trained model under attack |
| `judge_model` | — (required) | the scoring model; not the target |
| `pool` | packaged adversarial pool | the techniques to draw from (`technique_pool`); requires-benign techniques (fsh/dap) are dropped unless `benign_data` is given |
| `judge_system` | `SUCCESS_JUDGE_SYSTEM` | the judge's criterion (its system prompt) |
| `gen_model` | `uncensored_gen` default | applies model-calling techniques |
| `translation_model` / `check_model` | `None` / `gen_model` | the translation model, and the technique-output check model |
| `width` / `beam` / `depth` | 4 / 2 / 2 | round-0 candidates per input; frontier kept per input; mutation rounds after round 0 (`depth + 1` rounds in all) |
| `max_candidates` | 8 | most candidates tried per input in each mutation round (a seeded sample over the parents and the add/swap/drop kinds); `None` tries every valid mutation |
| `seed` | 42 | seeds the sampling, so a resume is identical |
| `sample_kwargs` | `{"exact_techniques": 1}` | round-0 sampler settings |
| `stop_on_success` | `True` | stop expanding an input once it succeeds |
| `resume` | `True` | reuse the saved rows and run only the unfinished candidates |
| `batch_size` | 64 | candidate units per model call |
| `data_dir` / `out_path` | — | the output is `Datasets/jailbreak_search.csv` under `data_dir`, or at `out_path` |

## Output

`jailbreak_search.csv` holds one row per tried candidate — the rows **are** the
tree. Every row has `sample_id` (the content id of
`search:{input_id}:{round}:{candidate}`, unique per unit and stable across a
resume) and `input_id`, plus `round`, `candidate` (the combination, techniques
joined by `+`, or `identity`), `parent` (the candidate it was mutated from, `""`
in round 0), `num_techniques`, `complexity`, `applied` (whether the combination
applied without a discard), `is_noop`, `jailbreak` (the applied text),
`technique_info`, `target_model`, `judge_model`, `target_response`, `success`,
`score`, `judge_reasoning`, `rules` (the `rules_id` of the pool),
`combination_spec_version`, and the input's `category`, `entry_type` and
`prompt`.

`search_combinations` returns `{"best": {input_id: BestCombo}, "rows": frame}`.
`BestCombo` is the best combination per input: `combination` (technique names in
applied order; empty when nothing beat the bare prompt), `score`, `success` and
the `round` it was found in. `search_tree(rows, input_id)` gives the input's
nodes round by round; `best_combinations(rows)` gives the per-input best from a
saved frame.

## Resume

The search is deterministic given the inputs, the seed, the pool and the
params: round 0's draws are seeded per `(seed, input_id, i)`, and a round's plan
(its beam and the sampled mutations) is a pure function of the rows of the
rounds **before** it alone. So a resume re-derives exactly the plan the first
run would have made — for each round, independent of what a killed run saved of
that round or a later one — and runs only the candidates the ledger has not
acked. A run killed mid-search, even after a success is saved mid-round, resumes
to the same rows with no duplicates. The ledger keys a unit `(input_id, round,
candidate)`; each round is committed before the next round's beam is read. The
plan file beside the output records the seed, `width`, `beam`, `depth`,
`max_candidates`, the pool names, the rules id, `sample_kwargs`, the target and
judge models and their system prompts, and `stop_on_success`; a resume with any
of them changed raises rather than mix two trees in one file (run with
`resume=False` to search again).

A prompt too long for the target or the judge on the async lane (an
OpenAI-compatible or vLLM binding) is recorded as a failed row for that
candidate, so one oversized prompt never stops the search; on an Anthropic or
introspection binding, as elsewhere, it stops the stage.

## Smoke run

`scripts/search_smoke.py` runs a small search against a local target with a
local judge and generation model, and prints counts only (never text):

```bash
python scripts/search_smoke.py --root /workspace/searchsmoke \
    --target llama-3.2-3b-debug --judge venice-uncensored-fp8
```
