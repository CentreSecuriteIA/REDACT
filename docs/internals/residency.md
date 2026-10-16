# residency/: the GPU planner

`residency/` says, before anything loads, whether a run's local models fit
on the machine's cards, where each goes, and with what share of its card.
It plans stage by stage, and loads, keeps and releases engines as the run
moves between stages. [running.md](../running.md#the-gpu-planner) describes
what a user sees; this page describes how.

## Responsibilities and boundaries

It is responsible for:

- each local model's footprint, computed without loading it;
- placing the models on cards and granting each vLLM engine its
  `gpu_memory_utilization`;
- one plan per stage, with the unloads and reloads that make every stage fit;
- handing the grants and cards to the vLLM loader, preloading, and unloading;
- explaining the plan in the log and in a telemetry event.

It must not:

- abort a run. A plan that does not fit, a planning error and a failed
  preload are logged; the stage that cannot load fails on its own;
- measure a model to size it. vLLM reserves `gpu_memory_utilization x card`
  at load, so a measurement would restate the setting;
- decide when a stage starts. The driver calls it at each stage change.

## Modules

| Module | Holds |
|---|---|
| `footprint.py` | `ModelFootprint`, `footprint(model, backend_type)` |
| `placement.py` | `plan_residency(models, ...)`; the card arithmetic and the search |
| `plan.py` | `ResidencyPlan` |
| `phases.py` | `Phase`, `plan_phases(stage_models, ...)` |
| `lifecycle.py` | `apply_plan`, `preload`, `unload_local`, `loaded_engines` |
| `explain.py` | `report(plan)`, the plan as text, the advice |
| `llms/resources/estimate.py` | Weights and KV cache from a checkpoint's HF config |
| `llms/resources/measure.py` | `detect_gpus`, `detect_gpu_memory_gib` (`nvidia-smi`), `free_total_gib` (torch) |

All sizes are GiB (bytes / 1024^3).

## Footprints

`footprint(model)` resolves the model's setup and returns `None` unless it is
local (`vllm` or `introspect`).

- **Weights**: the setup's `vram_gb` (source `declared`), else an estimate
  from the HF config (source `estimated`), else unknown. The estimate reads
  shapes only, takes the larger figure where the config leaves a choice, and
  sizes a quantized checkpoint (`awq`, `gptq` at 4 bits, `fp8` at 8).
- **KV cache** (vLLM only): always estimated, at `max_model_len`, else at
  10,000 tokens capped at the checkpoint's own maximum.
- **Padded need** per card: `(weights + KV) x 1.1 + 0.6`. The planning
  figure rounds it up to 0.5 GiB.
- **Engine**: models whose setups share a backend cache key (checkpoint,
  quantization, engine kwargs) share one engine and one footprint.

A transformers model gets no KV term: it preallocates none.

## The machine

The card count and name come from `nvidia-smi --query-gpu=name`, the size
from `nvidia-smi --query-gpu=memory.total` (the smallest card, when sizes
differ), else from torch. Only the cards `CUDA_VISIBLE_DEVICES` leaves count,
and cards are numbered over them. `plan_residency(..., gpus=, gib_per_gpu=)`
plans for another machine.

## Placement

`plan_residency(models, backend_types=None, *, gpus, gib_per_gpu, copies,
pinned)` places the models in the order given:

- a single-card vLLM engine on the card with the most room (lowest index on
  a tie), so engines spread before they share;
- a vLLM engine with `min_gpus` above 1, or with copies, on whole untouched
  cards;
- a transformers model where its `device_map` puts it: one card for
  `cuda:N`, none for `cpu`, the cards' combined room for `auto`,
  `balanced`, `balanced_low_0` or `sequential`;
- an engine already loaded (`pinned`) first, on its own cards.

A model that does not fit beside those placed starts a new **group**. More
than one group means the models cannot be resident together.

**Grants.** On each card, the vLLM engines split 0.9 of the card less what
transformers models hold. Each gets its padded need plus an equal share of
the spare, written to `planned_utilization` as a fraction of the card
(rounded down to 4 decimals, but never below the need). A grant below the
padded need makes the model `oversized`, which is a problem.

**Search.** When the order given leaves more than one group, `_Search` tries
other cards for the vLLM engines, and keeps an arrangement where everything
is one group: the fewest cards holding two engines, then the largest
smallest share. It tries every card for up to 8 single-card engines, and
stops after 20,000 tries with the best found. Beyond 8 engines it places them
once, largest first.

**Copies.** When all the models are one group and cards are left idle, the
first single-card vLLM engine in the order given gets a copy on each idle
card, provided it is alone on its card and fits it. A count in `copies`
replaces that: `copies={model: n}` takes `n` copies on whole cards, and `1`
turns copies off. The loader passes the copies to vLLM as
`data_parallel_size`.

## The plan

`ResidencyPlan` holds `groups`, `gpus_required`, `gpus_available`,
`gpu_name`, `per_gpu_gb`, `oversized` and `machine_given`.

| Property | Meaning |
|---|---|
| `co_resident` | One group or none |
| `sequential` | More than one group: the paraphrase stage reads it to unload between paraphrasers |
| `fits` | No problem |
| `problems` | More GPUs needed than present; several groups; an oversized model; a tensor-parallel size that does not divide the attention heads; a model of unknown size beside others; several vLLM engines on cards of unknown size |
| `warnings` | A model of unknown size; an unknown card size; a KV cache that could not be estimated |
| `as_dict()` | The plan as data, for the trace and the run summary |
| `explain()` | The plan as text, with each model's derivation, card and grant, the problems, and advice |

`report(plan, stages=...)` logs `explain()` at ERROR when the plan does not
fit, at WARNING when it has warnings, else at INFO, and records a
`residency` event.

## Phases

`plan_phases(stage_models, *, gpus, gib_per_gpu, copies)` makes one `Phase`
per stage: `stage`, `resident` (the models loaded while it runs), `plan`,
and `loadable` (the plan's first group).

- An engine is resident from the first stage that calls it to the last, so it
  loads once. Each distinct resident set is planned on its own; stages with
  the same set share one plan object.
- An engine keeps the cards it loaded on: later stages place it first, on
  them. When a stage does not fit that way, cards are searched for the run as
  a whole, and an engine takes other cards only when every stage fits on
  them.
- An engine gets the smallest share any of its stages plans for it, so the
  share it loads with holds while it is resident. Of its copies it keeps
  those on cards every one of its stages leaves it.

**Unload and load again.** When a stage still does not fit, it may hold
engines it does not call, kept for a later stage. The planner unloads each
such engine over its gap (from the stage after one call to the next call)
and loads it again for the later stage, if every stage then fits; otherwise
none. Of those unloads, each the run fits without is given back, the largest
engine's first. Each unload costs one more load and is logged.

## Lifecycle

| Function | Does |
|---|---|
| `apply_plan(plan, replace=True)` | Hands the plan's grants and cards to `backends/vllm.py` (`set_planned_utilization`, `set_planned_devices`), keyed by engine. `replace=False` keeps values already set |
| `preload(models, wanted=None)` | Loads local models one after another on a daemon thread and returns it. A model no longer in `wanted` when its turn comes is skipped. A failed load is logged at ERROR and recorded as `preload_failed` |
| `unload_local(keep=None, engines=None)` | Releases every cached local engine but those the kept models use |
| `loaded_engines(exclude=None)` | The engine keys loaded now, per setup |

The backends lock their engine caches, so a preload and a stage that need the
same engine load it once.

## How the run uses it

The driver (`driver.py`):

1. `_plan_phases`: before the first stage, `plan_phases` over each stage's
   models; `report` per distinct plan; `apply_plan` per phase.
2. `_enter_phase`, at each stage: unload the engines the previous phase held
   and this one does not; `apply_plan(phase.plan)`; preload the phase's
   loadable models, plus the next stage's when its plan is one group and
   holds everything this one holds.
3. At the end, also after a failure: wait for the preload threads, unload
   every engine, clear the planned shares and cards.

The paraphrase stage plans its own pool of paraphrasers (`plan_residency`
with one copy each). When they do not fit together it unloads each before
the next, keeping the check model and the engines of other stages.

## Limits

The `TODO(driver script)` block in `residency/__init__.py` lists what is not
done:

- the paraphraser pool is planned apart from the run's plan;
- one card size per plan, the smallest; no free memory per card;
- the unload search tries every candidate unloaded at once, and with many
  lifetimes it can miss a run that one unload would fit;
- a plan does not see engines loaded by another process;
- copies do not share a card with another model, do not fall back to one
  copy when the build fails, and a tensor-parallel engine gets copies only
  from a count;
- a transformers load that starts CUDA while a vLLM engine is built sees
  only that engine's cards.
