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
| `llms/resources/measure.py` | `gpu_cards`, `card_sizes_gib`, `card_use`, `detect_gpus` (`nvidia-smi`), `free_total_gib` (torch), `bus_order`, `visible_only` and `cuda_lock` |
| `llms/backends/vllm.py` | The loader: planned grants and cards, the engine processes each build starts, `terminate_engine_processes`, the exit hook |

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

The card count and name come from `nvidia-smi --query-gpu=name`, each card's
size from one `nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,name`
query (`gpu_cards`), else from torch. Only the cards `CUDA_VISIBLE_DEVICES`
leaves count, named by index or UUID, and cards are numbered over them in
the variable's order. Each card is planned at its own size. A card count
given without a size, or cards the size query does not name, plans every
card at the smallest. `plan_residency(..., gpus=, gib_per_gpu=)` plans for
another machine.

The plan numbers cards as `nvidia-smi` does, in PCI bus order, and CUDA's
own order puts the fastest card first. So a plan or a load sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` unless it is set, warning when CUDA has
started in this process already, and a load hands an engine its cards by
UUID.

A vLLM build narrows `CUDA_VISIBLE_DEVICES` for the whole process while it
runs (`measure.visible_only`). Meanwhile `visible_gpu_ids()` reads the value
from before, a transformers load waits (`measure.cuda_lock`, taken after a
backend's own load lock), and the torch probe for the card size does
without.

## Placement

`plan_residency(models, backend_types=None, *, gpus, gib_per_gpu, copies,
pinned)` places the models in the order given:

- a single-card vLLM engine on the emptiest card with room for it (lowest
  index on a tie), so engines spread before they share, or, where none has
  room, on the card with the most;
- one with copies given on that many cards with room, the largest first;
- a vLLM engine with `min_gpus` above 1 on whole untouched cards;
- a transformers model where its `device_map` puts it: one card for
  `cuda:N`, none for `cpu`, the cards' combined room for `auto`,
  `balanced`, `balanced_low_0` or `sequential`;
- an engine already loaded (`pinned`) first, on its own cards.

A model that does not fit beside those placed starts a new **group**. More
than one group means the models cannot be resident together.

**Grants.** On each card, the vLLM engines split 0.9 of the card less what
transformers models hold. Each gets its padded need plus an equal share of
the spare, as a fraction of the card (rounded down to 4 decimals, but never
below the need). An engine on several cards loads with one
`gpu_memory_utilization`: the smallest fraction any of its cards gives it,
written to `planned_utilization`, and its `reserved_gb` is what that holds
on its smallest card. A grant below the padded need makes the model
`oversized`, which is a problem.

**Search.** When the order given leaves more than one group, `_Search` tries
other cards for the vLLM engines, and keeps an arrangement where everything
is one group: the fewest cards holding two engines, then the largest
smallest share. It tries every card for up to 8 single-card engines,
counted once however many times a run loads them, and each set of cards
with room for an engine's copies given; of untouched cards of one size, as
many as an engine takes stand for the rest. It stops after 20,000 tries with
the best found, or, with none found, places the engines once more, largest
first. Beyond 8 engines it places them once, largest first.

**Copies.** When all the models are one group, the first vLLM engine in the
order given gets a copy on every other card with room for one, provided it
fits its own (vLLM's `data_parallel_size`). A copy of a single-card engine
is placed like the engine, beside other models under the ceiling, only
where every engine there keeps a share that covers its need, and never on a
card smaller than the engine's: its one share would grant less on every
card. A copy of a tensor-parallel engine takes `min_gpus` untouched cards,
whole; one that gets none leaves the cards to the first single-card engine.
A count in `copies` replaces that: `copies={model: n}` takes `n` copies,
which may share cards and may take smaller ones, and `1` turns copies off.
Copies of a tensor-parallel engine are planned, not yet loaded end to end
on a GPU.

## The plan

`ResidencyPlan` holds `groups`, `gpus_required`, `gpus_available`,
`gpu_name`, `per_gpu_gb` (the smallest card), `gib_per_card`, `oversized`
and `machine_given`.

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
  share it loads with holds while it is resident, and a share so lowered
  that it no longer covers the engine's need is a problem in every stage.
  Of its copies it keeps those on cards every one of its stages has room
  for: where stages differ, placement places them again with the copies on
  those cards only, so a copy that does not load leaves the models beside
  it their full share.

**Unload and load again.** When a stage still does not fit, it may hold
engines it does not call, kept for a later stage. The planner unloads each
such engine over its gap (from the stage after one call to the next call)
and loads it again for the later stage, if every stage then fits; otherwise
none. That attempt is exact up to 8 engines, since unloading more never
makes a fit harder. Of those unloads, each the run fits without is given
back, the largest engine's first. Each unload costs one more load and is
logged.

## Lifecycle

| Function | Does |
|---|---|
| `apply_plan(plan, replace=True)` | Hands the plan's grants and cards to `backends/vllm.py` (`set_planned_utilization`, `set_planned_devices`), keyed by engine. `replace=False` keeps values already set |
| `preload(models, wanted=None)` | Loads local models one after another on a daemon thread and returns it. A model no longer in `wanted` when its turn comes is skipped. A failed load is logged at ERROR and recorded as `preload_failed`; a load the way out ended is logged at INFO, recorded with `"cause": "exit"`, and ends the preload |
| `unload_local(keep=None, engines=None, *, wait=None)` | Releases every cached local engine but those the kept models use. With `wait`, a load in progress is waited for that long, then its vLLM build is ended |
| `loaded_engines(exclude=None)` | The engine keys loaded now, per setup |

The backends lock their engine caches, so a preload and a stage that need the
same engine load it once.

**Engine processes.** Each vLLM build records the child processes it starts
(its engine cores and, with copies, vLLM's coordinator). A release ends what
an engine's shutdown leaves running, and a failed build's processes are
ended at once. `terminate_engine_processes(processes, all=False)` sends
SIGTERM, then SIGKILL to those alive 5 s later, and lists their children
(tensor-parallel workers) with `psutil` before any is signalled; `all=True`
ends every engine process this process started and no other child, for the
forced exit of a second signal in `scripts/run.py`.

**A build that fails with copies.** When a build with the plan's copies
fails, one copy on the engine's first cards is built instead, once, with a
WARNING naming the error and `"copies": 1` as the way to make it permanent.
Not when a process the failed build started outlives SIGKILL, when the
process is exiting, or when the copies were handed to the engine directly.
The load event carries `planned_copies` and `fallback`.

**The exit.** The exit hook releases every engine, waiting for a load in
progress at most `EXIT_WAIT_S` (60 s). Past that it ends the engine
processes of the build, and vLLM's start-up wait fails as soon as one exits,
so the load lets go of the lock (`EngineLoadEnded`) and the release goes on.
A load that has not started its build is named, with nothing to end. No load
starts after the hook. A release mid-run still waits for a load in progress:
a first load takes up to 15 minutes.

## How the run uses it

The driver (`driver.py`):

1. `_plan_phases`: before the first stage, `plan_phases` over each stage's
   models; `report` per distinct plan; `apply_plan` per phase.
2. `_enter_phase`, at each stage: unload the engines the previous phase held
   and this one does not; `apply_plan(phase.plan)`; preload the phase's
   loadable models, plus the next stage's when its plan is one group and
   holds everything this one holds.
3. Before any model loads: `_check_cards` stops the run when another
   process holds memory on a planned card.
4. At the end, also after a failure: wait for the preload threads, at most
   `EXIT_WAIT_S` together, naming at ERROR a preload still loading; unload
   every engine with a 10 s wait, which ends that load; clear the planned
   shares and cards.

The paraphrase stage plans its own pool of paraphrasers (`plan_residency`
with one copy each). When they do not fit together it unloads each before
the next, keeping the check model and the engines of other stages.

## Limits

The `TODO(driver script)` block in `residency/__init__.py` lists what is not
done:

- the paraphraser pool is planned apart from the run's plan;
- each card is sized by its total and assumed empty; `run_pipeline` stops a
  run on a planned card another process holds, and a plan asked for outside
  it is not checked;
- the unload search tries every candidate unloaded at once, and with more
  than 8 engines it can miss a run that one unload would fit;
- a load that hangs is ended only on the way out or by a second signal; no
  load is timed out mid-run.
