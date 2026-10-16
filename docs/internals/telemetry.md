# telemetry: the trace and the cost roll-up

Two modules share the work. `llms/observe.py` is the hook the LLM layer emits
events through. `telemetry.py`, at the top level, collects them, writes them
out, and prices the run. [running.md](../running.md#telemetry-and-cost)
describes the output from a user's side.

## Responsibilities and boundaries

- Events are emitted where the data exists and decided on above.
  `llms/observe.py` imports nothing; `record()` is a no-op until an emitter
  is installed, and never raises.
- `telemetry.py` owns the sinks, the stage label, the roll-up and the
  prices. No vendor SDK is imported in `llms/`.
- The JSONL sink is a plain file writer, never a `logging.Handler`. Nothing
  here configures logging: the roll-up goes to the `redact.telemetry` logger
  at INFO, and the application's level decides whether it shows.
- A broken sink never stops a run: its exception is logged at DEBUG.

## The hook

| Function | Does |
|---|---|
| `observe.record(event)` | Hands an event to the emitter, filling `stage` |
| `observe.set_emitter(fn)` | Installs the emitter; `telemetry.install` calls it |
| `observe.set_stage(label)` | The stage later events carry; a module global, since thread pools do not pass context variables to their workers |

## Events

| `ev` | Emitted by | Fields |
|---|---|---|
| `call` | every backend, once per transport call | `model`, `backend`, `n_items`, `ms`, `in_tok`, `out_tok`, `out_tok_max`, `max_tokens`, `temperature`; `error` when it failed, `finish_reason` when the provider gives one, `cut: true` when the reply was cut at the token limit |
| `local`, `phase: load` | `backends/vllm.py`, `backends/introspection.py` | `backend`, `hf_model_id`, `load_ms`; for vLLM `quantization`, `tensor_parallel_size`, `data_parallel_size`, `gpu_memory_utilization`, `devices`; for transformers `device_map`, `torch_dtype`, and the measured `claimed_gib` and `weights_gib` |
| `local`, `phase: release` | the same, when an engine is released or at exit | `hf_model_id`, `held_s` |
| `residency` | `residency.report` | the plan's `as_dict()`, and `stages` |
| `preload_failed` | `residency.preload` | `model`, `error` |
| `stage` | `telemetry.stage` on leaving a stage | `stage`, `duration_s` |

Every event also carries `stage` and a UTC `ts`. A `call` is one transport
call, not one batch item: a hosted model makes one call per item, and vLLM
takes each item as its own engine request, so both give one event per item.

## Setup

| Function | Does |
|---|---|
| `install(data_dir=None, sink=None)` | Starts a `Collector` and installs it as the emitter. `sink` defaults to `REDACT_TRACE`, else `jsonl`. Writes `Datasets/run-<YYYYMMDD-HHMMSS>.trace.jsonl` under `data_dir` |
| `stage(label)` | A context manager: labels every event inside it, and records a `stage` event with its duration on exit; then logs a WARNING per model that had replies cut at the token limit in the stage, with their count |
| `summary()` | Rolls up the run and logs it; `{}` when telemetry is off |
| `collector()` | The installed `Collector`, or `None` |
| `uninstall()` | Stops collecting |
| `set_gpu_rate(usd_per_hour)` | Prices this machine's GPUs, over the environment and the table |

With `REDACT_TRACE=wandb` the same events also go to Weights & Biases (a
missing `wandb` package logs a warning and keeps the JSONL trace). With
`off` nothing is collected and `summary()` is empty.

## The roll-up

`Collector.summary()` returns `models`, `local_s`, `local_billed_s`, `gpu`,
`api_cost_usd`, `local_cost_usd`, `errors` and `cut`, the replies cut at the
token limit per stage and model
([running.md](../running.md#telemetry-and-cost) gives each).
`Collector.cut_in(stage)` gives one stage's counts.

A backend marks a cut reply with `cut` on its `call` event and warns for the
first cut reply of each model in each stage; later ones are only counted.

Two meters, because two things are bought:

- **Hosted calls** are priced per model: tokens x the entry's
  `price_per_1m_input` and `price_per_1m_output`. Only calls made through a
  hosted backend are priced; a model served locally is not, even when its
  entry has prices.
- **Local engines** are priced by time. A vLLM or transformers engine is held
  from the start of its load until its release, idle time included. The
  billed time is the union of every engine's spans, so two engines loaded at
  once are billed once. The cost is billed time x the hourly rate x the GPUs
  `nvidia-smi` reports on the machine, since a pod is rented whole.

The rate is, in order: `set_gpu_rate`, `REDACT_GPU_HOURLY_USD`, then
`configs/llm/gpu_pricing.json` under `REDACT_GPU_PROVIDER` (default `local`,
priced at 0), by the GPU name `nvidia-smi` reports or the provider's `*`
entry. A GPU the table does not price gives `local_cost_usd: null`: time is
reported, a cost is not guessed. When GPUs are held and the price comes from
the default provider, the roll-up warns once that the cost reads $0.

## How the run uses it

`run_pipeline` calls `install` first, so the residency plans and preloads are
recorded too; wraps each stage in `stage(name)`; and puts `summary()` in the
run summary under `telemetry`, with the trace's path under `trace`.
