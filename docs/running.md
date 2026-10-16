# Running a pipeline

This page covers a run from the outside: installing, writing a recipe,
choosing models, what the GPU planner does, resuming, the files a run leaves,
what it cost, and the failures you are likely to meet. Every key is listed in
[configuration.md](configuration.md).

## Install

Python 3.11 or later.

```bash
pip install -e .                   # core: pandas, openai, anthropic, python-dotenv, pyfiglet, filelock
pip install -e ".[vllm]"           # local models through vLLM
pip install -e ".[transformers]"   # the introspection backend (torch, transformers, accelerate)
pip install -e ".[hf]"             # loading published datasets from the HuggingFace Hub
pip install -e ".[dev]"            # pytest, ruff, mypy
```

`requirements.lock` pins an environment that runs vLLM on a GPU pod. Its
torch build needs a host driver for CUDA 13; on an older driver vLLM stops at
load with "The NVIDIA driver on your system is too old".

Put API keys in a `.env` file. `import redact` loads `.env` from the default
working root (`REDACT_OUTPUT_DIR`, else the source tree that holds
`pyproject.toml`, else the current directory), and its values win over the
shell's. The keys the shipped models read are `VENICE_API_KEY`,
`ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`. Local weights download to the
HuggingFace cache (`HF_HOME`); `scripts/download_models.py` fetches every
local checkpoint of the registry ahead of a run.

## Run a recipe

From the command line:

```bash
python scripts/run.py src/redact/configs/runs/training_example.json --data-dir ./runs/my_run
```

`--params` replaces the recipe's `params_file`, `--data-dir` its `data_dir`.
`--quiet` logs warnings only and `--debug` logs per-item detail. The script
prints the run summary as JSON. The first Ctrl-C, or `kill <pid>`, ends the
run through its cleanup, which saves the finished units and releases the
models. A second one ends the run's child processes (vLLM's engine among
them: SIGTERM, then SIGKILL after 2 seconds) and exits at once.

From Python:

```python
from redact import run_pipeline

if __name__ == "__main__":
    summary = run_pipeline("recipe.json", params="params.json", data_dir="./runs/my_run")
```

Keep the `if __name__ == "__main__":` guard in any script that may load a
local model. vLLM starts its workers with `spawn`, which imports the main
script again in each child; without the guard the child runs the pipeline
too.

The library logs under the `redact` logger and configures nothing. In your
own script, call `logging.basicConfig(level=logging.INFO)` to see progress.
`python scripts/inspect_run.py <data_dir>` prints each stage's run record.

## Write a recipe

A recipe says what to run. A params file holds each stage's own settings,
under the stage's name.

```json
{
  "dataset_type": "training",
  "data_dir": "./runs/first",
  "models": {
    "gen": {"name": "venice-uncensored", "backend_type": "vllm"},
    "check": null
  },
  "stages": ["constitution", "inputs", "outputs", "jailbreaks", "build"],
  "augmentations": {
    "adversarial": {"rounds": "escalation"},
    "natural": null
  },
  "params_file": "params.json",
  "resume": true
}
```

```json
{
  "constitution": {"num_taxonomy_categories": 3, "num_categories": 10},
  "inputs": {"samples_per_entry": 3, "style": "long"},
  "outputs": {"max_per_category": 20},
  "jailbreaks": {"seed": 42}
}
```

- `stages` run in the order listed. Keep the order of the stage table in
  [overview.md](overview.md#the-stages): each stage reads what earlier ones
  wrote. Without `stages` a run takes `inputs`, `outputs`, `jailbreaks`,
  `build`.
- A training run's `inputs` stage reads the constitution saved under
  `data_dir`. List `constitution` before it, or run it earlier on the same
  `data_dir`. An eval run's inputs are seeded the same way when its recipe
  lists `constitution` or a constitution is saved under `data_dir`; without
  one they are standalone.
- `taxonomy` names the taxonomy of the constitution and the standalone
  inputs (default `harmful`); `benign_taxonomy` the one whose subcategories
  the benign data of the `fsh` and `dap` techniques covers (default
  `benign`). Both are file names under `taxonomy_dir`, the packaged
  `configs/taxonomy/` by default.
- `params_file` is relative to the recipe's folder. `data_dir`,
  `prompt_dir` and `taxonomy_dir` are relative to the current directory.
- A params key must be a parameter of the stage's front, and not one the
  recipe passes itself (`model`, `data_dir`, `resume`, ...). Both rules are
  checked for every stage before the first runs.

The two example recipes under `src/redact/configs/runs/` share
`params_example.json`. Its `inputs` entry holds standalone settings
(`samples_per_category`, `num_categories`, ...), which a training run's
seeded inputs ignore.

## Choose models

A recipe names a model per **role**. A role left out, or `null`, takes the
default of its registry role: the first model in `configs/llm/models.json`
that lists that role.

| Recipe role | Called by | Registry role | Default today |
|---|---|---|---|
| `gen` | inputs, outputs, jailbreaks | `uncensored_gen` | `venice-uncensored`, hosted (Venice) |
| `check` | inputs, outputs | `uncensored_gen` | the `gen` model |
| `translation` | jailbreaks, only when a kind has `"translation": true` | `translation` | `deepseek-v3.2`, hosted (Venice) |
| `constitution` | constitution | `constitution_gen` | `claude-opus-4-6`, hosted (Anthropic) |
| `paraphraser` | paraphrase | `paraphraser` | `venice-paraphraser`, local vLLM (24B weights) |
| `paraphrase_check` | paraphrase | `uncensored_gen` | `venice-uncensored`, hosted |

A role's entry is a registry name, `null`, or an object that also says how
the model is served:

```json
"models": {
  "gen":   {"name": "venice-uncensored", "backend_type": "vllm", "copies": 2},
  "check": {"name": "olafangensan-glm-4.7-flash-heretic", "workers": 2}
}
```

| Key | Meaning |
|---|---|
| `name` | The registry name. Left out or `null`: the role's default |
| `backend_type` | Which setup of the entry serves it: `api`, `vllm` or `introspect`. Left out: the entry's own default |
| `copies` | vLLM only. Copies of the engine, one per card. Left out: a copy on each idle card; `1` turns copies off |
| `workers` | Hosted only. Requests in flight at once. Left out: the entry's `recommended_max_workers` |

One model is served one way in a run. Two roles that name the same model must
give the same keys, or only one of them gives any; a `check` left out is
served as `gen` is.

**Local or hosted.** A registry entry may carry both a hosted setup and local
weights. `venice-uncensored` does: its default is the Venice endpoint, and
`"backend_type": "vllm"` loads `dphn/Dolphin-Mistral-24B-Venice-Edition` on
the local GPU instead.

- A hosted model is limited by its `rpm` (a sliding 60-second window) and its
  workers. `venice-uncensored` allows 75 requests a minute and 3 in flight;
  `claude-opus-4-6` allows 5 a minute, one at a time, and its window is
  shared by every model on the same Anthropic key.
- A local vLLM model has no rate limit. The inputs, outputs and jailbreak
  stages keep as many requests in flight as the engine runs at once, times
  its copies. In the GPU check on an A100 80GB, every such stage ran its
  local model with at most 256 requests in flight. The paraphrase stage
  sends calls of `batch_size` instead, and the constitution stage one unit
  at a time.

Large runs belong on local models. A hosted model suits a small run, or the
role that makes few calls (the constitution).

## The GPU planner

When any stage calls a local model, the driver plans the GPUs before the
first stage runs. The plan is a forecast and a set of instructions to the
loader. It never aborts a run: a plan that does not fit is logged at ERROR
and the run goes on. [internals/residency.md](internals/residency.md) has the
mechanics.

**Footprint.** Each local model's need is worked out without loading it. The
weights are the setup's `vram_gb` when set, else an estimate from the
checkpoint's HuggingFace config, else unknown. A vLLM engine adds its KV cache
at its `max_model_len`, or at 10,000 tokens capped at the checkpoint's own
maximum. The padded need per card is (weights + KV) x 1.1 + 0.6 GiB. Two
registry entries with the same checkpoint, quantization and engine kwargs
share one engine and are counted once (`venice-paraphraser` and the local
setup of `venice-uncensored` do).

**Placement and grants.** The vLLM engines on a card split 0.9 of it: each
gets its padded need plus an equal share of what is left, and loads with that
fraction as its `gpu_memory_utilization`. A single-card engine goes to the
card with the most room; one with `min_gpus` above 1 takes whole cards
(`tensor_parallel_size`). When the models in the order given do not fit, the
planner searches other arrangements of cards.

**Phases.** The plan is made stage by stage. A model is resident from the
first stage that calls it to the last, so it loads once. Entering a stage,
the driver unloads the engines that stage does not keep, and preloads the
stage's models on a background thread (and the next stage's, when they fit
beside them). An engine keeps the cards it loaded on, and the smallest share
any of its stages plans for it.

In the GPU check of this release, a 24B generator and a 3B checker on one
A100 80GB give two plans: the 24B alone for `constitution` and `jailbreaks`,
and the 24B with the 3B for `inputs`, `outputs` and `paraphrase`. The 24B
loads at 0.71 in every stage, its share where the card is shared; the 3B
loads at 0.19. The 3B is unloaded before `jailbreaks` and the 24B before
`build`.

**Unload and load again.** A model kept for a later stage can crowd a stage
in between. The planner then unloads it for those stages and loads it again
for the later one, when every stage fits that way. It logs:
`<model> is unloaded for stage <a> and loaded again for stage <b>, so that
<c> fits. This costs one more load.`

**Copies.** When all of a stage's models fit together and cards are left
idle, the first single-card vLLM engine, when it is alone on its card, gets a
copy on each idle card (vLLM's `data_parallel_size`). More copies mean more requests in flight. A copy
lengthens the load, so for a short run set `"copies": 1`.

**Reading the plan.** The log shows, per plan, the stages it covers, the
cards found, and per model its size, how the size is derived, its card and
its grant:

```
[residency] stages inputs, outputs, paraphrase: plan needs 1 GPU(s); detected 1 x <GPU>, <size>GiB each
  group 1:
    <model>[vllm]  ~<size>GiB  (estimated: <weights> weights + <kv> KV, x1.1 +0.6 [bfloat16])  -> card 0, vLLM gets <n>GiB (gpu_memory_utilization=<share>)
```

A second group means the stage's models do not fit together. `PROBLEM:` lines
say why and what to change: a lower `max_model_len`, a quantized checkpoint,
`min_gpus`, the hosted setup, or the stages run as separate `run_pipeline`
calls on the same `data_dir`. `WARNING:` lines mark what the plan could not
check, such as a model of unknown size or a machine without `nvidia-smi`.

**Cards held by another process.** Before any model loads, the driver reads
`nvidia-smi` once. When another process holds memory on a card the plan puts
a model on, the run stops with a `RuntimeError` that names each such card,
its processes and their memory, and says how to free them; nothing has run
yet. These are most often the vLLM engine processes of an earlier run
killed with `kill -9`. A process on a card the plan leaves free, and memory
in use (1 GiB or more) on a planned card with no process listed, are only
warnings. Without `nvidia-smi` the check does nothing.

## Resume, fresh runs, and a kill partway through

**Resume is the default.** With `"resume": true` each stage skips the units
its ledger records as finished. A resumed stage:

1. takes the stage's lock, so a second run on the same output stops;
2. checks each output file against the size and checksum its ledger recorded
   at the last save;
3. cuts off what a save that did not finish left at the end of a file, and
   keeps those bytes in `<file>.cut`;
4. runs the units not yet finished, and saves them in batches (every 32
   units in most stages).

A unit is acked only after its rows are on disk, so a crash loses at most the
units finished since the last save, and those run again. [internals/dataset.md](internals/dataset.md#the-save-and-resume-model)
explains the mechanism.

**Stopping a run.** Ctrl-C, or `kill <pid>` on `scripts/run.py`, ends the run
cleanly: the finished units are saved, requests in flight are cancelled, and
the local engines are released. Rerun the same command to go on. A second
Ctrl-C (or `kill`) while that cleanup runs ends the run's child processes and
exits at once; the next resume cuts off a save it left half done.

`kill -9`, or a lost machine, may leave half a save. The next resume cuts it
off and runs those units again. vLLM's engine processes outlive `kill -9` and
keep their GPU memory; the next run stops before any stage, naming them and
their PIDs (see [the GPU planner](#the-gpu-planner)). Kill them and rerun.
The next load may log "Failed to reload cubin file" warnings from torch's
compile cache, and carries on.

In the GPU check, a jailbreak run of 432 units, killed with 32 saved, resumed
and ran the other 400. It ended with 432 rows, 433 ledger lines (the opening
`_saved` line and 432 acks), no duplicate, and an unchanged manifest.

**Starting over.** `"resume": false` starts every listed stage over: it
removes the stage's output, its `.cut` files, its ledger and its manifest,
then plans anew. The exceptions:

- The seeded `inputs` stage starts over only its current `style`. The rows
  and acks of other styles stay.
- The standalone `inputs` stage removes the files of the categories it runs.
- The `constitution` stage removes every CSV in `Data_cache/constitution/`.

The benign data (`Data_cache/benign/benign_samples.csv`) is never cleared:
it always resumes from its ledger, generating only the units (one subcategory
of the benign taxonomy and one answer type) it does not hold. A unit rejected
on every attempt is acked and skipped; delete
`benign_samples.state.jsonl` to try it again, or delete the CSV and its
sidecars to generate everything again.

To start one stage over and keep the others, run a recipe that lists only
that stage, with `"resume": false`, on the same `data_dir`; or call its front
with `resume=False`.

**What a resume does not change.**

- The jailbreak stage reuses its plan. A changed augmentation section, a new
  seed or new inputs take effect only with `"resume": false` for that stage.
  A resume logs a WARNING that counts the inputs its plan does not hold.
- A rejected unit is acked and not retried: a constitution unit or a seeded
  entry rejected on every attempt, a rejected output, a dropped paraphrase.
  Start the stage over to try them again.
- A file you edited stops the resume (see the table below).

## What lands on disk

A training run of all six stages, under `<data_dir>`:

```
Data_cache/
  constitution/
    harmful.csv  dual_use_harmful.csv  dual_use_benign.csv  benign.csv
    general_benign.csv                   # with include_standalone_benign
    merged.csv                           # every entry
    constitution.manifest.jsonl  constitution.state.jsonl
  benign/
    benign_samples.csv                   # benign Q&A pairs for fsh and dap
    benign_samples.manifest.jsonl  benign_samples.state.jsonl
Datasets/
  constitution_inputs/
    <category>/samples.csv               # one per category, accepted and rejected rows
    constitution_inputs.manifest.jsonl  constitution_inputs.state.jsonl
  constitution_inputs_merged.csv         # the accepted inputs, for build
  output_responses.csv                   # + .manifest.jsonl, .state.jsonl
  paraphrases_inputs.csv                 # + .manifest.jsonl, .state.jsonl
  paraphrases_outputs.csv                # + .manifest.jsonl, .state.jsonl
  jailbreaks.csv                         # + .manifest.jsonl, .state.jsonl, .coverage.csv
  variations.csv                         # + .manifest.jsonl, .state.jsonl, .coverage.csv
  complete_dataset.csv                   # the build
  constitution.run.json  inputs.run.json  outputs.run.json
  paraphrase.run.json  jailbreaks.run.json  build.run.json
  run-<YYYYMMDD-HHMMSS>.trace.jsonl      # one per run_pipeline call
```

An eval run with no constitution writes its inputs to
`Datasets/<category>/samples.csv` (no sidecars); with one, it writes them as
a training run does. Either way an eval build writes the paraphrases to
`Datasets/paraphrased.csv` in place of merging them.

Other files you may see:

| File | Meaning |
|---|---|
| `<ledger>.lock` | The stage's lock. The operating system holds it, so a run that died holds nothing; the file itself may stay |
| `<output>.cut` | Bytes a resume cut off an output: the tail of a save that did not finish |
| `<ledger>.dropping` | A fresh run of one seeded-inputs style that did not finish. Run it again |
| `*.tmp` | A file being replaced. It is gone once the write finishes |

A run record, `Datasets/<stage>.run.json`, holds `stage`, `timestamp`,
`params` (the params file's entry), `models` (the recipe's role names),
`counts` (`rows`, and `accepted` when the files have that column),
`sources` (the files the stage read its inputs from: the taxonomy file for
the constitution and standalone inputs, plus the seeds file without the
metaprompt; the saved constitution for seeded inputs; the input category
files, and `output_responses.csv` for paraphrases of outputs; the part files
of the build), `serving` (per role: name, setup, copies
or workers), and for the jailbreak stage `spec_version` and the resolved
`augmentation` section.

## Telemetry and cost

Every `run_pipeline` call writes a trace,
`Datasets/run-<YYYYMMDD-HHMMSS>.trace.jsonl` (UTC), one JSON event per line:
each model call, each engine load and release, each residency plan, each
stage's duration. `REDACT_TRACE=wandb` mirrors the events to Weights &
Biases (project `WANDB_PROJECT`, default `redact`), and `REDACT_TRACE=off`
writes nothing.

At the end of the run the driver logs a roll-up at INFO and returns it as
`summary["telemetry"]`:

| Key | Meaning |
|---|---|
| `models` | Per model: `calls`, `items`, `in` and `out` tokens, `ms`, and `cost_usd` for hosted calls of a priced model |
| `local_s` | Seconds each local checkpoint is held, from load to release |
| `local_billed_s` | The time any engine is held, overlaps counted once |
| `gpu` | The GPU name, the number of GPUs on the machine, the hourly rate used |
| `api_cost_usd` | Hosted tokens x the model's price per million |
| `local_cost_usd` | Billed time x rate x GPUs on the machine; `null` when the GPU is unpriced |
| `errors` | Failed calls recorded in the trace |
| `cut` | Per stage, per model: the replies cut at the token limit (`max_tokens` or the context), which are incomplete |

Hosted cost is per model, from `price_per_1m_input` and
`price_per_1m_output` in the registry. A model served locally is never priced
by its tokens. Local cost is the time an engine is alive, load and idle time
included, priced per GPU of the whole machine. The rate comes from, in order:
`telemetry.set_gpu_rate(...)`, `REDACT_GPU_HOURLY_USD`, then
`configs/llm/gpu_pricing.json` under `REDACT_GPU_PROVIDER` (default `local`,
priced at 0). A GPU the table does not list is reported as time only. The
table's prices are examples: in the GPU check the roll-up priced an A100 80GB
PCIe at the table's $1.19 an hour while the pod cost $1.79, so set your own
rate. [internals/telemetry.md](internals/telemetry.md) lists the events.

## Common failures

| What you see | What it means | What to do |
|---|---|---|
| `ValueError: No input samples found. Run generate_inputs() first.` | A stage that reads inputs found no category file under `data_dir` | Run `inputs` first, or point `data_dir` at the run that has them |
| WARNING `outputs skipped: no input sample was accepted ...` (or `paraphrase (...)`, `jailbreaks`) | No input is accepted; the stage writes nothing and records 0 rows. The run goes on | Read `rejection_reason` in the category files. A weak checker model rejects almost everything: in the GPU check a 3B checker accepted 2 of 144 seeded inputs, a 24B accepted 144 of 144 |
| WARNING `inputs: only N of M generated samples were accepted (under half)` | The checker rejects most samples | As above |
| `ValueError: No constitution at ...` | A run whose inputs are seeded has no saved constitution | List `constitution` before `inputs` |
| `RuntimeError: The run plans a model on a card another process holds memory on ...` | Before any stage: a card the plan uses is held, often by the engines of a run killed with `kill -9` | Kill the PIDs the message names and check `nvidia-smi` shows the card empty, or run on other cards (`CUDA_VISIBLE_DEVICES`) |
| `ValueError: benign_taxonomy '...' cannot be read` / `has no category` | At recipe load: the benign taxonomy a run of the jailbreak stage needs is missing or empty | Name a file in `taxonomy_dir`, or leave the key out for the packaged `benign` |
| `ValueError: No benign pairs at ...` | Every benign unit was rejected three times | Use a stronger `gen` model, then delete the ledger the message names |
| `KeyError: Model 'x' needs an API key, but the environment variable ... is not set` | A hosted model without its key | Add the key to `.env`, or serve the model locally |
| `KeyError: Model 'x' is not registered` | A name not in the registry | Fix the name, or add the model ([extending.md](extending.md#add-a-model)) |
| `ValueError: params['outputs'] names [...], which generate_outputs does not take` | A params key that is not a parameter of the front | Use a key from [configuration.md](configuration.md#params-keys-per-stage) |
| `ValueError: params['x'] sets [...], which the run passes ... itself` | The params file sets a key the recipe owns (`model`, `resume`, ...) | Set it in the recipe |
| `ValueError: augmentations.<kind> ...` | A bad augmentation section; the message names the key and what to write | Fix the section ([configuration.md](configuration.md#the-augmentation-section)) |
| `RuntimeError: ... is locked: another run, or this same process, is writing this stage's output` | Two runs on one output | Stop the other run, or use another `data_dir` |
| `RuntimeError: Cannot resume <file>: <ledger> recorded N bytes of it and the file is missing / is M bytes / holds other bytes there` | The output has been replaced, edited or copied incompletely since its last save | Restore the file; or delete the ledger, and the next resume takes over the rows in the file; or start the stage over |
| `RuntimeError: <file> is N bytes and <ledger> recorded M bytes after the last save: something else changed the file` | Another writer touched the output during the run | Stop the other writer, then resume |
| `RuntimeError: A fresh run of 'long' did not finish in ...` | A fresh run of one seeded-inputs style stopped half way | Run that style again with `resume: false` |
| ERROR `[residency] ...` with `PROBLEM:` lines | The local models of a stage do not fit the cards | Follow the plan's advice; the run continues and may run out of memory at load |
| ERROR `[preload] <model> failed to load` | A background load failed | The stage that needs the model fails when it calls it, with the cause in the log |
| `RuntimeError: The vLLM engine of ... was being built with N copies ...` | The engine failed to build with copies | Set `"copies": 1` for that role |
| vLLM logs `AsyncLLM output_handler failed` with `EngineDeadError` | An engine shutting down; logged at release and sometimes at exit | Nothing; REDACT logs a line saying so just before |
| INFO `[rate-limit] <model>: pausing Ns (N RPM)` | A hosted model is at its rate limit | Expected; serve the model locally for speed |
| WARNING `<model>: a reply was cut at the token limit (max_tokens=N) ...`, then `[<stage>] <model>: N replies were cut at the token limit and are incomplete.` | Replies hit `max_tokens` or the context: the first one per stage and model warns, the stage's end counts them, and `summary["telemetry"]["cut"]` holds the counts | Raise `default_max_tokens` or `max_model_len`, or ask for shorter replies |
| WARNING `... rejected, its prompt is too long` | One prompt is longer than the model's context; that unit gets a rejected row or ack | Raise `max_model_len`, or shorten the input |
| A stage stops on a provider error (for example 429, model overloaded) | A failed call is the transport failing, so it stops the stage; the SDK retries twice first | Resume: finished units are kept |
| WARNING `GPU cost reported as $0` | Local GPUs held, and no GPU price set | Set a rate (see above) |
| `DeprecationWarning` for flat augmentation keys | `include_*`, `sampling_probs` and the other flat keywords | Write them in the `adversarial` kind |
| `DeprecationWarning` naming `content_moderation_categories` or `content_moderation_seeds` | The former names of `harmful.json` in `configs/taxonomy/` and `configs/seeds/`, read for one release | Write `"taxonomy": "harmful"`; rename a file of your own `harmful.json` |
| WARNING `jailbreak: N of the M inputs are not in the plan ...` | A resume keeps the plan made before those inputs were accepted | Run the jailbreak stage with `"resume": false` to plan them |
| `ValueError: combination_spec.json and the technique registries differ` at import | A technique is registered without a spec entry, or the reverse | See [extending.md](extending.md#add-a-jailbreak-technique) |
