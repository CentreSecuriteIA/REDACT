# Running a pipeline

This page covers a run from the outside: installing, writing a recipe,
choosing models, what the GPU planner does, resuming, the files a run leaves,
what it cost, and the failures you are likely to meet. Every key is listed in
[configuration.md](configuration.md).

## Install

Python 3.11 or later.

```bash
pip install -e .                   # core: pandas, openai, anthropic, python-dotenv, pyfiglet, filelock
pip install -e ".[vllm]"           # local models through vLLM (vllm, hf_transfer, psutil, langid)
pip install -e ".[langid]"         # GlotLID, the language check of a machine translation and of a paraphrase
pip install -e ".[transformers]"   # the introspection backend (torch, transformers, accelerate)
pip install -e ".[hf]"             # loading published datasets from the HuggingFace Hub
pip install -e ".[dev]"            # pytest, ruff, mypy, ipykernel
pip install -e ".[docs]"           # the documentation site: mkdocs build --strict
```

`requirements.lock` pins an environment that runs vLLM on a GPU pod. Its
torch build needs a host driver for CUDA 13; on an older driver vLLM stops at
load with "The NVIDIA driver on your system is too old".

Put API keys in a `.env` file. `import redact` loads `.env` from the default
working root (`REDACT_OUTPUT_DIR`, else the first folder above the package
that holds `pyproject.toml` or `.git`, else the current directory), and its values win over the
shell's. The keys the shipped models read are `VENICE_API_KEY`,
`ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`. Local weights download to the
HuggingFace cache (`HF_HOME`); `scripts/download_models.py` fetches every
local checkpoint of the registry, whole (tokenizer files included, GGUF files
left out), at the revision an entry pins, ahead of a run. It fetches GlotLID,
the language identifier, with a model that has the `machine_translation` or
the `paraphraser` role. A model named as a recipe's paraphraser without that
role brings none: name a model with the role too, or the paraphrase check runs
without its language gate (one WARNING says so).

## Run a recipe

From the command line:

```bash
python scripts/run.py src/redact/configs/runs/training_example.json --data-dir ./runs/my_run
```

`--params` replaces the recipe's `params_file`, `--data-dir` its `data_dir`.
`--quiet` logs warnings only and `--debug` logs per-item detail. The script
prints the run summary as JSON. The first Ctrl-C, or `kill <pid>`, ends the
run through its cleanup, which saves the finished units and releases the
models. A second one ends vLLM's engine processes and exits at once.

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
`python scripts/inspect_run.py <data_dir>` prints one line per stage record:
its timestamp, counts and models.

To start a setup of your own, write a folder of recipes and the configs
they read:

```bash
python scripts/scaffold_setup.py ./my_setup
python scripts/run.py ./my_setup/eval_recipe.json
```

Each recipe there has `"config_dir": "."`: the folder's dataset type,
taxonomy, augmentation, generation axes and dimensions files are read in
place of the library's, file by file
([configuration.md](configuration.md#your-own-config-folder)). Delete every
file you do not change, so it keeps following the library.

The notebooks in `notebooks/` walk through a run step by step:
`eval_pipeline`, `training_pipeline`, `content_moderation` and
`jailbreak_augmentation`.

## Write a recipe

A recipe says what to run. Its `dataset_type`, `eval` or `training`, is
required: it names the packaged file that gives the run's default stages,
params and augmentation section
([configuration.md](configuration.md#dataset-types)), and the recipe and the
params file overlay it. A params file holds each stage's own settings, under
the stage's name.

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
  wrote. Without `stages` a run takes its dataset type's: all six,
  `constitution`, `inputs`, `outputs`, `paraphrase`, `jailbreaks`, `build`.
- Both dataset types add the general-benign part to the constitution:
  benign entries for each category of `benign_taxonomy`.
- An `eval` run generates only the `harmful` and `benign` entries and inputs,
  and its jailbreak stage attacks the harmful inputs with a small pool of hard
  techniques, in two rounds, with no natural variations; a `training` run
  takes every entry type and the packaged augmentation section. Anything the
  recipe or the params file gives wins.
- The `inputs` stage reads the constitution saved under `data_dir`, in an
  eval run as in a training one. List `constitution` before it, or run it
  earlier on the same `data_dir`; without either it stops with
  `No constitution at ...`.
- `taxonomy` names the taxonomy of the constitution (default `harmful`);
  `benign_taxonomy` the one whose categories the constitution's
  general-benign part is built from and whose subcategories the benign data
  of the `fsh` and `dap` techniques covers (default `benign`). Both are file
  names under `taxonomy_dir`, the packaged `configs/taxonomy/` by default.
- `params_file` and `config_dir` are relative to the recipe's folder.
  `data_dir`, `prompt_dir` and `taxonomy_dir` are relative to the current
  directory.
- `config_dir` names a folder of your own configs: a dataset type
  (`eval.json` or `training.json`), taxonomy, `jailbreak/augmentations.json`,
  `generation_axes.json` or `dimensions.json` file there is read in place of
  the packaged one; `taxonomy_dir`, when given, still wins for taxonomies.
- A params file holds one entry per stage, under the stage's name. Any
  other top-level key raises, except a comment key that starts with `_`.
  Inside a stage's entry every key must be a parameter of the stage's front,
  and not one the recipe passes itself (`model`, `data_dir`, `resume`, ...).
  These rules are checked for every stage before the first runs.
- `resume` is `true` or `false`. Any other value, the string `"false"` too,
  raises when the recipe loads.

The two example recipes under `src/redact/configs/runs/` each have a params
file: `eval_example.json` runs a small constitution (`params_eval_example.json`)
and leaves everything else to the eval type, `training_example.json` a
larger one (`params_training_example.json`). The eval params file sets no
`exclude_from`, since the training run it would name may not exist yet; its
`_comment` says what to add under `build` once it does
([build](stages/build.md#leaving-out-what-a-training-run-holds)).

## Choose models

A recipe names a model per **role**. A role left out, or `null`, takes the
default of its registry role: the first model in `configs/llm/models.json`
that lists that role.

| Recipe role | Called by | Registry role | Default today |
|---|---|---|---|
| `gen` | inputs, outputs, jailbreaks | `uncensored_gen` | `venice-uncensored`, hosted (Venice) |
| `check` | inputs, outputs, jailbreaks (what a technique's model writes; translation keeps its own checkers) | `uncensored_gen` | the `gen` model |
| `translation` | jailbreaks, only when a kind's pool holds a translation technique (`"translation": true`: the packaged attacks, and the eval type's) | `translation` | `translategemma-27b`, local vLLM (27B, FP8). Without a GPU, name the hosted `deepseek-v3.2`, or set `"translation": false` in the adversarial kind |
| `constitution` | constitution | `constitution_gen` | `claude-opus-4-6`, hosted (Anthropic) |
| `paraphraser` | paraphrase | `paraphraser` | `humanizer-12b`, local vLLM (12B, bf16) |
| `paraphrase_check` | paraphrase | `uncensored_gen` | `venice-uncensored`, hosted |

A key under `models` must be one of these six roles, whatever its value; a
key that starts with `_` is a comment. A role's entry is a registry name,
`null`, or an object that also says how the model is served:

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
| `copies` | vLLM only. Copies of the engine, one per card. Left out: a copy on each card with room; `1` turns copies off |
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
and the run goes on. So is a planning error, with its traceback; then every
local engine loads unplanned, at 0.9 of a whole card, so a second engine on
that card does not start. [internals/residency.md](internals/residency.md) has the
mechanics.

**Footprint.** Each local model's need is worked out without loading it. The
weights are the setup's `vram_gb` when set, else an estimate from the
checkpoint's HuggingFace config (the packaged copy under
`configs/llm/hf_configs/` when the setup has `hf_config`), else unknown. A
vLLM engine adds its KV cache
at its `max_model_len`, or at 10,000 tokens capped at the checkpoint's own
maximum. For a checkpoint whose attention shape varies by layer (Gemma 4,
whose full-attention layers have fewer, wider KV heads than its sliding
ones), the KV cache is summed layer by layer, every layer at the full
context, as vLLM allocates it; a config field that cannot be read (in
transformers 5.17 a per-layer attribute read globally raises) counts as
unset, so an estimate never fails a plan. The padded need per card is (weights + KV) x 1.1 + 0.6 GiB. Two
registry entries with the same checkpoint, quantization and engine kwargs
share one engine and are counted once (`venice-paraphraser` and
`venice-uncensored-fp8` do; the bf16 local setup of `venice-uncensored` is a
second load).

**Placement and grants.** The vLLM engines on a card split 0.9 of it: each
gets its padded need plus an equal share of what is left, and loads with that
fraction as its `gpu_memory_utilization`. A single-card engine goes to the
emptiest card with room for it; one with `min_gpus` above 1 takes whole
cards (`tensor_parallel_size`). Each card is planned at its own size. When
the models in the order given do not fit, the planner searches other
arrangements of cards.

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

**Copies.** When all of a stage's models fit together, the first vLLM
engine gets a copy on every other card with room for one (vLLM's
`data_parallel_size`), beside other models where they leave room, and not
on a card smaller than its own. A copy beside another model lowers the
engine's share on all its cards, and that model's share on that card. More
copies mean more requests in flight. A copy lengthens the load, so for a
short run set `"copies": 1`. When a build with copies fails, one copy is
loaded instead, with a WARNING.

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

The constitution stage is the exception: its ledger is plain, with no lock, no
recorded sizes and nothing cut back, and it saves each unit as it finishes. A
crash between a unit's rows and its ack makes the next run generate that unit
and append its rows again ([stages/constitution.md](stages/constitution.md#how-it-resumes)).

**Stopping a run.** Ctrl-C, or `kill <pid>` on `scripts/run.py`, ends the run
cleanly: the finished units are saved, requests in flight are cancelled, and
the local engines are released. A model still loading is waited for at most
60 seconds, then its engine processes are ended. Rerun the same command to
go on. A second Ctrl-C (or `kill`) while that cleanup runs ends the engine
processes at once: SIGTERM to vLLM's engine processes and their children,
then SIGKILL to those alive 5 seconds later; no other child process is
touched. It writes to stderr without logging, prints a `kill -9` line for
any process still alive, and exits. The next resume cuts off a save it left
half done.

`kill -9`, or a lost machine, may leave half a save. The next resume cuts it
off and runs those units again. vLLM's engine processes outlive `kill -9` and
keep their GPU memory; the next run stops before any stage. It names them,
their PIDs and the vLLM engine processes above them (titled `VLLM::`), with
one `kill -9` line for all of them (see [the GPU planner](#the-gpu-planner)).
Kill them and rerun.
The next load may log "Failed to reload cubin file" warnings from torch's
compile cache, and carries on.

In the GPU check, a jailbreak run of 432 units, killed with 32 saved, resumed
and ran the other 400. It ended with 432 rows, 433 ledger lines (the opening
`_saved` line and 432 acks), no duplicate, and an unchanged manifest.

**Starting over.** `"resume": false` starts every listed stage over: it
removes the stage's output, its `.cut` and `.tmp` files, its ledger and its
manifest (and for each jailbreak kind its `.coverage.csv`, `.plan.json` and
`.variants.csv`),
then plans anew. The exceptions:

- The `inputs` stage starts over only its current `style`. The rows and
  acks of other styles stay.
- The `constitution` stage removes every CSV in `Data_cache/constitution/`.

The benign data (`Data_cache/benign/benign_samples.csv`) is never cleared:
it always resumes from its ledger, generating only the units (one subcategory
of the benign taxonomy and one answer type) it does not hold. A unit rejected
on every attempt is acked and skipped; delete
`benign_samples.state.jsonl` to try it again, or delete the CSV and its
sidecars to generate everything again. With `"auto_generate_benign": false`
in the jailbreak stage's params, the file is loaded as it is; without one,
the stage stops before any unit runs.

To start one stage over and keep the others, run a recipe that lists only
that stage, with `"resume": false`, on the same `data_dir`; or call its front
with `resume=False`.

To plan one kind of the jailbreak stage again and keep the other, run a
recipe that lists only `jailbreaks`, with the kept kind `null` in
`augmentations` and `"resume": false`, then restore the section. A kind that
is off is not touched, and a resume of it afterwards goes on from its own
ledger and plan.

A stage that is skipped because none of its inputs was accepted still
starts over with `"resume": false`. Its earlier output, `.cut` and `.tmp`
files, ledger and manifest are removed through `dataset.clear_stage`; for the
jailbreak stage that is each kind that is on, with its `.coverage.csv`,
`.plan.json` and `.variants.csv`. So no stale rows reach the build. A jailbreak kind whose
`entry_types` match none of the inputs, or whose section leaves no technique
in its pool, is skipped with a WARNING and cleared the same way.

**What a resume does not change.**

- The jailbreak stage reuses each kind's plan. `<kind>.plan.json`, beside
  the manifest, records what the plan was made with: `seed`, `rounds` (with
  the budget keys and `iterations`), the pool's technique names,
  `family_weights`, `layer_rates`, `spec_version` (the spec version the
  pool was drawn at) and `rules` (a content id of the rules it was drawn
  by). A resume whose values differ stops
  with a `ValueError` that names the keys; run the stage with
  `"resume": false` to plan again. A pool that differs only by techniques
  the library added or retired after `spec_version` is the library's
  growth, not the run's change, and another `spec_version` or `rules` is
  the library's too: the plan is kept, with one WARNING that names both
  versions and what was added and retired, and counts the pending units
  whose combination today's rules refuse (the kept plan does not draw the
  new techniques, and runs those units as planned). Its rows record the
  plan's spec version. A plan without that file resumes unchecked, with a
  WARNING. Inputs accepted after the plan was made
  get no unit, and a WARNING counts them.
- A rejected unit is acked and not retried: a constitution unit or a seeded
  entry rejected on every attempt, a rejected output, a dropped paraphrase.
  Start the stage over to try them again.
- A file you edited stops the resume (see the table below).

**Folders and setups from before the answer modes of the outputs stage.**

- An outputs folder (`output_responses.*`) stops a resume before any call:
  its ledger acks units without an `iteration`, or its file lacks the
  `iteration` and answer-mode columns. Run the outputs stage with
  `"resume": false`, or in a new `data_dir`.
- The paraphrases of outputs stop the same way (acks without a
  `response_id`): run the paraphrase stage with `"resume": false` and
  `target: "outputs"`, which leaves the paraphrases of the inputs as they
  are.
- A prompt overlay of `output/generation/` or `output/quality_check/` is no
  longer read: each is now one prompt per answer mode, under `<mode>/`.
  `report_prompt_sources` warns `moved override: ...`; move the file to
  `harmful/`, and write a `helpful/` one if wanted.
- A `config_dir` written by `scaffold_setup` before the change holds a copy
  of the former `dataset_types/training.json`, which it reads in place of
  the packaged one: its training runs keep one response per input, with no
  answer of each mode for a borderline input, and nothing warns. Add
  `"outputs": {"outputs_per_input": 2}` to that file's `params`, or delete
  the file to follow the packaged one.
- `eval_outputs.csv`'s `label` is now `<entry_type>_<output_label>`, while
  the other eval files keep the entry types ([build](stages/build.md#eval-mode)).

## What lands on disk

A training run of all six stages, under `<data_dir>`:

```
Data_cache/
  constitution/
    harmful.csv  dual_use_harmful.csv  dual_use_benign.csv  benign.csv
    general_benign.csv                   # with include_general_benign
    merged.csv                           # every entry
    constitution.manifest.jsonl  constitution.state.jsonl
  benign/
    benign_samples.csv                   # benign Q&A pairs for fsh and dap
    benign_samples.manifest.jsonl  benign_samples.state.jsonl
Datasets/
  constitution_inputs/
    <category>/samples.csv               # one per category, accepted and rejected rows
    constitution_inputs.manifest.jsonl  constitution_inputs.state.jsonl
    constitution_inputs.plan.json  constitution_inputs.coverage.csv
  constitution_inputs_merged.csv         # the accepted inputs, for build
  output_responses.csv                   # + .manifest.jsonl, .state.jsonl, .coverage.csv, .plan.json
  paraphrases_inputs.csv                 # + .manifest.jsonl, .state.jsonl
  paraphrases_outputs.csv                # + .manifest.jsonl, .state.jsonl (target outputs or both)
  jailbreaks.csv                         # + .manifest.jsonl, .state.jsonl, .coverage.csv, .plan.json, .variants.csv
  variations.csv                         # + .manifest.jsonl, .state.jsonl, .coverage.csv, .plan.json, .variants.csv
  complete_dataset.csv                   # the build of a training run
  eval_inputs.csv  eval_outputs.csv  eval_jailbreaks.csv   # the build of an eval run
  eval_excluded.json                     # with exclude_from: the rows it left out
  constitution.run.json  inputs.run.json  outputs.run.json
  paraphrase.run.json  jailbreaks.run.json  build.run.json
  run-<YYYYMMDD-HHMMSS>.trace.jsonl      # one per run_pipeline call; -2, -3, ... when the name is taken
```

An eval run writes the same stage files; its build writes the three
`eval_*.csv` files in place of `complete_dataset.csv`
([stages/build.md](stages/build.md#eval-mode)).

Other files you may see:

| File | Meaning |
|---|---|
| `<ledger>.lock` | The stage's lock. The operating system holds it, so a run that died holds nothing; the file itself may stay |
| `<output>.cut` | Bytes a resume cut off an output: the tail of a save that did not finish |
| `<ledger>.dropping` | A fresh run of one seeded-inputs style that did not finish. Run it again |
| `<kind>.plan.json` | What a jailbreak kind's plan was made with: `seed`, `rounds`, `pool`, `family_weights`, `layer_rates`, `spec_version`, `rules`. A resume compares them with the run's, the pool against what the library added or retired after `spec_version`, and warns when `spec_version` or `rules` is not today's |
| `constitution_inputs.plan.json`, `output_responses.plan.json` | What the deal of the generation axes was made with: `seed`, the axes' weights, for the inputs `samples_per_entry`, and for the outputs `outputs_per_input` and `modes` (the answer modes of every entry type). A resume compares them with the run's |
| `constitution_inputs.coverage.csv`, `output_responses.coverage.csv` | Per axis value, the specs planned and accepted per category and entry type; the outputs' also per answer mode (`axis` `mode`) |
| `<kind>.coverage.csv` | Per layer, family and technique of a kind's plan, the count per category, entry type and iteration, with a zero row for each pool technique never drawn |
| `<kind>.variants.csv` | Per template technique and variant, the rows of a kind's CSV that drew it, written after each run, with a zero row for each variant of the pool never drawn ([stages/jailbreaks.md](stages/jailbreaks.md#files)) |
| `*.tmp` | A file being replaced. It is gone once the write finishes; a kill can leave one, which the next write replaces, and `resume: false` removes the output's |

A run record, `Datasets/<stage>.run.json`, holds `stage`, `timestamp`,
`params` (the stage's params: the dataset type's with the params file's
entry over them), `models` (the recipe's role names),
`counts` (`rows`, and `accepted` when the files have that column; the inputs
stage counts only the accepted rows of its `style`; an eval build with
`exclude_from` adds `excluded`; the paraphrase stage adds `skipped`: the units
under `min_words` words, per reason), `sources` (the files the stage
read its inputs from: the taxonomy file for the constitution, and the benign
taxonomy's with the general-benign part on; the saved constitution for the
inputs; the input category files, and `output_responses.csv` for paraphrases
of outputs; the part files of the build, and the files `exclude_from`
names), `serving` (per role the stage calls: `name`, `backend_type`,
and `copies` for a vLLM model or `workers` for a hosted one; `copies` is the
count the plan made before any stage ran, and when a build with copies fails
and one copy loads instead, the trace's `load` event says so with `fallback`
and `planned_copies`), `dataset_type` and `dataset_type_hash` (the content
id of the type file the run read), with a `config_dir` that folder and
`config_files` (the files read from it), and for the jailbreak stage
`spec_version` and the resolved `augmentation` section.

## Telemetry and cost

Every `run_pipeline` call writes a trace,
`Datasets/run-<YYYYMMDD-HHMMSS>.trace.jsonl` (UTC; `-2`, `-3`, ... after the
time when two runs start in one second), one JSON event per line:
each model call, each engine load and release, each residency plan, each
stage's duration. `REDACT_TRACE=wandb` mirrors the events to Weights &
Biases (project `WANDB_PROJECT`, default `redact`; `pip install wandb`, which
no extra brings), and `REDACT_TRACE=off` writes nothing and leaves
`summary["telemetry"]` empty.

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
| `ValueError: No input samples found. Run generate_inputs() first.` | `generate_outputs` or `generate_jailbreaks` found no category file under `<data_dir>/Datasets/constitution_inputs/` (`generate_paraphrases` skips the target at INFO instead) | Run `inputs` first, or point `data_dir` at the run that has them |
| WARNING `outputs skipped: no input sample was accepted ...` (or `paraphrase (...)`, `jailbreaks`) | No input is accepted; the stage writes nothing, and the run goes on. With `"resume": false` it also removes what an earlier run of the stage left, so its record counts 0 rows; on a resume its earlier rows stay | Read `rejection_reason` in the category files. A checker's reply accepts only when its first word is `yes` (or `ok`, `accept`, `pass`), so a checker that opens with anything else rejects. A weak checker model rejects almost everything: in a GPU check made before the checker prompts stated the verdict as the reply's first word, a 3B checker accepted 2 of 144 seeded inputs, a 24B accepted 144 of 144 |
| WARNING `inputs: only N of M generated samples were accepted (under half)` | The checker rejects most samples | As above |
| `ValueError: No constitution at ...` | A run whose inputs are seeded has no saved constitution | List `constitution` before `inputs` |
| `RuntimeError: The run plans a model on a card another process holds memory on ...` | Before any stage: a card the plan uses is held, often by the engines of a run killed with `kill -9` | Kill the PIDs the message names and check `nvidia-smi` shows the card empty, or run on other cards (`CUDA_VISIBLE_DEVICES`) |
| `ValueError: benign_taxonomy '...' cannot be read` / `has no category` | At recipe load: the benign taxonomy the constitution's general-benign part or the jailbreak stage's benign data needs is missing or empty | Name a file in `taxonomy_dir` or the `config_dir`'s `taxonomy/`, or leave the key out for the packaged `benign` |
| `ValueError: The recipe names no dataset_type` / `dataset_type must be "eval" or "training", got ...` | At recipe load: `dataset_type` is required and is `eval` or `training`; a file of another name in the `config_dir`'s `dataset_types/` is no new type | Write `"dataset_type": "eval"` or `"training"`; to change what one means, edit its file in the `config_dir` |
| `FileNotFoundError: exclude_from names ...` | An eval build's `exclude_from` names a path that does not exist, or a folder with no `Datasets/*.csv` and no constitution; nothing is written | Name the training run's `data_dir` or one of its CSVs |
| `ValueError: No benign pairs at ...` | Every benign unit was rejected three times | Use a stronger `gen` model, then delete the ledger the message names |
| `KeyError: Model 'x' needs an API key, but the environment variable ... is not set` | A hosted model without its key | Add the key to `.env`, or serve the model locally |
| `ImportError: The 'vllm' package is required for VLLMBackend` | A stage calls a local model and the `vllm` extra is not installed. The role defaults for `translation` (when a kind translates) and `paraphraser` are local | Install the extra. On a machine without a GPU, name `deepseek-v3.2` for `translation` (or set `"translation": false` in the adversarial kind) and take `paraphrase` out of `stages` |
| WARNING `jailbreaks: <model> is a machine-translation model and GlotLID is not available ...` | The `langid` extra, or GlotLID's model file, is missing, so the language of the translations is not checked (the other checks run) | `pip install -e ".[langid]"`, then `python scripts/download_models.py` |
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
| WARNING `[vllm] The engine of ... failed to build with N copies ...; loading one copy on ... instead` | The engine failed to build with copies | The run goes on with one copy; set `"copies": 1` for that role to skip the attempt |
| `RuntimeError: The vLLM engine of ... was being built with N copies ...` | The engine failed to build with copies, and one copy failed too or was not tried | The message says what failed; set `"copies": 1` for that role |
| ERROR `[residency] the preload of ... has not finished ...` or `[vllm] The load of ... has not returned` | A model load hung, and the run is ending | Its engine processes are ended; check `nvidia-smi` |
| vLLM logs `AsyncLLM output_handler failed` with `EngineDeadError` | An engine shutting down; logged at release and sometimes at exit | Nothing; REDACT logs a line saying so just before |
| INFO `[rate-limit] <model>: pausing Ns (N RPM)` | A hosted model is at its rate limit | Expected; serve the model locally for speed |
| WARNING `<model>: a reply was cut at the token limit (max_tokens=N) ...`, then `[<stage>] <model>: N replies were cut at the token limit and are incomplete.` | Replies hit `max_tokens` or the context: the first one per stage and model warns, the stage's end counts them, and `summary["telemetry"]["cut"]` holds the counts | Raise `default_max_tokens` or `max_model_len`, or ask for shorter replies |
| The same WARNING for the paraphraser, in the `paraphrase` stage | Expected: each paraphrase is capped at about 2.5 times its original's tokens (`reply_cap`), so a reply that runs on into a post is cut there. With the check on, the length gate usually rejects it (a cut reply in Chinese or Japanese can pass the gate and go to the check model) | Nothing, unless many are cut: then read the rejected rows' `reasoning` |
| WARNING `Entry ... rejected, its prompt is too long` (inputs) or `Input ... rejected, its prompt is too long` (outputs) | One prompt is longer than the model's context. On the async lane (an OpenAI-compatible endpoint or vLLM) the inputs stage acks the entry rejected and the outputs stage writes a rejected row; the jailbreak stage writes an error row for that unit | Raise `max_model_len`, or shorten the input |
| ERROR `[residency] planning failed (...)`, then an engine fails with `Free memory on device ... is less than desired GPU memory utilization` | The planner raised, so every local engine loaded unplanned at 0.9 of a card, and the second engine on that card had no room | Read the traceback after the ERROR and report it; meanwhile serve one local model per card, or a hosted model for one role |
| `PromptTooLongError` stops a stage | A prompt longer than the context of an Anthropic or introspection model, in any stage (grouped calls; the failed `generate()` stops the stage). The constitution stage and the benign data stop on any model, since they do not handle it per unit; on the async lane the paraphrase stage acks the unit `prompt_too_long` | Shorten the input, or serve that role on a model with a longer context |
| A stage stops on a provider error (for example 429, model overloaded) | A failed call is the transport failing, so it stops the stage; the SDK retries it `max_retries` times first (2 by default) | Resume: finished units are kept; raise `max_retries` or `timeout_s` on the model's `api` setup |
| WARNING `GPU cost reported as $0` | Local GPUs held, and no GPU price set | Set a rate (see above) |
| `DeprecationWarning` for flat augmentation keys | `include_*`, `sampling_probs` and the other flat keywords | Write them in the `adversarial` kind |
| `DeprecationWarning` naming `content_moderation_categories` | The former name of `harmful.json` in `configs/taxonomy/`, read for one release | Write `"taxonomy": "harmful"`; rename a file of your own `harmful.json` |
| WARNING `jailbreak: N of the M inputs are not in the plan ...` | A resume keeps the plan made before those inputs were accepted | Run the jailbreak stage with `"resume": false` to plan them |
| `ValueError: jailbreak: this run's [...] differ from those the plan jailbreaks.manifest.jsonl was made with (see jailbreaks.plan.json) ...` | A resume with another augmentation section or seed than the plan's; also an adversarial plan made before the packaged attacks dealt a request obfuscation, whose rounds now carry `allow_request_obfuscation` (the differing key is `rounds`) | Run the jailbreak stage with `"resume": false`, or with the plan's section and seed (`"allow_request_obfuscation": false` in the adversarial kind for a plan made before) ([release notes](configuration.md#release-notes)) |
| WARNING `jailbreak: since the plan jailbreaks.manifest.jsonl was made (spec version ..., now ...), the library added [...] and retired [...] ...` | The library added or retired techniques after the plan was made; the kept plan does not draw the added ones | Nothing, or `"resume": false` to plan again and draw them |
| `DeprecationWarning`: `techniques.allow names '...', which spec version ... deprecated` | A kind's `techniques.allow` names a retired technique, which it still draws for this release | Take the name out of `techniques.allow` |
| WARNING `jailbreak: the plan ... has no jailbreaks.plan.json beside it ...` | A plan made before plan files were written; its section and seed are not checked | Nothing, or `"resume": false` to plan again |
| `ValueError: .../output_responses.state.jsonl acks units without an iteration ...` or `Cannot append to .../output_responses.csv: its header has no column 'iteration', 'response_mode', ... It was written before an input had several responses, each in an answer mode.` | A resume onto an outputs folder written when a unit was an input, before the answer modes | Run the outputs stage with `"resume": false`, or in a new `data_dir` |
| `ValueError: Outputs: this run's ['outputs_per_input'] differ from those the responses in output_responses.csv were dealt with ...` (or `['modes']`) | A resume with another `outputs.outputs_per_input` or `outputs.modes` (or seed, or axes' weights) than the plan file's; a training folder of one response per input resumed with the type's 2 is one | Run the outputs stage with `"resume": false`, or with the plan's settings |
| `ValueError: Outputs: N input(s) have no entry type (a missing, NaN or empty entry_type cell), first [...] ...` | Inputs given to the outputs stage with blank `entry_type` cells, such as a merged frame or a hand-edited `samples.csv`; raised before anything is cleared | Give each input one of the four entry types (a frame without the column at all is answered as `harmful`, with a WARNING) |
| `ValueError: Outputs: the inputs hold the entry type(s) [...], which outputs.modes does not map ...` | An input whose `entry_type` is none of the four; raised before anything is cleared | Fix the inputs' entry types |
| WARNING `moved override: output/generation` (or `output/quality_check`) | A prompt overlay of the single output prompt or checker; the outputs stage reads one of each per answer mode | Move it to `output/generation/harmful/` (or `output/quality_check/harmful/`), and write `helpful/` if wanted |
| A training run writes one response per input | Its recipe's `config_dir` holds a `dataset_types/training.json` scaffolded before `outputs.outputs_per_input` 2 | Add `"outputs": {"outputs_per_input": 2}` to that file's `params`, or delete the file |
| WARNING `paraphrase: .../paraphrases_outputs.csv holds paraphrases of responses from an earlier run, and the build merges them ...` | The paraphrase stage ran on the inputs only (the default), and an earlier run of the outputs target left its file | Delete the file, or run the paraphrase stage with `target: "outputs"` and `"resume": false` to paraphrase the current responses |
| `ValueError: .../paraphrases_outputs.state.jsonl acks units without a response_id ...` or `Cannot append to .../paraphrases_outputs.csv: its header has no column 'response_id' ...` | A resume of the outputs target onto paraphrases written when its units were keyed on the input answered | Run the paraphrase stage with `"resume": false` and `target: "outputs"` (the inputs' paraphrases stay), or in a new `data_dir` |
| WARNING `Outputs: rows without a response_label, written before the answer modes, take their entry_type as their label in eval_outputs.csv.` | An eval build from an output or paraphrase file written before the answer modes | Run the outputs (and paraphrase) stage again with `"resume": false` for the compound labels |
| `ValueError: variation: this run's ['pool'] differ from those the plan variations.manifest.jsonl was made with ...` | A natural plan made before the spec's `kinds` narrowed the natural pool: it names techniques the natural kind no longer draws | Run the natural kind with `"resume": false` ([techniques.md](techniques.md#the-natural-pool)) |
| `ValueError: augmentations.natural.families.allow names 'structural', which the natural kind does not draw ...` | A `config_dir` copy of `augmentations.json` from before; its natural `families.allow` names `structural`, `temporal`, `continuation` | Take them out of that list |
| INFO `configs: the natural kind of jailbreak/augmentations.json in ... does not allow ..., which the library's allows ...` | A `config_dir` copy of `augmentations.json` made before the library allowed those natural families; its variations do not draw them | Add them to the copy's natural `families.allow`, delete the copy, or `scaffold_setup(..., overwrite=True)`, which replaces your edits ([configuration.md](configuration.md#your-own-config-folder)) |
| WARNING `configs: the eval attacks of ... draw N technique(s) the library's eval type does not: ...` | An eval run whose `config_dir` holds a copy of `dataset_types/eval.json` made before the library added those techniques to a family it allows (21 for a copy of the previous release); the eval pool is wider than the held 36 | Delete the copy, `scaffold_setup(..., overwrite=True)`, which replaces your edits, or add the names to its adversarial `techniques.deny` ([configuration.md](configuration.md#release-notes)) |
| WARNING `Variations: left out N benign or dual-use benign rows made with a technique the natural kind does not draw (...)` | `variations.csv` holds rows of a natural plan made before the `kinds` gate; the build leaves the benign ones out | Run the natural kind with `"resume": false` and build again |
| `ValueError: The plan names the technique '...', which no registered technique implements` | A kept plan names a technique that was renamed or removed | Run the jailbreak stage with `"resume": false` |
| `ValueError: Benign data not found and no client provided for generation ...` | `auto_generate_benign` is false and there is no benign file | Run once with it on, or set `benign_path` |
| WARNING `adversarial kind skipped: entry_types [...] match none of the inputs' (...)` | No input of the kind's `entry_types` | Fix `entry_types` in the augmentation section |
| WARNING `adversarial kind skipped: its section leaves no technique to draw (...)` | The kind's `layers`, `families`, `techniques`, `pure_only` and `translation` leave its pool empty | Allow a layer, family or technique in the augmentation section |
| WARNING `jailbreak: N input rows share their id with another (...)` | Two input rows hold one text without ids, or one id given twice | Nothing to do (the last row of each id is kept), or drop the copies; under two entry types the stage raises instead |
| `ValueError: The params file has key(s) [...], which are not stages ...` | A top-level params key that is not a stage name, such as a typo | Write only stage names, or a comment key starting with `_` |
| `ValueError: resume must be true or false, got 'false'.` | `resume` is not a JSON bool | Write `true` or `false` |
| `ValueError: models['x'] is not a model role; write one of [...]` | A key under `models` that is no recipe role | Use one of the six roles ([Choose models](#choose-models)) |
| `ValueError: combination_spec.json and the technique registries differ` at import | A technique is registered without a spec entry, or the reverse | See [extending.md](extending.md#add-a-jailbreak-technique) |
