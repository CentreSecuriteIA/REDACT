# REDACT — Red-team Dataset Automation & Construction Toolkit

A modular Python library for generating, validating, and managing synthetic red-teaming datasets. Built for content moderation and jailbreak research, but designed to be extensible to any synthetic data generation task.

## Overview

REDACT automates the full lifecycle of red-teaming dataset construction:

1. **Constitution** — generate structured category hierarchies (harmful, benign, dual-use) using Claude Opus
2. **Generate** harmful content samples across configurable harm categories
3. **Validate** each sample via a checker LLM with feedback-driven retry
4. **Transform** inputs into jailbreak attacks using 152 techniques, dealt evenly across categories, layers and families
5. **Merge** every stage's rows into one dataset, resumable at each stage

The library is **model-agnostic** (API or local vLLM), **prompt-agnostic** (all prompts are external JSON files), and **category-agnostic** (new categories require only a taxonomy entry and prompt file).

---

## Installation

```bash
pip install -e .             # core (Venice API, pandas, openai)
pip install -e ".[anthropic]"  # + Anthropic Claude support
pip install -e ".[vllm]"       # + local vLLM inference
pip install -e ".[dev]"        # + pytest, ruff, mypy
```

Requires Python 3.11+. See [pyproject.toml](pyproject.toml) for full dependency list.

---

## Quick Start

**High-level API** — the common path. Every function takes one `data_dir` working root (all
`Datasets/` and `Data_cache/` output lands under it); models default to their registry **role**
(pass `None`), and a model name resolves to a ready-to-call `ModelClient`. A single `resume` flag
(default `True`) makes every stage restartable.

```python
from redact import (
    generate_constitution, generate_inputs, generate_outputs,
    generate_jailbreaks, build_dataset,
)

DATA_DIR = "./runs/training"                       # one root for the whole run
constitution = generate_constitution(data_dir=DATA_DIR, num_taxonomy_categories=3)
inputs = generate_inputs(data_dir=DATA_DIR, constitution_df=constitution, samples_per_entry=3)
outputs = generate_outputs(data_dir=DATA_DIR, inputs=inputs)         # responses + output checker
jailbreaks = generate_jailbreaks(data_dir=DATA_DIR, inputs=inputs)   # jailbreaks + natural variations
dataset = build_dataset(data_dir=DATA_DIR)                           # merge everything on disk
```

For a cheap eval set without a constitution, `generate_inputs(data_dir=DATA_DIR, samples_per_category=15, num_categories=3)`
runs the standalone meta-prompt path instead. It is **deprecated** (emits `DeprecationWarning`) but functional;
constitution-seeded inputs give better and more adjustable coverage.

`generate_jailbreaks` writes two kinds of augmentation: **attacks** (`jailbreaks.csv`) and **natural
variations** (`variations.csv`), both on by default. One `augmentation=` section says what each draws;
a kind you give overlays its packaged default, and `None` switches it off:

```python
jailbreaks = generate_jailbreaks(data_dir=DATA_DIR, inputs=inputs, augmentation={
    "adversarial": {"entry_types": ["harmful"], "rounds": "escalation"},  # 4 rounds per input
    "natural": None,                                                     # no variations
})
```

See [The augmentation section](#the-augmentation-section) for every key.

**Paraphrase / fingerprint removal** — `generate_paraphrases(data_dir=DATA_DIR, target="both")` adds reworded ("defingerprinted") copies of the base inputs and *accepted* base outputs as **additive rows** (`paraphrases_inputs.csv` / `paraphrases_outputs.csv`). A **separate** meaning-preservation checker drops paraphrases that don't preserve meaning (check→drop), duplicates are deduped, and each artifact gets a `*.manifest.jsonl` plan + `*.state.jsonl` resume ledger like every other stage (see [Resume model](#resume-model-one-modular-sidecar-system)). `build_dataset(mode="training")` merges the paraphrases into the dataset while `mode="eval"` writes a separate `paraphrased.csv`.

> **⚠️ Disclaimer — placeholder paraphraser.** The actual defingerprinting / paraphraser model is trained in a separate repository (see [Out of Scope](#out-of-scope)) and is **not shipped here**. The `paraphraser` role is a stand-in that loads the same Dolphin-Mistral-24B weights as `venice-uncensored`'s local setup; register your real paraphraser with `register_model(..., role="paraphraser")` for production fingerprint removal.

**Config-driven runs** — define a whole run in a **recipe** JSON and drive it with one call:

```python
from redact import run_pipeline

summary = run_pipeline("src/redact/configs/runs/eval_example.json")
# or from the CLI:  python scripts/run.py src/redact/configs/runs/eval_example.json --data-dir ./runs/eval
```

A recipe names:

| Key | Meaning |
|---|---|
| `dataset_type` | `"eval"` (standalone inputs, no constitution) or `"training"` (inputs seeded from the constitution saved under `data_dir`) |
| `data_dir` | The working root (`--data-dir` or `run_pipeline(data_dir=...)` overrides it) |
| `stages` | In order, from `constitution`, `inputs`, `outputs`, `paraphrase`, `jailbreaks`, `build`; default `["inputs", "outputs", "jailbreaks", "build"]` |
| `models` | Per role (`gen`, `check`, `translation`, `constitution`, `paraphraser`, `paraphrase_check`): a registry name, `null` for the role's default, or `{"name": ..., "backend_type": ..., "copies": ..., "workers": ...}` to also say how it is served |
| `taxonomy`, `taxonomy_dir`, `prompt_dir` | The taxonomy by name, and where to read taxonomies and prompt overrides from |
| `augmentations` | The jailbreak stage's [augmentation section](#the-augmentation-section): `{"adversarial": {...}, "natural": {...}}`, each kind overlaying its default, `null` to switch one off |
| `params_file` | A JSON of each stage's own parameters under the stage's name (the keyword arguments of its `generate_*` function), relative to the recipe; `--params` overrides it |
| `resume` | `true` (default) resumes every stage; `false` starts each over |

An unknown recipe key, a params key that is not a parameter of its stage, and a bad augmentation
section all fail before any stage runs. Each stage writes a
`{stage}.run.json` run record under `{data_dir}/Datasets/` (params, models, how they were served,
row counts); `python scripts/inspect_run.py DATA_DIR` prints them. When the inputs stage accepted no
sample, the outputs, paraphrase and jailbreak stages each log a warning, write nothing and record 0
rows, and the run goes on; the inputs stage itself warns, with the counts, when it accepts under half
of what it generated. Local models are planned, loaded
and unloaded stage by stage (see [Telemetry, cost, and GPU planning](#telemetry-cost-and-gpu-planning)).

See `notebooks/eval_pipeline.ipynb` and `notebooks/training_pipeline.ipynb` for the full walkthroughs.

### Resume model (one modular sidecar system)

Every batched stage plans and resumes through **one shared mechanism** — two sidecar
JSONL files next to the stage's output CSV, both built on a single
`redact.dataset.JsonlSidecar` foundation (path/mkdir/reset/bad-line-robust reads):

- a **`<artifact>.manifest.jsonl`** (`redact.dataset.Manifest`) — the **plan**: one JSON
  object per unit, written *before* generation so a run's full intended scope is
  inspectable up front. Planning is idempotent (overwrite).
- a **`<artifact>.state.jsonl`** (`redact.dataset.Ledger`) — the **resume state**: one
  object per *completed* unit, appended **only after** that unit's rows are flushed to
  CSV, so a crash loses at most one chunk and resume never trusts a large/hand-edited CSV.

`resume=True` (default) skips ledger-recorded units; `resume=False` clears the artifact,
its ledger, **and** its manifest, then restarts. Each stage keys both files on the *same*
stable unit id — and the multi-turn stages use the **same vocabulary as the rest of the
library** (`input_id` for the source row, `iteration` for the per-source index), so every
derived artifact joins, dedups, plans, and resumes identically:

| Stage | Manifest | Ledger | Unit key |
|---|---|---|---|
| Constitution generation | `constitution.manifest.jsonl` | `constitution.state.jsonl` | `source_category::entry_type` |
| Content-mod input (from constitution) | `constitution_inputs.manifest.jsonl` | `constitution_inputs.state.jsonl` | `(sample_description, entry_type, style)` |
| Content-mod output | `output_responses.manifest.jsonl` | `output_responses.state.jsonl` | `input_id` |
| Paraphrase | `paraphrases_*.manifest.jsonl` | `paraphrases_*.state.jsonl` | `(input_id, iteration)` |
| Jailbreaks | `jailbreaks.manifest.jsonl` | `jailbreaks.state.jsonl` | `(input_id, iteration)` |
| Natural variations | `variations.manifest.jsonl` | `variations.state.jsonl` | `(input_id, iteration)` |
| Conversations | `conversations.manifest.jsonl` | `conversations.state.jsonl` | `(input_id, iteration)` |
| Conversation scoring | — (scores existing rows) | `conversations_scored.state.jsonl` | `sample_id` |

A resumed run reads the ledger alone: it checks each output file against the size and checksum
the ledger recorded at its last save and cuts off what an unfinished save left behind. Plan-backed
stages (jailbreaks, conversations) keep their manifest on a resume and stream their units from it.
The one intentional exception is **standalone meta-prompt input generation** — it has no discrete
idempotent units (it generates open-endedly toward a per-category sample count), so it gets
neither a manifest nor a ledger and resumes by extending its CSVs. That path is now
**deprecated** (emits `DeprecationWarning`) in favour of constitution-seeded generation, but
remains fully functional.

**Lower-level building blocks:**

```python
# 1. Auto-select backend from model name
from redact.llms import ModelClient, load_prompt

client = ModelClient.create("venice-uncensored")  # ready to call (via VENICE_API_KEY)

# 2. Expand constitution entries into checked input samples
from redact.content_moderation import InputPipeline

pipeline = InputPipeline(gen=client, check=client,
                         dataset_dir="./runs/eval/Datasets/constitution_inputs")
result = pipeline.run_from_constitution(
    constitution_df=constitution,                     # from generate_constitution()
    prompt_config=load_prompt("input", "generation/from_constitution/long"),
    samples_per_entry=3, style="long",
)
print(f"{result.total_prompts_accepted} accepted samples")

# 3. Apply jailbreak techniques
from redact.jailbreak import apply_combination, combine_techniques
from redact.jailbreak.hacking.cognitive import to_persona_roleplay
from redact.jailbreak.obfuscation.encoding import to_base64

# Pure technique (no LLM) — plain (str) -> (text, info)
obfuscated, info = to_base64("How to pick a lock")

# LLM-dependent techniques are generators (they yield LLMRequests). Drive one
# prompt through one or more synchronously with apply_combination():
text, info, accepted, reasoning = apply_combination(
    combine_techniques(to_persona_roleplay, to_base64), "How to pick a lock",
    client=client,
)
```

> LLM-dependent techniques (cognitive, personas, translation, typos, tokenbreak, FSH/DAP-selected) are **technique generators** — calling them directly returns a generator, not a result. Use `apply_combination()` for one sample, or let `generate_jailbreaks()` drive them in batches. Pure transforms (encoding, structural, suffixes, ascii_art, casual_writing, framing, all `requests/`) are plain callables.

See [`notebooks/`](notebooks/) for walkthroughs: `eval_pipeline`, `training_pipeline`, `content_moderation`, `jailbreak_augmentation`.

---

## Architecture

```
src/redact/
├── llms/                          # Model-agnostic LLM abstraction
│   ├── backends/                  # One file per provider, each a *configured model*
│   │   ├── base.py                # Abstract LLMBackend + ComputeConfig capability flags
│   │   ├── capabilities.py        # BACKEND_TYPES + resolve_setup/transport_for/backend_for
│   │   ├── openai.py              # Generic OpenAI-compatible API backend (Venice today)
│   │   ├── anthropic.py           # Anthropic Claude backend (native SDK)
│   │   ├── vllm.py                # Local vLLM backend: AsyncLLM in-process, one engine request per item
│   │   └── introspection.py       # Local transformers backend with internals capture
│   ├── observe.py                 # Telemetry emit hook (no-op by default; keeps llms/ dep-free)
│   ├── aio.py                     # One event loop on a daemon thread for every async model call
│   ├── client.py                  # ModelClient: a configured backend + its dispatch strategy
│   ├── router.py                  # generate_sample(), batch_generate_samples(), assert_single_sample_per_call()
│   ├── wrappers.py                # RateLimiter + BatchCaller, both wired from a backend
│   ├── prompting/                 # load_prompt, PromptTemplate, reply extraction, scaffold_prompt_tree
│   └── model_config.py            # Model registry (RPM, backend_type, capability flags, roles)
│
├── llm_pipeline/                  # LLMRequest; drive_sync / drive_generators / drive_units; checks and retries
├── residency/                     # Which local models fit, per stage; preload and unload
│
├── constitution/                  # Constitution generation for classifiers
│   ├── generation.py              # ConstitutionPipeline (4 severity types), read_constitution
│   ├── units.py                   # One call per (taxonomy category, entry type): its messages and rows
│   └── parse.py                   # The [END] marker and the 3-layer markdown parser
│
├── content_moderation/            # Content moderation stages
│   ├── inputs.py                  # InputPipeline.run_from_constitution — the seeded inputs stage
│   ├── standalone.py              # The deprecated standalone inputs path (run_standalone_inputs)
│   ├── outputs.py                 # run_output_generation — one checked response per input
│   ├── paraphrase.py              # Paraphrase stage (placeholder paraphraser)
│   ├── checker.py                 # Input, output and paraphrase checkers
│   ├── feedback.py                # Rejection feedback and the do-not-repeat note
│   ├── metaprompt.py              # Category description + seed generation (standalone path)
│   └── results.py                 # Result classes
│
├── jailbreak/                     # Jailbreak technique library
│   ├── obfuscation/               # Text transformation attacks
│   │   ├── encoding.py            # base64, rot13/18/47, unicode, ordinal, separator, leetspeak, morse, braille
│   │   ├── structural.py          # JSON, XML, markdown wrapping
│   │   ├── ascii_art.py           # pyfiglet-based text art (16 fonts)
│   │   ├── suffixes.py            # Adversarial suffix generators
│   │   ├── tokenbreak.py          # Token-breaking of the words a model marks harmful (LLM-dependent)
│   │   ├── sensitive_words.py     # Encoding/splitting/substitution of those words (LLM-dependent)
│   │   ├── typos.py               # LLM-rewritten typos at 4 density levels
│   │   ├── casual_writing.py      # Casing, spacing, punctuation, abbreviations, accents
│   │   └── translation.py         # 20 languages across resource tiers
│   ├── hacking/                   # Cognitive/psychological manipulation
│   │   ├── cognitive.py           # 5 techniques (persona, framing, AVI, authority, inception)
│   │   ├── personas.py            # 14 named persona archetypes + invented persona
│   │   └── framing.py             # 5 scenario-modifying directives (pure transforms, multi-template)
│   ├── manipulation/              # Context manipulation with benign examples
│   │   ├── benign.py              # Benign sample generation + caching
│   │   ├── pairs.py               # Subcategory and pair selection both families share
│   │   ├── fsh.py                 # Few-Shot Hacking (4 variants)
│   │   └── dap.py                 # Distract and Persuade (4 variants)
│   ├── requests/                  # Request-structure attacks (all pure transforms)
│   │   ├── answer.py              # 9 output-format + conditioning directives
│   │   ├── answer_language.py     # 20 ask-answer-in-language functions (generated from JSON)
│   │   ├── continuation.py        # 4 continuation-attack functions
│   │   ├── indirect.py            # 6 task-embedding functions
│   │   ├── distractor.py          # 4 distractor prefix/suffix functions
│   │   ├── impersonation.py       # 1 good-person impersonation function
│   │   ├── temporal.py            # 1 past-tense reframing function
│   │   └── asking.py              # 2 question-framing functions
│   ├── technique.py               # The technique contract, the spec and its tags
│   ├── pool.py                    # The techniques a run may draw from
│   ├── rules.py                   # The spec's compatibility rules
│   ├── sampling.py                # Draw one combination: by budget or by exact count
│   ├── plan.py                    # plan_run (dealt evenly per group), coverage table, escalation schedule
│   ├── chain.py                   # Run one input through its techniques; combine_techniques, apply_combination
│   ├── units.py                   # A planned unit as the driver runs it, and its row
│   ├── pipeline.py                # run_augmentation — the stage
│   ├── augmentation.py            # The augmentation section (two kinds)
│   └── directives.py              # Template files of the pure techniques
│
├── multi_turn/                    # Conversation datasets: Setting, actors, generate/evaluate_conversations
├── multiturn_attacks/             # Crescendo / PAIR attacks against a safety-trained target
├── optimization/                  # Beam search over conversation moves toward a judge's yes
│
├── dataset/                       # Rows on disk: identity, sidecar files, resume, the per-category tree
│   ├── identity.py                # content_id (content hash of a text), row_id / row_ids
│   ├── rows.py                    # The rows of one CSV: append_rows, read_rows, iter_rows, count_rows, has_rows; write_concat
│   ├── sidecar.py                 # JsonlSidecar, Ledger (*.state.jsonl), Manifest (*.manifest.jsonl)
│   ├── resume.py                  # resume_state (lock, check, cut back), release_lock
│   ├── save.py                    # commit / CommitBuffer: a save (rows, then acks)
│   ├── drop.py                    # drop_units: a fresh run of part of a stage
│   ├── categories.py              # The per-category tree: append_samples, merge_all, merge_inputs
│   ├── loading.py                 # HuggingFace dataset loading
│   └── taxonomy.py                # create_taxonomy, taxonomy and seed file loading, category iteration
│
├── configs/                       # JSON configuration files (package data)
│   ├── content_moderation_input.json
│   ├── llm/                       # models.json (the registry), roles.json, gpu_pricing.json
│   ├── runs/                      # Example recipes + params file
│   ├── seeds/                     # Hand-written seed prompts
│   └── taxonomy/                  # Category/technique taxonomies
│
├── prompts/                       # Prompt JSON per pipeline/step (input, output, paraphrase, jailbreak, constitution, ...)
│
├── stages.py                      # The stage functions (generate_*, build_dataset) and the STAGES table
├── recipe.py                      # load_recipe, load_params: what a recipe and its params file say
├── driver.py                      # run_pipeline and the per-stage run records
├── paths.py                       # The on-disk layout under a data_dir
├── telemetry.py                   # Run trace and cost roll-up
├── exceptions.py                  # RedactError, PromptTooLongError
├── types.py                       # EntryType
├── __init__.py                    # Package exports
└── py.typed                       # PEP 561 type marker

scripts/                           # run.py, inspect_run.py, scaffold_prompts.py, download_models.py
notebooks/                         # Walkthroughs
Datasets/                          # Generated output
Data_cache/                        # Intermediate data (benign samples, constitution, etc.)
```

---

## Module Reference

### LLMs — Model-Agnostic Abstraction

Everything above this layer calls a unified interface and is backend-agnostic.

| Component | Purpose |
|---|---|
| `LLMBackend` | Abstract base class. A backend **is one configured model** — name, generation defaults, provider params, rpm and worker budget are bound at construction, so `generate(messages_list, *, max_tokens=..., temperature=...)` takes only what varies per call. Always batch-shaped (a single sample is a batch of one); a backend with async calls also has the coroutine `agenerate(messages)` for one item |
| `OpenAIBackend` | Generic OpenAI-compatible API backend (Venice AI is the provider registered against it today) |
| `AnthropicBackend` | Anthropic Claude (native SDK, separate system param) |
| `VLLMBackend` | Local vLLM for self-hosted GPU inference: vLLM's `AsyncLLM` in this process, one engine request per item |
| `TransformersIntrospectionBackend` | Local raw-`transformers` inference with hidden-state/attention/logprob capture (research/interpretability) |
| `ModelClient.create()` | Resolve a model name to a ready-to-call client (backend + limiter + batch strategy) |
| `backend_for()` | Build the finished backend for one model on one of its setups |
| `ModelClient` | A configured backend plus its dispatch strategy, bound once at construction |
| `RateLimiter` | Per-model sliding-window RPM enforcement (thread-safe); inert when the backend carries no rpm |
| `BatchCaller` | Parallel/sequential fan-out + rate limiting for backends without native batching. Native batching (vLLM) is dispatched by `ModelClient` straight to the engine, bypassing this class entirely |
| `generate_sample()` | Single generation with rate limiting |
| `batch_generate_samples()` | Generate multiple samples in chunks of `batch_size`, in input order |
| `check_sample()` / `batch_check_samples()` | Validate one sample / many against a `(original, sample) -> messages` checker (`redact.llm_pipeline`) |
| `checked()` / `extracted()` | A unit's retry steps: generate, check, regenerate with the verdict / generate until a reply parses (`redact.llm_pipeline`) |
| `drive_units()` | Drive many such generators as a pool topped up from a lazy source, batching their calls per model; a request to a backend with async calls goes out alone on the async lane instead (`redact.llm_pipeline`) |
| `telemetry.install()` | Start the run trace (JSONL by default) and cost roll-up |
| `telemetry.stage()` | Label every event inside a block with its pipeline stage |
| `residency.plan_residency()` | Which local models fit on this machine, and in what order |
| `residency.preload()` | Warm local models in the background while API stages run |
| `load_prompt()` | Load prompt JSON by pipeline/category |
| `PromptTemplate` | Two-stage prompt render: system prompt built once, template rendered per call |
| `extract_and_clean()` | Extract a numbered list from LLM output and clean each sample (Q&A pairs: `extract_structured_qa()`) |

**Backend auto-routing** — just pass a model name:

```python
from redact.llms import ModelClient

client = ModelClient.create("venice-uncensored")   # -> client over OpenAIBackend
client = ModelClient.create("claude-opus-4-6")     # -> client over AnthropicBackend
```

**Direct instantiation** (when you need custom config):

```python
from redact.llms import AnthropicBackend, ModelClient, OpenAIBackend

# Custom API endpoint. A backend is one configured model, so the name comes
# first and the model's facts (defaults, rpm) are bound here, not per call.
backend = OpenAIBackend("my-model", api_key="...",
                        base_url="https://api.example.com/v1", rpm=60)
client = ModelClient(backend)

# Anthropic
backend = AnthropicBackend("claude-opus-4-6", api_key=os.environ["ANTHROPIC_API_KEY"])
```

Anything you leave out here falls back to `LLMBackend`'s defaults — including
`rpm=None`, which means no rate limiting at all. For anything you'll reuse,
`register_model()` it instead: the setup is validated up front and
`ModelClient.create(name)` then wires it the same way as every shipped model.

**Local inference via vLLM** — select the local setup of a registered model with `backend_type="vllm"`, or register any HuggingFace model ID:

```python
from redact import generate_constitution, generate_inputs

# Use the registered local model — backend is auto-initialized from the name
client = ModelClient.create("venice-uncensored", backend_type="vllm")   # local weights
constitution = generate_constitution(model="llama-3.2-3b-debug")
generate_inputs(constitution_df=constitution, model="llama-3.2-3b-debug")
```

When the `vllm` setup of `venice-uncensored` is requested, `ModelClient.create()` automatically creates a `VLLMBackend` for `dphn/Dolphin-Mistral-24B-Venice-Edition`. On first use vLLM downloads the model weights from HuggingFace and caches them at the path set by `HF_HOME` in your `.env`. Subsequent runs load directly from cache — no re-download.

Prompts are tokenized with the chat template of the engine's tokenizer, so any checkpoint whose tokenizer ships a usable `chat_template` — Dolphin, Hermes, and most instruct models — works with no extra flag. For one that doesn't (Mistral-3.x), pass vLLM's native `vllm_kwargs={"tokenizer_mode": "mistral"}`; there is no hand-rolled prompt formatting in this library.

The engine is cached per checkpoint, not per model name, so two registry rows naming the same `hf_model_id` (as `venice-uncensored`'s `.vllm` setup and `venice-paraphraser` do) share one load instead of filling GPU memory twice. That cache is private to `VLLMBackend`: the introspection backend loads the same weights through `transformers`, a different runtime, and keeps its own.

There is deliberately **no separate `venice-uncensored-vllm` row**: the hosted endpoint and the local weights are the same model, so one entry carries both `.api` and `.vllm`, and `ModelClient.create("venice-uncensored", backend_type="vllm")` selects the local one. Which setup is bound also decides the budget: a backend built from the `.vllm` setup carries no rpm and no `extra_body`, so running locally never inherits the endpoint's rate limit or its provider params.

Both generation and checker paths go through a `ModelClient` (never the raw backend), so every call is rate-limited. In the stages, a request to a backend with async calls (vLLM, OpenAI-compatible) goes on the **async lane** of `drive_units`: it is sent the moment its unit needs it, with up to the model's `max_workers` in flight (for vLLM, the sequences the engine runs at once, times its copies), and its reply is handed back the moment it is done, so no unit waits for a longer reply. Requests to the other backends are grouped into `generate()` calls of up to `batch_size`: a thread pool sized by the registry's `recommended_max_workers` for parallel-safe APIs, or sequential for series-only backends (Anthropic). Which path a model takes is read off its backend's capability flags, never per call.

For a custom model, register it (or an existing HF id) so the backend auto-resolves from the name:

```python
from redact.llms import register_model

from redact.llms import VLLMConfig

register_model("my-model", backend_type="vllm", roles=["uncensored_gen"],
               vllm=VLLMConfig(hf_model_id="mistralai/Mistral-7B-v0.3"))
generate_inputs(model="my-model")   # backend auto-selected from the registry
```

**Registering a new model:**

```python
from redact.llms import register_model

from redact.llms import APIConfig

register_model("my-model", backend_type="api", default_max_tokens=4000,
               api=APIConfig(backend_type="openai", api_key_env="MY_API_KEY",
                             base_url="https://api.example.com/v1", rpm=50))
```

**Model registry** — pre-configured models with rate limits and backend routing. RPM only
applies to API backends (`openai`/`anthropic`) — local backends (`vllm`) aren't rate-limited,
so they have no RPM at all, not an unlimited one:

| Model | RPM | Backend | Notes |
|---|---|---|---|
| `venice-uncensored` | 75 | openai | Venice AI API (generic OpenAI-compatible backend) |
| `venice-paraphraser` | — | vllm | Paraphraser-role placeholder — loads the same weights as `venice-uncensored`'s local setup, its own registry identity (see the paraphrase section's disclaimer) |
| `llama-3.2-3b-debug` | — | vllm | Small (3B), fast-loading local debug model (`huihui-ai/Llama-3.2-3B-Instruct-abliterated`) for iterating on pipeline mechanics against a real vLLM engine, not for judging generation quality |
| `deepseek-v3.2` | 20 | openai | Stronger multilingual (used for translation) |
| `claude-opus-4-6` | 5 | anthropic | Set `max_workers=1` to avoid TPM limits |

**Deep-internals logging (research/interpretability)** — a fourth backend,
`TransformersIntrospectionBackend`, captures hidden states / attention / logprobs
during generation. Not a vLLM extension: vLLM's continuous batching discards
intermediate activations by design, so this runs a standalone local HF
`transformers` model instead (used *instead of* vLLM for whichever run needs
capture, not alongside a live vLLM instance of the same weights).

```python
from redact.llms import IntrospectConfig, register_model

register_model(
    "my-model-introspect", backend_type="introspect",
    introspect=IntrospectConfig(
        hf_model_id="mistralai/Mistral-7B-v0.3",
        log_dir="./runs/internals",
        capture={"logprobs": True, "hidden_states": "last", "attention": False},
    ),
)
generate_outputs(model="my-model-introspect")   # capture is automatic once the backend supports it
```

Capture is a **pure side effect** — it never adds a CSV column or changes what a
pipeline returns. Every dataset row already has a `sample_id` (this row's own
content-hash identity) and, where traceable, an `input_id` (the origin row's
`sample_id`); captures are filed under `{log_dir}/{input_id}/...` using those
same ids, so a CSV row and its captured internals always cross-reference with no
separate bookkeeping — e.g. `generate_outputs()` writes to
`{log_dir}/{input_id}/output/` and `.../val_out/` (generation and checker calls
respectively), each with an always-on `meta.json` (resolved settings, messages,
output text) alongside whatever tensors `capture` asked for. Wired into
`generate_outputs`, constitution-seeded input generation, paraphrase, and the
jailbreak engine; not yet multi-turn conversations. Full design:
`.claude/introspection_backend_plan.md`.

---

### Constitution — Category Hierarchy Generation

Generates structured constitutions for constitutional classifier training. Each constitution spans 4 severity levels:

| Entry Type | Description | CSV File |
|---|---|---|
| `harmful` | Absolutely harmful — always flag | `harmful.csv` |
| `dual_use_harmful` | Borderline harmful framing — debatable | `dual_use_harmful.csv` |
| `dual_use_benign` | Borderline benign framing — could look harmful | `dual_use_benign.csv` |
| `benign` | Absolutely benign — never flag (hard negatives) | `benign.csv` |

```python
from redact import generate_constitution

# Generate constitution for all taxonomy categories
constitution = generate_constitution(
    taxonomy="content_moderation_categories",
    num_categories=10,          # constitution categories per type per taxonomy category
    model="claude-opus-4-6",
    num_taxonomy_categories=3,  # limit to first 3 taxonomy categories (None = all)
)
print(f"{len(constitution)} constitution entries")
```

Output saved to `Data_cache/constitution/` as 4 type-based CSVs + `merged.csv` (`read_constitution(dir)` reads it back). A reply cut off before its `[END]` marker is generated again, up to 3 times. Each entry later seeds N input samples for classifier training.

**Expanding constitution entries into input prompts** — pass the returned DataFrame to `generate_inputs()`:

```python
from redact import generate_inputs

inputs = generate_inputs(
    constitution_df=constitution,
    style="long",                   # "long" (2-5 sentences) or "short" (5-20 words)
    samples_per_entry=3,
    model="llama-3.2-3b-debug",
    resume=True,                    # True = skip entries already done;
                                    # False = start this style over
)
```

Both `generate_constitution()` and `generate_inputs(constitution_df=...)` are **resumable by default**: re-running with `resume=True` (the default) skips work already saved to disk, so an interrupted run picks up where it left off. Set `resume=False` to clear previous output and regenerate from scratch.

**Entry-type-aware checker** — `build_quality_checker(category, entry_type, subcategory)` in `content_moderation/checker.py` loads `prompts/input/quality_check/` with all three fields, so benign and dual-use samples are evaluated correctly rather than rejected for "not belonging to the harm category." It serves both input paths; `build_output_quality_checker` is the output-side equivalent.

---

### Content Moderation — Input/Output Generation

**Seeded inputs** (`generate_inputs(constitution_df=...)`, previous section) are the recommended path: each
constitution entry asks for `samples_per_entry` samples, a reply with another count is generated again
(up to `max_attempts`), and each sample is checked by the entry-type-aware checker. Rejected samples are
saved with their reason, so the rejection rate per category is visible in the CSVs.

**Standalone inputs** (`generate_inputs(samples_per_category=..., num_categories=...)`) are **deprecated**
(`DeprecationWarning`) but functional, and need no constitution. Per category:

```
1. generate_category_description()   category name -> LLM -> description
   generate_seeds()                  category + description -> LLM -> seed prompts
   (use_metaprompt=False: the taxonomy description and hand-written seeds from configs/seeds/)
2. turns until samples_per_category are accepted:
     generate N samples in one call -> extract -> drop texts the category already has
     -> check each sample -> save all (accepted + rejected) -> rejection reasons feed the next turn
```

**Outputs** (`generate_outputs`) generate one response per input and check it with the entry-type-aware
output checker: a refusal of a harmful input is rejected, a benign input's reply is judged as any answer.
`max_attempts > 1` regenerates a rejected response with the checker's verdict as feedback;
`max_per_category` caps the inputs per `(category, entry_type)`.

**Key classes and functions:** `InputPipeline`, `run_standalone_inputs`, `run_output_generation`,
`build_quality_checker`, `build_output_quality_checker`, `build_paraphrase_checker`, `ConstitutionInputResult`.

---

### Jailbreak — Technique Library

152 single-turn jailbreak techniques, segmented into **4 layers** and **22 families** by `configs/jailbreak/combination_spec.json`. Technique definitions are taxonomy-driven where applicable — adding a new variant means adding a JSON entry, not a new function.

#### Layers and families

A combination takes at most one technique per family, and techniques are always applied in layer order: **hacking → manipulation → obfuscation → requests**. The per-layer tables further down list every function.

| Layer | Max per combo | Families (technique count) |
|---|---|---|
| 1. Hacking | 1 | `framing` (5) · `cognitive` (5) · `persona` (15) |
| 2. Manipulation | 1 | `fsh` (4) · `dap` (4) |
| 3. Obfuscation | no cap (budget: `max_obfuscations`) | applied in this order: `sensitive_words` (16) → `tokenbreak` (3) → `translation` (20) → `typos` (4) → `casual_writing` (9) → `suffixes` (4) → `encode` (12) → `structural` (3) → `ascii_art` (1) |
| 4. Requests | 1 | `answer_format` (9) · `answer_language` (20) · `continuation` (4) · `indirect` (6) · `distractor_prefix` (2) · `distractor_suffix` (2) · `impersonation` (1) · `temporal` (1) · `asking` (2) |

The ten `to_sensitive_words_encode_*` functions belong to both `sensitive_words` and `encode`, so picking one blocks both families. The layer caps mean that, for example, framing, persona and cognitive never appear together, and neither do FSH and DAP.

#### Combination rules

Enforced by `compatible()` in `jailbreak/rules.py`; every rule binds both ways, whichever technique was picked first:

| Rule | Effect |
|---|---|
| One per family | A family that has already been picked can't be picked again |
| Layer caps | 1 hacking, 1 manipulation, 1 request technique |
| `fsh` / `dap` | ✗ `continuation`, `indirect`, `framing`, `persona`, `cognitive` |
| `continuation` | ✗ `indirect` |
| `structural` | ✗ `ascii_art` |
| `ascii_art` | ✗ `structural`, `framing`, `persona`, `cognitive`, `fsh`, `dap` (output size) |
| Manipulation blocking | `fsh`/`dap` never sit with `translation`, `ascii_art`, or *heavy* full-text encoding (base64, unicode_escape, ascii_ordinal, morse, braille). Word-level `sensitive_words_encode_*` is allowed |
| Complexity budget | Each technique scores 0–3; budget mode stops adding techniques once `max_complexity` is reached |

`combine_techniques()` on its own does **not** check these rules. It only sorts techniques into layer order, so chaining techniques by hand (see below) can produce combinations the sampler never would.

#### Sampling and the plan

- **Family-first:** each pick chooses a family, then a technique within it, so the 20 translation languages count as one choice, not 20.
- **Dealt evenly:** the whole run is planned before anything generates (`jailbreaks.manifest.jsonl`). Within each `(category, entry_type, iteration)` group, each layer joins at its rate (`layer_rates`; the spec's are hacking 0.5, manipulation 0.33, requests 0.7) and families are dealt by count over their `family_weights` (1 each by default), so every group gets its share rather than a random draw. A `<out>.coverage.csv` beside each manifest (`jailbreaks.coverage.csv`, `variations.coverage.csv`) counts every layer, family and technique drawn per category, entry type and iteration, with a zero row for each pool technique never drawn. A layer that falls two or more picks short of its rate in a group (the pool or the rules left no fitting technique) is logged as a WARNING.
- **Budget-driven** rounds pick hacking, manipulation and requests at their rates and up to `max_obfuscations` obfuscation families within `max_complexity`; with `allow_request_obfuscation`, one light encoding or translation may wrap the finished request.
- **Count-driven** rounds (`exact_techniques=N`) pick exactly N compatible techniques and ignore the complexity budget.
- **Reproducible:** each unit's draw is seeded by `(seed, sample_id, iteration)` and each technique's random choices by the unit and the technique, so a replayed plan gives the same text.

#### The augmentation section

`generate_jailbreaks(augmentation=...)` and a recipe's `augmentations` take the same section, with two **kinds**:

- **`adversarial`**: the attacks, written to `jailbreaks.csv`. By default one round of combinations from every technique but translation, by the rules and budget above.
- **`natural`**: natural variations, written to `variations.csv`. By default one technique per input (`exact_techniques: 1`) from the list of mild families and techniques in `augmentations.json`: typos, casual writing, structural wrapping, answer format and language, asking, indirect, temporal, continuation, basic leetspeak and two suffixes (three answer-format directives are denied). A variation that leaves its input unchanged is rejected.

A kind you give overlays its default from `configs/jailbreak/augmentations.json`; a kind left out is the default, and a kind given as `null` is off. In a recipe:

```json
"augmentations": {
  "adversarial": {
    "entry_types": ["harmful", "dual_use_harmful"],
    "families": {"deny": ["translation", "ascii_art"]},
    "rounds": "escalation",
    "family_weights": {"persona": 2}
  },
  "natural": null
}
```

| Key | Kind | Meaning |
|---|---|---|
| `entry_types` | both | The inputs to augment, by entry type; `null` for all |
| `layers.deny` | both | Layers to leave out |
| `families.allow`, `families.deny`, `techniques.allow`, `techniques.deny` | both | With an allow list, the kind draws only the families and techniques it names; a deny list takes names out. The natural kind must have an allow list |
| `pure_only` | both | Only techniques that call no model and need no benign data |
| `family_weights` | both | A family's share within its layer when dealing, e.g. `{"persona": 2}` (1 each by default) |
| `iterations` | both | Augmentations per input (with `rounds`, their number) |
| `max_complexity`, `max_obfuscations`, `layer_rates`, `allow_request_obfuscation` | adversarial | The budget-driven settings above; `layer_rates` (`hacking`, `manipulation`, `requests`, `request_obfuscation`) override the spec's rates |
| `rounds` | adversarial | `null` (every round the settings above), `"escalation"` (four rounds: exactly 1 technique, exactly 2, then two rising budgets), or a list of rounds, each merged over the kind's settings (`exact_techniques`, `max_complexity`, `max_obfuscations`, `layer_rates`, `allow_request_obfuscation`) |
| `exact_techniques` | natural | Techniques per variation (default 1) |

**Overlay rules.** A value you give replaces the default's whole value, a list too. `layers`, `families` and `techniques` are overlaid list by list: an `allow` or `deny` you give replaces the default's list of that name and keeps the other, and a `deny` given alone also takes its names out of the default's `allow` (`"natural": {"families": {"deny": ["typos"]}}` is the natural default without typos). A deny you write replaces the default's deny: the adversarial default denies `translation`, so keep it in a deny of your own to keep translation off. The families of `natural_only_families` (top of `augmentations.json`; `casual_writing` today) never enter the attacks unless the adversarial `families.allow` names them. An unknown key, a name the spec does not have, a name under both allow and deny, or a natural kind without an allow list raises when the recipe loads, naming the kind and the key.

**Translation is off by default** in both kinds: each translation costs 2–8 calls to the translation-role model, a hosted model today. Turn it on by leaving `translation` out of the adversarial deny (`"families": {"deny": []}`) or by naming it in the natural allow; once a local translation model is registered, it can be on in the packaged default.

**The variations file.** `variations.csv` has the same columns as `jailbreaks.csv` (rows keep their input's `entry_type`), with its own manifest, ledger and coverage table. It goes beside `output_path` when one is given, else to `{data_dir}/Datasets/variations.csv`; `variations_path=` overrides both. `build_dataset` adds its **accepted** rows as the `natural_variation` part; `include_variations=False` leaves them out and `variations_path=` reads another file.

**Deprecated aliases.** The twelve flat keywords of `generate_jailbreaks` (`include_hacking`, `include_manipulation`, `include_obfuscation`, `include_requests`, `include_translation`, `pure_only`, `max_complexity`, `max_obfuscations`, `sampling_probs`, `iterations`, `settings_per_iteration`, `entry_types`) are read into the adversarial kind for one release, with a `DeprecationWarning`: `include_<layer>` is a `layers.deny`, `include_translation` a `families.deny`, `sampling_probs` is `layer_rates` and `settings_per_iteration` is `rounds`. The same keys are read, and warn, at the top of a recipe's `augmentations` (the flat form) and in `params["jailbreaks"]`; a key only the section holds (such as `family_weights`) in the params raises.

**Known limitation:** the phase-5 request obfuscation isn't counted against `max_obfuscations`.

**Execution model.** LLM-dependent techniques are **generators** that `yield` an `LLMRequest` and resume via `.send(response)` (`technique.py`); pure transforms are plain callables. `generate_jailbreaks()` plans the run, then drives every pending unit through one pool (`llm_pipeline.drive_units`): a request to a model with async calls (vLLM, OpenAI-compatible) goes out alone the moment its unit needs it, and the others are grouped per model into calls of up to `batch_size`. Finished units are appended every 32. It resumes from the sidecar `jailbreaks.state.jsonl` ledger, reusing the plan (see [Resume model](#resume-model-one-modular-sidecar-system)). A unit that fails gets an error row with its own `sample_id`. Benign Q&A data for FSH/DAP is loaded, or generated once and cached, when a planned unit needs it. Each translation makes 2–8 calls to the translation-role model (translate, check, retry), which is why translation is off by default. The natural kind runs the same way, to `variations.csv`.

#### Obfuscation (72 functions)

| Type | Module | Functions | LLM Required |
|---|---|---|---|
| Encoding | `encoding.py` | `to_base64`, `to_rot13`, `to_rot18`, `to_rot47`, `to_unicode_escape`, `to_ascii_ordinal`, `to_separator`, `to_leetspeak_{basic,intermediate,advanced}`, `to_morse`, `to_braille` | No |
| Structural | `structural.py` | `to_json`, `to_xml`, `to_markdown` | No |
| ASCII Art | `ascii_art.py` | `to_ascii_art` (random one of 16 pyfiglet fonts) | No |
| Suffixes | `suffixes.py` | `to_adversarial_suffix_{punctuation,fragments,unicode,emoji}` | No |
| TokenBreak | `tokenbreak.py` | `to_tokenbreak_{prepend,split,delimiter}` | Yes |
| Sensitive Words | `sensitive_words.py` | `to_sensitive_words_encode_{base64,rot13,rot18,rot47,unicode,ascii,separator,leetspeak_*}`, `to_sensitive_words_{split,star,hyphen,underscore,variables}`, `to_synonym_substitution` | Yes |
| Typos | `typos.py` | `to_rewrite_with_typos_{low,medium,high,insane}` | Yes |
| Casual writing | `casual_writing.py` | `to_lowercase`, `to_uppercase`, `to_no_sentence_capitals`, `to_dropped_spaces`, `to_doubled_spaces`, `to_dropped_punctuation`, `to_doubled_punctuation`, `to_abbreviations`, `to_stripped_accents` | No |
| Translation | `translation.py` | 20 languages across resource tiers: French, Japanese, Russian, Spanish, German, Arabic, Turkish, Czech, Vietnamese, Greek, Croatian, Swahili, Thai, Khmer, Maori, Nepali, Zulu, Scots Gaelic, Bengali, Javanese | Yes |

Sensitive-words and TokenBreak functions share one LLM step that marks the harmful words (`tokenbreak.harmful_words_gen`, prompt `jailbreak/extract_harmful`); the encoding is then applied only to those words rather than the whole prompt. Casual writing is how a person typing quickly writes: casing, spacing, punctuation, abbreviations, accents. Typo rewriting uses a single LLM call with a level-description injected into the template.

#### Hacking (25 functions)

| Type | Module | Functions | LLM Required |
|---|---|---|---|
| Cognitive | `cognitive.py` | `to_persona_roleplay`, `to_hypothetical_framing`, `to_authority_obedience`, `to_avi`, `to_deep_inception` | Yes |
| Named Personas | `personas.py` | `to_invented_persona` + 14 named archetypes (psychopath, alien, cult_leader, very_advanced_ai, cartel_leader, artist, mentally_ill, deformed_scientist, politician, deformed_professor, religious_figure, radical_politician, actor, someone_from_the_future) | Yes |
| Framing | `framing.py` | `to_fictional_world`, `to_noble_goal`, `to_nefarious_goal`, `to_high_stake`, `to_no_moral_constraints` | No |

Cognitive and persona techniques are two-step LLM processes: **scenario generation → jailbreak construction**. Named personas use a persona-grounded scenario prompt that grounds the scenario in the persona's character before construction. Cognitive technique definitions are loaded from `configs/taxonomy/cognitive_techniques.json`; persona definitions from `configs/taxonomy/personas.json` — both are editable without touching code.

Framing directives are pure transforms covering scenario and intent modification. Each has 4 named template variants stored in `configs/jailbreak/hacking/framing_templates.json`; one is randomly selected per call with the variant name recorded in `additional_info`. Answer-format directives live in `requests/answer.py`.

#### Manipulation (8 functions)

| Technique | Variants | Description |
|---|---|---|
| FSH (Few-Shot Hacking) | random_short, random_long, selected_short, selected_long | Benign Q&A pairs prime helpful pattern before harmful query |
| DAP (Distract & Persuade) | random_short, random_long, selected_short, selected_long | Harmful query hidden among benign pairs at random position |

#### Requests (47 functions — all pure transforms)

Modifies *how* a request is posed rather than obfuscating its text or attacking the model's mindset. All functions are pure transforms with no LLM dependency.

| Type | Module | Functions |
|---|---|---|
| Answer format | `answer.py` | `to_refusal_then_comply`, `to_apology_prevention`, `to_prefix_injection`, `to_answer_in_markdown`, `to_answer_as_tutorial`, `to_answer_as_code`, `to_answer_as_dialogue`, `to_answer_as_story`, `to_answer_as_structured_data` (9) |
| Answer language | `answer_language.py` | `to_ask_answer_in_{language}` for 20 languages — French, Japanese, Russian, Spanish, German, Arabic, Turkish, Czech, Vietnamese, Greek, Croatian, Thai, Swahili, Khmer, Maori, Nepali, Zulu, Scots Gaelic, Bengali, Javanese (20) |
| Continuation | `continuation.py` | `to_continue_passage`, `to_complete_story`, `to_fill_in_blank`, `to_continue_instructions` (4) |
| Indirect | `indirect.py` | `to_summarize`, `to_proofread`, `to_critique`, `to_translate_task`, `to_explain_to_child`, `to_rewrite` (6) |
| Distractor | `distractor.py` | `to_distractor_prefix_related`, `to_distractor_prefix_unrelated`, `to_distractor_suffix_related`, `to_distractor_suffix_unrelated` (4) |
| Impersonation | `impersonation.py` | `to_impersonate_good_person` — 8 profession variants (nurse, doctor, security researcher, teacher, etc.) (1) |
| Temporal | `temporal.py` | `to_use_past_tense` — 4 historical/retrospective framings (1) |
| Asking | `asking.py` | `to_innocuous_question`, `to_ask_for_details` (2) |

Each function selects randomly from 4 named template variants stored in per-module JSON config files; the chosen variant is logged in `additional_info` for traceability. Answer language functions are generated dynamically from the language list in `answer_language_templates.json` — adding a new language requires only a JSON entry.

#### Reference List Coverage

The library is benchmarked against a reference set of **73 instruction primitives** and **74 request primitives**. Coverage:

| Status | Primitives |
|---|---|
| Covered | All encoding/sensitive_words/typos, all 16 translation languages, all 14 named personas, all framing directives (fictional_world, noble/nefarious_goal, high_stake, no_moral_constraints, refusal_then_comply, prefix_injection, apology_prevention, answer_in_markdown, answer_as_tutorial), all request primitives (innocuous_question, impersonate_good_person, distractor ×4, use_past_tense, ask_for_details, ask_answer_in ×16) |
| Excluded by design | `agent_context_additional_instr` (requires system-prompt access), `fine_tuning` (out of scope), `use_highly_specialized_language` (unclear implementation path), `direct_question` (no-op) |

**Beyond the reference list** — techniques not in the reference set but included:
- Extra encodings: rot18, rot47, braille, morse, ascii_ordinal
- ASCII art obfuscation (16 pyfiglet fonts)
- Adversarial suffixes (punctuation, fragments, unicode, emoji)
- Structural wrapping (JSON, XML, markdown)
- Cognitive hacking (5 two-step LLM techniques: persona_roleplay, hypothetical_framing, authority_obedience, AVI, deep_inception)
- Manipulation (FSH + DAP, 8 variants using benign Q&A caching)
- Extra answer formats: code, dialogue, story, structured data
- Continuation attacks (4 variants)
- Indirect task embedding (6 task types)
- Extra answer languages: Zulu, Scots Gaelic, Bengali, Javanese

**Technique chaining:**

```python
from redact.jailbreak import combine_techniques
from redact.jailbreak.obfuscation.encoding import to_base64, to_rot13

combo = combine_techniques(to_rot13, to_base64)  # manual chaining: order is sorted, compatibility is NOT checked
result, info = combo("some harmful prompt")
```

---

### Multi-turn Conversations

A **use-case-agnostic** subsystem for generating multi-turn **conversation datasets**. Define a `Setting` (its participants, `max_turns`, an optional `stop`) and play it on a seed to produce a `Trajectory` — a **typed step log** that captures every step, including an actor's *strategy* notes alongside the visible turns (a multi-turn transcript, not just a final answer). The seed is the first participant's message; the participants then take turns in order.

Layering (dependency direction strictly downward): `llm_pipeline/` (`LLMRequest`, the drivers) ← **`multi_turn/`** (the general core) ← **`multiturn_attacks/`** and **`optimization/`** (applications, each its own top-level folder). Design is Inspect-oriented (seed ≈ `Sample`, messages ≈ `ChatMessage`, `Setting` ≈ solver, judge ≈ scorer).

**Actors** — an actor's turn is a generator `turn(transcript) -> yields LLMRequest, returns list[Step]`:
- `ScriptedActor` — fixed/templated turns or replay a given transcript (no LLM)
- `ModelActor` — one LLM turn (renders the transcript, calls its model)
- `StrategyActor` — sends what a `propose(transcript, turn_index)` generator builds (it may call a model, e.g. an attacker), logging its note as a `strategy` step

**Generate conversations** (batched, resumable):

```python
import pandas as pd
from redact import generate_conversations
from redact.multi_turn import ScriptedActor, ModelActor, Setting

seeds = pd.DataFrame({"seed": ["opening question one", "opening question two"]})
# pass a Setting factory (fresh actors per conversation) or a template Setting (deep-copied)
make_setting = lambda: Setting(
    participants=[ScriptedActor("user", ["go deeper"]), ModelActor("bot", "venice-uncensored")],
    max_turns=6,
)
convos = generate_conversations(seeds, make_setting, data_dir="./runs/convo", iterations=1)
# -> Datasets/conversations.csv (sample_id, input_id, iteration, setting, turns_used, stop_reason, transcript[JSON], …)
# resumable via a full-plan *.manifest.jsonl + a sidecar *.state.jsonl ledger; not a recipe stage
```

**Evaluate — optional and separable** (generate → *then* judge; nothing about success is hardwired into generation):

```python
from redact import evaluate_conversations

scored = evaluate_conversations(
    data_dir="./runs/convo", judge_model="claude-opus-4-6",
    judge_system="Did the assistant accomplish the GOAL?", scope="last_reply",  # or "transcript"
)   # -> Datasets/conversations_scored.csv (sample_id, success, judge_reasoning, …)
```

#### Multi-turn jailbreak attacks (`multiturn_attacks/`)

Attacker-vs-target conversations that probe a **safety-trained target** over several turns (**crescendo** template escalation or **PAIR-lite** attacker-LLM), optionally composing jailbreak augmentations from `jailbreak/` into each turn. An attack is just a `Setting` fed to `generate_conversations`, scored by a jailbreak-success judge.

```python
from redact.multiturn_attacks import generate_attacks, score_attacks
from redact.jailbreak.obfuscation.encoding import to_rot13

harmful_seeds = pd.DataFrame({"seed": ["<a harmful request>"]})
attacks = generate_attacks(
    harmful_seeds, target_model="claude-opus-4-6",   # target is REQUIRED (must be safety-trained)
    attack="crescendo", jailbreak_technique=to_rot13, max_turns=6, data_dir="./runs/attack",
)
scored = score_attacks(data_dir="./runs/attack", judge_model="venice-uncensored")  # attack-success rate
```

> ⚠️ Attacking an *uncensored* model is meaningless (it complies on turn 1). The target must be a **safety-trained** model that resists single-shot; there is no default — you pass it explicitly.

#### Optimization / search (`optimization/`)

A general **optimize-toward-an-objective** beam search (`optimization.optimize`): extend every frontier node by each candidate move, have the target reply to all of them in one round, score the replies with a judge, **keep the whole tree**, and continue with the `beam` best. Objective-general; `beam=1` refines a single line (PAIR-like), `beam>1` explores (TAP-like), and `optimize_attack` uses jailbreak techniques as the moves.

```python
from redact.multiturn_attacks import optimize_attack
from redact.jailbreak.obfuscation.encoding import to_rot13, to_base64

best, tree = optimize_attack(
    "<a harmful request>", target_model="claude-opus-4-6",
    techniques=[to_rot13, to_base64], judge_model="venice-uncensored",
    beam=2, depth=3,
)
# best = highest-scoring conversation found; tree = every candidate + judge score (the full trajectory)
```

Everything above is **batched, resumable, and offline-testable** (drive it with a fake router / mock judge — see `tests/test_multi_turn.py`, `tests/test_multiturn_attacks.py`, `tests/test_optimization.py`).

---

### Extraction Utilities

Multi-format extraction from LLM output (`redact.llms`), plus constitution parsing (`redact.constitution.parse`):

| Function | Purpose |
|---|---|
| `extract_numbered_list()` | `"1. sample"` / `"2) sample"` / `"3: sample"` |
| `extract_structured_qa()` | `**Prompt N:** **Question:** ... **Answer:** ...` |
| `parse_constitution()` | 3-layer markdown hierarchy -> `ParsedEntry` list (`redact.constitution.parse`) |
| `clean_sample()` | Strip markdown formatting and ChatML tokens (`<\|im_end\|>`) |
| `get_format_instruction()` | Format instructions to append to system prompts — loaded from `prompts/format_instructions/{style}/`, one of `EXTRACTION_STYLES` |

Constitution parsing example:

```python
from redact.constitution.parse import parse_constitution

entries = parse_constitution("""
## 1. Violence
### 1.1 Physical Violence
- (A person punching another person)
### 1.2 Verbal Threats
- (Someone threatening to harm another)
""")

for e in entries:
    print(f"{e.category} / {e.subcategory} / {e.sample}")
```

---

### Dataset Functions — Data Handling

| Function | Purpose |
|---|---|
| `content_id()` / `row_id()` | A row's id: the content hash of its text, or the id the row already carries |
| `append_rows()` / `read_rows()` | Append rows to a stage's CSV in one synced write; read the CSV back |
| `Ledger` | Sidecar `*.state.jsonl` of finished units — `.completed()` / `.record()` / `.reset()` / `.sidecar()` (see [Resume model](#resume-model-one-modular-sidecar-system)) |
| `Manifest` | Sidecar `*.manifest.jsonl` run plan — `.write()` / `.load()` / `.sidecar()` |
| `JsonlSidecar` | Shared base for `Ledger` + `Manifest` (the path beside the artifact, reset, a read that skips bad lines) |
| `resume_state()` | The units a run skips. Takes the stage's lock and puts its files back at the last finished save, or clears them for a fresh run |
| `CommitBuffer` | Saves finished units in batches: their rows first, their acks in the ledger after |
| `append_samples()` | Add samples to a category's `samples.csv`, skipping ids already in it |
| `merge_all()` / `merge_inputs()` | Merge the accepted rows of the category files under a directory / of a run |
| `write_concat()` | One CSV from the rows of several, copied in chunks (what `build_dataset` uses) |
| `create_taxonomy()` / `load_taxonomy()` | Write a taxonomy JSON of your own / load one by name |
| `iter_categories()` | Iterate categories for generation loops |
| `load_hf_dataset()` | Load from HuggingFace Hub with filtering |

---

## Extensibility Guide

### Adding a New Harm Category

1. Add to taxonomy — `configs/taxonomy/content_moderation_categories.json`
2. Create prompt template — `prompts/input/generation/template.json`
3. Run the pipeline — the category appears automatically in `iter_categories()`

### Adding a Custom Jailbreak Technique

Any function matching the signature can be chained by hand with `combine_techniques()`:

```python
def my_technique(prompt: str, **kwargs) -> tuple[str, str]:
    """Transform prompt. Return (transformed, additional_info)."""
    return f"[OVERRIDE] {prompt}", "my_technique_v1"
```

For LLM-dependent techniques, write a **technique generator**: `yield` an `LLMRequest` and resume with the model's reply. This lets the stage batch your call with every other unit's call to the same model.

```python
from redact.jailbreak.technique import TechniqueGen
from redact.llm_pipeline import LLMRequest

def my_llm_technique(prompt: str, *, gen_model: str | None = None, **kwargs) -> TechniqueGen:
    """Reframe the prompt via one LLM round."""
    messages = [{"role": "user", "content": f"Reframe: {prompt}"}]
    result = yield LLMRequest(gen_model, messages)   # batched with other units' calls
    return result.strip(), "llm_reframed"
```

Multi-round techniques (translate→check→retry, scenario→construction) simply `yield` more than once; `apply_combination()` runs the same generator synchronously for one prompt. For `generate_jailbreaks()` to draw it, return it from its layer's getter (e.g. `get_all_obfuscation_functions()`) and add a matching entry to `combination_spec.json` (`families`, `complexity`, `requires_llm`, ...); importing `redact.jailbreak` fails while the two disagree.

### Swapping Backends

```python
from redact.llms import ModelClient, OpenAIBackend, VLLMBackend

# Auto-routing (recommended) — a registered name resolves to a ready client
client = ModelClient.create("venice-uncensored")                       # Venice API
client = ModelClient.create("claude-opus-4-6")                         # Anthropic API
client = ModelClient.create("venice-uncensored", backend_type="vllm")  # same model, local weights

# Direct instantiation, when the registry doesn't describe your endpoint. A backend
# is one configured model, so the model name comes first.
backend = OpenAIBackend("my-model", api_key="...",
                        base_url="https://api.example.com/v1", rpm=60)
client = ModelClient(backend)

# A local vLLM engine built by hand (loads the weights on construction)
client = ModelClient(VLLMBackend("my-local-model", hf_model_id="mistralai/Mistral-7B-v0.3"))
```

For anything you'll reuse, `register_model()` it instead, so `ModelClient.create(name)` wires it like every shipped model.

---

## Configuration

### Environment Variables

| Variable | Required For | Description |
|---|---|---|
| `VENICE_API_KEY` | Venice models | Venice AI API key |
| `ANTHROPIC_API_KEY` | Claude models | Anthropic API key |
| `HF_TOKEN` | HuggingFace loading | HuggingFace access token |
| `HF_HOME` | vLLM models | Directory where vLLM downloads and caches model weights. Defaults to `~/.cache/huggingface` if unset. Set this to a path with sufficient disk space (`venice-uncensored`'s local setup requires ~48 GB). |
| `REDACT_OUTPUT_DIR` | Optional | Default working root for `Datasets/` and `Data_cache/` when no `data_dir` is given (else the source tree holding `pyproject.toml`, else the current directory) |
| `REDACT_TRACE` | Optional | Telemetry sink: `jsonl` (default), `wandb`, or `off` |
| `REDACT_GPU_PROVIDER` | Optional | Key into `configs/llm/gpu_pricing.json` (e.g. `runpod`) for costing local GPU time. Defaults to `local`, which prices at 0 |

Place in a `.env` file in the working root. It is loaded on `import redact` (via `python-dotenv`), its values winning over the shell's.

### Telemetry, cost, and GPU planning

Every run writes a trace beside its other sidecars — no configuration, no account, no
dependency:

```
Datasets/
  constitution.manifest.jsonl
  constitution.state.jsonl
  run-20260829-142211.trace.jsonl     <- what was called, and what it cost
```

One event per **transport call** (for vLLM, one per engine request, so a batch of 32 is
32 events), labelled with the pipeline stage it belongs to. Set
`REDACT_TRACE=wandb` to mirror the same events to Weights & Biases, or `off` for nothing.

`run_pipeline` also logs a cost roll-up at the end. **Two meters, because two different
things are being bought:** API cost is tokens × price per *model* (set
`price_per_1m_input`/`price_per_1m_output` on the model's `APIConfig`), while local cost is
engine wall-clock lifetime × GPU rate per *checkpoint* — you lease the card for as long as
an engine is alive, not just while it generates. Rates live in
`configs/llm/gpu_pricing.json`; the GPU is detected with `nvidia-smi`. An unpriced GPU
reports **time only** rather than guessing a number.

Before anything loads, a residency plan says whether the run fits:

```
[residency] plan needs 1 GPU(s); detected 1 x NVIDIA RTX A5000, 24.0GiB each
  group 1:
    gen-model[vllm]  ~18.5GiB  (declared: 15.0 weights + 1.20 KV, x1.1 +0.6)  -> card 0, vLLM gets 21.6GiB (gpu_memory_utilization=0.90)
  group 2:  -> cannot be resident with the above
    other-model[vllm]  ~18.5GiB  (declared: 15.0 weights + 1.20 KV, x1.1 +0.6)  -> card 0, vLLM gets 21.6GiB (gpu_memory_utilization=0.90)
  PROBLEM: these groups cannot be resident together, and nothing is unloaded
  between stages, so a later group would load on top of the earlier ones. ...
```

A footprint is estimated from the checkpoint's HF config (weights + KV cache, plus
headroom) without loading anything. Declare `vram_gb` on a `VLLMConfig` or
`IntrospectConfig` to override the estimated weights (the KV cache is always estimated on
top, at `max_model_len` or a default of 10,000 tokens), and `min_gpus` on a `VLLMConfig` for a model too large for one
card. The vLLM engines on a card split 0.9 of it: each is granted its need with overhead ((weights + KV) x1.1 + 0.6 GiB)
plus an equal share of the spare, and loads with that `gpu_memory_utilization`
(it is not a `vllm_kwargs` setting). A grant below that padded need means the plan does not fit.

`run_pipeline` plans **per stage**: as it enters a stage it unloads the local engines that stage
does not keep, and **preloads** the stage's models (and the next stage's, when they fit beside
them) on a background thread, so a multi-minute 24B load overlaps with API work instead of
waiting behind it. Every local engine is released when the run ends. A preload that fails logs the reason and lets the run continue — an API-only stage is
ledger-backed and its output is saved as it goes, so finishing it beats killing it.

### Prompt JSON Schema

```json
{
    "system_prompt": "System instruction...",
    "template": "User template with {variables}",
    "seed_fields": ["variable1", "variable2"],
    "few_shot_examples": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "metadata": {"version": "1.0"}
}
```

### Taxonomy JSON Schema

```json
{
    "name": "taxonomy_name",
    "categories": {
        "Category": {
            "description": "Used in generation prompts",
            "subcategories": ["Sub1", "Sub2"]
        }
    },
    "aliases": {"Alternative Name": "Category"},
    "groups": {"group_name": ["Category"]}
}
```

---

## Key Design Decisions

- **Prompts are external** — prompt text lives in JSON files that ship with the library; a `prompt_dir` overrides any of them file by file (`scripts/scaffold_prompts.py` writes an editable copy).
- **Backend auto-routing** — `ModelClient.create(model)` resolves the registry entry, picks the setup, and builds a fully-configured backend for it. Manual instantiation available for custom setups.
- **Inputs are per-category** — input samples land in `{category}/samples.csv`; every other stage writes one CSV under `Datasets/`. Merging is explicit (`build_dataset`).
- **Checker feedback feeds back** — rejection reasoning is injected into the next generation call for directed improvement.
- **Technique functions are pure** — most obfuscation functions take a string and return a string. No side effects, no hidden state.
- **Spec-driven registry** — every technique is returned by its layer's getter and named in `combination_spec.json`, which holds its families, complexity and the combination rules; import fails when the two differ.
- **`Data_cache/` is internal** — intermediate artifacts (benign samples, scenarios) go here, never in `Datasets/`.

---

## Safety & Ethics

This library is designed for **defensive AI safety research** — building datasets to evaluate and improve content moderation systems.

- **Generated data should be handled responsibly.** Clear `Datasets/` and `Data_cache/` before sharing.
- All generation involves a **checker LLM** that validates sample quality and rejects low-quality or off-category outputs.

---

## Requirements

- Python 3.11+
- `pandas >= 2.0`
- `openai >= 1.0` (Venice / OpenAI-compatible backends)
- `python-dotenv >= 1.0`
- `pyfiglet >= 0.8` (ASCII art jailbreaks)

Optional:
- `anthropic >= 0.30` — Anthropic Claude backend (`pip install redact[anthropic]`)
- `vllm >= 0.4` — local GPU inference (`pip install redact[vllm]`)
- `datasets >= 2.0` — HuggingFace loading (`pip install redact[hf]`)

## License

MIT License. See [LICENSE](LICENSE).
