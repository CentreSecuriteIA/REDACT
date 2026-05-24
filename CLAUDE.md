# REDACT — Red-team Dataset Automation & Construction Toolkit

## Project Overview

A modular, extensible pipeline for automated red teaming dataset generation. Built from lessons learned across multiple BELLS-O dataset iterations (content moderation input/output, jailbreaks). Designed to be reusable beyond red teaming — applicable to any synthetic data generation task.

The library generates, checks, filters, and stores samples across configurable harm categories. It is built to be model-agnostic (API or local vLLM), prompt-agnostic (prompts are external JSON files), and category-agnostic (new categories require only new prompt files and a runner script).

**Prompts are not bundled in public releases for safety reasons.** The `prompts/` directory will be redacted or replaced with documented placeholders before open-sourcing.

---

## Architecture

```
REDACT/
├── src/
│   └── redact/                      # Main package
│       ├── __init__.py              # Config, seed, get_output_dir(), subpackage imports
│       ├── types.py                 # Shared dependency-free types (EntryType) — avoids circular imports
│       ├── pipelines.py             # High-level pipeline functions
│       ├── exceptions.py            # Custom exception hierarchy
│       ├── py.typed                 # PEP 561 type marker
│       │
│       ├── llms/                    # LLM abstraction layer (model-agnostic)
│       │   ├── base.py              # Abstract LLMBackend + capability flags (supports_native_batching / supports_parallel_calls)
│       │   ├── api.py               # get_backend(model) — backend auto-router + per-type cache
│       │   ├── venice_backend.py    # Venice / OpenAI-compatible API backend
│       │   ├── anthropic_backend.py # Anthropic Claude backend (native SDK, series-only)
│       │   ├── vllm_backend.py      # Local vLLM backend (native single-pass batching)
│       │   ├── router.py            # ModelRouter — process-wide RateLimiter + per-model BatchCaller cache; role lookup
│       │   ├── wrappers.py          # RateLimiter, retry wrappers, BatchCaller (capability-aware dispatch)
│       │   ├── calls.py             # High-level generate/check call pairs
│       │   ├── translator.py        # Translation calls (jailbreak pipeline)
│       │   ├── prompts.py           # JSON prompt loader and template renderer
│       │   ├── model_config.py      # Model registry (RPM, backend_type, capability flags, roles)
│       │   └── extraction.py        # Regex extraction utilities
│       │
│       ├── content_moderation/      # Content moderation dataset generation
│       │   ├── generation.py        # InputPipeline: standalone (run_category) + constitution-seeded (run_from_constitution)
│       │   ├── checker.py           # Entry-type-aware quality + output + category checkers
│       │   ├── metaprompt.py        # Meta-prompt generation (description + seeds)
│       │   └── paraphrase.py        # Fingerprint removal (stub)
│       │
│       ├── dataset/                 # Data handling utilities
│       │   ├── io.py                # CSV read/write per category folder (+ _hash_text content id)
│       │   ├── merge.py             # Merge category CSVs into unified dataset
│       │   ├── split.py             # Split datasets for processing
│       │   ├── taxonomy.py          # Taxonomy loading and category iteration
│       │   ├── dedup.py             # Deduplication utilities
│       │   └── loading.py           # HuggingFace dataset loading
│       │
│       ├── constitution/            # Constitution generation for classifiers
│       │   ├── generation.py        # ConstitutionPipeline (4 severity types) → Data_cache CSVs
│       │   └── input_generation.py  # ConstitutionInputPipeline — wraps InputPipeline.run_from_constitution
│       │
│       ├── jailbreak/               # Jailbreak dataset generation
│       │   ├── utils.py             # Technique combination, tagging, sampling, deterministic assignment
│       │   ├── protocol.py          # LLMRequest + technique-generator contract (yield/send), run_sync
│       │   ├── engine.py            # batch_apply_combinations — round-by-round batched engine
│       │   ├── manifest.py          # plan_run / load_plan / resume ledger (content-hash ids)
│       │   ├── distribution.py      # Balanced splitting re-exports
│       │   ├── obfuscation/         # Text transformation attacks
│       │   │   ├── encoding.py      # base64, rot13/18/47, unicode, ordinal, separator, leetspeak, morse, braille
│       │   │   ├── translation.py   # 20 languages across resource tiers
│       │   │   ├── structural.py    # JSON/XML/markdown wrapping
│       │   │   ├── ascii_art.py     # ASCII art obfuscation (19 fonts)
│       │   │   ├── tokenbreak.py    # Token-level splitting + sensitive-word encoding (16 functions)
│       │   │   ├── typos.py         # LLM-rewritten typos at 4 density levels
│       │   │   └── suffixes.py      # Adversarial suffix injection
│       │   ├── hacking/             # Cognitive/psychological attacks
│       │   │   ├── cognitive.py     # 5 techniques; definitions loaded from cognitive_techniques.json
│       │   │   ├── personas.py      # 14 named persona archetypes + invented persona; loaded from personas.json
│       │   │   └── framing.py       # 5 scenario-modifying directives (pure transforms); templates in framing_templates.json
│       │   ├── manipulation/        # Few-shot manipulation
│       │   │   ├── benign.py        # Benign sample generation
│       │   │   ├── fsh.py           # Few-Shot Hacking
│       │   │   └── dap.py           # Distract and Persuade
│       │   └── requests/            # Request-structure attacks (all pure transforms)
│       │       ├── answer.py          # 9 output-format + conditioning directives; answer_templates.json
│       │       ├── answer_language.py # 20 ask-answer-in-language functions; answer_language_templates.json
│       │       ├── continuation.py    # 4 continuation-attack functions; continuation_templates.json
│       │       ├── indirect.py        # 6 task-embedding functions; indirect_templates.json
│       │       ├── distractor.py      # 4 prefix/suffix distractor functions; distractor_templates.json
│       │       ├── impersonation.py   # 1 good-person impersonation function; impersonation_templates.json
│       │       ├── temporal.py        # 1 past-tense reframing function; temporal_templates.json
│       │       └── asking.py          # 2 question-framing functions; asking_templates.json
│       │
│       ├── configs/                 # Configuration data (package data)
│       │   ├── content_moderation_input.json
│       │   ├── jailbreak/
│       │   │   ├── hacking/
│       │   │   │   └── framing_templates.json         # 5 scenario-modifying framing directives
│       │   │   └── requests/
│       │   │       ├── answer_templates.json          # 9 output-format + conditioning directives
│       │   │       ├── answer_language_templates.json # 20 languages + 4 template variants
│       │   │       ├── continuation_templates.json    # 4 continuation-attack templates
│       │   │       ├── indirect_templates.json        # 6 task-embedding templates
│       │   │       ├── distractor_templates.json      # prefix/suffix × related/unrelated × 4 variants
│       │   │       ├── impersonation_templates.json   # 8 good-person profession variants
│       │   │       ├── temporal_templates.json        # 4 past-tense framing variants
│       │   │       └── asking_templates.json          # innocuous_question + ask_for_details variants
│       │   ├── seeds/               # Hand-written seed prompts
│       │   └── taxonomy/            # Category taxonomy definitions
│       │       ├── content_moderation_categories.json
│       │       ├── jailbreak_techniques.json
│       │       ├── cognitive_techniques.json  # Cognitive hacking technique definitions (editable)
│       │       └── personas.json              # Named persona archetypes (editable)
│       │
│       └── prompts/                 # [REDACTED IN PUBLIC RELEASE]
│           ├── content_moderation/  # Prompt templates per pipeline step
│           ├── jailbreak/           # Jailbreak prompt templates
│           └── constitution/        # Constitution generation prompts (4 severity types)
│
├── tests/                           # Test suite
├── full_pipeline.ipynb              # Complete pipeline walkthrough
├── Datasets/                        # Generated output (gitignored)
├── Data_cache/                      # Intermediate cache (gitignored)
├── pyproject.toml                   # Build config and dependencies
├── README.md
└── .env                             # API keys (gitignored)
```

---

## Module Details

### `llms/` — LLM Abstraction Layer

The core abstraction. Everything above this layer calls a unified interface and is backend-agnostic.

**Base class** (`base.py`): Abstract `LLMBackend` with `generate(messages, model)` and `batch_generate(messages_list, model)`, plus two capability properties — `supports_native_batching` (vLLM: one engine pass) and `supports_parallel_calls` (False for Anthropic/vLLM). These flags are how every layer above stays backend-agnostic: callers never branch on backend type, they read the flags.

**Backends** (one class per provider; `api.get_backend(model)` auto-selects by `backend_type`/name and caches per type):
- `venice_backend.py` — Venice / OpenAI-compatible API (uncensored hosted models)
- `anthropic_backend.py` — Claude via native SDK (system param split out); series-only
- `vllm_backend.py` — Local vLLM for self-hosted inference; native single-pass batching

**`model_config.py`**: `MODEL_REGISTRY` of `ModelConfig` entries — RPM, default gen params, `backend_type`, capability flags, `recommended_max_workers`, and a logical `role` (`uncensored_gen`, `translation`, `constitution_gen`, `uncensored_local`). `register_model()` adds entries at runtime; `default_model_for_role()` powers role lookup.

**`router.py`**: `ModelRouter` (process-wide via `get_router()`) owns the single shared `RateLimiter` and a lazy per-model `BatchCaller` cache, and resolves roles → `(backend, model)`. It is the single entry point for rate-limited, capability-aware generation. Both generation and checker paths now dispatch through a `BatchCaller` (either the router's or one built via `BatchCaller.from_model(backend, model, rate_limiter=...)` that wraps an explicitly-passed backend) — never `backend.batch_generate()` directly — so per-model RPM is enforced everywhere. New code should keep this invariant: route batch dispatch through a `BatchCaller`/the router, not the raw backend.

**`wrappers.py`**: `RateLimiter` (thread-safe sliding-window RPM per model), retry wrappers, and `BatchCaller` which dispatches by capability flag: vLLM → native batch, parallel-safe API → thread pool (`recommended_max_workers`), series-only → sequential. Misconfiguration (e.g. `max_workers>1` on a series-only backend) raises rather than silently degrading.

**`calls.py`**: High-level paired calls — `generate_sample()`, `check_sample()`, and `batch_check_samples()`. Checker receives generated output and returns accept/reject with reasoning (acceptance = response starts with yes/ok/accept/pass). Reasoning is passed back to generator on rejection for directed improvement. `batch_check_samples()` sends all checker prompts in one `batch_generate()` call — the primary speed lever for vLLM. For API backends, `batch_generate()` falls back to sequential.

**`prompts.py`**: Loads prompt JSON files by category. Renders templates with seed injections.

**`translator.py`**: Wraps generate calls for multilingual output. Used by jailbreak pipeline for language-based attack vectors.

---

### `dataset/` — Data Handling

All output is saved as CSVs in `Datasets/{category}/`. Functions:

- `io.py` — Read/write category CSVs, append samples incrementally
- `merge.py` — Merge all category CSVs into a single unified dataset
- `split.py` — Split datasets into subcategories or processing batches
- `taxonomy.py` — Load taxonomy JSONs, iterate categories, load seeds
- `dedup.py` — Exact and normalized deduplication
- `loading.py` — HuggingFace dataset loading with config-driven filtering

---

### `jailbreak/` — Jailbreak Generation

Four attack families:
- **obfuscation/** — Pure text transforms (encoding, structural) + LLM-based (translation, tokenbreak, typos)
- **hacking/** — Two-step cognitive attacks: scenario generation → jailbreak construction. `framing.py` covers the 5 scenario-modifying directives (fictional world, noble/nefarious goal, urgency, constraint removal).
- **manipulation/** — FSH (few-shot hacking) and DAP (distract and persuade) using benign Q&A pairs
- **requests/** — Pure transforms that modify request structure. 8 subtype modules, all no-LLM, each backed by its own JSON config. Subtypes: `format` (9), `answer_language` (20, generated dynamically), `continuation` (4), `indirect` (6), `distractor` (4), `impersonation` (1), `temporal` (1), `asking` (2). Total: 47 functions.

Benign samples for FSH/DAP are cached in `Data_cache/benign/`.

**Execution model (the refactor core).** Generation no longer applies one technique chain at a time. Instead:

- Every LLM-dependent technique is a **generator** (`protocol.py`): it `yield`s an `LLMRequest(model, messages)` and resumes via `.send(response)`. Pure transforms stay plain `(str, **kwargs) -> (str, str)` functions and run inline. Multi-round techniques (translation translate→check→retry, cognitive scenario→construction) simply yield more than once.
- `engine.batch_apply_combinations()` drives a whole chunk of samples **round by round**: it collects every live sample's pending request, groups by `request.model`, and dispatches **one batch per model per round** through the router. So all translations for a round go out together, then all checks, etc.
- `combine_techniques()` chains techniques and re-orders them by `(layer_order, within_layer_order)`; `sample_combination()` / `assign_combination()` pick a valid combination per sample using `combination_spec.json` (family caps, cross-incompatibilities, complexity budget). Assignment is seeded by SHA-256 of `(seed, content-id, iteration)` so it is reproducible and chunk-independent. Any technique whose `__name__` is absent from the spec gets no metadata and is silently never sampled — keep `combination_spec.json` in sync when adding techniques.
- `pipelines.generate_jailbreaks()` runs two phases: **plan** (`manifest.plan_run` writes one JSONL line per sample×iteration before any generation) then **execute** (stream the manifest in chunks through the engine, appending output per chunk). Resume is driven by the output CSV as source of truth — `(input_id, iteration)` pairs already present are skipped. `run_sync` (in `protocol.py`) is the single-sample equivalent used by `apply_combination` and tests.

**Reference list coverage:** Benchmarked against 73 instruction primitives + 74 request primitives. All feasible primitives are covered. Intentionally excluded: `agent_context_additional_instr` (system-prompt access required), `fine_tuning` (out of scope), `use_highly_specialized_language` (unclear path), `direct_question` (no-op). The library is a strict superset of the reference list on everything else, with additional techniques not in the reference set (ASCII art, adversarial suffixes, structural wrapping, cognitive hacking, manipulation, continuation attacks, indirect embedding, extra encodings, extra languages).

---

### `content_moderation/` — Content Moderation Generation

`InputPipeline` (in `generation.py`) is the single driver for both input modes:

- **Standalone** (`run_category` / `pipelines.generate_inputs`): meta-prompt generation (description + seeds) → multi-turn generation → per-sample checking with the entry-type-aware quality checker (defaults to `harmful`). Rejection feedback and a prohibited-sample list feed the next turn.
- **Constitution-seeded** (`run_from_constitution`, also reached via `generate_inputs(constitution_df=...)`): each constitution entry becomes one batched generation request; one `batch_generate` per chunk for generation, one flat batch for checking.

**Output pipeline** (`pipelines.generate_outputs`): batched generation of model responses for input samples → entry-type-aware output checker (refusals on harmful inputs are rejected; benign inputs judged normally) → incremental append to `output_responses.csv`. Resumable like the jailbreak run, but completion is tracked in a **sidecar `output_responses.state.jsonl` ledger** (keyed by the same content-hash id as inputs/jailbreaks), kept separate from the CSV so a large or hand-edited CSV can't corrupt resume state. `resume=True` (default) skips already-completed ids; `fresh=True` clears both the CSV and the ledger. Optional paraphrase / fingerprint removal (`paraphrase.py`) is a pass-through stub; the removal model is trained in a separate repository.

---

### `constitution/` — Constitution Generation

Generates structured category hierarchies for constitutional classifier training. Each constitution spans 4 severity levels:

1. **Absolutely harmful** — clear-cut violations, always flag
2. **Dual-use harmful** — borderline, harmful framing, debatable
3. **Dual-use benign** — borderline, benign framing, could look harmful
4. **Absolutely benign** — clearly safe, never flag (hard negatives)

`ConstitutionPipeline` (in `generation.py`) generates entries per taxonomy category using Claude Opus. Uses `parse_constitution()` from `llms/extraction.py` to parse the 3-layer markdown output. Entries are saved to `Data_cache/constitution/` as per-type CSVs plus `merged.csv`. Each entry later seeds N input samples for classifier training.

`ConstitutionInputPipeline` (in `input_generation.py`) expands constitution entries into full prompts. It is a thin wrapper that loads the constitution CSVs from disk and delegates to `InputPipeline.run_from_constitution()`; prefer the composable `generate_inputs(constitution_df=generate_constitution(...))` in new code.

**Unified checker.** `build_quality_checker()` in `content_moderation/checker.py` is now entry-type-aware — it accepts `category`, `entry_type`, and `subcategory` and loads the shared template `prompts/input/quality_check/template.json`, so it evaluates harmful, benign, and dual-use samples correctly. `_build_constitution_checker()` in `input_generation.py` is a backward-compatible alias that forwards to it. `build_output_quality_checker()` is the output-side equivalent.

---

## Prompt JSON Schema

All prompts are stored as `.json` files in `src/redact/prompts/{pipeline}/{category}/`, loaded at runtime via `load_prompt(pipeline, category)` in `llms/prompts.py`. `load_prompt` resolves to the single `.json` in that directory, else `template.json`. Pipeline names in use: `input` (e.g. `input/generation/standalone`, `input/generation/from_constitution/{style}`, `input/quality_check`, `input/category_check`), `output` (`output/generation`, `output/quality_check`), `jailbreak` (per-technique: `scenario_generation`, `jailbreak_construction`, `extract_harmful`, `benign_generation`, `category_selection`, `rewrite_with_typos`, `synonym_substitution`, ...), and `constitution/generation` (one per entry type).

```json
{
  "category": "violence",
  "pipeline": "input",
  "system_prompt": "...",
  "template": "... {Category} ... {SeedPrompts} ...",
  "seed_fields": ["Category", "SeedPrompts"],
  "few_shot_examples": [],
  "metadata": {
    "version": "1.0",
    "notes": ""
  }
}
```

`build_messages(config, **kwargs)` renders both `system_prompt` and `template` with the same kwargs (via `str.format_map`), so any placeholder works in either field. `seed_fields` is documentation only — render kwargs are whatever the caller passes.

---

## Installation

```bash
pip install -e .           # editable install
pip install -e ".[dev]"    # with dev dependencies
pip install -e ".[vllm]"   # with vLLM support
```

## Usage

```python
from redact import (
    generate_constitution, generate_inputs, generate_outputs,
    generate_jailbreaks, build_dataset,
)

# Optional: constitution-seeded inputs (Claude Opus → entries → prompts)
constitution = generate_constitution(num_categories=10, num_taxonomy_categories=3)
inputs = generate_inputs(constitution_df=constitution, samples_per_entry=3)

# Or standalone meta-prompt inputs
inputs = generate_inputs(samples_per_category=15, num_categories=3)

jailbreaks = generate_jailbreaks(inputs=inputs)   # plan → batched execute, resumable
outputs = generate_outputs(inputs=inputs)         # model responses + output checker
dataset = build_dataset()                         # merge inputs + jailbreaks + outputs
```

Backends are auto-selected from the model name (`get_backend`), and all model-name strings resolve through the registry in `llms/model_config.py`. Pass `backend=` explicitly only for custom endpoints or a manually-constructed `VLLMBackend`.

---

## Key Design Decisions

- **Prompts are external** — No prompts hardcoded in library code. Adding a new category = adding a JSON file.
- **Backends are swappable** — Switching from API to vLLM is a config change, not a code change.
- **Data is per-category** — All output lands in `Datasets/{category}/` as CSV. Merging is explicit and on-demand.
- **Checker reasoning feeds back to generator** — Rejection is not just a signal; the checker's reasoning is passed into the next generation call for directed improvement.
- **`Data_cache/` is internal** — Intermediate artifacts (benign samples, partial batches) go here, never in `Datasets/`.
- **Concurrency is capability-driven** — a bare `BatchCaller()` defaults to `max_workers=1` (sequential, for easy debugging), but `BatchCaller.from_model()` (what the pipelines and router use) sizes workers from the model's `recommended_max_workers` and dispatches by capability flag (vLLM native batch / parallel API thread-pool / series-only sequential). Misconfiguring concurrency on a backend that can't support it raises rather than silently degrading.
- **src/ layout** — Prevents accidental imports of local code during testing. Package is properly installable via `pip install -e .`.

---

## Runner Scripts

Each generation task has a runner script (see `jailbreak/` for examples). A runner:

1. Selects backend and loads config
2. Loads prompt JSON for the target category
3. Runs generation loop with seed injection
4. Calls checker on each sample
5. Logs acceptance rate
6. Saves accepted samples to `Datasets/{category}/samples.csv`

Rejection rate is the primary quality signal. Sustained high rejection rate after N turns indicates either a bad prompt (improvable) or model capability ceiling (not improvable).

---

## Generation Strategy

Each category is generated across multiple turns (~15 samples per turn). Diversity is maintained through:

1. **Semantic seed injection** — Each turn gets a different seed combination (setting, actor type, method, etc.) drawn deterministically via `hash(category + turn_index)`. Reproducible and auditable.
2. **Template rotation** — Prompt templates rotate across framing styles (journalistic, fictional, instructional) across turns.
3. **Random prefix injection** — Short random prefix added for surface-level phrasing diversity within a turn.

Coverage is tracked via pairwise embedding distances. Mode collapse is detected via forward-backward category recovery: generate a sample, ask a judge LLM to infer its category, flag if it doesn't match.

---

## Reference Implementations

Four reference projects informed this architecture. All located in `../reference_files/` relative to this repo. Listed in order of influence:

1. **Jailbreak library** (`../reference_files/BELLS_Jailbreak_Test/`) — Most advanced. Take the most inspiration here: LLM call wrappers, rate limiting, retry logic, regex utilities, dataset splitting getter functions, runner script structure.
2. **Content moderation output** (`../reference_files/output dataset/`) — vLLM integration pattern. Paraphrasing/fingerprint removal call pattern (training pipeline is out of scope for this repo).
3. **Content moderation input** (`../reference_files/input dataset/`) — Original meta-prompt + filtering pattern. Most code is deprecated (old HF API hosting). Useful only for understanding the initial generation + filtering flow.
4. **Constitutional classifier** (`../reference_files/constitutional_classifier/`) — Constitution generation for constitutional AI classifiers. Contains `constitution_gen.ipynb` (generation notebook) and `constitution.csv` (output). This pipeline will be integrated into REDACT as a new generation module.

---

## Out of Scope (Separate Repositories)

- **Defingerprinting pipeline** — Reverse paraphraser training, fingerprint removal model, finetuning loop. Called optionally from this library but trained and maintained separately.
- **Prompt finetuner / auto-optimizer** — Automated prompt optimization loop (task-dataset gen, running task, feedback). Designed to consume and produce prompt JSONs compatible with this library's schema.

---

## Public Release Notes

Before open-sourcing:
- Redact or replace all files in `prompts/` with documented placeholders
- Clear `Datasets/` and `Data_cache/`
- Confirm no category-specific generation logic leaks harmful prompt content into code
