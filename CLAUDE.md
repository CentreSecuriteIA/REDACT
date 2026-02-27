# BELLS-O Artificial Data Generation Library

## Project Overview

A modular, extensible pipeline for automated red teaming dataset generation. Built from lessons learned across multiple BELLS-O dataset iterations (content moderation input/output, jailbreaks). Designed to be reusable beyond red teaming — applicable to any synthetic data generation task.

The library generates, checks, filters, and stores samples across configurable harm categories. It is built to be model-agnostic (API or local vLLM), prompt-agnostic (prompts are external JSON files), and category-agnostic (new categories require only new prompt files and a runner script).

**Prompts are not bundled in public releases for safety reasons.** The prompts/ directory will be redacted or replaced with documented placeholders before open-sourcing.

---

## Architecture

```
Redact_Library/
├── Dataset_Functions/        # Data handling utilities
│   ├── io.py                 # CSV read/write per category folder
│   ├── merge.py              # Merge category CSVs into unified dataset
│   ├── split.py              # Split datasets into subcategories for processing
│   └── distribution.py      # Distribution analysis and statistics
│
├── LLMs/                     # LLM abstraction layer (model-agnostic)
│   ├── base.py               # Abstract LLMBackend base class
│   ├── api.py                # API backend (OpenAI-compatible endpoints)
│   ├── vllm_backend.py       # Local vLLM backend
│   ├── wrappers.py           # Rate limiting, retry, multithreaded batch calls
│   ├── calls.py              # High-level generate/check call pairs
│   ├── translator.py         # Translation calls (used by jailbreak pipeline)
│   └── prompts.py            # JSON prompt loader and template renderer
│
├── Jailbreak/                # Jailbreak dataset generation
│   ├── obfuscation.py        # Obfuscation-based jailbreak generation
│   ├── hacking.py            # Prompt hacking techniques
│   ├── manipulation.py       # Manipulation-based jailbreaks
│   ├── benign_generator.py   # Benign sample generation (for FSH and DAP attacks)
│   └── runners/              # Category-specific run scripts
│
├── Content_Moderation/       # Content moderation dataset generation
│   ├── input/
│   │   ├── metaprompt.py     # Meta-prompt generation
│   │   ├── generation.py     # Iterative sample generation with per-step filtering
│   │   └── checker.py        # Sample validation and rejection logic
│   └── output/
│       └── generation.py     # Output sample generation (optionally calls fingerprint removal)
│
├── Prompts/                  # [REDACTED IN PUBLIC RELEASE]
│   ├── Jailbreak/
│   │   └── {category}/
│   │       ├── system.json
│   │       └── template.json
│   └── Content_Moderation/
│       └── {category}/
│           ├── system.json
│           └── template.json
│
├── Datasets/                 # Generated output
│   └── {category}/
│       └── samples.csv
│
└── Data_cache/               # Internal/intermediate data (not for release)
    └── benign/               # Cached benign samples for jailbreak construction
```

---

## Module Details

### `llm/` — LLM Abstraction Layer

The core abstraction. Everything above this layer calls a unified interface and is backend-agnostic.

**Base class** (`base.py`): Abstract `LLMBackend` with `generate(prompt)` and `batch_generate(prompts)`. Rate limiting, retry logic, and multithreaded batch dispatch live here so they are not duplicated per backend.

**Backends**:
- `api.py` — OpenAI-compatible API (used for uncensored hosted models)
- `vllm_backend.py` — Local vLLM for self-hosted inference

**`wrappers.py`**: Rate limiter, exponential backoff retry, multithreaded batch caller. Taken and generalized from jailbreak reference implementation.

**`calls.py`**: High-level paired calls — `generate_sample()` and `check_sample()`. Checker receives generated output and returns accept/reject with reasoning. Reasoning is passed back to generator on rejection for directed improvement.

**`prompts.py`**: Loads prompt JSON files by category. Renders templates with seed injections. Schema defined below.

**`translator.py`**: Wraps generate calls for multilingual output. Used by jailbreak pipeline for language-based attack vectors (models have weaker safety training on non-English text).

---

### `dataset/` — Data Handling

All output is saved as CSVs in `data/{category}/`. Functions:

- `io.py` — Read/write category CSVs, append samples incrementally
- `merge.py` — Merge all category CSVs into a single unified dataset
- `split.py` — Split datasets into subcategories or processing batches (getter functions modeled on Jailbreak reference)
- `distribution.py` — Category counts, coverage statistics, pairwise embedding diversity metrics

---

### `jailbreak/` — Jailbreak Generation

Modeled on the jailbreak reference implementation (most advanced, most lessons learned).

**`benign_generator.py`**: Generates benign-looking wrapper content for FSH (few-shot hijacking) and DAP (disguised as prompt) attack construction. Output cached in `data_cache/benign/`.

Runner scripts in `runners/` show how category-specific generation is configured and invoked.

---

### `content_moderation/` — Content Moderation Generation

**Input pipeline**: Meta-prompt generation → iterative sample generation → per-step or post-hoc filtering. Optionally calls fingerprint removal LLM before saving.

**Output pipeline**: Run model on input samples → optionally paraphrase through fingerprint removal LLM → save to `data/{category}/`.

Fingerprint removal is called as an optional LLM pass. The full training pipeline for the removal model lives in a separate repository.

---

## Prompt JSON Schema

All prompts are stored as `.json` files in `prompts/{pipeline}/{category}/`. The prompt loader in `llm/prompts.py` reads these at runtime.

```json
{
  "category": "violence",
  "pipeline": "content_moderation_input",
  "system_prompt": "...",
  "template": "...",
  "seed_fields": ["setting", "actor_type", "method"],
  "few_shot_examples": [],
  "metadata": {
    "version": "1.0",
    "notes": ""
  }
}
```

Seed fields are injected at generation time from the seed bank for that category, enabling deterministic diversity across turns. See `llm/prompts.py` for rendering logic.

---

## Generation Strategy

Each category is generated across multiple turns (~15 samples per turn). Diversity is maintained through:

1. **Semantic seed injection** — Each turn gets a different seed combination (setting, actor type, method, etc.) drawn deterministically via `hash(category + turn_index)`. Reproducible and auditable.
2. **Template rotation** — Prompt templates rotate across framing styles (journalistic, fictional, instructional) across turns.
3. **Random prefix injection** — Short random prefix added for surface-level phrasing diversity within a turn.

Coverage is tracked via pairwise embedding distances. Mode collapse is detected via forward-backward category recovery: generate a sample, ask a judge LLM to infer its category, flag if it doesn't match.

---

## Runner Scripts

Each generation task has a runner script (see `Jailbreak/runners/` for examples). A runner:

1. Selects backend and loads config
2. Loads prompt JSON for the target category
3. Runs generation loop with seed injection
4. Calls checker on each sample
5. Logs acceptance rate
6. Saves accepted samples to `Datasets/{category}/samples.csv`

Rejection rate is the primary quality signal. Sustained high rejection rate after N turns indicates either a bad prompt (improvable) or model capability ceiling (not improvable).

---

## Key Design Decisions

- **Prompts are external** — No prompts hardcoded in library code. Adding a new category = adding a JSON file.
- **Backends are swappable** — Switching from API to vLLM is a config change, not a code change.
- **Data is per-category** — All output lands in `Datasets/{category}/` as CSV. Merging is explicit and on-demand.
- **Checker reasoning feeds back to generator** — Rejection is not just a signal; the checker's reasoning is passed into the next generation call for directed improvement.
- **`Data_cache/` is internal** — Intermediate artifacts (benign samples, partial batches) go here, never in `Datasets/`.
- **Multithreading is opt-in** — `wrappers.py` exposes both sequential and multithreaded batch generation. Sequential is default for easier debugging.

---

## Reference Implementations

Three reference projects informed this architecture. Listed in order of influence:

1. **Jailbreak library** (`BELLS_Jailbreak/`) — Most advanced. Take the most inspiration here: LLM call wrappers, rate limiting, retry logic, regex utilities, dataset splitting getter functions, runner script structure.
2. **Content moderation output** (`output dataset/`) — vLLM integration pattern. Paraphrasing/fingerprint removal call pattern (training pipeline is out of scope for this repo).
3. **Content moderation input** (`input dataset/`) — Original meta-prompt + filtering pattern. Most code is deprecated (old HF API hosting). Useful only for understanding the initial generation + filtering flow.

---

## Out of Scope (Separate Repositories)

- **Defingerprinting pipeline** — Reverse paraphraser training, fingerprint removal model, finetuning loop. Called optionally from this library but trained and maintained separately.
- **Prompt finetuner / auto-optimizer** — Automated prompt optimization loop (task-dataset gen, running task, feedback). Designed to consume and produce prompt JSONs compatible with this library's schema.

---

## Public Release Notes

Before open-sourcing:
- Redact or replace all files in `Prompts/` with documented placeholders
- Clear `Datasets/` and `Data_cache/`
- Confirm no category-specific generation logic leaks harmful prompt content into code