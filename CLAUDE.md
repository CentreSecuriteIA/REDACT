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
│       ├── pipelines.py             # High-level pipeline functions
│       ├── exceptions.py            # Custom exception hierarchy
│       ├── py.typed                 # PEP 561 type marker
│       │
│       ├── llms/                    # LLM abstraction layer (model-agnostic)
│       │   ├── base.py              # Abstract LLMBackend base class
│       │   ├── api.py               # API backend (OpenAI-compatible endpoints)
│       │   ├── vllm_backend.py      # Local vLLM backend
│       │   ├── wrappers.py          # Rate limiting, retry, multithreaded batch calls
│       │   ├── calls.py             # High-level generate/check call pairs
│       │   ├── translator.py        # Translation calls (jailbreak pipeline)
│       │   ├── prompts.py           # JSON prompt loader and template renderer
│       │   ├── model_config.py      # Model registry and config
│       │   └── extraction.py        # Regex extraction utilities
│       │
│       ├── content_moderation/      # Content moderation dataset generation
│       │   ├── generation.py        # Iterative sample generation with filtering
│       │   ├── checker.py           # Sample validation and rejection logic
│       │   ├── metaprompt.py        # Meta-prompt generation (description + seeds)
│       │   └── paraphrase.py        # Fingerprint removal (stub)
│       │
│       ├── dataset/                 # Data handling utilities
│       │   ├── io.py                # CSV read/write per category folder
│       │   ├── merge.py             # Merge category CSVs into unified dataset
│       │   ├── split.py             # Split datasets for processing
│       │   ├── taxonomy.py          # Taxonomy loading and category iteration
│       │   ├── dedup.py             # Deduplication utilities
│       │   └── loading.py           # HuggingFace dataset loading
│       │
│       ├── constitution/            # Constitution generation for classifiers
│       │   └── pipeline.py         # ConstitutionPipeline, EntryType, results
│       │
│       ├── jailbreak/               # Jailbreak dataset generation
│       │   ├── utils.py             # Technique combination utilities
│       │   ├── distribution.py      # Balanced splitting re-exports
│       │   ├── obfuscation/         # Text transformation attacks
│       │   │   ├── encoding.py      # Base64, ROT13, hex, etc.
│       │   │   ├── translation.py   # Low-resource language translation
│       │   │   ├── structural.py    # JSON/XML/markdown wrapping
│       │   │   ├── ascii_art.py     # ASCII art obfuscation
│       │   │   ├── tokenbreak.py    # Token-level splitting
│       │   │   └── suffixes.py      # Adversarial suffix injection
│       │   ├── hacking/             # Cognitive/psychological attacks
│       │   │   └── cognitive.py     # Persona, framing, authority, AVI, inception
│       │   └── manipulation/        # Few-shot manipulation
│       │       ├── benign.py        # Benign sample generation
│       │       ├── fsh.py           # Few-Shot Hacking
│       │       └── dap.py           # Distract and Persuade
│       │
│       ├── configs/                 # Configuration data (package data)
│       │   ├── content_moderation_input.json
│       │   ├── seeds/               # Hand-written seed prompts
│       │   └── taxonomy/            # Category taxonomy definitions
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

**Base class** (`base.py`): Abstract `LLMBackend` with `generate(prompt)` and `batch_generate(prompts)`. Rate limiting, retry logic, and multithreaded batch dispatch live here so they are not duplicated per backend.

**Backends**:
- `api.py` — OpenAI-compatible API (used for uncensored hosted models)
- `vllm_backend.py` — Local vLLM for self-hosted inference

**`wrappers.py`**: Rate limiter, exponential backoff retry, multithreaded batch caller.

**`calls.py`**: High-level paired calls — `generate_sample()`, `check_sample()`, and `batch_check_samples()`. Checker receives generated output and returns accept/reject with reasoning. Reasoning is passed back to generator on rejection for directed improvement. `batch_check_samples()` sends all checker prompts in one `batch_generate()` call — this is the primary speed lever for vLLM. For API backends, `batch_generate()` falls back to sequential; use `BatchCaller(max_workers=N)` for parallelism there.

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

Three attack families:
- **obfuscation/** — Pure text transforms (encoding, structural) + LLM-based (translation, tokenbreak)
- **hacking/** — Two-step cognitive attacks: scenario generation → jailbreak construction
- **manipulation/** — FSH (few-shot hacking) and DAP (distract and persuade) using benign Q&A pairs

Benign samples for FSH/DAP are cached in `Data_cache/benign/`.

---

### `content_moderation/` — Content Moderation Generation

**Input pipeline**: Meta-prompt generation → iterative sample generation → per-step or post-hoc filtering.

**Output pipeline**: Run model on input samples → optionally paraphrase through fingerprint removal LLM → save to `Datasets/`.

Fingerprint removal is called as an optional LLM pass. The full training pipeline for the removal model lives in a separate repository.

---

### `constitution/` — Constitution Generation

Generates structured category hierarchies for constitutional classifier training. Each constitution spans 4 severity levels:

1. **Absolutely harmful** — clear-cut violations, always flag
2. **Dual-use harmful** — borderline, harmful framing, debatable
3. **Dual-use benign** — borderline, benign framing, could look harmful
4. **Absolutely benign** — clearly safe, never flag (hard negatives)

`ConstitutionPipeline` generates entries per taxonomy category using Claude Opus. Uses `parse_constitution()` from `llms/extraction.py` to parse the 3-layer markdown output. Entries saved to `Data_cache/constitution/` as 4 type-based CSVs. Each entry later seeds N input samples for classifier training.

`ConstitutionInputPipeline` expands constitution entries into full prompts. Uses a dedicated checker at `prompts/constitution/checker/template.json` (via `_build_constitution_checker()`) that injects `category`, `subcategory`, and `entry_type` — so it correctly evaluates benign and dual-use samples, not just harmful ones.

**Known limitation:** `content_moderation/checker.py` `build_quality_checker()` is harmful-only. When content moderation benign/dual-use sample generation is added, extend it with an `entry_type` parameter following the same pattern as `_build_constitution_checker()` in `constitution/input_generation.py`.

---

## Prompt JSON Schema

All prompts are stored as `.json` files in `src/redact/prompts/{pipeline}/{category}/`. The prompt loader in `llms/prompts.py` reads these at runtime.

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

Seed fields are injected at generation time from the seed bank for that category.

---

## Installation

```bash
pip install -e .           # editable install
pip install -e ".[dev]"    # with dev dependencies
pip install -e ".[vllm]"   # with vLLM support
```

## Usage

```python
from redact import generate_inputs, generate_jailbreaks, build_dataset

inputs = generate_inputs(samples_per_category=15, num_categories=3)
jailbreaks = generate_jailbreaks(inputs=inputs)
dataset = build_dataset()
```

---

## Key Design Decisions

- **Prompts are external** — No prompts hardcoded in library code. Adding a new category = adding a JSON file.
- **Backends are swappable** — Switching from API to vLLM is a config change, not a code change.
- **Data is per-category** — All output lands in `Datasets/{category}/` as CSV. Merging is explicit and on-demand.
- **Checker reasoning feeds back to generator** — Rejection is not just a signal; the checker's reasoning is passed into the next generation call for directed improvement.
- **`Data_cache/` is internal** — Intermediate artifacts (benign samples, partial batches) go here, never in `Datasets/`.
- **Multithreading is opt-in** — `wrappers.py` exposes both sequential and multithreaded batch generation. Sequential is default for easier debugging.
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
