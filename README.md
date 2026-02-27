# REDACT — Red-team Dataset Automation & Construction Toolkit

A modular Python library for generating, validating, and managing synthetic red-teaming datasets. Built for content moderation and jailbreak research, but designed to be extensible to any synthetic data generation task.

## Overview

REDACT automates the full lifecycle of red-teaming dataset construction:

1. **Generate** harmful content samples across configurable harm categories
2. **Validate** each sample via a checker LLM with feedback-driven retry
3. **Transform** inputs into jailbreak attacks using 30+ techniques
4. **Split, merge, and manage** datasets with balanced distribution across techniques

The library is **model-agnostic** (API or local vLLM), **prompt-agnostic** (all prompts are external JSON files), and **category-agnostic** (new categories require only a taxonomy entry and prompt file).

---

## Quick Start

```python
# 1. Set up an LLM backend
from Redact_Library.LLMs import APIBackend, RateLimiter

backend = APIBackend(api_key="your-key", base_url="https://api.venice.ai/api/v1")
rate_limiter = RateLimiter()

# 2. Generate content moderation samples
from Redact_Library.Content_Moderation import InputPipeline
from Redact_Library.Content_Moderation.checker import build_quality_checker
from Redact_Library.LLMs import load_prompt

pipeline = InputPipeline(
    gen_backend=backend, gen_model="venice-uncensored",
    check_backend=backend, check_model="venice-uncensored",
    rate_limiter=rate_limiter,
)

prompt_config = load_prompt("Content_Moderation", "generation")
result = pipeline.run_category(
    category="Physical Harm",
    prompt_config=prompt_config,
    build_check_messages=build_quality_checker("Physical Harm"),
    num_turns=2, samples_per_request=5,
)
print(f"Generated {result.total_accepted} accepted samples")

# 3. Apply jailbreak techniques
from Redact_Library.Jailbreak.obfuscation.encoding import to_base64
from Redact_Library.Jailbreak.hacking.cognitive import to_persona_roleplay

# Pure technique (no LLM)
obfuscated, info = to_base64("How to pick a lock")

# LLM-dependent technique
jailbreak, info, scenario = to_persona_roleplay(
    "How to pick a lock",
    backend=backend, model="venice-uncensored", rate_limiter=rate_limiter,
)
```

See [`examples/`](examples/) for complete pipeline scripts.

---

## Architecture

```
Redact_Library/
├── LLMs/                          # Model-agnostic LLM abstraction
│   ├── base.py                    # Abstract LLMBackend base class
│   ├── api.py                     # OpenAI-compatible API backend
│   ├── vllm_backend.py            # Local vLLM backend
│   ├── wrappers.py                # Rate limiter, retry, batch caller
│   ├── calls.py                   # generate_sample(), check_sample()
│   ├── prompts.py                 # JSON prompt loader + template renderer
│   ├── extraction.py              # Multi-sample extraction from LLM output
│   ├── translator.py              # Translation with fidelity checking
│   └── model_config.py            # Model registry (RPM, defaults)
│
├── Content_Moderation/            # Content moderation generation pipeline
│   ├── generation.py              # InputPipeline — the main driver
│   ├── checker.py                 # Quality + category validation checkers
│   ├── metaprompt.py              # Automated description + seed generation
│   └── paraphrase.py              # Fingerprint removal (placeholder)
│
├── Jailbreak/                     # Jailbreak technique library
│   ├── obfuscation/               # Text transformation attacks
│   │   ├── encoding.py            # base64, rot13, leetspeak, morse, braille
│   │   ├── structural.py          # JSON, XML, markdown wrapping
│   │   ├── ascii_art.py           # pyfiglet-based text art
│   │   ├── suffixes.py            # Adversarial suffix generators
│   │   ├── tokenbreak.py          # Token-breaking (LLM-dependent)
│   │   └── translation.py         # Low-resource language translation
│   ├── hacking/                   # Cognitive/psychological manipulation
│   │   └── cognitive.py           # 5 techniques (persona, framing, etc.)
│   ├── manipulation/              # Context manipulation with benign examples
│   │   ├── benign.py              # Benign sample generation + caching
│   │   ├── fsh.py                 # Few-Shot Hacking (4 variants)
│   │   └── dap.py                 # Distract and Persuade (4 variants)
│   ├── utils.py                   # combine_techniques() for chaining
│   └── distribution.py            # Re-exports from Dataset_Functions
│
├── Dataset_Functions/             # Data handling utilities
│   ├── io.py                      # CSV read/write per category folder
│   ├── merge.py                   # Merge + normalize CSVs (general + presets)
│   ├── split.py                   # Balanced splitting across techniques
│   ├── dedup.py                   # Exact + normalized deduplication
│   ├── loading.py                 # HuggingFace dataset loading
│   └── taxonomy.py                # Taxonomy loading, filtering, iteration
│
├── Dataset_Configs/               # JSON configuration files
│   ├── content_moderation_input.json
│   └── taxonomy/
│       ├── content_moderation_categories.json
│       └── jailbreak_techniques.json
│
├── Prompts/                       # Prompt templates (redacted for safety)
│   ├── Content_Moderation/
│   │   ├── category_description/  # Step 1: category name -> rich description
│   │   ├── seed_generation/       # Step 2: description -> seed prompts
│   │   ├── generation/            # Step 3: seeds + description -> samples
│   │   ├── quality_check/         # Quality validation checker
│   │   ├── output_generation/     # Response generation
│   │   └── metaprompt/            # [DEPRECATED] seed abstraction (see below)
│   └── Jailbreak/
│       ├── benign_generation/     # Benign sample generation
│       ├── category_selection/    # Category matching for FSH/DAP
│       ├── extract_harmful/       # Harmful word extraction (tokenbreak)
│       ├── jailbreak_construction/  # Hacking technique construction
│       └── scenario_generation/   # Scenario generation (hacking)
│
├── Datasets/                      # Generated output (per-category CSVs)
└── Data_cache/                    # Intermediate data (benign samples, etc.)
```

---

## Module Reference

### LLMs — Model-Agnostic Abstraction

Everything above this layer calls a unified interface and is backend-agnostic.

| Component | Purpose |
|---|---|
| `LLMBackend` | Abstract base class — `generate(messages, model)` |
| `APIBackend` | OpenAI-compatible API (Venice AI, OpenAI, Together AI, etc.) |
| `VLLMBackend` | Local vLLM for self-hosted GPU inference |
| `RateLimiter` | Per-model sliding-window RPM enforcement (thread-safe) |
| `with_retries()` | Simple retry decorator |
| `with_feedback_retries()` | Retry with checker feedback passed to generator |
| `BatchCaller` | Sequential or multithreaded batch dispatch |
| `generate_sample()` | Single generation with rate limiting |
| `check_sample()` | Validate a sample (yes/no + reasoning) |
| `generate_with_check()` | Full generate → check → feedback loop |
| `load_prompt()` | Load prompt JSON by pipeline/category |
| `render_template()` | Render template with `{variable}` substitution |
| `build_messages()` | Build OpenAI-format message list from prompt config |
| `extract_and_clean()` | Extract numbered lists / Q&A / delimited from LLM output |
| `translate_with_check()` | Translation with fidelity validation |

**Swapping backends** — change one line:

```python
# API backend
backend = APIBackend(api_key="...", base_url="https://api.venice.ai/api/v1")

# Local vLLM backend
backend = VLLMBackend(model="mistralai/Mistral-7B-v0.3")
```

**Registering a new model:**

```python
from Redact_Library.LLMs import register_model

register_model("my-model", rpm=50, default_max_tokens=4000)
```

---

### Content_Moderation — Input/Output Generation

The pipeline operates in two modes, controlled by `USE_METAPROMPT` in example 01:

**Automated mode (`USE_METAPROMPT=True`, default)** — three LLM steps per category:

```
Step 1. generate_category_description()
        category name -> LLM -> rich description (3-5 sentences)
        Template: Prompts/Content_Moderation/category_description/

Step 2. generate_seeds()
        category + description -> LLM -> numbered list of seed prompts
        Template: Prompts/Content_Moderation/seed_generation/

Step 3. InputPipeline.run_category()
        description + seeds -> LLM -> samples, checked per-sample
        Template: Prompts/Content_Moderation/generation/
```

**Simple mode (`USE_METAPROMPT=False`)** — no extra LLM calls:

```
Description: short one-liner from taxonomy JSON
Seeds:       hand-written list from Dataset_Configs/seeds/content_moderation_seeds.json
Step 3:      same InputPipeline.run_category() as above
```

The `InputPipeline` generation loop (Step 3 in both modes):

```
For each turn:
  1. Build messages (prompt + format instruction + prohibited list + feedback)
  2. Generate N samples in one LLM call
  3. Extract individual samples via regex
  4. Dedup against existing samples
  5. Check each sample individually via checker LLM
  6. Save all samples (accepted + rejected) to CSV
  7. Collect rejection feedback for next turn
```

**Key classes and functions:**

| Symbol | Purpose |
|---|---|
| `InputPipeline` | Main generation driver — `run_category()`, `run_turn()` |
| `SampleResult` | Single sample outcome (text, accepted, reasoning) |
| `TurnResult` | Turn aggregate (extracted_count, acceptance_rate) |
| `CategoryResult` | Category aggregate across all turns |
| `generate_category_description()` | Step 1 — category name -> rich description |
| `generate_seeds()` | Step 2 — description -> seed prompt list |
| `build_quality_checker()` | Quality + category fit validator |
| `build_category_checker()` | Forward-backward category recovery (mode collapse detection) |

**Feedback loop** — rejection reasoning from turn N is injected into turn N+1's prompt, directing the generator to avoid similar mistakes.

**Deprecated:** `generate_abstract_seeds()` (alias: `generate_instruction_prompt`) and `Prompts/Content_Moderation/metaprompt/` predate the two-step pipeline. They take *existing* seeds and abstract them into variations via an LLM call. The `metaprompt/template.json` is a placeholder — fill it in if you want to use this pattern. Prefer `generate_seeds()` for new code.

---

### Jailbreak — Technique Library

30+ jailbreak techniques organized in three families:

#### Obfuscation (22 functions)

Text transformations that obscure harmful content:

| Type | Functions | LLM Required |
|---|---|---|
| Encoding | `to_base64`, `to_rot13`, `to_leetspeak`, `to_morse`, `to_braille` | No |
| Structural | `to_json`, `to_xml`, `to_markdown` | No |
| ASCII Art | `to_ascii_art` (16 pyfiglet fonts) | No |
| Suffixes | `to_adversarial_suffix_{punctuation,fragments,unicode,emoji}` | No |
| TokenBreak | `to_tokenbreak_{prepend,split,delimiter}` | Yes |
| Translation | `to_{zulu,scots_gaelic,bengali,swahili,thai,javanese}` | Yes |

#### Hacking (5 functions)

Cognitive/psychological prompt manipulation:

| Function | Description |
|---|---|
| `to_persona_roleplay` | Assign AI an unfiltered persona |
| `to_hypothetical_framing` | Frame in fictional context |
| `to_authority_obedience` | Authority directives override |
| `to_avi` | Anthropomorphic vulnerability exploitation |
| `to_deep_inception` | Nested multi-layer scenarios |

Each is a two-step LLM process: **scenario generation** → **jailbreak construction**. Scenarios are reusable and can be cached to `Data_cache/scenarios/`.

#### Manipulation (8 functions)

Context manipulation using benign examples:

| Technique | Variants | Description |
|---|---|---|
| FSH (Few-Shot Hacking) | random_short, random_long, selected_short, selected_long | Benign Q&A pairs prime helpful pattern before harmful query |
| DAP (Distract & Persuade) | random_short, random_long, selected_short, selected_long | Harmful query hidden among benign pairs at random position |

"Selected" variants use an LLM to choose the most thematically relevant benign subcategory. "Random" variants need no LLM.

**Technique chaining** with `combine_techniques()`:

```python
from Redact_Library.Jailbreak import combine_techniques
from Redact_Library.Jailbreak.obfuscation.encoding import to_base64, to_rot13

combo = combine_techniques(to_rot13, to_base64)
result, info = combo("some harmful prompt")
# result is base64(rot13(prompt)), info tracks both steps
```

**Registry pattern** — every technique family exposes a registry:

```python
from Redact_Library.Jailbreak import get_type_to_getter

registry = get_type_to_getter()  # All obfuscation types
# {"encoding_cyphering": get_encoding_functions, "ascii_art": get_ascii_art_functions, ...}

all_funcs = registry["encoding_cyphering"]()
# [to_base64, to_rot13, to_leetspeak, to_morse, to_braille]
```

---

### Dataset_Functions — Data Handling

| Function | Purpose |
|---|---|
| `append_samples()` | Incremental CSV save with MD5 dedup |
| `merge_category_csvs()` | Merge per-category CSVs into one DataFrame |
| `merge_csvs_from_dirs()` | General directory-based CSV merge with normalization |
| `merge_technique_csvs()` | Jailbreak preset — renames type columns, drops DISCARDED |
| `merge_content_mod_csvs()` | Content mod preset — filters accepted, normalizes columns |
| `deterministic_balanced_assign()` | Stratified splitting across N bins |
| `split_by_functions()` | Split by technique function registry |
| `load_hf_dataset()` | Load from HuggingFace Hub with filtering |
| `load_from_config()` | Config-driven HF loading |
| `load_taxonomy()` | Load taxonomy JSON |
| `iter_categories()` | Iterate categories for generation loops |
| `normalize_categories()` | Apply taxonomy aliases |
| `filter_by_group()` | Filter to a taxonomy group |

---

## Pipeline Walkthroughs

### Content Moderation Input Generation

See [`examples/01_generate_content_moderation.py`](examples/01_generate_content_moderation.py)

Toggle `USE_METAPROMPT` in the config section of the script:

```
USE_METAPROMPT=True (default — fully automated):
  1. Load taxonomy -> get category names
  2. For each category:
     a. Step 1: LLM generates a rich category description
     b. Step 2: LLM generates seed prompts from the description
     c. Step 3: Run N turns of generate -> extract -> check -> save
        - Feedback from rejections improves next turn
  3. Results saved to Datasets/{category}/samples.csv

USE_METAPROMPT=False (simple — no extra LLM calls):
  1. Load taxonomy -> get category names + short descriptions
  2. Load hand-written seeds from Dataset_Configs/seeds/content_moderation_seeds.json
  3. For each category:
     a. Use taxonomy description + seeds JSON directly (no LLM prep steps)
     b. Run N turns of generate -> extract -> check -> save
  4. Results saved to Datasets/{category}/samples.csv
```

### Output Response Generation

See [`examples/02_generate_output_responses.py`](examples/02_generate_output_responses.py)

```
1. Load accepted input samples from Datasets/
2. For each sample, generate an AI response
3. Optionally paraphrase (fingerprint removal)
4. Save input+output pairs to CSV
```

### Jailbreak Generation

See [`examples/03_generate_jailbreaks.py`](examples/03_generate_jailbreaks.py)

```
1. Load harmful prompts (from HF or generated dataset)
2. Split across technique functions (balanced by category + origin)
3. Apply each technique to its data slice:
   - Pure obfuscation: direct transform, no LLM
   - TokenBreak/Translation: LLM-dependent obfuscation
   - Hacking: scenario generation → jailbreak construction
   - Manipulation: load benign data → build FSH/DAP prompts
4. Save per-technique CSVs
5. Merge into unified dataset with merge_technique_csvs()
```

---

## Extensibility Guide

### Adding a New Harm Category

1. **Add to taxonomy** — edit `Dataset_Configs/taxonomy/content_moderation_categories.json`:
   ```json
   "My New Category": {
       "description": "What this category covers",
       "subcategories": ["Sub1", "Sub2"]
   }
   ```

2. **Create prompt template** — add `Prompts/Content_Moderation/generation/template.json` (or a category-specific one)

3. **Run the pipeline** — the category appears automatically in `iter_categories()`:
   ```python
   taxonomy = load_taxonomy("content_moderation_categories")
   for name, info in iter_categories(taxonomy):
       # Your new category is here
       pipeline.run_category(category=name, ...)
   ```

### Adding a Custom Jailbreak Technique

Any function matching the signature works:

```python
def my_technique(prompt: str, **kwargs) -> tuple[str, str]:
    """Transform prompt. Return (transformed, additional_info)."""
    transformed = f"[OVERRIDE] {prompt}"
    return transformed, "my_technique_v1"
```

For LLM-dependent techniques, accept `backend`, `model`, `rate_limiter` in `**kwargs`:

```python
def my_llm_technique(prompt, backend=None, model=None, rate_limiter=None, **kwargs):
    from Redact_Library.LLMs import generate_sample
    messages = [{"role": "user", "content": f"Reframe: {prompt}"}]
    result = generate_sample(backend, model, messages, rate_limiter)
    return result, "llm_reframed"
```

Register in a custom registry and use with `split_by_functions()`:

```python
registry = {"my_techniques": lambda: [my_technique, my_llm_technique]}
splits = split_by_functions(df, registry)
```

### Adding a Custom Taxonomy

Create a JSON file following the schema:

```json
{
    "name": "my_taxonomy",
    "description": "...",
    "categories": {
        "Category Name": {
            "description": "...",
            "subcategories": ["Sub1", "Sub2"]
        }
    },
    "aliases": {"Alt Name": "Category Name"},
    "groups": {"group_name": ["Category Name"]}
}
```

Save to `Dataset_Configs/taxonomy/` and load:

```python
taxonomy = load_taxonomy("my_taxonomy")
```

See [`examples/04_custom_taxonomy_pipeline.py`](examples/04_custom_taxonomy_pipeline.py) for a complete example.

### Swapping LLM Backends

```python
# API backend (OpenAI-compatible)
from Redact_Library.LLMs import APIBackend
backend = APIBackend(api_key="...", base_url="https://api.example.com/v1")

# Local vLLM backend
from Redact_Library.LLMs import VLLMBackend
backend = VLLMBackend(model="mistralai/Mistral-7B-v0.3")

# Everything else stays the same — just pass the backend to the pipeline
```

---

## Configuration Reference

### Dataset Config JSON (`Dataset_Configs/*.json`)

```json
{
    "name": "config_name",
    "dataset_id": "huggingface/dataset-id",
    "split": "train",
    "columns": ["id", "prompt", "category"],
    "drop_columns": ["unwanted_col"],
    "filters": {"column": ["included_value"]},
    "exclude": {"column": ["excluded_value"]},
    "metadata": {"description": "...", "version": "1.0"}
}
```

Load with: `df = load_from_config("config_name")`

### Taxonomy JSON (`Dataset_Configs/taxonomy/*.json`)

```json
{
    "name": "taxonomy_name",
    "description": "...",
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

### Prompt JSON (`Prompts/{pipeline}/{category}/template.json`)

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

---

## Key Design Decisions

- **Prompts are external** — no prompts hardcoded in library code. Adding a category = adding a JSON file. Prompts are redacted in public releases for safety.
- **Backends are swappable** — switching from API to vLLM is a one-line change.
- **Data is per-category** — all output lands in `Datasets/{category}/` as CSV. Merging is explicit.
- **Checker feedback feeds back** — rejection reasoning is injected into the next generation call for directed improvement.
- **Technique functions are pure** — most obfuscation functions take a string and return a string. No side effects, no hidden state. LLM-dependent techniques receive the backend explicitly.
- **Registry pattern** — technique families expose `get_type_to_getter()` registries for uniform access and automatic dataset splitting.
- **`Data_cache/` is internal** — intermediate artifacts (benign samples, scenarios) go here, never in `Datasets/`.

---

## Safety & Ethics

This library is designed for **defensive AI safety research** — building datasets to evaluate and improve content moderation systems.

- **Prompts are redacted** in public releases. The `Prompts/` directory contains placeholder templates. Replace with your own prompts for your specific research context.
- **Generated data should be handled responsibly.** Clear `Datasets/` and `Data_cache/` before sharing the codebase.
- All generation involves a **checker LLM** that validates sample quality and rejects low-quality or off-category outputs.

---

## Requirements

- Python 3.10+
- `pandas`
- `openai` (for API backend)
- `pyfiglet` (for ASCII art jailbreaks)
- `datasets` (optional, for HuggingFace loading)
- `vllm` (optional, for local inference)
