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

```python
# 1. Auto-select backend from model name
from redact.llms import get_backend, RateLimiter, generate_sample

backend = get_backend("venice-uncensored")  # -> VeniceBackend (via VENICE_API_KEY env var)
rate_limiter = RateLimiter()

# 2. Generate content moderation samples
from redact.content_moderation import InputPipeline
from redact.content_moderation.checker import build_quality_checker
from redact.llms import load_prompt

pipeline = InputPipeline(
    gen_backend=backend, gen_model="venice-uncensored",
    check_backend=backend, check_model="venice-uncensored",
    rate_limiter=rate_limiter,
)

prompt_config = load_prompt("content_moderation", "generation")
result = pipeline.run_category(
    category="Physical Harm",
    prompt_config=prompt_config,
    build_check_messages=build_quality_checker("Physical Harm"),
    num_turns=2, samples_per_request=5,
)
print(f"Generated {result.total_accepted} accepted samples")

# 3. Apply jailbreak techniques
from redact.jailbreak.obfuscation.encoding import to_base64
from redact.jailbreak.hacking.cognitive import to_persona_roleplay

# Pure technique (no LLM)
obfuscated, info = to_base64("How to pick a lock")

# LLM-dependent technique
jailbreak, info, scenario = to_persona_roleplay(
    "How to pick a lock",
    backend=backend, model="venice-uncensored", rate_limiter=rate_limiter,
)
```

See [`full_pipeline.ipynb`](full_pipeline.ipynb) for a complete pipeline walkthrough.

---

## Architecture

```
src/redact/
├── llms/                          # Model-agnostic LLM abstraction
│   ├── base.py                    # Abstract LLMBackend base class
│   ├── api.py                     # Backend router: get_backend(model) -> auto-select
│   ├── venice_backend.py          # Venice AI / OpenAI-compatible API backend
│   ├── anthropic_backend.py       # Anthropic Claude backend (native SDK)
│   ├── vllm_backend.py            # Local vLLM backend for self-hosted inference
│   ├── wrappers.py                # Rate limiter, retry, batch caller
│   ├── calls.py                   # generate_sample(), check_sample()
│   ├── prompts.py                 # JSON prompt loader + template renderer
│   ├── extraction.py              # Multi-sample + constitution extraction
│   ├── translator.py              # Translation with fidelity checking
│   └── model_config.py            # Model registry (RPM, backend_type, defaults)
│
├── content_moderation/            # Content moderation generation pipeline
│   ├── generation.py              # InputPipeline — the main driver
│   ├── checker.py                 # Quality + category validation checkers
│   ├── metaprompt.py              # Automated description + seed generation
│   └── paraphrase.py              # Fingerprint removal (placeholder)
│
├── jailbreak/                     # Jailbreak technique library
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
│   └── distribution.py            # Re-exports from dataset module
│
├── dataset/                       # Data handling utilities
│   ├── io.py                      # CSV read/write per category folder
│   ├── merge.py                   # Merge + normalize CSVs (general + presets)
│   ├── split.py                   # Balanced splitting across techniques
│   ├── dedup.py                   # Exact + normalized deduplication
│   ├── loading.py                 # HuggingFace dataset loading
│   └── taxonomy.py                # Taxonomy loading, filtering, iteration
│
├── configs/                       # JSON configuration files (package data)
│   ├── content_moderation_input.json
│   ├── seeds/                     # Hand-written seed prompts
│   └── taxonomy/                  # Category/technique taxonomies
│
├── prompts/                       # Prompt templates (redacted for safety)
│   ├── content_moderation/        # Per-step prompt templates
│   └── jailbreak/                 # Per-technique prompt templates
│
├── exceptions.py                  # RedactError, ConfigError, etc.
├── pipelines.py                   # High-level pipeline functions
├── __init__.py                    # Config, PROJECT_ROOT, package exports
└── py.typed                       # PEP 561 type marker

Datasets/                          # Generated output (per-category CSVs)
Data_cache/                        # Intermediate data (benign samples, etc.)
full_pipeline.ipynb                # Complete pipeline walkthrough
```

---

## Module Reference

### LLMs — Model-Agnostic Abstraction

Everything above this layer calls a unified interface and is backend-agnostic.

| Component | Purpose |
|---|---|
| `LLMBackend` | Abstract base class — `generate(messages, model)` |
| `VeniceBackend` | Venice AI / OpenAI-compatible API backend |
| `AnthropicBackend` | Anthropic Claude (native SDK, separate system param) |
| `VLLMBackend` | Local vLLM for self-hosted GPU inference |
| `get_backend()` | Auto-select backend from model name |
| `RateLimiter` | Per-model sliding-window RPM enforcement (thread-safe) |
| `BatchCaller` | Sequential or multithreaded batch dispatch |
| `generate_sample()` | Single generation with rate limiting |
| `check_sample()` | Validate a sample (yes/no + reasoning) |
| `generate_with_check()` | Full generate -> check -> feedback loop |
| `load_prompt()` | Load prompt JSON by pipeline/category |
| `extract_and_clean()` | Extract numbered lists / Q&A / delimited from LLM output |
| `parse_constitution()` | Parse 3-layer markdown constitution into structured entries |
| `extract_bold_prompt_answer()` | Extract bold-formatted prompt-answer pairs |
| `translate_with_check()` | Translation with fidelity validation |

**Backend auto-routing** — just pass a model name:

```python
from redact.llms import get_backend

backend = get_backend("venice-uncensored")   # -> VeniceBackend
backend = get_backend("claude-opus-4-6")      # -> AnthropicBackend
```

**Direct instantiation** (when you need custom config):

```python
from redact.llms import VeniceBackend, AnthropicBackend

# Custom API endpoint
backend = VeniceBackend(api_key="...", base_url="https://api.example.com/v1")

# Anthropic
backend = AnthropicBackend.from_env("ANTHROPIC_API_KEY")
```

**Local inference via vLLM** — create manually and pass to pipelines:

```python
from redact.llms import VLLMBackend

# vLLM requires manual init (model path, quantization, GPU config)
backend = VLLMBackend(model="mistralai/Mistral-7B-v0.3")

# Pass directly — overrides auto-routing
generate_inputs(model="venice-uncensored", backend=backend)
```

**Registering a new model:**

```python
from redact.llms import register_model

register_model("my-model", rpm=50, default_max_tokens=4000, backend_type="venice")
```

**Model registry** — pre-configured models with rate limits and backend routing:

| Model | RPM | Backend | Notes |
|---|---|---|---|
| `venice-uncensored` | 75 | venice | Can also run locally via vLLM |
| `deepseek-v3.2` | 20 | venice | Stronger multilingual (used for translation) |
| `claude-opus-4-6` | 5 | anthropic | Set `max_workers=1` to avoid TPM limits |

---

### Content Moderation — Input/Output Generation

The pipeline operates in two modes:

**Automated mode (`USE_METAPROMPT=True`)** — three LLM steps per category:

```
Step 1. generate_category_description()
        category name -> LLM -> rich description (3-5 sentences)

Step 2. generate_seeds()
        category + description -> LLM -> numbered list of seed prompts

Step 3. InputPipeline.run_category()
        description + seeds -> LLM -> samples, checked per-sample
```

**Simple mode (`USE_METAPROMPT=False`)** — no extra LLM calls:

```
Description: short one-liner from taxonomy JSON
Seeds:       hand-written list from configs/seeds/
Step 3:      same InputPipeline.run_category()
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

**Key classes:** `InputPipeline`, `SampleResult`, `TurnResult`, `CategoryResult`

**Feedback loop** — rejection reasoning from turn N is injected into turn N+1's prompt.

---

### Jailbreak — Technique Library

30+ jailbreak techniques organized in three families:

#### Obfuscation (22 functions)

| Type | Functions | LLM Required |
|---|---|---|
| Encoding | `to_base64`, `to_rot13`, `to_leetspeak`, `to_morse`, `to_braille` | No |
| Structural | `to_json`, `to_xml`, `to_markdown` | No |
| ASCII Art | `to_ascii_art` (16 pyfiglet fonts) | No |
| Suffixes | `to_adversarial_suffix_{punctuation,fragments,unicode,emoji}` | No |
| TokenBreak | `to_tokenbreak_{prepend,split,delimiter}` | Yes |
| Translation | `to_{zulu,scots_gaelic,bengali,swahili,thai,javanese}` | Yes |

#### Hacking (5 functions)

| Function | Description |
|---|---|
| `to_persona_roleplay` | Assign AI an unfiltered persona |
| `to_hypothetical_framing` | Frame in fictional context |
| `to_authority_obedience` | Authority directives override |
| `to_avi` | Anthropomorphic vulnerability exploitation |
| `to_deep_inception` | Nested multi-layer scenarios |

Each is a two-step LLM process: **scenario generation** -> **jailbreak construction**.

#### Manipulation (8 functions)

| Technique | Variants | Description |
|---|---|---|
| FSH (Few-Shot Hacking) | random_short, random_long, selected_short, selected_long | Benign Q&A pairs prime helpful pattern before harmful query |
| DAP (Distract & Persuade) | random_short, random_long, selected_short, selected_long | Harmful query hidden among benign pairs at random position |

**Technique chaining:**

```python
from redact.jailbreak import combine_techniques
from redact.jailbreak.obfuscation.encoding import to_base64, to_rot13

combo = combine_techniques(to_rot13, to_base64)
result, info = combo("some harmful prompt")
```

---

### Extraction Utilities

Multi-format extraction from LLM output, plus constitution parsing:

| Function | Purpose |
|---|---|
| `extract_numbered_list()` | `"1. sample"` / `"2) sample"` / `"3: sample"` |
| `extract_structured_qa()` | `**Prompt N:** **Question:** ... **Answer:** ...` |
| `extract_delimited()` | Samples separated by `---`, `===`, blank lines |
| `parse_constitution()` | 3-layer markdown hierarchy -> `ConstitutionEntry` list |
| `extract_bold_prompt_answer()` | `**Prompt:** ... **Answer:** ...` pairs |
| `clean_sample()` | Strip markdown formatting and meta-commentary |
| `get_format_instruction()` | Format instructions to append to system prompts |

Constitution parsing example:

```python
from redact.llms import parse_constitution

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
| `append_samples()` | Incremental CSV save with MD5 dedup |
| `merge_technique_csvs()` | Jailbreak preset — renames type columns, drops DISCARDED |
| `merge_content_mod_csvs()` | Content mod preset — filters accepted, normalizes columns |
| `deterministic_balanced_assign()` | Stratified splitting across N bins |
| `load_taxonomy()` | Load taxonomy JSON |
| `iter_categories()` | Iterate categories for generation loops |
| `load_hf_dataset()` | Load from HuggingFace Hub with filtering |

---

## Extensibility Guide

### Adding a New Harm Category

1. Add to taxonomy — `configs/taxonomy/content_moderation_categories.json`
2. Create prompt template — `prompts/content_moderation/generation/template.json`
3. Run the pipeline — the category appears automatically in `iter_categories()`

### Adding a Custom Jailbreak Technique

Any function matching the signature works:

```python
def my_technique(prompt: str, **kwargs) -> tuple[str, str]:
    """Transform prompt. Return (transformed, additional_info)."""
    return f"[OVERRIDE] {prompt}", "my_technique_v1"
```

For LLM-dependent techniques, accept `backend`, `model`, `rate_limiter`:

```python
def my_llm_technique(prompt, backend=None, model=None, rate_limiter=None, **kwargs):
    from redact.llms import generate_sample
    messages = [{"role": "user", "content": f"Reframe: {prompt}"}]
    result = generate_sample(backend, model, messages, rate_limiter)
    return result, "llm_reframed"
```

### Swapping Backends

```python
from redact.llms import get_backend, VeniceBackend, VLLMBackend

# Auto-routing (recommended)
backend = get_backend("venice-uncensored")   # Venice API
backend = get_backend("claude-opus-4-6")      # Anthropic API

# Direct instantiation
backend = VeniceBackend(api_key="...", base_url="https://api.example.com/v1")

# Local vLLM (manual init, pass directly to pipelines)
backend = VLLMBackend(model="mistralai/Mistral-7B-v0.3")
```

---

## Configuration

### Environment Variables

| Variable | Required For | Description |
|---|---|---|
| `VENICE_API_KEY` | Venice models | Venice AI API key |
| `ANTHROPIC_API_KEY` | Claude models | Anthropic API key |
| `HF_TOKEN` | HuggingFace loading | HuggingFace access token |
| `REDACT_OUTPUT_DIR` | Optional | Base directory for `Datasets/` and `Data_cache/` (defaults to script directory) |

Place in a `.env` file in the project root. Loaded automatically via `python-dotenv`.

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

- **Prompts are external** — no prompts hardcoded in library code. Adding a category = adding a JSON file. Prompts are redacted in public releases for safety.
- **Backend auto-routing** — `get_backend(model)` picks the right backend from model name. Manual instantiation available for custom setups. vLLM requires manual init (heavy GPU setup).
- **Data is per-category** — all output lands in `Datasets/{category}/` as CSV. Merging is explicit.
- **Checker feedback feeds back** — rejection reasoning is injected into the next generation call for directed improvement.
- **Technique functions are pure** — most obfuscation functions take a string and return a string. No side effects, no hidden state.
- **Registry pattern** — technique families expose `get_type_to_getter()` registries for uniform access and automatic dataset splitting.
- **`Data_cache/` is internal** — intermediate artifacts (benign samples, scenarios) go here, never in `Datasets/`.

---

## Safety & Ethics

This library is designed for **defensive AI safety research** — building datasets to evaluate and improve content moderation systems.

- **Prompts are redacted** in public releases. The `prompts/` directory contains placeholder templates.
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
