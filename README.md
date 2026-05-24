# REDACT — Red-team Dataset Automation & Construction Toolkit

A modular Python library for generating, validating, and managing synthetic red-teaming datasets. Built for content moderation and jailbreak research, but designed to be extensible to any synthetic data generation task.

## Overview

REDACT automates the full lifecycle of red-teaming dataset construction:

1. **Constitution** — generate structured category hierarchies (harmful, benign, dual-use) using Claude Opus
2. **Generate** harmful content samples across configurable harm categories
3. **Validate** each sample via a checker LLM with feedback-driven retry
4. **Transform** inputs into jailbreak attacks using 100+ techniques
5. **Split, merge, and manage** datasets with balanced distribution across techniques

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

**High-level API** — the common path; backends are auto-selected from model names:

```python
from redact import generate_inputs, generate_jailbreaks, generate_outputs, build_dataset

inputs = generate_inputs(samples_per_category=15, num_categories=3)
jailbreaks = generate_jailbreaks(inputs=inputs)   # plan → batched execute, resumable
outputs = generate_outputs(inputs=inputs)         # model responses + output checker, resumable
dataset = build_dataset()                         # merge everything on disk
```

`generate_jailbreaks(inputs, settings_per_iteration=default_escalation_schedule())` instead runs a multi-round escalation — each sample augmented 4× from a single technique up to high-complexity combinations (see the [Jailbreak section](#jailbreak--technique-library)).

**Lower-level building blocks:**

```python
# 1. Auto-select backend from model name
from redact.llms import get_backend, RateLimiter, load_prompt

backend = get_backend("venice-uncensored")  # -> VeniceBackend (via VENICE_API_KEY env var)
rate_limiter = RateLimiter()

# 2. Generate content moderation samples
from redact.content_moderation import InputPipeline
from redact.content_moderation.checker import build_quality_checker

pipeline = InputPipeline(
    gen_backend=backend, gen_model="venice-uncensored",
    check_backend=backend, check_model="venice-uncensored",
    rate_limiter=rate_limiter,
)

prompt_config = load_prompt("input", "generation/standalone")
result = pipeline.run_category(
    category="Physical Harm",
    prompt_config=prompt_config,
    build_check_messages=build_quality_checker("Physical Harm"),
    num_turns=2, samples_per_request=5,
)
print(f"Generated {result.total_accepted} accepted samples")

# 3. Apply jailbreak techniques
from redact.jailbreak.obfuscation.encoding import to_base64
from redact.jailbreak.utils import apply_combination
from redact.jailbreak.hacking.cognitive import to_persona_roleplay

# Pure technique (no LLM) — plain (str) -> (text, info)
obfuscated, info = to_base64("How to pick a lock")

# LLM-dependent techniques are generators (they yield LLMRequests). Drive a
# single one synchronously with apply_combination():
text, info, accepted, reasoning = apply_combination(
    to_persona_roleplay, "How to pick a lock",
    backend=backend, model="venice-uncensored", rate_limiter=rate_limiter,
)
```

> LLM-dependent techniques (cognitive, personas, translation, typos, tokenbreak, FSH/DAP-selected) are **technique generators** — calling them directly returns a generator, not a result. Use `apply_combination()` for one sample, or let `generate_jailbreaks()` / the batched engine drive them. Pure transforms (encoding, structural, suffixes, ascii_art, framing, all `requests/`) are still plain callables.

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
│   ├── router.py                  # ModelRouter — shared RateLimiter + per-model BatchCaller + role lookup
│   ├── wrappers.py                # RateLimiter, retry, capability-aware BatchCaller
│   ├── calls.py                   # generate_sample(), check_sample(), batch_check_samples()
│   ├── prompts.py                 # JSON prompt loader + template renderer
│   ├── extraction.py              # Multi-sample + constitution extraction
│   ├── translator.py              # Translation with fidelity checking
│   └── model_config.py            # Model registry (RPM, backend_type, capability flags, roles)
│
├── constitution/                  # Constitution generation for classifiers
│   ├── generation.py              # ConstitutionPipeline (4 severity types)
│   └── input_generation.py        # ConstitutionInputPipeline (→ InputPipeline.run_from_constitution)
│
├── content_moderation/            # Content moderation generation pipeline
│   ├── generation.py              # InputPipeline — the main driver
│   ├── checker.py                 # Quality + category validation checkers
│   ├── metaprompt.py              # Automated description + seed generation
│   └── paraphrase.py              # Fingerprint removal (placeholder)
│
├── jailbreak/                     # Jailbreak technique library
│   ├── obfuscation/               # Text transformation attacks
│   │   ├── encoding.py            # base64, rot13/18/47, unicode, ordinal, separator, leetspeak, morse, braille
│   │   ├── structural.py          # JSON, XML, markdown wrapping
│   │   ├── ascii_art.py           # pyfiglet-based text art (19 fonts)
│   │   ├── suffixes.py            # Adversarial suffix generators
│   │   ├── tokenbreak.py          # Token-breaking + sensitive-word encoding (LLM-dependent)
│   │   ├── typos.py               # LLM-rewritten typos at 4 density levels
│   │   └── translation.py         # 20 languages across resource tiers
│   ├── hacking/                   # Cognitive/psychological manipulation
│   │   ├── cognitive.py           # 5 techniques (persona, framing, AVI, authority, inception)
│   │   ├── personas.py            # 14 named persona archetypes + invented persona
│   │   └── framing.py             # 5 scenario-modifying directives (pure transforms, multi-template)
│   ├── manipulation/              # Context manipulation with benign examples
│   │   ├── benign.py              # Benign sample generation + caching
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
│   ├── utils.py                   # combine/sample/assign techniques, tagging, spec rules
│   ├── protocol.py                # LLMRequest + technique-generator contract, run_sync
│   ├── engine.py                  # batch_apply_combinations — round-by-round batched engine
│   ├── manifest.py                # plan_run / load_plan / resumable JSONL ledger
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
├── types.py                       # Shared dependency-free types (EntryType)
├── pipelines.py                   # High-level pipeline functions
├── __init__.py                    # Config, get_output_dir(), package exports
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
| `get_router()` | Process-wide `ModelRouter` — one shared `RateLimiter`, a per-model `BatchCaller` cache, and `for_role()` lookup. The intended single entry point for rate-limited, capability-aware generation |
| `RateLimiter` | Per-model sliding-window RPM enforcement (thread-safe) |
| `BatchCaller` | Capability-aware dispatch: vLLM native batch / API thread pool / series-only sequential |
| `generate_sample()` | Single generation with rate limiting |
| `check_sample()` | Validate a single sample (yes/no + reasoning) |
| `batch_check_samples()` | Validate multiple samples in one `batch_generate()` pass — used automatically by all pipelines |
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

**Local inference via vLLM** — use the pre-registered `venice-uncensored-vllm` model or any HuggingFace model ID:

```python
from redact import generate_inputs_from_constitution

# Use the registered local model — backend is auto-initialized
generate_inputs_from_constitution(model="venice-uncensored-vllm", ...)
```

When `venice-uncensored-vllm` is requested, `get_backend()` automatically creates a `VLLMBackend` for `dphn/Dolphin-Mistral-24B-Venice-Edition`. On first use vLLM downloads the model weights from HuggingFace and caches them at the path set by `HF_HOME` in your `.env`. Subsequent runs load directly from cache — no re-download.

Both generation and checker paths dispatch through a `BatchCaller` (never the raw backend), so every batch is rate-limited and uses the right execution mode for its backend: one vLLM engine pass for native backends, a thread pool sized by the registry's `recommended_max_workers` for parallel-safe APIs, or sequential for series-only backends (Anthropic). The `batch_size` parameter (default 32) controls how many entries are grouped per pass.

For a custom model, instantiate `VLLMBackend` directly and pass it to any pipeline:

```python
from redact.llms import VLLMBackend

backend = VLLMBackend(model="mistralai/Mistral-7B-v0.3")
generate_inputs(model="my-model", backend=backend)
```

**Registering a new model:**

```python
from redact.llms import register_model

register_model("my-model", rpm=50, default_max_tokens=4000, backend_type="venice")
```

**Model registry** — pre-configured models with rate limits and backend routing:

| Model | RPM | Backend | Notes |
|---|---|---|---|
| `venice-uncensored` | 75 | venice | Venice AI API |
| `venice-uncensored-vllm` | 999 | vllm | Local self-hosted version of `venice-uncensored` (`dphn/Dolphin-Mistral-24B-Venice-Edition`) |
| `deepseek-v3.2` | 20 | venice | Stronger multilingual (used for translation) |
| `claude-opus-4-6` | 5 | anthropic | Set `max_workers=1` to avoid TPM limits |

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

Output saved to `Data_cache/constitution/` as 4 type-based CSVs + `merged.csv`. Each entry can later seed N input samples for classifier training.

**Entry-type-aware checker** — `build_quality_checker(category, entry_type, subcategory)` in `content_moderation/checker.py` loads the unified template `prompts/input/quality_check/template.json` and injects all three fields, so benign and dual-use samples are evaluated correctly rather than rejected for "not belonging to the harm category." The same checker serves standalone content-moderation, constitution-seeded inputs, and (via `build_output_quality_checker`) output checking. `_build_constitution_checker()` is a thin backward-compatible alias.

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

140+ jailbreak techniques organized in four families. Technique definitions are taxonomy-driven where applicable — adding a new variant means adding a JSON entry, not a new function.

**Execution model.** LLM-dependent techniques are **generators** that `yield` an `LLMRequest` and resume via `.send(response)` (`protocol.py`); pure transforms are plain callables. `engine.batch_apply_combinations()` advances a chunk of samples **round by round**, grouping pending requests by model and dispatching one batch per model per round through the router. `generate_jailbreaks()` first **plans** the whole run to a JSONL manifest (`manifest.py`), then **executes** it in chunks — resumable from the output CSV. `combination_spec.json` defines layers, family caps, cross-incompatibilities, and complexity budgets; combinations are assigned deterministically (SHA-256 of seed + content-id + iteration). Any technique whose name is missing from the spec is silently never sampled — keep them in sync.

**Multi-round escalation.** Pass `settings_per_iteration` (a `list[dict]`, one dict of sampler kwargs per round) to augment every sample across an escalating schedule — the list length sets the round count. `default_escalation_schedule()` is the built-in 4-round default (each sample targeted 4×): round 1 = exactly 1 technique, round 2 = exactly 2, rounds 3–4 = rising complexity budgets. Rounds 1–2 are *count-driven* (`exact_techniques`, which ignores the complexity budget — count is the only constraint); rounds 3–4 are *budget-driven* (`max_complexity` / `max_obfuscations` / `sampling_probs`). The run is still planned up front from one shared pool; `settings_per_iteration=None` (default) keeps single-round behavior.

```python
from redact import generate_jailbreaks, default_escalation_schedule

# 4 rounds per sample: 1 technique → 2 techniques → higher → even higher complexity
jailbreaks = generate_jailbreaks(inputs=inputs, settings_per_iteration=default_escalation_schedule())

# Or a custom schedule — exact-count rounds carry only exact_techniques,
# budget rounds carry only the budget knobs:
schedule = [{"exact_techniques": 1}, {"exact_techniques": 3}, {"max_complexity": 9, "max_obfuscations": 3}]
jailbreaks = generate_jailbreaks(inputs=inputs, settings_per_iteration=schedule)
```

#### Obfuscation

| Type | Module | Functions | LLM Required |
|---|---|---|---|
| Encoding | `encoding.py` | `to_base64`, `to_rot13`, `to_rot18`, `to_rot47`, `to_unicode_escape`, `to_ascii_ordinal`, `to_separator`, `to_leetspeak_{basic,intermediate,advanced}`, `to_morse`, `to_braille` | No |
| Structural | `structural.py` | `to_json`, `to_xml`, `to_markdown` | No |
| ASCII Art | `ascii_art.py` | `to_ascii_art` (19 pyfiglet fonts) | No |
| Suffixes | `suffixes.py` | `to_adversarial_suffix_{punctuation,fragments,unicode,emoji}` | No |
| TokenBreak | `tokenbreak.py` | `to_tokenbreak_{prepend,split,delimiter}` | Yes |
| Sensitive Words | `tokenbreak.py` | `to_sensitive_words_encode_{base64,rot13,rot18,rot47,unicode,ascii,separator,leetspeak_*}`, `to_sensitive_words_{split,star,hyphen,underscore,variables}`, `to_synonym_substitution` | Yes |
| Typos | `typos.py` | `to_rewrite_with_typos_{low,medium,high,insane}` | Yes |
| Translation | `translation.py` | 20 languages across resource tiers: French, Japanese, Russian, Spanish, German, Arabic, Turkish, Czech, Vietnamese, Greek, Croatian, Swahili, Thai, Khmer, Maori, Nepali, Zulu, Scots Gaelic, Bengali, Javanese | Yes |

Sensitive-words functions share the `extract_harmful()` LLM detection step from TokenBreak — encoding is then applied only to the detected harmful words rather than the whole prompt. Typo rewriting uses a single LLM call with a level-description injected into the template.

#### Hacking

| Type | Module | Functions | LLM Required |
|---|---|---|---|
| Cognitive | `cognitive.py` | `to_persona_roleplay`, `to_hypothetical_framing`, `to_authority_obedience`, `to_avi`, `to_deep_inception` | Yes |
| Named Personas | `personas.py` | `to_invented_persona` + 14 named archetypes (psychopath, alien, cult_leader, very_advanced_ai, cartel_leader, artist, mentally_ill, deformed_scientist, politician, deformed_professor, religious_figure, radical_politician, actor, someone_from_the_future) | Yes |
| Framing | `framing.py` | `to_fictional_world`, `to_noble_goal`, `to_nefarious_goal`, `to_high_stake`, `to_no_moral_constraints` | No |

Cognitive and persona techniques are two-step LLM processes: **scenario generation → jailbreak construction**. Named personas use a persona-grounded scenario prompt that grounds the scenario in the persona's character before construction. Cognitive technique definitions are loaded from `configs/taxonomy/cognitive_techniques.json`; persona definitions from `configs/taxonomy/personas.json` — both are editable without touching code.

Framing directives are pure transforms covering scenario and intent modification. Each has 4 named template variants stored in `configs/framing_templates.json`; one is randomly selected per call with the variant name recorded in `additional_info`. Answer-format directives that were previously part of framing have moved to `requests/answer.py`.

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
- ASCII art obfuscation (19 pyfiglet fonts)
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
| `clean_sample()` | Strip markdown formatting, meta-commentary, and ChatML tokens (`<\|im_end\|>`) |
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

For LLM-dependent techniques, write a **technique generator**: `yield` an `LLMRequest` and resume with the model's reply. This lets the batched engine pool your call with every other sample's call for the same model. Add a matching entry to `combination_spec.json` (with a `families`/`complexity`) or the technique will never be sampled.

```python
from redact.jailbreak.protocol import LLMRequest, TechniqueGen

def my_llm_technique(prompt: str, *, gen_model: str | None = None, **kwargs) -> TechniqueGen:
    """Reframe the prompt via one LLM round."""
    messages = [{"role": "user", "content": f"Reframe: {prompt}"}]
    result = yield LLMRequest(gen_model, messages)   # engine batches this call
    return result.strip(), "llm_reframed"
```

Multi-round techniques (translate→check→retry, scenario→construction) simply `yield` more than once. The engine drives every generator round-by-round; for a single sample, `apply_combination()` runs the same generator synchronously.

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
| `HF_HOME` | vLLM models | Directory where vLLM downloads and caches model weights. Defaults to `~/.cache/huggingface` if unset. Set this to a path with sufficient disk space (the `venice-uncensored-vllm` model requires ~48 GB). |
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
