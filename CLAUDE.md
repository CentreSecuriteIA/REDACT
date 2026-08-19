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
│       ├── paths.py                 # Single source of truth for on-disk layout (data_dir → all default paths)
│       ├── pipelines.py             # High-level pipeline functions (one data_dir root, models by role)
│       ├── runconfig.py             # Config-driven runs: recipe + input-params + per-stage run manifests
│       ├── exceptions.py            # Custom exception hierarchy
│       ├── py.typed                 # PEP 561 type marker
│       │
│       ├── llms/                    # LLM abstraction layer (model-agnostic)
│       │   ├── base.py              # Abstract LLMBackend + capability flags (supports_native_batching / supports_parallel_calls / supports_internals)
│       │   ├── api.py               # get_backend(model) — backend auto-router + per-type cache
│       │   ├── venice_backend.py    # Venice / OpenAI-compatible API backend
│       │   ├── anthropic_backend.py # Anthropic Claude backend (native SDK, series-only)
│       │   ├── vllm_backend.py      # Local vLLM backend (native single-pass batching)
│       │   ├── introspection_backend.py # Local transformers backend — captures hidden states/attention/logprobs, keyed on caller-supplied internals_id(s)
│       │   ├── router.py            # ModelRouter — process-wide RateLimiter + per-model BatchCaller cache; role lookup
│       │   ├── wrappers.py          # RateLimiter, retry wrappers, BatchCaller (capability-aware dispatch)
│       │   ├── calls.py             # High-level generate/check call pairs
│       │   ├── translator.py        # Translation calls (jailbreak pipeline)
│       │   ├── prompts.py           # JSON prompt loader and template renderer
│       │   ├── model_config.py      # Model registry (RPM, backend_type, capability flags, roles)
│       │   ├── conversation.py      # Model-layer conversation primitives: LLMRequest, Step/Transcript (typed step log), drive_sync
│       │   └── extraction.py        # Regex extraction utilities
│       │
│       ├── multi_turn/              # General multi-turn conversation datasets (Setting → Trajectory); Actors, runner
│       │
│       ├── content_moderation/      # Content moderation dataset generation
│       │   ├── generation.py        # InputPipeline core + constitution-seeded (run_from_constitution) + run_output_generation
│       │   ├── standalone_generation.py # _StandaloneGenerationMixin: run_category/run_standalone (DEPRECATED path), split out of generation.py
│       │   ├── results.py           # SampleResult/TurnResult/CategoryResult/ConstitutionInputResult — dependency-free, shared by both above
│       │   ├── checker.py           # Entry-type-aware quality + output + category checkers
│       │   ├── metaprompt.py        # Meta-prompt generation (description + seeds)
│       │   └── paraphrase.py        # Batched paraphrase / fingerprint removal (real; paraphraser role) + generate_paraphrases orchestration
│       │
│       ├── dataset/                 # Data handling utilities
│       │   ├── io.py                # CSV read/write per category folder (+ _hash_text content id)
│       │   ├── sidecar.py           # JsonlSidecar — shared base for the ledger + manifest (path/mkdir/reset/bad-line-robust read)
│       │   ├── ledger.py            # Shared sidecar resume-ledger (Ledger) — single impl behind every stage's *.state.jsonl
│       │   ├── manifest.py          # Shared sidecar run-manifest (Manifest) — the plan-ahead *.manifest.jsonl for every stage
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
│       │   ├── utils.py             # Backward-compat re-export shim (was ~820 lines mixing 7 concerns; now re-exports from the 4 files below)
│       │   ├── spec.py              # load_spec / tag_all_functions — combination_spec.json loading + tagging
│       │   ├── chain.py             # combine_techniques, make_combination_gen, apply_combination, is_noop — run one sample through a (combined) technique chain
│       │   ├── sampling.py          # get_compatible_remaining, sample_combination, sample_exact_combination, default_escalation_schedule
│       │   ├── assignment.py        # assign_combination, build_combination, build_function_registry — manifest planning + reconstruction
│       │   ├── dynamic_functions.py # bind_functions — shared "inject dynamically-named functions into module globals" helper
│       │   ├── directives.py        # load_templates, apply_template_directive — shared template-directive machinery (requests/* + hacking/framing.py)
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
├── notebooks/                       # Walkthroughs: eval_pipeline, training_pipeline, content_moderation, jailbreak_augmentation
├── scripts/                         # run.py (recipe → pipeline), inspect_run.py (summarize manifests)
├── src/redact/configs/runs/         # Example recipe + input-params JSON
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

**Base class** (`base.py`): Abstract `LLMBackend` with `generate(messages, model)` and `batch_generate(messages_list, model)`, plus three capability properties — `supports_native_batching` (vLLM: one engine pass), `supports_parallel_calls` (False for Anthropic/vLLM), and `supports_internals` (True only for `TransformersIntrospectionBackend`). These flags are how every layer above stays backend-agnostic: callers never branch on backend type, they read the flags.

**Backends** (one class per provider; `api.get_backend(model)` auto-selects by `backend_type`/name and caches per type):
- `venice_backend.py` — Venice / OpenAI-compatible API (uncensored hosted models)
- `anthropic_backend.py` — Claude via native SDK (system param split out); series-only
- `vllm_backend.py` — Local vLLM for self-hosted inference; native single-pass batching. Supports two prompt formats: ChatML (default, for Dolphin/Hermes) and Mistral `[INST]/[SYSTEM_PROMPT]` (set `use_mistral_format=True` via `vllm_kwargs` in the model config, required for Mistral-3.x family models). Also strips the `|>` artifact that appears in responses when ChatML tokens are absent from the tokenizer vocabulary (vLLM ≥ 0.16.0 with Mistral-3.x tokenizers).
- `introspection_backend.py` — `TransformersIntrospectionBackend`: local raw-HF-`transformers` inference (not a vLLM extension — vLLM's continuous batching discards intermediate activations by design) for research/interpretability logging. Config-driven `capture` (`logprobs` / `hidden_states`: `False`/`"last"`/`"all"` / `attention`), defaulting to the lightest option (logprobs + last-layer hidden state) since attention scales `layers × heads × seq_len²` and there's no batching to amortize storage over. Registered via `backend_type="transformers_introspect"` + `hf_model_id` + `introspect_kwargs={"log_dir": ..., "capture": {...}}`; `supports_native_batching=False`, `supports_parallel_calls=False`, `supports_internals=True`. Capture is opt-in per call: pass `internals_id` (single) / `internals_ids` (batch, one per item) — an id the *caller* computes and internals are written under `{log_dir}/{internals_id}/`; `introspection_backend.py` never interprets the id beyond that (no CSV columns, no ids of its own — see "Internals capture is a non-interfering hook" below). `BatchCaller` is the enforcement point: it raises `ValueError` if `internals_ids` is passed to a backend with `supports_internals=False`, rather than letting it silently reach (and crash inside) e.g. Venice/Anthropic's SDK call. Every capture also gets an always-written `meta.json` alongside whatever tensors `capture` asks for — resolved model/settings, input messages, output text — self-contained for analysis, and the only place multi-round jailbreak techniques' discarded intermediate prompts/completions survive at all. `LLMBackend.rename_capture(old_id, new_id)` (default no-op, real move-the-folder implementation here) lets a caller capture under a pre-call-known **provisional** id and relabel it once a stage's real post-hoc identity (its `sample_id`) is known — needed wherever the resulting text's own hash isn't knowable before the call returns (paraphrase, jailbreak). Wired into `generate_outputs()` (`f"{input_id}/output"` / `.../val_out`, no rename needed — the row's identity is already `input_id`), `run_from_constitution` (`input`/`val_in`, gated to one sample per LLM call — see the design doc), paraphrase (`{input_id}/paraphrase/{sample_id}/`, provisional-then-renamed), and the jailbreak engine (`{input_id}/jailbreak/{sample_id}/{technique}/{n}/`, `_tag_yields` in `jailbreak/chain.py` tags each technique's LLM calls per-request, checking that *specific* request's own target model's backend — so a chain mixing an internals-capable `gen_model` with a non-capable `translate_model` captures exactly what's capturable). Not yet wired: multi-turn conversations (subsystem itself still settling). Full design/status: `.claude/introspection_backend_plan.md`.

**`model_config.py`**: `MODEL_REGISTRY` of `ModelConfig` entries — RPM, default gen params, `backend_type`, capability flags, `recommended_max_workers`, and a logical `role` (`uncensored_gen`, `translation`, `constitution_gen`, `uncensored_local`). `register_model()` adds entries at runtime; `default_model_for_role()` powers role lookup.

**`router.py`**: `ModelRouter` (process-wide via `get_router()`) owns the single shared `RateLimiter` and a lazy per-model `BatchCaller` cache, and resolves roles → `(backend, model)`. It is the single entry point for rate-limited, capability-aware generation. Both generation and checker paths now dispatch through a `BatchCaller` (either the router's or one built via `BatchCaller.from_model(backend, model, rate_limiter=...)` that wraps an explicitly-passed backend) — never `backend.batch_generate()` directly — so per-model RPM is enforced everywhere. New code should keep this invariant: route batch dispatch through a `BatchCaller`/the router, not the raw backend.

**`wrappers.py`**: `RateLimiter` (thread-safe sliding-window RPM per model), retry wrappers, and `BatchCaller` which dispatches by capability flag: vLLM → native batch, parallel-safe API → thread pool (`recommended_max_workers`), series-only → sequential. Misconfiguration (e.g. `max_workers>1` on a series-only backend) raises rather than silently degrading — same treatment for `internals_ids`/`internals_id` on a backend with `supports_internals=False`. `assert_single_sample_per_call(backend, samples_per_call)` is the shared guard for pipelines shaped like `run_from_constitution` (one LLM call can return several samples) — raises when internals capture is requested against a multi-sample-per-call setting, since one forward pass's internals can't be attributed to any single resulting sample. Not yet wired into any pipeline (see the introspection-backend plan).

**`calls.py`**: High-level paired calls — `generate_sample()`, `check_sample()`, and `batch_check_samples()`. Checker receives generated output and returns accept/reject with reasoning (acceptance = response starts with yes/ok/accept/pass). Reasoning is passed back to generator on rejection for directed improvement. `batch_check_samples()` sends all checker prompts in one `batch_generate()` call — the primary speed lever for vLLM. For API backends, `batch_generate()` falls back to sequential.

**`prompts.py`**: Loads prompt JSON files by category. Renders templates with seed injections.

**`translator.py`**: Wraps generate calls for multilingual output. Used by jailbreak pipeline for language-based attack vectors.

**`conversation.py`**: Model-layer conversation primitives, shared so `jailbreak/` and `multi_turn/` reuse them without a cross-dependency. `LLMRequest` (model + messages; the engine's routing key) is **defined here** and re-exported from `jailbreak/protocol.py` for back-compat. `Step`/`Transcript` are a **typed step log** — visible turns (`message`/`reply`, ChatMessage-like, rendered via `as_messages()`) plus provenance events (`strategy`/`analysis`/`evaluation`) that are logged but never sent to models. `drive_sync` drives one generator with a blocking call fn (single-conversation path; the batched engine drives many).

---

### `multi_turn/` — General multi-turn conversation datasets

Use-case-agnostic conversation generation: define a **`Setting`** (how it's *driven* — `conversation` (plain) vs `feedback` (adaptive) — plus participants, `max_turns`, optional `stop`) and play it on a seed to produce a **`Trajectory`** (the full typed step log). An **`Actor`**'s turn is a generator `turn(transcript) -> yields LLMRequest, returns list[Step]`: `ScriptedActor` (seed/replay, no LLM), `ModelActor` (one LLM turn), `StrategyActor` (strategize→ask→analyze, all logged). `run_conversation(setting, seed, call)` is the single-conversation path; `generate_conversations(seeds, setting, data_dir, iterations, resume, …)` is the batched, resumable pipeline (writes `conversations.csv`), and `evaluate_conversations(...)` is the separable judge pass (writes `conversations_scored.csv`).

**Library-consistent naming.** Conversation rows/units use the same vocabulary as every other stage — `input_id` (source seed row's id, mirroring jailbreak/output/paraphrase), `iteration` (0-based per-source index; independent stochastic variants, *not* escalation rounds — the `iterations` param sets the count), `source="conversation"`, and the `(input_id, iteration)` ledger key. `Trajectory.input_id` matches. So conversation datasets join / dedup / resume identically to inputs, jailbreaks, and paraphrases — the subsystem is a first-class part of the library, not an appendix.

Dependency direction: `multi_turn/` → `llms/` only. The jailbreak multi-turn **attacks** and **optimization** loops live in their *own* top-level folders (`multiturn_attacks/`, `optimization/`) that build on `multi_turn/` — never inside it. Oriented toward Inspect (seed≈Sample, messages≈ChatMessage, step-log≈events, Setting.drive≈solver, judge≈scorer). Design/plan: `.claude/theme4_multiturn_plan.md`.

---

### `dataset/` — Data Handling

All output is saved as CSVs in `Datasets/{category}/`. Functions:

- `io.py` — Read/write category CSVs, append samples incrementally
- `sidecar.py` — `JsonlSidecar`: the shared base for both the ledger and the manifest — path resolution (`.sidecar(artifact)`), `.exists()`/`.reset()`, mkdir-on-write, and bad-line-robust JSONL reads. `Ledger` and `Manifest` add only their differing semantics on top, so the "resume/plan subsystem" is one modular system, not two copies.
- `ledger.py` — `Ledger(JsonlSidecar)`: the **single** sidecar-**resume** implementation behind every stage's `*.state.jsonl` (see **Resume model** below). One JSON object per completed unit, keyed on one or more fields (`Ledger(path, key_fields, casters)`), `.completed()` / `.record(rows)` / `.reset()` / `.sidecar(artifact)`. Composite-key and status-field aware, append-only.
- `manifest.py` — `Manifest(JsonlSidecar)`: the **single** sidecar-**plan** implementation behind every stage's `*.manifest.jsonl` — the run plan written *before* generation (one row per unit). `.write(rows)` (idempotent overwrite) / `.load()` (list[dict]) / `.sidecar(artifact)`.
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
- **Family-first sampling.** Every technique draw goes through `_pick_by_family()` (in `sampling.py`): it picks a **family** uniformly first, then a technique within it. This stops over-sized families (e.g. 20 translation languages, 15 personas) from dominating a flat `rng.choice` over individual functions — translation competes as *one* obfuscation family, not 20. Used by all four pick sites: the hacking/manipulation/requests layer picks, the obfuscation loop, the Phase-5 request-obfuscation pick, and `sample_exact_combination`.
- **Translation is the cost hot-spot.** Each translation technique is a translate→check→retry generator (`obfuscation/translation.py`, `num_retries=4`), so a *single* translation makes **2–8 LLM calls** on the translation-role model (DeepSeek), not one — both the translate and the faithfulness check route there. `generate_jailbreaks(include_translation=False)` drops the whole translation family from the pool at build time (after the `pure_only` filter, matched by `"translation" in fn.families`) so it is never sampled — the recommended first-pass setting.
- `pipelines.generate_jailbreaks()` runs two phases: **plan** (`manifest.plan_run` writes one JSONL line per sample×iteration before any generation) then **execute** (stream the manifest in chunks through the engine, appending output per chunk). Resume is driven by the sidecar `jailbreaks.state.jsonl` **ledger** (keyed `(input_id, iteration)`, acked per chunk after the CSV append), **unioned** with the output CSV so pre-ledger runs still resume; `resume=False` clears the CSV and the ledger. `run_sync` (in `protocol.py`) is the single-sample equivalent used by `apply_combination` and tests.
- **Multi-round escalation.** `generate_jailbreaks(settings_per_iteration=...)` augments every sample across a schedule of rounds — one dict of sampler kwargs per round; the list length sets `iterations` (it wins over the scalar). `default_escalation_schedule()` (in `jailbreak/sampling.py`, re-exported from `jailbreak/utils.py` and from `redact`) is the built-in 4-round default (each sample targeted 4×): round 1 = exactly 1 technique, round 2 = exactly 2, rounds 3–4 = rising complexity budgets. Rounds 1–2 are **count-driven** via `sample_exact_combination()` / `sample_combination(exact_techniques=N)`, which pick exactly N compatible techniques and **ignore the complexity budget** (`remaining_complexity=None`; only layer caps + cross-incompatibilities apply, best-effort if the pool runs out); rounds 3–4 are **budget-driven** (the probabilistic `max_complexity`/`max_obfuscations`/`sampling_probs` path). Keep the two modes' knobs disjoint in a custom schedule — `max_complexity` on an `exact_techniques` round is a silent no-op. The whole run is still planned up front from one shared pool, built once from top-level `include_*` flags (the superset); per-round `include_*`/`sampling_probs` only restrict sampling. `settings_per_iteration=None` (default) keeps single-round behavior, and the manifest still freezes choices (re-run with `resume=False` after changing the schedule).

**Reference list coverage:** Benchmarked against 73 instruction primitives + 74 request primitives. All feasible primitives are covered. Intentionally excluded: `agent_context_additional_instr` (system-prompt access required), `fine_tuning` (out of scope), `use_highly_specialized_language` (unclear path), `direct_question` (no-op). The library is a strict superset of the reference list on everything else, with additional techniques not in the reference set (ASCII art, adversarial suffixes, structural wrapping, cognitive hacking, manipulation, continuation attacks, indirect embedding, extra encodings, extra languages).

---

### `content_moderation/` — Content Moderation Generation

`InputPipeline` (core in `generation.py`) is the single driver for both input modes. It's one class split across two files via a mixin — `class InputPipeline(_StandaloneGenerationMixin)` — so every method call site (`pipeline.run_category(...)`, `pipeline.run_from_constitution(...)`, etc.) is unaffected by which file actually defines the method:

- **Standalone** (`run_category`/`run_standalone`, in `standalone_generation.py`; reached via `pipelines.generate_inputs` with `constitution_df=None`): meta-prompt generation (description + seeds) → multi-turn generation → per-sample checking with the entry-type-aware quality checker (defaults to `harmful`). Rejection feedback and a prohibited-sample list feed the next turn. **Deprecated** (both entry points emit `DeprecationWarning`) in favour of constitution-seeded generation, which gives better/more adjustable coverage — but kept fully functional as the cheap "quick eval" path. Has no discrete idempotent units, so it is the one stage with neither a manifest nor a ledger (resumes by extending its CSVs). `run_category` (fixed `num_turns`, caller-supplied per-turn seeds) and `run_standalone` (target-driven, generates its own seeds, multi-category) are two deliberately-separate primitives, not accidental duplication — see `run_category`'s docstring for why merging them wasn't worth the risk. `generate_batch`/`check_samples`/`run_turn` (the per-turn primitives) live alongside them in the same file/mixin; `run_from_constitution` never calls them.
- **Constitution-seeded** (`run_from_constitution`, in `generation.py` — the active, recommended path; also reached via `generate_inputs(constitution_df=...)`): each constitution entry becomes one batched generation request; one `batch_generate` per chunk for generation, one flat batch for checking. A `constitution_inputs.manifest.jsonl` plan (one row per entry, keyed like the ledger) is written before the batch loop. Resume uses the sidecar **`constitution_inputs.state.jsonl` ledger** (keyed `(sample_description, entry_type, style)`, acked per entry after its rows are saved), **unioned** with the existing per-category CSVs so pre-ledger runs still resume. The `fresh` parameter (forwarded from `generate_inputs` as `fresh=not resume`): `fresh=False` (default) resumes; `fresh=True` clears the per-category CSVs and the ledger first (style-aware fresh keeps other styles' acks). This is the one CM stage where the *expansion* is cheap — the expensive Opus work upstream is the ledger-protected constitution generation.

Result dataclasses (`SampleResult`, `TurnResult`, `CategoryResult`, `ConstitutionInputResult`) live in `results.py`, dependency-free (mirrors `redact/types.py`'s role) so `generation.py` and `standalone_generation.py` can both depend on them without importing each other — `SampleResult` in particular is used by both the standalone mixin's `check_samples` and `run_from_constitution`. Both `generation.py` and `standalone_generation.py` re-export names existing code imports directly from them (`InputPipeline`, `SampleResult`, etc.) via an explicit `__all__`.

**Output pipeline** (`run_output_generation`, in `generation.py`; called from `pipelines.generate_outputs`): batched generation of model responses for input samples → entry-type-aware output checker (refusals on harmful inputs are rejected; benign inputs judged normally) → incremental append to `output_responses.csv`. A `output_responses.manifest.jsonl` plan (one row per `input_id`) is written before the batch loop. Resumable like the jailbreak run, with completion tracked in a **sidecar `output_responses.state.jsonl` ledger** (keyed by the same content-hash id as inputs/jailbreaks), kept separate from the CSV so a large or hand-edited CSV can't corrupt resume state. `resume=True` (default) skips already-completed ids; `fresh=True` clears both the CSV and the ledger. Optional paraphrase / fingerprint removal (`paraphrase.py`, which also now owns the `generate_paraphrases` plan/ledger/execute orchestration, moved there from `pipelines.py`) drives the prompt with a stand-in model; the real removal model is trained in a separate repository.

---

### `constitution/` — Constitution Generation

Generates structured category hierarchies for constitutional classifier training. Each constitution spans 4 severity levels:

1. **Absolutely harmful** — clear-cut violations, always flag
2. **Dual-use harmful** — borderline, harmful framing, debatable
3. **Dual-use benign** — borderline, benign framing, could look harmful
4. **Absolutely benign** — clearly safe, never flag (hard negatives)

`ConstitutionPipeline` (in `generation.py`) generates entries per taxonomy category using Claude Opus. Uses `parse_constitution()` from `llms/extraction.py` to parse the 3-layer markdown output. Entries are saved to `Data_cache/constitution/` as per-type CSVs plus `merged.csv`. Each entry later seeds N input samples for classifier training.

**Resumable + incremental.** A `constitution.manifest.jsonl` plan (one row per `(source_category, entry_type)` unit + the standalone-benign unit) is written before the category loop. Each `(source_category, entry_type)` LLM call (plus the standalone-benign call, keyed `general::general_benign`) is one resumable unit. Entries are flushed to the per-type / `merged.csv` CSVs as each category completes — never held in memory for the whole run — and the unit is recorded in a sidecar `constitution.state.jsonl` ledger (`_state_path` / `_read_state` / `_append_state` in `generation.py`, thin adapters over the shared `dataset.Ledger`) only after its rows are saved. `resume=True` (default) skips ledger-recorded units and appends; a unit that fails all retries stays un-acked so it retries next run, so a crash loses at most one unit. `fresh=True` clears the CSVs and the ledger first. `run()` / `generate_constitution()` return the on-disk `merged.csv` (read back via `_load_saved_result()`), mirroring how `generate_outputs` returns `pd.read_csv(out_path)`. Same sidecar-ledger pattern as the content-moderation output pipeline.

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

Every public function takes one **`data_dir`** working root (all `Datasets/`/`Data_cache/` output
derives from it via `paths.py`), a **role-defaulted `model`** (pass `None` → the registry role;
constitution → Opus, generation → uncensored), and a single **`resume`** flag (default `True`).
The public `backend` param was removed — backends auto-resolve from the model name.

```python
from redact import (
    generate_constitution, generate_inputs, generate_outputs,
    generate_jailbreaks, generate_paraphrases, build_dataset,
    default_escalation_schedule,
)

DATA_DIR = "./runs/training"

# Training path: constitution-seeded inputs (Claude Opus → entries → prompts).
# Constitution-seeded inputs auto-route to Datasets/constitution_inputs/.
constitution = generate_constitution(data_dir=DATA_DIR, num_taxonomy_categories=3)
inputs = generate_inputs(data_dir=DATA_DIR, constitution_df=constitution, samples_per_entry=3)

# Eval path: standalone meta-prompt inputs (DEPRECATED — emits DeprecationWarning, still
# functional; prefer the constitution-seeded path above for better/adjustable coverage).
inputs = generate_inputs(data_dir=DATA_DIR, samples_per_category=15, num_categories=3)

jailbreaks = generate_jailbreaks(data_dir=DATA_DIR, inputs=inputs)   # plan → batched execute

# Or escalate: each sample augmented 4× (1 technique → 2 → higher complexity → even higher)
jailbreaks = generate_jailbreaks(
    data_dir=DATA_DIR, inputs=inputs,
    translation_model=None,                               # None → translation role (DeepSeek)
    settings_per_iteration=default_escalation_schedule(),  # or a custom list[dict]
)

outputs = generate_outputs(data_dir=DATA_DIR, inputs=inputs)  # model responses + output checker

# Paraphrase (fingerprint removal) — additive copies of base inputs + accepted base outputs.
# K 1:1 calls round-robin over the `paraphraser` role; a SEPARATE meaning-preservation checker
# (check→drop); deduped. Writes paraphrases_{inputs,outputs}.csv, resumable via a mapping file.
paraphrases = generate_paraphrases(data_dir=DATA_DIR, target="both", paraphrases_per_sample=1)

# Merge. mode="training" merges paraphrases into complete_dataset.csv; mode="eval" writes them
# to a separate paraphrased.csv. Jailbreaks are only ever built from the base inputs.
dataset = build_dataset(data_dir=DATA_DIR, mode="training")   # base + jailbreaks + outputs (+ paraphrases)
```

**Config-driven** — a whole run from a recipe (`dataset_type` `"eval"`/`"training"`) + input-params:

```python
from redact import run_pipeline
summary = run_pipeline("src/redact/configs/runs/eval_example.json")   # or scripts/run.py
```

Backends are auto-selected from the model name (`get_backend`), and all model-name strings resolve through the registry in `llms/model_config.py`. Single-file `*_path` overrides (`output_path`, `jailbreak_path`, `benign_path`, `inputs_path`, `responses_path`, `manifest_path`) remain for power users; `taxonomy_dir`/`prompt_dir` relocate config/prompt inputs. For a genuinely custom endpoint, `register_model()` it (the backend resolves from the name) or construct a `VLLMBackend` and use it via the lower-level pipeline classes.

---

## Key Design Decisions

- **Prompts are external** — No prompts hardcoded in library code. Adding a new category = adding a JSON file.
- **Backends are swappable** — Switching from API to vLLM is a config change, not a code change.
- **Data is per-category** — All output lands in `Datasets/{category}/` as CSV. Merging is explicit and on-demand.
- **Checker reasoning feeds back to generator** — Rejection is not just a signal; the checker's reasoning is passed into the next generation call for directed improvement.
- **`Data_cache/` is internal** — Intermediate artifacts (benign samples, partial batches) go here, never in `Datasets/`.
- **Resume model — one modular sidecar system** — Every batched stage plans + resumes through the *same* two sidecar files, both built on `redact.dataset.JsonlSidecar` (the shared base owning path/mkdir/reset/bad-line-robust reads): a `<artifact>.manifest.jsonl` **plan** (`redact.dataset.Manifest`, one row per unit, written *before* generation — idempotent overwrite) and a `<artifact>.state.jsonl` **ledger** (`redact.dataset.Ledger`, one row per *completed* unit, appended **only after** that unit's rows are flushed to CSV — crash-safe: a crash loses ≤ one chunk; resume never trusts a large/hand-edited CSV). `resume=True` skips ledger-recorded units; `resume=False` clears artifact, ledger **and** manifest. Both files key on the *same* per-stage unit id: constitution `source_category::entry_type`; CM-input-from-constitution `(sample_description, entry_type, style)`; CM-output `input_id`; paraphrase / jailbreaks / conversations `(input_id, iteration)`; conversation-scoring `sample_id` (ledger only — it scores existing rows). Plan-backed stages (jailbreaks, conversations) stream the manifest back to drive execution and union the ledger with the output CSV for back-compat. **Only exception:** standalone meta-prompt input generation has no discrete idempotent units (open-ended toward a sample-count target) — no manifest, no ledger; it resumes by extending its CSVs and is now **deprecated** (emits `DeprecationWarning`) in favour of constitution-seeded generation, though still functional. When adding a stage, use `Manifest` + `Ledger`, don't hand-roll `*.jsonl` read/write.
- **Concurrency is capability-driven** — a bare `BatchCaller()` defaults to `max_workers=1` (sequential, for easy debugging), but `BatchCaller.from_model()` (what the pipelines and router use) sizes workers from the model's `recommended_max_workers` and dispatches by capability flag (vLLM native batch / parallel API thread-pool / series-only sequential). Misconfiguring concurrency on a backend that can't support it raises rather than silently degrading.
- **Every dataset row has a `sample_id`** (its own content-hash identity) **and, where it has a traceable origin, an `input_id`** (the origin row's `sample_id`) — the same pairing pattern everywhere: CM input's `sample_id` is the root of the chain (no `input_id` — it *is* the origin); CM output, jailbreak, and paraphrase rows carry both (jailbreak's `sample_id` = `_hash_text(jailbreak_text)`, computed in `jailbreak/engine.py: _finalize()` — not inherited from the input row, which was a real bug fixed alongside this: every technique/iteration variant used to silently share one `sample_id` with `input_id`); conversations carry both too. Constitution generation's entries also get a `sample_id` (hash of `sample_description`) despite having no downstream consumer yet. When adding a stage, give every row a `sample_id`; add `input_id` only when there's a real origin row to point at.
- **Internals capture is a non-interfering hook, never a CSV column** — it must never touch a pipeline's CSVs, schema, control flow, ledger, or manifest; it only writes to `{log_dir}/...`. `internals_id(s)` is never a generic kwarg threaded through every layer's signature; it's an explicit param only `TransformersIntrospectionBackend`, `BatchCaller`, and (for jailbreak) `LLMRequest`/`drive_generators` know about, and the path is rooted at the stage's own `input_id` so no new id scheme exists anywhere. `BatchCaller` is the single place that checks `backend.supports_internals` and raises `ValueError` before dispatch if unsupported — never a silent crash three layers down inside a third-party SDK call. Wired across `generate_outputs`, `run_from_constitution`, paraphrase, and jailbreak (see the `introspection_backend.py` bullet above for the per-stage shapes); not yet multi-turn conversations. Full design/status: `.claude/introspection_backend_plan.md`.
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
