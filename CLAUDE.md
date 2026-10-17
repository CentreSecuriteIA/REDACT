# REDACT — Red-team Dataset Automation & Construction Toolkit

> Developer documentation: [docs/](docs/README.md), one page per stage and per layer, published at <https://centresecuriteia.github.io/REDACT/>.

## Project Overview

A modular pipeline for automated red-teaming dataset generation, built from the BELLS-O dataset iterations (content moderation input/output, jailbreaks) and reusable for other synthetic data tasks.

A run takes a taxonomy and a recipe and gives one checked dataset: a constitution (sample descriptions at four severities per category, plus a general-benign part per benign-taxonomy category), input samples seeded from it and checked by a second model, model responses, paraphrases, jailbreak attacks and natural variations, merged into one CSV. Every stage saves as it goes and resumes where it stopped. The library is model-agnostic (hosted APIs or local vLLM engines, planned onto the GPUs), prompt-agnostic (prompts are JSON files, overlaid per file) and category-agnostic (a new category is a taxonomy entry). The product goal: a taxonomy and a task in, a dataset out, with defaults rather than settings.

Prompts and template files ship with the library. A `prompt_dir` overlays them file by file (`scripts/scaffold_prompts.py` writes an editable copy).

---

## Architecture

```
REDACT/
├── src/
│   └── redact/                      # The package
│       ├── __init__.py              # NullHandler, .env load, set_seed, re-exports with __all__
│       ├── types.py                 # EntryType, ALL_ENTRY_TYPES; imports nothing from the package
│       ├── exceptions.py            # RedactError, PromptTooLongError
│       ├── paths.py                 # get_output_dir(); every file under a data_dir; the package's own folders
│       ├── stages.py                # The fronts (generate_*, build_dataset), RunContext, Stage, STAGES
│       ├── recipe.py                # load_recipe, load_params, load_dataset_type, serving_of, augmentations_of, benign_taxonomy_of, resolve_role
│       ├── driver.py                # run_pipeline; run records (record_path/write_record/read_record); file_counts; the card check
│       ├── telemetry.py             # Run trace (JSONL, or mirrored to W&B) + cost roll-up, stage labels, GPU rates
│       ├── pipelines.py, runconfig.py # Re-exports of stages / recipe + driver, kept one release
│       ├── py.typed                 # PEP 561 marker
│       │
│       ├── llms/                    # Model calls, backend-agnostic
│       │   ├── backends/
│       │   │   ├── base.py          # LLMBackend (a configured model), ComputeConfig, the system-prompt helpers
│       │   │   ├── capabilities.py  # BACKEND_TYPES, resolve_setup/transport_for/backend_for, validate_concurrency, set_run_serving, clear_transport_caches
│       │   │   ├── openai.py        # OpenAIBackend: any OpenAI-compatible endpoint (Venice, OpenRouter); sync and async SDK clients
│       │   │   ├── anthropic.py     # AnthropicBackend: the Messages API, one call at a time
│       │   │   ├── vllm.py          # VLLMBackend: AsyncLLM in-process; the engine cache, planned grants, engine processes, the exit hook
│       │   │   └── introspection.py # TransformersIntrospectionBackend: transformers, capturing hidden states/attention/logprobs
│       │   ├── resources/           # estimate.py (a model's GPU need from its HF config), measure.py (nvidia-smi probes, visible cards)
│       │   ├── prompting/           # prompts.py (load, overlay, PromptTemplate, format instructions), extraction.py (parse replies), scaffold.py
│       │   ├── client.py            # ModelClient: a backend + its dispatch; ModelClient.create() is the factory
│       │   ├── router.py            # generate_sample, batch_generate_samples, assert_single_sample_per_call
│       │   ├── wrappers.py          # RateLimiter, shared_limiter, BatchCaller
│       │   ├── model_config.py      # ModelConfig + APIConfig/VLLMConfig/IntrospectConfig; MODEL_REGISTRY; register_model, default_model_for_role
│       │   ├── observe.py           # The telemetry hook: record, set_emitter, set_stage, current_stage
│       │   ├── aio.py               # One event loop on a daemon thread for every async model call
│       │   └── progress.py          # ProgressReporter: throttled progress lines
│       │
│       ├── llm_pipeline/            # Units of model calls and their drivers
│       │   ├── request.py           # LLMRequest, LLMRequest.for_client
│       │   ├── drive.py             # drive_sync, drive_generators, drive_units (the pool and the async lane), pool_size
│       │   └── check.py             # is_accepted, check_sample, batch_check_samples, checked(), extracted()
│       │
│       ├── residency/               # The GPU planner
│       │   ├── footprint.py         # ModelFootprint, footprint(model): weights + KV, the engine a model loads on
│       │   ├── placement.py         # plan_residency: cards, grants, copies, the search
│       │   ├── plan.py              # ResidencyPlan
│       │   ├── phases.py            # Phase, plan_phases: one plan per stage
│       │   ├── lifecycle.py         # apply_plan, preload, unload_local, loaded_engines, loaded_grants
│       │   └── explain.py           # The plan as text; report() logs it and records the residency event
│       │
│       ├── constitution/            # The constitution stage
│       │   ├── generation.py        # ConstitutionPipeline.run, read_constitution
│       │   ├── units.py             # Unit (one call), plan_units, messages_of, rows_of, file_of
│       │   └── parse.py             # strip_end_marker ([END]), parse_constitution (3-level markdown)
│       │
│       ├── content_moderation/      # The inputs, outputs and paraphrase stages
│       │   ├── inputs.py            # InputPipeline.run_from_constitution: the inputs stage and its message building
│       │   ├── outputs.py           # run_output_generation: one checked response per input row
│       │   ├── paraphrase.py        # paraphrase_pool, run_paraphrase_target, paraphrase_batch
│       │   ├── checker.py           # build_quality_checker, build_output_quality_checker, build_paraphrase_checker
│       │   ├── feedback.py          # Run-time prompt text: rejection feedback, the do-not-repeat note
│       │   └── results.py           # ConstitutionInputResult: the inputs stage's totals
│       │
│       ├── jailbreak/               # The technique library and the augmentation engine
│       │   ├── technique.py         # The technique contract, DISCARDED, load_spec, tag
│       │   ├── pool.py              # all_techniques, technique_pool(kind), languages, registry, uses_translation; spec ⇔ registries checked at import
│       │   ├── rules.py             # compatible(): the spec's rules, both ways; request_obfuscation_candidates
│       │   ├── sampling.py          # sample_combination (budget), sample_exact_combination (count), family-first picks
│       │   ├── plan.py              # plan_run (dealt per group), assign_combination, combination_of, coverage, plan_path/plan_inputs/changed_plan_inputs, default_escalation_schedule
│       │   ├── chain.py             # run_chain, in_layer_order, combine_techniques, apply_combination
│       │   ├── units.py             # pending_units, unit_gen, row_of, error_row_of
│       │   ├── pipeline.py          # run_augmentation: the stage core, once per kind
│       │   ├── augmentation.py      # The augmentation section: KINDS, resolve, rounds_of, the deprecated flat keywords
│       │   ├── directives.py        # load_templates, apply_template_directive
│       │   ├── hacking/             # cognitive.py (5), personas.py (invented + 14 named), framing.py (5, templates)
│       │   ├── manipulation/        # benign.py (the benign Q&A pairs), pairs.py (what fsh and dap share), fsh.py, dap.py
│       │   ├── obfuscation/
│       │   │   ├── encoding.py      # base64, rot13/18/47, unicode escapes, ASCII ordinals, separator, leetspeak, morse, braille
│       │   │   ├── sensitive_words.py # Encode, split or substitute the words a model marks (16)
│       │   │   ├── tokenbreak.py    # Token splitting of the marked words (3); the strict word-list parse
│       │   │   ├── translation.py   # 20 languages (translate → check → retry)
│       │   │   ├── typos.py         # Model-written typos at 4 densities
│       │   │   ├── casual_writing.py # Casing, spacing, punctuation, abbreviations, accents (9, pure)
│       │   │   ├── suffixes.py      # Adversarial suffixes (4)
│       │   │   ├── structural.py    # JSON, XML, markdown wrapping
│       │   │   └── ascii_art.py     # A pyfiglet font drawn from the 16 that draw every character
│       │   └── requests/            # answer.py (9), answer_language.py (20), continuation.py (4), indirect.py (6), distractor.py (4), impersonation.py, temporal.py, asking.py (2)
│       │
│       ├── multi_turn/              # Conversation datasets: a Setting played on seeds, saved and scored
│       │   ├── transcript.py        # Step, StepType, Transcript
│       │   ├── actors.py            # Actor, ScriptedActor, ModelActor, StrategyActor
│       │   ├── conversation.py      # Setting, Trajectory, conversation_gen, run_conversation
│       │   ├── pipeline.py          # generate_conversations
│       │   └── evaluate.py          # evaluate_conversations, judge_prompt
│       ├── multiturn_attacks/       # setting.py (build_attack_setting), attacks.py (attacker strategies), pipeline.py (generate_attacks, score_attacks), optimize.py (optimize_attack)
│       ├── optimization/            # search.py: optimize(), a beam search over conversation moves
│       │
│       ├── dataset/                 # Rows on disk
│       │   ├── identity.py          # content_id, row_id, row_ids
│       │   ├── rows.py              # append_rows, check_columns, read_rows, iter_rows, count_rows, has_rows, keys_in, accepted_mask; write_concat over Parts
│       │   ├── sidecar.py           # JsonlSidecar, Ledger (*.state.jsonl), Manifest (*.manifest.jsonl)
│       │   ├── resume.py            # resume_state, clear_stage, release_lock
│       │   ├── save.py              # commit, CommitBuffer: rows, then acks
│       │   ├── drop.py              # drop_units: a fresh run of part of a stage
│       │   ├── categories.py        # The inputs tree constitution_inputs/{category}/samples.csv: sample_row, discover_categories, merge_all, input_files, merge_inputs, take_per_group
│       │   ├── taxonomy.py          # load_taxonomy, create_taxonomy, iter_categories
│       │   └── loading.py           # load_hf_dataset, load_from_config
│       │
│       ├── configs/                 # Package data
│       │   ├── llm/                 # models.json (the registry), roles.json, gpu_pricing.json
│       │   ├── dataset_types/       # training.json, eval.json: what each dataset_type means (stages, params, augmentations)
│       │   ├── runs/                # eval_example.json, training_example.json, params_example.json, params_training_example.json
│       │   ├── datasets/            # content_moderation_input.json: a HuggingFace dataset to load
│       │   ├── taxonomy/            # harmful.json (the default taxonomy), benign.json (the benign pairs' categories)
│       │   └── jailbreak/
│       │       ├── combination_spec.json # Layers, families, rules, layer_rates, per-technique tags: the source of truth for sampling
│       │       ├── augmentations.json    # The packaged augmentation section: the adversarial and natural kinds
│       │       ├── hacking/              # framing_templates.json, cognitive_techniques.json, personas.json
│       │       ├── obfuscation/          # structural_templates.json
│       │       ├── requests/             # answer, answer_language, continuation, indirect, distractor, impersonation, temporal, asking templates
│       │       └── multi_turn/           # crescendo_followups.json
│       │
│       └── prompts/                 # Prompt JSON per pipeline/step: constitution/, input/, output/, paraphrase/, jailbreak/, conversation/, format_instructions/
│
├── tests/                           # The test suite; fixtures/ pins the technique registry and each stage's files
├── notebooks/                       # eval_pipeline, training_pipeline, content_moderation, jailbreak_augmentation
├── scripts/                         # run.py, inspect_run.py, scaffold_prompts.py, download_models.py, gpu_check_residency.py
├── docs/                            # Developer docs and the mkdocs site: overview, running, configuration, techniques, extending, stages/, internals/, api/
├── .github/workflows/docs.yml       # Builds and publishes the docs site
├── mkdocs.yml                       # The docs site
├── pyproject.toml                   # Build config, dependencies, extras
├── requirements.lock                # A pinned environment that runs vLLM on a GPU pod
├── README.md
├── LICENSE
├── Datasets/                        # Stage output (gitignored)
├── Data_cache/                      # The constitution, the benign pairs (gitignored)
└── .env                             # API keys (gitignored)
```

---

## Module Details

### Top level — `stages.py`, `recipe.py`, `driver.py`

A stage's public function (its **front**) may be called alone, from a notebook or a test; everything about a *run* (serving, residency, telemetry, run records, counting) lives in the driver, which reads what it needs about each stage from one table.

**Fronts** (`stages.py`): `generate_constitution`, `generate_inputs`, `generate_outputs`, `generate_paraphrases`, `generate_jailbreaks`, `build_dataset`. Each stage front but `build_dataset` takes `data_dir` (the working root; every path derives from it through `paths.py`), a model name per role (`None` is the registry role's default), `prompt_dir`, `resume` (default `True`), `verbose`, `read_back` (`False` returns an empty frame) and its tunables; `build_dataset` takes `data_dir`, `mode`, its part paths and `include_*` flags, `verbose` and `read_back`. A front resolves its inputs (the frame given, else `dataset.merge_inputs(data_dir)`), its clients (`ModelClient.create`) and its output path (`paths.*(data_dir)` unless an `*_path` override is given), then calls its core: `ConstitutionPipeline.run`, `InputPipeline.run_from_constitution`, `run_output_generation`, `run_paraphrase_target` (per target), `jailbreak.pipeline.run_augmentation` (per augmentation kind), `dataset.write_concat`. The core owns the manifest, the ledger, `resume_state` and `CommitBuffer`; the front never touches a ledger.
- When the inputs hold no accepted row (all rejected, or an empty frame given), the outputs, jailbreak and paraphrase fronts log a WARNING (`<stage> skipped: ...`), write nothing and return an empty frame, so a recipe run goes on; with `resume=False` they first remove what an earlier run left (`dataset.clear_stage`: the output, its `.cut` and `.tmp` files, ledger, manifest, and a jailbreak kind's coverage table and plan file). A jailbreak kind whose `entry_types` match no input is skipped and cleared the same way. With no input tree at all, `generate_outputs` and `generate_jailbreaks` raise `ValueError`; `generate_paraphrases` skips the target at INFO.
- `generate_inputs` takes the constitution frame (`None`: the constitution saved under `data_dir`, else `ValueError`), then copies the accepted rows of every category file into `constitution_inputs_merged.csv` for `build_dataset`, removes that file when no row is accepted, and warns when under half of the generated samples are accepted. `generate_paraphrases` returns `{"inputs": df, "outputs": df}`.

**The `Stage` table**: `STAGES` maps each recipe stage name, in run order (`constitution`, `inputs`, `outputs`, `paraphrase`, `jailbreaks`, `build`), to a frozen `Stage(name, front, roles, files, recipe_kwargs, sources, folded=())`: the front's name, looked up on `redact` when the stage runs (`Stage.run`); the `(recipe role, registry role)` pairs it calls; `files(ctx)`, the CSVs that hold its rows and the column values a counted row holds; `recipe_kwargs(ctx)`, what the run passes the front (a value an earlier stage saves is a `Saved`, read when its stage starts: the constitution that seeds the inputs); `sources(ctx)`, the files it reads its inputs from (those that exist go in its run record; `[]` for none); `folded`, the params keys the run folds into a kwarg it passes itself (the jailbreak stage's flat keywords, into its `augmentation`). `Stage.params(ctx)` is the dataset type file's params for the stage with the params file's laid over them key by key. `Stage.check(ctx)` raises for a params key the front does not take or one the run passes itself, reading no stage file; `run_pipeline` calls it for every listed stage before the first runs, and `Stage.kwargs(ctx)` lays the checked params over the recipe kwargs. `RunContext`: `data_dir`, `dataset_type`, `models` (recipe role → name), `prompt_dir`, `taxonomy`, `taxonomy_dir`, `params`, `augmentations` (both kinds resolved), `resume`, `verbose`, `benign_taxonomy`. A new stage is a core, a front and a row (`docs/extending.md`).

**Recipe** (`recipe.py`): `load_recipe(path | dict)` reads and checks the keys; an unknown key raises, and `_`-prefixed keys are comments.
- `dataset_type` (`eval` | `training`, required): names the packaged file `configs/dataset_types/<type>.json` (`load_dataset_type`, cached, a copy per call; exactly the keys `stages`, `params`, `augmentations`, `_` comments aside) and is the build's mode (paraphrases merged only for `training`). The file's `stages` are the default, its `augmentations` the base the recipe's kinds overlay (`augmentation.resolve(section, kind, base=)`, over the packaged default), its `params` the base the params file overlays per stage. Both files list all six stages and set `constitution.include_general_benign` true with `general_benign_categories` 10; `eval.json` also sets `constitution`/`inputs` `entry_types` `["harmful", "benign"]`, `jailbreaks.seed` 43, an adversarial kind on the harmful inputs only (the families `encode` less the light encodings and `sensitive_words`, `fsh`, `dap`, `framing`, `structural`, `ascii_art`, `continuation`, `temporal`; the nine low-resource translations, `"translation": true`, `family_weights` `{"translation": 3}`; `to_refusal_then_comply`; rounds exactly 1 then 2 techniques) and `natural` off. Every run seeds its inputs from the constitution saved under `data_dir`, written by a `constitution` stage in `stages` or an earlier run; the `inputs` stage raises without one.
- `stages` (default: the dataset type's; `[]` runs nothing; an unknown or repeated name raises), `data_dir`, `taxonomy` (default `"harmful"`; the former name `"content_moderation_categories"` is read as `harmful` with a `DeprecationWarning` for one release), `taxonomy_dir`, `prompt_dir`, `benign_taxonomy` (default `"benign"`; passed to the constitution and jailbreak stages; `benign_taxonomy_of` reads it and checks it has a category when `constitution` runs with the type's `include_general_benign` on, or `jailbreaks` runs a kind that draws a `requires_benign` technique), `augmentations` (resolved by `augmentations_of(recipe, params["jailbreaks"])`), `params_file` (relative to the recipe), `resume` (a bool; any other value, `"false"` too, raises). A top-level `copies` other than null raises.
- `models`: a key per recipe role (`gen`, `check`, `translation`, `constitution`, `paraphraser`, `paraphrase_check`; any other key raises, a plain name too). A value is a registry name, `null` (the role's default), or `{name, backend_type, copies, workers}`, which also says how the model is served; `serving_of` checks it (`copies` only for a vLLM setup, `workers` only for an API one), and one model is served one way per run.
- `load_params` reads the params file: each front's kwargs under its stage's name. A top-level key that is not a stage name raises (`_` keys are comments); a `jailbreaks` key only the augmentation section holds raises, and a flat keyword warns.
- `resolve_role` gives the models a role stands for, as its stage resolves them: `check` left out is `gen`'s model, `paraphraser` may be `"distribution"` (every paraphraser), and `translation` is no model when no kind's pool has a translation technique.

**Driver** (`driver.py`): `run_pipeline(recipe, params=None, verbose=True, data_dir=None)`:
1. builds the `RunContext` and checks every stage's params; installs telemetry; with `verbose`, logs the prompt overlay (`report_prompt_sources`);
2. notes the engines loaded already (`residency.loaded_engines()`), sets the run's serving (`capabilities.set_run_serving`) and plans residency per stage (`residency.plan_phases`; a plan that does not fit is logged, never fatal);
3. before any model loads, `_check_cards` reads `nvidia-smi` once (`measure.card_use`) and raises `RuntimeError` when another process holds memory on a planned card, naming the cards, the processes and, found with `psutil`, the vLLM engine processes above them (titled `VLLM::`) in one `kill -9` line; a process on an unplanned card, or 1 GiB or more in use on a planned card with no process listed, only warns;
4. per stage: enters the stage's phase (unloads the engines the stage's plan leaves out, keeping those loaded before the run that no stage calls; preloads the stage's local models and the next stage's when they fit beside them), runs `stage.run(read_back=False, **stage.kwargs(ctx))` inside `telemetry.stage(name)`, counts rows from `stage.files(ctx)` (`file_counts`; no stage reads its rows back), and writes the run record `{data_dir}/Datasets/{stage}.run.json` (`write_record`: `stage`, `timestamp`, `params` (the merged ones), `models`, `counts`, `sources`, `serving` (per role: `name`, `backend_type`, and `copies` as planned, or `workers`), `dataset_type`, `dataset_type_hash` (`recipe.dataset_type_hash`: the content id of the type file as parsed); the `jailbreaks` record adds `spec_version` and `augmentation`);
5. `finally`: clears the serving; with a plan, waits for the preload threads at most `vllm.EXIT_WAIT_S` (60 s) together, unloads the engines the run loaded with a 10 s wait (which ends a load still running; engines loaded before the run stay) and clears the planned shares and cards; rolls telemetry up, after a failed stage too, without masking the stage's error.

It returns `{dataset_type, data_dir, stages: {name: {counts, manifest}}, serving, telemetry}`, plus `residency` and `trace` when present (`"manifest"` is the record's path, a key kept by that name). `scripts/run.py RECIPE [--params P] [--data-dir D] [--quiet|--debug]` is the CLI: it configures logging; the first Ctrl-C or SIGTERM exits through the `finally`, and a second calls `terminate_engine_processes(all=True, quiet=True)` (SIGTERM to the engine processes and their descendants, SIGKILL 5 s later, no other child; written to stderr without logging, with a `kill -9` line for any pid still alive) and exits with `os._exit`. `scripts/inspect_run.py DATA_DIR` prints each record. `redact.pipelines`, `redact.runconfig` and the names `manifest_path`/`write_manifest`/`read_manifest` are aliases kept for one release.

**Package root**: `__init__.py` attaches a `NullHandler` to the `redact` logger and sets no level, loads `.env` from `get_output_dir()` (its values win over the shell's) before the subpackages import, and lists its re-exports in `__all__`. Importing seeds nothing; `set_seed(seed=42)` seeds `random` and `numpy` when the caller asks. `paths.get_output_dir()` is `REDACT_OUTPUT_DIR`, else the first folder above the package holding `pyproject.toml` or `.git`, else the current directory. `exceptions.py` holds `RedactError` and `PromptTooLongError(RedactError)`, which a backend raises for a prompt over the model's context (see `llm_pipeline/`); elsewhere the library raises built-in errors.

### `llms/` — LLM abstraction layer

Everything above this layer calls one interface and reads capability flags; nothing branches on a backend's type. `llms/` imports nothing above it.

**Backends** (`backends/base.py`). A backend **is a configured model**: `from_config(config)` binds everything fixed about one model on one setup — name, `default_max_tokens`/`default_temperature`, system-prompt policy, `rpm`, `max_workers`, provider parameters, sampling defaults. Calls take only what varies: `generate(messages_list, *, max_tokens=None, temperature=None, internals_ids=None, **kwargs) -> list[str]` is batch-shaped (one sample is a batch of one), and backends with `supports_async_calls` add the coroutine `agenerate(messages, *, max_tokens=None, temperature=None, internals_id=None, **kwargs) -> str`. `_prepare()` splits system-role messages out (an empty system prompt counts as none) or, for a model with `supports_system_prompt=False`, folds them into the first user message (the first message when none is the user's), joined by a blank line, dropping empty ones; `_resolve()` fills `max_tokens`/`temperature` from the model's defaults. `max_workers` is clamped to 1 in `__init__` when the class has no `supports_parallel_calls`. Capabilities are one `compute_config` ClassVar, a frozen `ComputeConfig(supports_native_batching, supports_parallel_calls, supports_internals, supports_async_calls)`, read off the class without building a transport (which for a local one loads weights).

| Backend (`backends/`) | native batching | parallel calls | internals | async calls |
|---|---|---|---|---|
| `OpenAIBackend` (`openai.py`) | no | yes | no | yes |
| `AnthropicBackend` (`anthropic.py`) | no | no | no | no |
| `VLLMBackend` (`vllm.py`) | yes | yes | no | yes |
| `TransformersIntrospectionBackend` (`introspection.py`) | no | no | yes | no |

- `openai.py` — any OpenAI-compatible endpoint (the shipped entries use Venice and OpenRouter); the class has no provider logic, the endpoint lives in the entry's `APIConfig`. `ModelClient` fans a batch out through `BatchCaller`, one call per item; `agenerate()` runs on a cached async SDK client, so these models use the async lane. It sends `default_extra_body` with a call that passes no `extra_body`, maps a context-length error to `PromptTooLongError`, and warns on an empty reply and on a content-filter stop.
- `anthropic.py` — Claude through the native SDK: the system prompt goes as `system`, one call at a time; a 400 saying the prompt is too long (or that it and `max_tokens` exceed the context) raises `PromptTooLongError`; it warns on an empty reply.
- Both API backends: the SDK retries a failed call (a 429, a 5xx, a lost connection) `APIConfig.max_retries` times (default 2) with its own backoff, unseen by the rate limiter and the trace, and gives up after `APIConfig.timeout_s` (default `None`: the SDK's 10 minutes). A call that still fails stops the stage.
- `vllm.py` — vLLM's `AsyncLLM`, in this process. Every item is its own engine request, awaited on the shared loop of `llms/aio.py`, and the engine batches whatever is in flight: `agenerate()` makes one request, `generate()` one per item and waits for all (one failure cancels the rest). `max_workers` is the engine's `max_num_seqs` times its copies: what the async lane keeps in flight. The engine loads with the residency plan's `gpu_memory_utilization` and cards (`set_planned_utilization`/`set_planned_devices`, kept out of the cache key), else 0.9 of the card; planned cards beyond one copy's span become `data_parallel_size` copies inside one engine, and each engine records the share and cards it was built with (`engine_grants()`). Without `max_model_len` the engine gets `estimate.effective_max_model_len()`: 10,000 tokens, capped at the checkpoint's longest context from its HF config, else from the first context field its `config.json` sets (read from the checkpoint folder or the hub cache, never downloaded); 10,000 when neither names one. Prompts go through the engine tokenizer's chat template; for a checkpoint without one (Mistral-3.x) pass `tokenizer_mode="mistral"` in `vllm_kwargs`. A prompt longer than the context raises `PromptTooLongError` for its request alone; a request aborted by a release raises. Every request records one telemetry `call` event, a failed or cancelled one too (with `error`). Defaults: `temperature` 0.7, `top_p` 0.85, unless the entry sets them. A leading `|>` artifact (ChatML tokens missing from the vocabulary) is stripped.
- `introspection.py` — raw `transformers` inference that saves a model's internals (vLLM's continuous batching discards activations). `IntrospectConfig(hf_model_id, log_dir, capture={logprobs, hidden_states: False|"last"|"all", attention})`, default logprobs plus the last hidden state; capturing attention loads with `attn_implementation="eager"` unless `extra_kwargs` names one. Capture is opt-in per call: `internals_id(s)` is an id the caller computes, and the files go to `{log_dir}/{internals_id}/`, with an always-written `meta.json` (model, settings, input messages, output text); a capture replaces the files an earlier one left under the same id. `rename_capture(old_id, new_id)` (a no-op on the other backends) moves a capture made under a provisional id once the row's `sample_id` is known. A too-long prompt fails with whatever `transformers` raises.

**Caches.** A backend is cheap; what it uses is cached on that resource's own identity, behind double-checked locking: OpenAI SDK clients per `(base_url, api_key, max_retries, timeout_s)`, sync and async apart; the Anthropic client per `(api_key, max_retries, timeout_s)`; vLLM engines per `(hf_model_id, quantization, engine kwargs)`, so `venice-uncensored` served with vLLM and `venice-paraphraser` share one load; transformers models per `(hf_model_id, device_map, torch_dtype, load kwargs)`, apart from vLLM's. A local backend holds only its engine's key and reads the engine on each call, so it never keeps a released engine alive. `clear_transport_caches()` clears all of them: an SDK client is closed (a backend built before fails on its next call), a local engine is shut down and its memory freed (a client built on it raises).

**`backends/capabilities.py`**: `BACKEND_TYPES` (transport name → class), `API_BACKEND_TYPES`, `SETUP_TYPES`; `resolve_setup(config, requested)` picks a setup — the one requested, the run's choice (`set_run_serving`, from the recipe), the entry's `backend_type`, the only setup present; `transport_for` maps it to a `BACKEND_TYPES` key (`api` becomes the `APIConfig`'s own `backend_type`); `backend_for(config, setup)` calls the class's `from_config`, with the run's `workers` in place of `recommended_max_workers` for an API setup; `validate_concurrency`, `compute_config_for`.

**`client.py`**: `ModelClient` is a configured backend plus its dispatch, chosen once in `__init__` from `compute_config`: a native-batching backend gets the whole list (the limiter charged once per batch); any other fans out through a `BatchCaller`. `ModelClient.create(model, backend_type=None)` resolves the setup, builds the backend and resolves its rate limit. There is no client cache: everything expensive is cached on its own identity, so a fresh client is microseconds, `clear_transport_caches()` frees memory, and a new API key takes effect. `generate(messages_list, ...)` is batch in, batch out; `agenerate(messages, ...)` makes one async call, charging the limiter per call (`RateLimiter.reserve()`), and raises without `supports_async_calls`; both raise `ValueError` when capture ids reach a backend without `supports_internals`. `rename_capture` delegates to the backend. The client resolves no params, builds no messages or checkers and reads no verdicts.

**Rate limits and fan-out** (`wrappers.py`). `RateLimiter` is a thread-safe 60-second sliding window; `shared_limiter(key)` gives one window per key: the model name, or under `rate_limit_scope="endpoint"` (Anthropic meters the account) the endpoint id, so every model on one provider, URL and key variable shares a window. Models sharing an endpoint window must declare the same `rpm`, checked in `ModelClient.create()`. `reserve()` asks without blocking, `wait_if_needed()` blocks; a backend with `rpm=None` (any local binding) is never limited. `BatchCaller` runs one call per item over `backend.max_workers` threads; it raises at construction for `max_workers` above 1 on a series-only backend, and for `internals_ids` on a backend without `supports_internals`. Both take a finished backend and import no `ModelClient`.

**`model_config.py`**: `MODEL_REGISTRY`, loaded from `configs/llm/models.json` at import (every invalid entry is reported in one `ValueError`; `notes` is ignored). A `ModelConfig` is an identity — `name`, `roles` (a list; each must be in `configs/llm/roles.json`), `is_uncensored`, `supports_system_prompt`, `default_max_tokens`, `default_temperature` — plus one or more setups: `APIConfig` (`backend_type` `openai`|`anthropic`, `api_key_env`, `rpm`; `base_url`, required for `openai`; `api_model_id`, `rate_limit_scope`, `recommended_max_workers`, `default_extra_body`, `price_per_1m_input`/`price_per_1m_output`, `max_retries`, `timeout_s`), `VLLMConfig` (`hf_model_id`; `quantization`, `vllm_kwargs`, `sampling`, `vram_gb`, `min_gpus`), `IntrospectConfig` (`hf_model_id`, `log_dir`; `capture`, `device_map`, `torch_dtype`, `sampling`, `vram_gb`, `extra_kwargs`). Every config validates itself in `__post_init__` and raises (never asserts); the required fields make a setup without weights or without a log folder unrepresentable. One entry may carry several setups: `venice-uncensored` has `.api` and `.vllm`, and `ModelClient.create("venice-uncensored", backend_type="vllm")` binds the local one, which has no `rpm` and no `extra_body`. `backend_type` names the default setup (`api`, `vllm`, `introspect`, each matching its field; `None` means the only one present). The registry name is the stable identity (roles, rate-limit windows, telemetry rows, prices); `api_model_id` is what the provider receives when its id differs (`venice-uncensored` is sent as `venice-uncensored-1-2`), as `hf_model_id` is for the local setups. `default_model_for_role(role)` is the first entry in `models.json` order that lists the role; `get_models_by_role`, `available_roles`, `get_model_config` (`KeyError` for an unregistered name), and `model_compute_config(model, backend_type=None)`, the capability flags of a model's transport read off the class. `register_model(name, backend_type=None, roles=None, is_uncensored=False, supports_system_prompt=True, default_max_tokens=2000, default_temperature=None, api=None, vllm=None, introspect=None)` adds or replaces an entry for the running process. The roles: `uncensored_gen`, `translation`, `constitution_gen`, `paraphraser`, `debug_local`.

**`router.py`**: `generate_sample()` (a batch of one, unwrapped), `batch_generate_samples()` (chunks of `batch_size`, a progress label per chunk, `internals_ids` sliced per chunk; a `batch_size` below 1 raises), and `assert_single_sample_per_call(client, n)`, which raises when the backend can capture internals and `n` is not 1 (one forward pass's internals cannot be split between samples; `run_from_constitution` calls it).

**`prompting/`**: `load_prompt(pipeline, category, prompt_dir=None)` and `resolve_prompt` read a prompt JSON, the user's overlay file first, else the packaged one. `PromptTemplate` renders the system prompt and few-shot once (build-time kwargs) and the `template` per call; `PromptTemplate.load(pipeline, category, *, prompt_dir=None, few_shot=True, format_style=None, **build_kwargs)`. `build_messages(prompt_config, few_shot=True, format_style=None, **kwargs)` renders both at once. `format_style` (`numbered`, `structured_qa`) appends the format instruction from `prompts/format_instructions/{style}/` (`get_format_instruction`). `report_prompt_sources(prompt_dir)` logs the overrides and warns about override folders that match nothing. `extraction.py`: `extract_numbered_list(text, strict=False)`, `extract_structured_qa`, `extract_and_clean` (with `expected_count`, a reply with another count is rejected whole), `clean_sample`. `scaffold.py`: `scaffold_prompt_tree(target_dir, source_dir=None, mode="copy")`.

**Also here**: `aio.py` (one event loop on a daemon thread for every async call: `submit`, `run`), `progress.py` (`ProgressReporter`, throttled progress lines), `observe.py` (the telemetry hook, below).

### `llm_pipeline/` — units of model calls, and how they are driven

Between the stages and `llms/`; imports only `redact.llms` and `redact.exceptions`, and reads or writes no file. `LLMRequest(model, messages, internals_id=None, *, client=None, max_tokens=None)` is a pending call a generator yields; `LLMRequest.for_client(client, messages, *, internals_id=None, max_tokens=None)` binds it to a client. Requests share a call only when their reply limits agree.

- `drive_sync(gen, call)` drives one generator with a blocking call (`apply_combination`, `run_conversation`); `drive_generators(gens, *, finalize, ...)` drives many round by round, one call per client (or model) and reply limit per round (`optimization/search.py`).
- `drive_units(source, *, on_done, on_error=None, pool_size, cap=None, resolve=None, progress=None)` drives a pool of `(key, generator)` units topped up from a lazy source, at most `pool_size` live, reporting each to `on_done` as it finishes. Each round it groups the pending requests by model, setup and reply limit; a group of `cap` requests goes out as one `generate()`, and the largest group goes when none is full and no async request is in flight.
- **The async lane**: a request whose client has `supports_async_calls` (OpenAI-compatible, vLLM) is never grouped. It goes to the `aio` loop the moment its unit yields it, at most `backend.max_workers` per model in flight, and its reply returns the moment it lands. `pool_size(batch_size, *clients)` is `max(2 * batch_size, widest lane among the clients)`, so a local engine is kept full and `batch_size` only sizes the grouped calls. Requests in flight when `drive_units` exits (an error, Ctrl-C) are cancelled.
- **Errors**: a unit's own exception goes to `on_error` (or propagates without one). A failed `generate()` or async request propagates and stops the stage, with one exception: an async request's `PromptTooLongError` is raised inside its unit at the `yield`. So on the lane the inputs stage acks that entry rejected, outputs writes a rejected row, jailbreaks and conversations an error row, scoring scores it false; on the grouped path (Anthropic, introspection) the error comes out of `generate()` and stops the stage. The constitution and benign units take no `on_error`, and paraphrase is not on the pool, so a too-long prompt stops them on any backend.
- `check.py`: `is_accepted(reply)` (the first word, after whitespace, markdown and quotes, is yes/ok/okay/accept/accepted/pass/passed, any case), `check_sample`, `batch_check_samples`, and the unit steps `checked(gen, build_gen_messages, check=None, build_check_messages=None, *, original="", internals_ids=(None, None), max_tokens=(None, None), max_attempts=1)` (generate, check, regenerate with the verdict) and `extracted(gen, build_gen_messages, extract, *, internals_id=None, max_tokens=None, max_attempts=3)` (generate until a reply parses).

Every stage on the pool passes `pool_size=pool_size(batch_size, *its clients)` and `cap=batch_size`; the constitution runs one unit at a time (`pool_size=1`).

---

### `telemetry.py`, `llms/resources/`, `residency/` — what a run cost, and what fits on the GPU

Instrument where the data exists, decide where the policy belongs: `llms/` emits facts and imports no telemetry vendor SDK; the top level owns configuration, aggregation and export.

**`llms/observe.py`** is the hook: `record(event)`, `set_emitter(fn)`, `set_stage(label)`, `current_stage()`. With no emitter `record()` returns at once; it never raises (an emitter's exception is logged at DEBUG). It is a module function because `vllm._engine()`, cached per checkpoint rather than per backend, emits too. A `call` event is one transport call, with `n_items` the items it carried: a hosted model makes one call per item and vLLM one engine request per item, so both give one event per item.

**`telemetry.py`**: `install(data_dir=None, sink=None)` registers a `Collector`; `REDACT_TRACE=jsonl|wandb|off` picks the sink (an unknown value warns and uses `jsonl`). The JSONL trace is `<data_dir>/Datasets/run-<YYYYMMDD-HHMMSS>.trace.jsonl` (UTC), or `-2`, `-3`, ... after the run id when that name is taken; it is written for `jsonl` and `wandb` alike, and W&B only replays the same events. `stage(label)` labels everything inside it (a module global, not a `ContextVar`, since thread pools do not pass context to workers), records the stage's duration, and on leaving logs per model the replies cut at the token limit (a `call` event's `cut: true`; the first per stage and model warns at once; `summary()["cut"]`). `summary()` rolls up `models` (calls, items, tokens, `ms`, `cost_usd`), `local_s`, `local_billed_s`, `gpu`, `api_cost_usd`, `local_cost_usd`, `errors`, `cut`. **Logging**: the library attaches only a `NullHandler` and sets no level; the JSONL sink is a plain file writer, never a `logging.Handler`; the roll-up goes to `redact.telemetry` at INFO; nothing calls `basicConfig()` or `setLevel()` (a test asserts it).

**Cost** is metered as it is bought. Hosted: tokens × `price_per_1m_input`/`price_per_1m_output`, per model; a model served locally is never priced by tokens. Local: engine wall-clock lifetime, per checkpoint (the timer lives with `_engine()` in `backends/vllm.py`, from load to `clear_cache()`/exit, so two models sharing a load share one meter), load, compile and idle time included; billed time is the union of engine spans (`local_billed_s`) × the hourly rate × the GPUs on the machine (rented, not used). The rate: `set_gpu_rate()`, else `REDACT_GPU_HOURLY_USD`, else `configs/llm/gpu_pricing.json` under `REDACT_GPU_PROVIDER` (default `local`, at 0) by the GPU name `nvidia-smi` reports. An unpriced GPU reports time only.

**`llms/resources/`**: `estimate.py` sizes a local model from its HF config's shapes, without loading it (weights, with quantization and MoE; KV cache at `max_model_len`, else `DEFAULT_MAX_MODEL_LEN` = 10,000 capped at the checkpoint's context; padded need `(weights + KV) × 1.1 + 0.6` GiB per card). Sizing is computed, not measured: vLLM preallocates `gpu_memory_utilization × card`, so a card's memory delta restates the setting, and the V1 engine core runs in another process, hiding torch's allocator. `vram_gb` in the registry overrides the weights when known better (an MoE checkpoint, a measured figure). `measure.py` holds the `nvidia-smi` probes — `gpu_cards()` (index, UUID, used and total memory, name, in one query), `card_use()` (plus compute processes), `detect_gpus()`, `card_sizes_gib()`, `visible_gpu_ids()` — and `visible_only`, `cuda_lock`, `bus_order`, `free_total_gib()` (torch, the fallback size), and `Measurement` (`claimed_gib`/`weights_gib` for the introspection load event). Everything is GiB.

**`residency/`** answers "does this fit?" before anything loads (`footprint.py`, `placement.py`, `plan.py`, `phases.py`, `lifecycle.py`, `explain.py`).
- **Footprint** (`footprint(model)`): weights declared (`vram_gb`) → estimated → unknown; KV for vLLM only. Models sharing a backend cache key are one engine, counted once.
- **Cards**: each card is planned at its own size from one `nvidia-smi` query (torch only when that fails, since it starts CUDA here; with neither the plan warns). Only the cards `CUDA_VISIBLE_DEVICES` leaves count, named by index, UUID or UUID prefix; as CUDA does, an entry that names no card ends the list. Planning and loading set `CUDA_DEVICE_ORDER=PCI_BUS_ID` unless it is set, and a build gets its cards by UUID, narrowing `CUDA_VISIBLE_DEVICES` for the process while it builds (`measure.visible_only` under `measure.cuda_lock`).
- **Placement** (`plan_residency(models, backend_types=None, *, gpus=None, gib_per_gpu=None, copies=None, pinned=None)` → `ResidencyPlan`): an engine already loaded (`pinned`) first, on its own cards; a single-GPU vLLM engine on the emptiest card with room (lowest index on a tie), else the card with the most room; a tensor-parallel one (`min_gpus > 1`, equal to its `tensor_parallel_size`) on whole untouched cards; a transformers model where its `device_map` puts it. A model that does not fit starts a new group; with more than one group, `_Search` tries every card for up to 8 single-card engines (at most 20,000 tries) for one group with the fewest shared cards.
- **Copies** (vLLM `data_parallel_size`, inside one engine): when everything is one group, the first vLLM engine gets a copy on every other card with room, beside other models only where every engine keeps its need and never on a card smaller than its home card; the recipe's `copies` replaces that (`1` turns copies off). Tensor-parallel copies are planned, not yet loaded end to end on a GPU.
- **Grants**: on each card the vLLM engines split 0.9 of it, less what transformers models hold, each granted its padded need plus an equal share of the spare; an engine loads with the smallest grant any of its cards gives. `apply_plan()` hands grants and cards to the loader; `VLLMConfig` refuses a `gpu_memory_utilization` in `vllm_kwargs`. A grant below the padded need makes the engine `oversized` (listed once, by its first model), a problem.
- **`ResidencyPlan`**: `groups`, `gpus_required`, `gpus_available`, `gpu_name`, `per_gpu_gb` (the smallest card), `gib_per_card`, `oversized`, `machine_given`, `loaded`; `co_resident`, `sequential`, `fits`, `problems`, `warnings`, `footprints`, `explain()`, `as_dict()`. **An engine loaded already**: for the detected machine, a planned vLLM engine that is loaded is recorded in `loaded` with the share and cards it was built with (`lifecycle.loaded_grants`, from `vllm.engine_grants()`) and named in a WARNING, also in `warnings`: the plan's share and cards for it apply only after it is released. A loaded engine not in the plan is warned about too.
- **Per stage** (`plan_phases`): one `Phase` per stage (resident models, plan, loadable first group). An engine is resident from the first stage that calls it to the last, keeps the cards it loaded on, and loads with the smallest share any of its stages plans; one kept for a later stage is unloaded for a stage it crowds and loaded again, when every stage then fits. `unload_local(keep=, engines=, wait=)` releases what a stage leaves out (vLLM's shutdown hook, garbage collection, `torch.cuda.empty_cache()`). At the run's end every engine the run loaded is released; those loaded before it (a notebook's) are kept. The paraphrase stage plans its own pool of paraphrasers and releases each before the next when they cannot be resident together. Plans go to the `residency` trace event and `summary["residency"]`; a plan that does not fit is logged at ERROR and never aborts the run.
- **Preload** (`preload(models, ..., wanted=None)`): one daemon thread loads the stage's local models, one after another, so a long load overlaps the API calls of an earlier stage; a model no longer `wanted` is skipped. A failed preload logs at ERROR and records `preload_failed`, never aborting the run: the stage that needs the model fails at the point of use. Double-checked locking in every transport cache (`vllm._engine`, `introspection._load`, both `_sdk_client`s) keeps a preload and a first call from building two engines; a regression test spawns four threads and asserts one load.

**Engine processes and the exit** (`backends/vllm.py`): each build records the processes it starts (engine cores and, with copies, the DP coordinator); a release ends what the shutdown leaves, and a failed build's processes are ended at once. `terminate_engine_processes(processes=(), *, all=False, quiet=False)` lists their children with `psutil`, sends SIGTERM, then SIGKILL to those alive 5 s later, and returns the pids still alive; `all=True` ends every engine process this process started and their descendants, and no other child. When a build with the plan's copies fails, one copy on the engine's first cards is built instead, once, with a WARNING naming `"copies": 1` (not when `data_parallel_size` came in `vllm_kwargs`, when exiting, or when the failed build's processes outlive SIGKILL); the `load` event carries `planned_copies` and `fallback`. The exit hook releases every engine, waiting at most `EXIT_WAIT_S` (60 s) for a load in progress, then ends that build's processes so the load fails with `EngineLoadEnded`; no load starts after the hook. A release mid-run waits for a load however long it takes.

**Not done** (the `TODO(driver script)` block in `residency/__init__.py`): the paraphraser pool is planned apart from the run's plan; each card is sized by its total and assumed empty (`_check_cards` stops a `run_pipeline` run on a planned card another process holds; a plan made outside it is not checked); the unload search tries every candidate unloaded at once, so with more than 8 engines it can miss a run one unload would fit; a load that hangs is ended only on the way out or by a second signal. Besides: a planned engine that is loaded already is warned about, not yet planned around. Offline tests cover the hang and second-signal paths.

---

### `multi_turn/` — General multi-turn conversation datasets

Use-case-agnostic; not a recipe stage (call the functions); full reference in `docs/stages/conversations.md`. A **`Setting`** (`participants`, `max_turns`, optional `stop(transcript)`, `name`) played on a seed gives a **`Trajectory`** (`input_id`, `setting`, `transcript`, `turns_used`, `stop_reason`): the seed is the first participant's message, then the participants take turns until `max_turns` (the seed counts as one) or `stop` is true. A **`Transcript`** is a typed step log: `message`/`reply` steps carry a chat role and are sent to models (`as_messages()`); `strategy`/`analysis`/`evaluation` steps are notes, only logged. An **`Actor`**'s turn is a generator `turn(transcript)` that yields `LLMRequest`s and returns its `Step`s: `ScriptedActor` (the given messages in order, the last repeated), `ModelActor` (one call on the visible transcript), `StrategyActor` (sends what a `propose(transcript, turn_index)` generator builds, logging its note as a `strategy` step). `conversation_gen(setting, seed)` is the generator; `run_conversation(setting, seed, call)` drives one with a blocking call.

**The stages.** `generate_conversations(seeds, setting, data_dir, iterations=1, resume=True, batch_size=256, ...)` plays the setting (a factory, or a template deep-copied per conversation) on every seed (a `seed`/`sample`/`prompt`/`text` column) `iterations` times. A unit is `(input_id, iteration)`: planned in `conversations.manifest.jsonl`, driven through `drive_units` (pool sized by the `ModelActor`s' clients), saved to `conversations.csv` (`sample_id` = `content_id("{input_id}:{iteration}")`, `input_id`, `iteration`, `setting`, `turns_used`, `stop_reason`, `transcript` as JSON, `category`, `entry_type`, `source="conversation"`) and acked in the ledger keyed `(input_id, iteration)`. A unit that raises gets a row with `stop_reason="ERROR: ..."`, an empty `setting`, `turns_used` 0 and transcript `[]`, and is acked; a too-long prompt is such an error on the async lane, and stops the stage on an Anthropic or introspection binding. `iteration` numbers independent variants, not escalation rounds. `evaluate_conversations(data_dir=..., judge_model=..., judge_system=None, scope="last_reply" | "transcript", batch_size=32, ...)` is the separable judge pass: the criterion is the judge's system prompt (default `prompts/conversation/judge`), the goal is the seed. It writes `conversations_scored.csv` (`sample_id`, `success`, `judge_reasoning`, `judge_model`, `scope`), ledger keyed `sample_id`, no manifest. The judge and its prompt are resolved before `resume_state`, so a fresh scoring run with a bad judge stops before it clears anything. With the judge on the async lane a transcript too long for it is scored false and acked; on an Anthropic or introspection binding the stage stops. A conversation's `sample_id` does not change with its transcript: after a fresh conversation run, score with `resume=False` too. Internals capture is not wired in.

**Applications**, each its own top-level folder building on `multi_turn/` (never inside it): `multiturn_attacks/` — `build_attack_setting(target_model, "crescendo" | "pair", ...)` returns a Setting factory, an attacker `StrategyActor` then the target `ModelActor`. Crescendo sends `DEFAULT_CRESCENDO_FOLLOWUPS` (`configs/jailbreak/multi_turn/crescendo_followups.json`), optionally through a pure jailbreak technique; PAIR has an attacker model write each message (`prompts/conversation/pair_attacker`). `generate_attacks` / `score_attacks` wrap the two stages, scoring with `SUCCESS_JUDGE_SYSTEM` (`prompts/conversation/success_judge`). The target must be safety-trained and has no default. `optimization/` — `optimize(seed, target_model, candidates, judge_model, judge_system, beam=2, depth=3)` extends every frontier node by each candidate move, has the target reply to all of them in one round, scores the replies, keeps the `beam` best, and returns `(best_node, tree)`; `multiturn_attacks.optimize_attack` feeds it jailbreak moves.

Dependency direction: `multi_turn/` imports `llm_pipeline/`, `llms/`, `dataset/`, `paths` and `exceptions` only. Oriented toward Inspect (seed≈Sample, messages≈ChatMessage, step log≈events, Setting≈solver, judge≈scorer).

---

### `dataset/` — Data Handling

Rows on disk. A stage's output is CSV: under `Datasets/` (one file for most stages, one per paraphrase target and per jailbreak kind, one `constitution_inputs/{category}/samples.csv` per category for the input samples), and under `Data_cache/` for the constitution and the benign pairs. `dataset/` sees no prompt, client or model.

- `identity.py` — `content_id(text)`: the MD5 of the text, cut to 16 hex characters, a row's identity. `row_id(row, text_col, id_col="sample_id")` gives the id a row carries, else the content id of its text; `row_ids(frame, text_col)` does it for a frame.
- `rows.py` — the rows of one CSV. `append_rows(path, rows)` appends in one synced write, in the column order of the file's header, and returns the file's size; it raises on a column the header lacks (`check_columns` checks without writing), and no rows write nothing. `read_rows`, `iter_rows`, `keys_in` read `sample_id`/`input_id` as text; `read_rows` reads a missing text cell as `""`; a missing or 0-byte file reads as no rows; `count_rows` and `has_rows` read in chunks. `accepted_mask(series)` is the one predicate for an accepted cell (`true` in any case, booleans or text). `write_concat(target, parts, *, keep_empty=True, unique=None)` writes `Part`s (a file or a frame, `dataset_type`, `accepted_only`, `label`) one after another, in chunks, through `<target>.tmp`, synced, then a rename; `unique="sample_id"` leaves out a row whose id an earlier part wrote and logs the count; a frame part's gapped whole-number float column is written as `Int64`.
- `sidecar.py` — `JsonlSidecar` (the path beside the artifact, `.exists()`, `.reset()`, a read that skips an unparseable line), `Ledger` (`*.state.jsonl`, the finished units: `Ledger(path, key_fields, casters)`, `.record(records, saved=)` in one synced write, `.read()`, `.completed()`), `Manifest` (`*.manifest.jsonl`, the planned units: `.write(rows)` through `<manifest>.tmp`, synced, and a rename; `.load()`).
- `resume.py` — `resume_state(ledger, *, resume, artifact=None, pattern=None, manifest=None, extra=None, units_in=None)` returns the keys of the finished units, or on a fresh run removes the output, its `.cut` and `.tmp` files, the `.dropping` marker, the ledger and the manifest. `clear_stage(artifact, *, manifest_path=None, also=())` does the same clear for a stage skipped with nothing to run. `release_lock` gives back the stage's lock. See **Resume model** below.
- `save.py` — `commit` and `CommitBuffer(append=, ledger=, every=32)`: the rows of finished units, then their acks.
- `drop.py` — `drop_units(ledger, *, artifact, name, rows, acks, pattern=None, units_in=None)`: a fresh run of part of a stage (the inputs stage, one style).
- `categories.py` — the tree `Datasets/constitution_inputs/{category}/samples.csv`, the default folder of its functions. `sample_row` builds one input row in `SAMPLE_COLUMNS` order; `read_category_csv` reads one file (ids as text, missing text as `""`); `discover_categories` lists the folders with a `samples.csv` of at least one byte; `merge_all(dir, accepted_only=True)` merges them; `input_files(data_dir)` and `merge_inputs(data_dir)` give the run's category files and accepted inputs, which the outputs, jailbreak and paraphrase stages read when given none, and the build when it finds no merged file. `take_per_group(frame, n)` caps a frame per `(category, entry_type)`, a missing group value forming its own group.
- `taxonomy.py` — `load_taxonomy(name, config_dir=None)` reads `configs/taxonomy/<name>.json` (a name with or without `.json`; the former name `content_moderation_categories` reads `harmful`, with a `DeprecationWarning`); `create_taxonomy(name, categories, description="", taxonomy_dir=None, verbose=True)` writes `<name>.json` and returns it; `iter_categories` yields `(name, info)`.
- `loading.py` — `load_hf_dataset` loads a HuggingFace dataset with column and row filters (the `hf` extra); `load_from_config` does so from a config under `configs/datasets/`.

---

### `jailbreak/` — Jailbreak Generation

**152 single-turn techniques** in **4 layers → 23 families**, as `configs/jailbreak/combination_spec.json` names them. `pool.py` checks at import that the spec's `techniques` and the registries (`get_all_{hacking,manipulation,obfuscation,request}_functions()`: 25 + 8 + 72 + 47) name the same techniques, and raises otherwise; `technique.tag()` sets each function's spec tags when `redact.jailbreak` is imported: `families`, `layer` and `within_layer_order` (from its **first** family), `complexity` (0–3), `requires_llm`, `requires_benign`, `encode_weight`, `max_tokens`, `cross_incompatible_with`.

| Layer (apply order) | Families (count) — LLM? |
|---|---|
| 1. `hacking` (max 1 per combo) | `framing` (5, pure) · `cognitive` (5, LLM, scenario→construction) · `persona` (15 = invented + 14 named, LLM) |
| 2. `manipulation` (max 1) | `fsh` (4) · `dap` (4) — `random_*` pure, `selected_*` LLM; all `requires_benign` |
| 3. `obfuscation` (no layer cap; ordered by `within_layer_order`) | ① `sensitive_words` (16, LLM; the 10 `to_sensitive_words_encode_*` are *also* in `encode`) ② `tokenbreak` (3, LLM) ③ `translation` (20, LLM) ④ `typos` (4, LLM) ⑤ `casual_writing` (9, pure) ⑥ `suffixes` (4) ⑦ `encode` (12; `encode_weight` light/heavy) ⑧ `structural` (3) ⑨ `ascii_art` (1) |
| 4. `requests` (max 1; all pure) | `answer_format` (9) · `answer_language` (20, generated from JSON) · `continuation` (4) · `indirect` (6) · `distractor_prefix` (2) · `distractor_suffix` (2) · `impersonation` (1) · `temporal` (1) · `asking` (2) |

**A technique** (`technique.py`) is a transform `(prompt, **kw) -> (text, info)` or, when it calls a model, a generator that yields `LLMRequest`s, is resumed with each reply and returns `(text, info)`. `info` is `""` or `key=value;...`; a rejection's starts with `DISCARDED`. A technique with a random choice draws it from its `rng` keyword. The pure techniques' template text lives in `configs/jailbreak/{hacking,obfuscation,requests}/*_templates.json` (`directives.load_templates`); the model-calling ones load `prompts/jailbreak/...` and take the `prompt_dir` overlay. Families drawn from a list (translation, answer_language, the named personas, the sensitive-word encodings and separators) are generated and bound with `globals().update`. Results a technique rejects: a word list that is not at most ten of the prompt's words (`malformed word list`, the `sensitive_words`/`tokenbreak` marking call); a text no ASCII-art font draws whole (the font is drawn by `rng` from the 16 that have a glyph for every character; any non-ASCII character rejects); a translation failing its check 4 times. `to_invented_persona` makes the same calls as `to_persona_roleplay` until it gets prompts of its own. To add a technique: write it in its layer's module, return it from the module's getter, and give it a spec entry; import fails until both agree.

**Compatibility rules** (`rules.compatible`), each binding both ways (a forbidden pair is refused whichever was picked first):
- One technique per family; a multi-family technique blocks every family it belongs to.
- Layer caps from `layer_max_picks`: 1 hacking / 1 manipulation / 1 request.
- `cross_incompatible_with`, read from either family's list: fsh/dap ⊥ continuation, indirect, framing, persona, cognitive (so a combination holds a hacking or a manipulation technique, never both); continuation ⊥ indirect; structural, translation ⊥ ascii_art; ascii_art ⊥ framing, persona, cognitive, fsh, dap. A technique's own entry may hold the key too: `to_adversarial_suffix_emoji`, `to_adversarial_suffix_unicode`, `to_braille` and `to_tokenbreak_split` ⊥ ascii_art.
- `manipulation_blocked_obfuscation`: fsh/dap never sit with `translation`, `ascii_art` or heavy full-text `encode` (base64, unicode_escape, ascii_ordinal, morse, braille); `to_sensitive_words_encode_*` stays allowed.
- Complexity filters only in budget mode (`remaining_complexity`).

**Sampling** (`sampling.py`). Every pick is **family-first** (`_pick_by_family`): a family, then a technique within it, so the 20 languages count as one choice. *Budget-driven* `sample_combination`: hacking, manipulation and requests, in that order, each join at their layer rate (the spec's `layer_rates`, overridable per call), then the obfuscations fill what the `max_complexity` budget leaves, up to `max_obfuscations` families; phase 5 (`allow_request_obfuscation`) may add a light `encode` (the light `to_sensitive_words_encode_*` included) or a `translation` **after** the request, obfuscating the finished request. *Count-driven* `sample_exact_combination` (`exact_techniques=N`) picks exactly N compatible techniques and ignores `max_complexity`, `max_obfuscations`, `layer_rates` and `allow_request_obfuscation`. The apply order is the layers' whatever the pick order.

**The plan** (`plan.py`). `plan_run` writes the whole plan before anything generates: `<stem>.plan.json` (`plan_inputs`: `seed`, `rounds`, sorted pool names, `family_weights`, `layer_rates`), then one manifest row per input and round, `{sample_id, iteration, combination, settings, category, entry_type, status}`, a round's settings merged over the base key by key (a round's `layer_rates` replaces the kind's whole). Each `(category, entry_type, iteration)` group is **dealt** in `sample_id` order by a `_Dealer`: a layer joins while its picks are below `floor(rate × position + offset)` (a seeded offset per group and layer), and a family is picked by lowest count over its `family_weights` weight. A layer two or more picks short of its target in a group is logged as a WARNING naming the coverage file. `assign_combination` seeds each draw by `(seed, sample_id, iteration)`, so the plan does not depend on input order or chunking, and redraws without the dealer while the combination equals one the input already got (up to 8 draws in all). `coverage()` writes `<stem>.coverage.csv`: per layer, family and technique, the count per category, entry type and iteration, with a zero row for each pool technique never drawn. `combination_of(names, registry)` replays a row in its recorded order through the registry of **every** technique; an unknown name raises `ValueError` naming `resume=False`. `default_escalation_schedule()`: exactly 1 technique, exactly 2, then budget rounds `max_complexity` 6 / 9, `max_obfuscations` 2 / 3, rates hacking 0.7 / 0.9, manipulation 0.3 / 0.1, requests 0.8 / 0.9, request_obfuscation 0.5 / 0.7, `allow_request_obfuscation`.

**Application order** (`chain.py`). `in_layer_order` sorts by `(layer_order index, within_layer_order)`: hacking → manipulation → obfuscation ①…⑨ → requests, so request templates wrap obfuscated text and structural/ascii_art are the outermost obfuscations; phase 5's pick goes after the request. `run_chain(techniques, text, ...)` runs transforms inline and yields a generator's requests, capped at the technique's `max_tokens` and, with capture, tagged `{root}/{technique}/{n}` when the request's model can capture. Each technique gets the kwargs it takes (`gen_model`, `translate_model`, `benign_data`, `prompt_dir`) and its own `random.Random` seeded by `(seed, label, input id, iteration, technique)`, the label `jailbreak` or `variation`, so a replayed plan gives the same text and the two kinds draw apart. A step that changed nothing is recorded as `noop=<name>`; a step that leaves the text empty is a rejection (`reason=empty text`); the first rejection stops the chain. `combine_techniques(*fns)` and `apply_combination(fn, prompt, client=...)` run one prompt synchronously; manual chaining checks no rule.

**The stage** (`pipeline.run_augmentation`; front `generate_jailbreaks`). `resume_state` (lock, ledger keyed `(input_id, iteration)`), then the plan: kept on a resume (it is sampled) and checked with `changed_plan_inputs` against `<stem>.plan.json` — a changed seed, pool, `rounds`, `family_weights` or `layer_rates` raises naming the keys and `resume=False`, a plan without the file runs unchecked with a WARNING, and a WARNING counts the inputs the kept plan does not hold. Then benign data when a pending unit's combination has a `requires_benign` technique (`manipulation.get_or_generate_benign_data`: the pairs the cache in `Data_cache/benign/` lacks for the run's `benign_taxonomy`, one unit per `(main_category, sub_category, answer_type)` on the worker pool, up to 3 calls each, with its own manifest and ledger and always resumed; a unit rejected every time is acked `rejected`, and a cache with no pair afterwards raises; with `auto_generate_benign` off the cache is loaded as it is, and a missing one raises before any unit). Then `drive_units` over the pending units (`units.pending_units` builds a unit's generator only as the driver takes it; `pool_size(batch_size, client)` live). `CommitBuffer` saves every 32 units. A row is the input's columns plus `sample_id` (`content_id` of the jailbreak text; an accepted adversarial no-op has its input's id, and the build leaves it out), `input_id`, `jailbreak`, `technique`, `technique_info`, `complexity`, `num_techniques`, `is_noop`, `accepted`, `reasoning`, `iteration`, `combination_spec_version`. A unit that raises (on the async lane a too-long prompt too) gets an error row (the prompt unchanged, not accepted, `sample_id` `content_id("{label}:{input_id}:{iteration}")`) and is acked; on the grouped path a failed call stops the stage. The natural kind runs with `accept_noop=False`. Translation is the cost hot spot: translate → check → retry, up to 4 rounds, so 2–8 calls on the translation-role model per translation. Internals are captured under `{input_id}/{label}/attempt_{iteration}` and renamed to `{input_id}/{label}/{sample_id}` once the text is known.

**The augmentation section** (`augmentation.py`): `generate_jailbreaks(augmentation=...)`, or a recipe's `augmentations`, is `{"adversarial": {...}, "natural": {...}}`. `resolve(section, kind)` overlays a kind on its default from `configs/jailbreak/augmentations.json` and checks it (every error names the kind and key; `_` keys are comments); a kind left out is the default, `null` switches it off. The stage runs `run_augmentation` once per kind that is on, with `technique_pool(kind)` and `rounds_of(kind)`: the **adversarial** kind (attacks) to `jailbreaks.csv`, the **natural** kind (natural variations) to `variations.csv` (beside `output_path` when one is given, else `paths.variations_csv(data_dir)`, unless `variations_path` says otherwise); each file has its own manifest, plan file, ledger and coverage table. A kind whose `entry_types` match no input warns and writes nothing.
- Keys of both kinds: `entry_types` (`null` all), `translation` (bool), `layers` (`deny`), `families` and `techniques` (`allow`, `deny`; `allow` `null` is everything), `pure_only`, `family_weights`, `iterations`. Adversarial only: `max_complexity`, `max_obfuscations`, `layer_rates` (`hacking`, `manipulation`, `requests`, `request_obfuscation`, over the spec's), `allow_request_obfuscation`, `rounds` (`null`; `"escalation"`; or a list of rounds, which sets `iterations`). Natural only: `exact_techniques` (default 1).
- **Overlay**: a value replaces the default's whole value, lists included, except `layers`/`families`/`techniques`, whose `allow` and `deny` overlay one by one; a `deny` given alone also takes its names out of the default's `allow`. The natural kind must keep an allow list. `natural_only_families` (`casual_writing`) are added to the adversarial `families.deny` unless its `families.allow` names them.
- **Translation** has its own switch: no family list changes it (`layers.deny`, `techniques.deny` and `pure_only` still apply). On, it admits the whole family unless `techniques.allow` names some of its techniques (`pool.languages()`): then only those. Off, the translation an allow list or `family_weights` names is left out of the resolved kind with a WARNING (`_switch_off_translation`; an allow list it empties becomes `null` beside the other, and a kind that allowed only translation raises), so `"translation": false` overrides the eval type in one key. The packaged default is `false` in both kinds; once a local translation model is registered, flipping it is one line of `augmentations.json`. `resolve_role` plans the translation model only when a kind's pool has a translation technique.
- Defaults: the adversarial kind denies no family (only `natural_only_families`); the natural kind allows a list of mild families and techniques and denies three answer-format directives.
- The twelve flat keywords (`include_*` ×5, `pure_only`, `max_complexity`, `max_obfuscations`, `sampling_probs`, `iterations`, `settings_per_iteration`, `entry_types`; `FLAT_KEYWORDS`) are deprecated aliases into the adversarial kind for one release, each with a `DeprecationWarning` (`apply_flat_keywords`), wherever they are written: `generate_jailbreaks` keywords, the flat form of a recipe's `augmentations`, or `params["jailbreaks"]`.

**Known issues (open):**
- Phase 5's request obfuscation is not counted against `max_obfuscations`.
- The spec's per-family `max_picks` (and `description`) is not read: a combination holds one technique per family.

**Reference list coverage:** benchmarked against 73 instruction primitives + 74 request primitives; every feasible one is covered. Left out by design: `agent_context_additional_instr` (needs the system prompt), `fine_tuning` (out of scope), `use_highly_specialized_language` (no clear way to build it), `direct_question` (changes nothing). Beyond the list: ASCII art, adversarial suffixes, structural wrapping, cognitive hacking, manipulation, continuation attacks, indirect embedding, casual writing, extra encodings and languages.

---

### `content_moderation/` — Content Moderation Generation

Three stage cores, each called by its front in `stages.py`:

- **Inputs** (`InputPipeline.run_from_constitution`, `inputs.py`; front `generate_inputs`), seeded from the constitution in every run. An entry is checked against its `source_category` (a benign-taxonomy category for the general-benign part), and its `source_taxonomy` is copied onto its rows when the constitution has the column. Each constitution entry is one unit: a generation call for `samples_per_entry` samples, made again while the reply holds another count (up to `max_attempts`; a reply is rejected whole, never salvaged), then one check call per sample with the entry-type-aware `build_quality_checker`. A rejected sample is saved with its reason; an entry rejected on every attempt is acked `rejected` with no rows, and so is one whose generation or check prompt is too long, on the async lane (on an Anthropic or introspection binding that stops the stage); an entry whose samples were all in the folder already is acked `no_samples`. The generation prompt lists the first 20 texts this run saved as do-not-repeat and counts the rest; a text is written once per folder. Rows go to `Datasets/constitution_inputs/{category}/samples.csv` (`SAMPLE_COLUMNS` plus `source_sample_description`, `constitution_category`, `source_group_tag`, `template_style`); `style` (`long`, `short`, or any directory under `prompts/input/generation/from_constitution/`) picks the template. The ledger `constitution_inputs.state.jsonl` is keyed `(sample_description, entry_type, style)`; `resume=False` starts that style over and keeps the other styles' rows and acks (`drop_units`). It returns a `ConstitutionInputResult` (totals). With internals capture, one sample per call is enforced (`assert_single_sample_per_call`).
- **Outputs** (`run_output_generation`, `outputs.py`; front `generate_outputs`). One `llm_pipeline.checked` unit per input row with an entry-type-aware `build_output_quality_checker`: a refusal of a harmful input is rejected, a benign input's reply is judged as any answer. `max_attempts > 1` regenerates a rejected response with the verdict as feedback. On the async lane an input whose generation or check prompt is too long gets a rejected row with an empty response; on an Anthropic or introspection binding the stage stops. Writes `output_responses.csv` (`input_id`, `input_prompt`, `category`, `subcategory`, `entry_type`, `model`, `source`, `sample_id`, `output_response`, `accepted`, `rejection_reason`), ledger keyed `input_id`. The front caps inputs per `(category, entry_type)` with `max_per_category` (`take_per_group`), then with `max_samples`.
- **Paraphrase** (`run_paraphrase_target`, `paraphrase.py`; front `generate_paraphrases`). `k` paraphrases per base input and per accepted base output (`accepted_mask`); each unit `(input_id, iteration)` gets a paraphraser from `paraphrase_pool(paraphraser)` (`"distribution"` is every `paraphraser`-role model) by a seeded hash. A separate checker (`build_paraphrase_checker`, by default on the `uncensored_gen` model; the front warns when it is also a paraphraser) drops a paraphrase that changes the meaning, and a duplicate is dropped; every attempted unit is acked, a dropped one too (`dropped_dedup`). Writes `paraphrases_inputs.csv` / `paraphrases_outputs.csv`. It runs one paraphraser at a time in `batch_size` chunks, not on the unit pool, so a too-long prompt stops it on any backend; when the pool's models cannot be resident together, it releases their engines between them. The paraphraser stands in for the defingerprinting model trained in another repository.

`checker.py`'s builders return `(original, sample) -> messages`, the contract of `llm_pipeline.check`, each loading its prompt once as a `PromptTemplate`. `feedback.py` gathers the prompt text the stages add at run time (rejection feedback, the do-not-repeat note); it moves to prompt JSON later.

---

### `constitution/` — Constitution Generation

Structured category hierarchies for constitutional classifier training, at 4 severity levels:

1. **Absolutely harmful** (`harmful`) — clear-cut violations, always flag
2. **Dual-use harmful** (`dual_use_harmful`) — borderline, harmful framing
3. **Dual-use benign** (`dual_use_benign`) — borderline, benign framing, could look harmful
4. **Absolutely benign** (`benign`) — clearly safe, never flag (hard negatives)

`ConstitutionPipeline(client, output_dir=None, prompt_dir=None).run(taxonomy, ...)` (front: `generate_constitution`, role `constitution_gen`, `claude-opus-4-6` by default) makes one call per **unit**: a `(source_category, entry_type)` pair for each of the first `num_taxonomy_categories` taxonomy categories and each entry type (all four by default, or `entry_types` in their order), asking for `num_categories` constitution categories, plus, with `include_general_benign` (on in both dataset types), one unit per category of the first `num_taxonomy_categories` of `benign_taxonomy` (the recipe's, passed by the run; default `benign.json`, 11 categories), through the `general_benign` prompt with the same kwargs a taxonomy unit gets (`Category`, `description`, `subcategories`, `group_tag`, `num_categories` = `general_benign_categories`, 10). A `Unit` has `prompt` (its folder under `prompts/constitution/`, the entry type unless given) and `source_taxonomy`; its key is `source_category::prompt`. A general-benign unit's rows are `entry_type` `benign`, `source_category` the benign category, `source_taxonomy` `benign` (taxonomy rows say `harmful`). The front reads the former names `include_standalone_benign`/`standalone_benign_categories` for one release with a `DeprecationWarning` (given, they win). `units.py` plans the units (`plan_units`), builds each call from `prompts/constitution/{prompt}/` (`messages_of`) and turns a reply into rows (`rows_of`, via `parse.parse_constitution`). A reply without its `[END]` marker (stopped at the token limit or by a provider filter) or with no entry that parses is rejected whole and the call made again, up to 3 attempts (`llm_pipeline.extracted`); a unit rejected every time is acked `rejected` with no rows, skipped on resume and named in a warning. The units run one at a time (`drive_units(pool_size=1)`); a failed call, or a too-long prompt on any backend, stops the run and the units saved before it stay acked.

Rows (`sample_id` = `content_id(sample_description)`, `constitution_category`, `constitution_subcategory`, `sample_description`, `entry_type`, `source_category`, `source_taxonomy`, `source_group_tag`, `model`) go to `{entry_type}.csv` (`general_benign.csv` for a benign-taxonomy row, `units.file_of`) and `merged.csv` under `Data_cache/constitution/`, saved per unit through `CommitBuffer(every=1)` before the unit's ack; a resume of a `merged.csv` without one of these columns raises before any call, naming `resume=False`. The plan is `constitution.manifest.jsonl` (`source_category`, `entry_type`, `prompt`, `status`), the ledger `constitution.state.jsonl`, keyed `source_category::prompt`; it is a plain ledger (no lock, no recorded sizes), so a crash between a unit's two appends leaves it un-acked and the next run appends its rows again. `resume=False` removes the CSVs, the ledger and the plan first. `run()` returns `read_constitution(output_dir)`: `merged.csv` with every cell as text and an empty one as `""` (an empty frame with the columns when there is none), rows of earlier runs included; the driver reads the constitution the same way to seed a run's inputs.

---

## Prompt JSON Schema

All prompts are `.json` files in `src/redact/prompts/{pipeline}/{category}/`, loaded at runtime via `load_prompt(pipeline, category)` in `llms/prompting/prompts.py`, which resolves to the single `.json` in that directory, else `template.json`.

**`prompt_dir` is an overlay, not a replacement.** Each prompt is read from the user's directory if present and from the packaged tree otherwise, so customizing one checker means copying one file, and the rest keep the library's later improvements. `report_prompt_sources(prompt_dir)` logs the resulting picture once at startup (what is overridden, and any directory matching no packaged prompt). `scripts/scaffold_prompts.py TARGET` writes an editable starting tree: real prompts by default, `--empty` for TODO skeletons whose key set mirrors each real file's. Pipelines: `constitution` (one per entry type, plus `general_benign`), `input` (`generation/from_constitution/{style}`, `quality_check`), `output` (`generation`, `quality_check`), `paraphrase` (`generation`, `check`), `jailbreak` (per technique step: `scenario_generation`, `persona_scenario_generation`, `jailbreak_construction`, `extract_harmful`, `benign_generation`, `category_selection`, `rewrite_with_typos`, `synonym_substitution`, `translate`, `translate_retry`, `translate_check`), `conversation` (`judge`, `pair_attacker`, `success_judge`), and `format_instructions` (`numbered`/`structured_qa` — not a chat prompt, a `{"instruction": "..."}` snippet `PromptTemplate`'s `format_style=` appends to another prompt's `system_prompt`). The tree also holds `input/category_check`, `jailbreak/benign_check`, `jailbreak/scenario_check` and `jailbreak/typo_check`, which no code loads yet.

```json
{
  "system_prompt": "... {Category} ...",
  "template": "... {sample} ...",
  "seed_fields": ["Category", "sample"],
  "few_shot_examples": [],
  "metadata": {
    "version": "1.0",
    "notes": ""
  }
}
```

A prompt file carries no `category` or `pipeline` field: its directory is its identity. `build_messages(prompt_config, **kwargs)` renders both `system_prompt` and `template` with the same kwargs (via `str.format_map`), so any placeholder works in either field; `seed_fields` documents the placeholders and nothing reads it. Some prompts carry extra keys the code reads (`level_descriptions` for the typo rewrite, `answer_styles` for benign generation).

---

## Installation

```bash
pip install -e .                   # core: pandas, openai, anthropic, python-dotenv, pyfiglet, filelock
pip install -e ".[vllm]"           # local models through vLLM (+ hf_transfer, psutil)
pip install -e ".[transformers]"   # the introspection backend (torch, transformers, accelerate)
pip install -e ".[hf]"             # published datasets from the HuggingFace Hub
pip install -e ".[dev]"            # pytest, ruff, mypy, ipykernel
pip install -e ".[docs]"           # the documentation site: mkdocs build --strict
```

Tests: `PYTHONPATH=src python -m pytest tests -q`.

## Usage

Every stage front but `build_dataset` takes one **`data_dir`** working root, a **role-defaulted model** per role (`None` is the registry role's default: constitution → `constitution_gen` (Claude Opus), generation and checking → `uncensored_gen`, translation → `translation`, paraphrase → `paraphraser`), `prompt_dir`, and **`resume`** (default `True`). A model name resolves through the registry (`llms/model_config.py`) to a `ModelClient`; for a custom endpoint, `register_model()` it.

```python
from redact import (
    generate_constitution, generate_inputs, generate_outputs,
    generate_jailbreaks, generate_paraphrases, build_dataset, run_pipeline,
)

DATA_DIR = "./runs/training"

# Inputs are seeded from a constitution (eval and training alike); they land in
# Datasets/constitution_inputs/. Without constitution_df, the saved one is read.
constitution = generate_constitution(data_dir=DATA_DIR, num_taxonomy_categories=3)
inputs = generate_inputs(data_dir=DATA_DIR, constitution_df=constitution, samples_per_entry=3)

outputs = generate_outputs(data_dir=DATA_DIR, inputs=inputs)        # responses + output checker
jailbreaks = generate_jailbreaks(data_dir=DATA_DIR, inputs=inputs)  # attacks + natural variations, defaults
# A changed section plans the kind again: resume=False starts it over (a resume would raise).
jailbreaks = generate_jailbreaks(data_dir=DATA_DIR, inputs=inputs, resume=False, augmentation={
    "adversarial": {"entry_types": ["harmful"], "rounds": "escalation"},
    "natural": None,                         # off: variations.csv is left as it is
})
paraphrases = generate_paraphrases(data_dir=DATA_DIR, target="both")
dataset = build_dataset(data_dir=DATA_DIR, mode="training")         # paraphrases merged in

summary = run_pipeline("src/redact/configs/runs/training_example.json")   # or scripts/run.py
```

`build_dataset(mode="eval")` writes the paraphrases to `paraphrased.csv` instead of merging them. The `*_path` overrides (`output_path`, `variations_path`, `manifest_path`, `benign_path`, `inputs_path`, `outputs_path`, `jailbreak_path`, `responses_path`, ...) move one file; `taxonomy_dir`/`prompt_dir` relocate config and prompt inputs.

---

## Key Design Decisions

- **Prompts are external** — no prompt text in library code; the stages' run-time additions sit in `content_moderation/feedback.py`. A new category is a taxonomy entry.
- **Backends are swappable** — switching from API to vLLM is a recipe's `backend_type`, not a code change.
- **Inputs are per-category** — input samples land in `Datasets/constitution_inputs/{category}/samples.csv`. The other stages write their CSVs under `Datasets/` (one per paraphrase target and per jailbreak kind), and the constitution under `Data_cache/constitution/`. Merging is explicit (`build_dataset`): it takes the accepted rows of every stage (`accepted_mask`) and leaves out a row whose `sample_id` an earlier part wrote. Rejected and error rows stay in the stage files.
- **Checker reasoning feeds back to the generator** — a rejection's reasoning goes into the next generation call (`checked()`, the outputs stage's `max_attempts`).
- **Reject over accept** — a reply with the wrong count or format is rejected whole and asked again, never salvaged.
- **`Data_cache/` is internal** — the constitution and the benign pairs go there, never in `Datasets/`.
- **Resume model — one sidecar system** — every batched stage plans and resumes through the same two files, built on `redact.dataset.JsonlSidecar`: a `<artifact>.manifest.jsonl` **plan** (`Manifest`, one row per unit, written whole before generation) and a `<artifact>.state.jsonl` **ledger** (`Ledger`, one record per finished unit). Both key on the same per-stage unit id: constitution `source_category::prompt` (the entry type, or `general_benign`); inputs `(sample_description, entry_type, style)`; outputs `input_id`; paraphrase / jailbreaks / variations / conversations `(input_id, iteration)`; the benign data `(main_category, sub_category, answer_type)`; conversation scoring `sample_id` (ledger only).
  - **A save** is two synced appends (`commit` / `CommitBuffer`): the rows of the finished units with `append_rows`, then their acks, the last carrying `_saved` (the size and a CRC32 of the last 4 kB of every file of the stage). A unit is acked only once its rows are on disk; a new ledger opens with an empty `_saved` line.
  - **A resume reads the ledger alone.** `resume_state(ledger, resume=True, artifact=...)` takes the stage's lock (`<ledger name>.lock`, `filelock`), checks each output file against the recorded size and checksum (missing, shorter or different raises a `RuntimeError` naming the remedies), cuts off what an unfinished save left (kept in `<file>.cut`), cuts the ledger back to its last `_saved` line, and returns the finished keys; those units' rows are never written twice. A folder written before the ledger recorded sizes is taken over once. Each save checks that every file still has its recorded size, so a second writer is caught.
  - **A fresh run** (`resume=False`) removes the output, its `.cut` and `.tmp` files, the `.dropping` marker, the ledger and the manifest, then plans anew; `clear_stage` does the same for a stage skipped for want of accepted rows. A stage of several files names their folder and pattern (inputs: `artifact=folder, pattern="*/samples.csv"`), and a fresh run of one input style removes that style's rows and acks alone (`drop_units`, which leaves `<ledger name>.dropping` until it is done). The constitution names no output to `resume_state`: its ledger is plain (no lock, no sizes, nothing cut), and it clears its own CSVs.
  - **The jailbreak stage** keeps its manifest on a resume (its plan is sampled) and keeps beside it `<kind>.plan.json` (what the plan was made with; a resume with another section or seed raises) and `<kind>.coverage.csv`; the other stages write their manifest again from their inputs. `Manifest.write`, `write_concat` and `drop_units` write through `<file>.tmp`, synced, then a rename. The driver adds `<stage>.run.json` per stage and one trace per run.
  - A failed batch **dispatch** propagates (the transport failing, never acked rows); only a unit's own exception, and on the async lane its `PromptTooLongError`, is isolated. When adding a stage, use `Manifest`, `Ledger`, `resume_state` and `CommitBuffer`; don't hand-roll `*.jsonl` read/write.
- **Concurrency is capability-driven, and settled at construction** — a backend binds `max_workers` from its setup's `recommended_max_workers` (or the run's `workers`), clamped against its class's `supports_parallel_calls`; `ModelClient` picks the dispatch of `generate()` once from `supports_native_batching`; `drive_units` sends a request on the async lane exactly when its backend has `supports_async_calls`. Overriding concurrency on a backend that can't support it raises.
- **Every dataset row has a `sample_id`** (its own content hash) **and, where it has an origin row, an `input_id`** (that row's `sample_id`): an input is the root (no `input_id`); output, jailbreak, variation, paraphrase and conversation rows carry both. A jailbreak's `sample_id` is `content_id` of its text (`units.row_of`), so every technique and iteration variant has its own and an accepted no-op equals its input's (the build leaves it out); an error row's is `content_id("{label}:{input_id}:{iteration}")`; a conversation's is `content_id("{input_id}:{iteration}")`. Constitution entries get `content_id(sample_description)`. When adding a stage, give every row a `sample_id`; add `input_id` only when there is an origin row.
- **Internals capture is a non-interfering hook, never a CSV column** — it writes only under `{log_dir}/...` and never touches a stage's rows, ledger or manifest. `internals_id(s)` is an explicit parameter of the calls that carry it (`LLMBackend.generate`/`agenerate`, `ModelClient`, `BatchCaller`, the router's batch helpers, `checked`/`extracted`, `LLMRequest`), never passed through `**kwargs`; only `TransformersIntrospectionBackend` acts on it, and `ModelClient`/`BatchCaller` raise `ValueError` before dispatch when a backend without `supports_internals` gets ids. Ids are rooted at an id the stage already has: outputs `{input_id}/output` and `/val_out` (later attempts under `/attempt_<n>`); inputs `<entry content id>/input_<style>` and `<sample_id>/val_in_<style>` (one sample per call enforced); paraphrase `{input_id}/paraphrase/...` and jailbreak `{input_id}/{label}/attempt_{iteration}`, captured under a provisional id and renamed to the row's `sample_id` (`rename_capture`). Conversations are not captured.
- **Telemetry is emitted low and decided high** — `llms/observe.py` is a bare hook, so `llms/` imports nothing upward and no optional dependency; `telemetry.py` owns sinks, aggregation and cost. A `call` event is one transport call (one per item for hosted models and vLLM). The JSONL trace is a plain file writer, never a `logging.Handler`; the library sets no log level and attaches no handler.
- **Cost is metered as it is bought** — tokens per *model* for APIs, engine wall-clock lifetime per *checkpoint* for local (two models sharing a load share one meter). Same principle as the caches: endpoint → SDK client, checkpoint → engine, model → rate-limit window.
- **A footprint is computed, not measured** — planning runs on `llms/resources/estimate.py` (shapes from the HF config, explicit headroom), since vLLM preallocates `gpu_memory_utilization × card` and its engine core runs in another process; `vram_gb` in the registry overrides the weights when known better. Everything is GiB.
- **src/ layout** — prevents accidental imports of local code during testing; the package installs with `pip install -e .`.

---

## Runner Scripts

A run is a recipe (plus its params file) handed to `run_pipeline`, or to `scripts/run.py RECIPE [--params P] [--data-dir D] [--quiet|--debug]`, which configures logging and prints the summary; `src/redact/configs/runs/` holds an eval and a training example, each with its params file. Each stage logs its acceptance counts and leaves a `{stage}.run.json` record; `scripts/inspect_run.py DATA_DIR` prints them. `scripts/scaffold_prompts.py TARGET [--empty]` writes an editable prompt overlay; `scripts/download_models.py [MODELS] [--dry-run]` fetches the local weights of the registry ahead of a run; `scripts/gpu_check_residency.py STEP --root R` runs the residency checks on a GPU pod.

Rejection rate is the primary quality signal. A sustained high rejection rate means either a bad prompt (improvable) or a model capability ceiling (not improvable): a weak checker rejects almost everything.

---

## Generation Strategy

Diversity comes from the constitution: each entry (a category, subcategory, severity and sample description) drives its own generation call, and `style` picks the template (`long`, `short`). Within a run the inputs stage lists the first texts it saved as do-not-repeat, and a text is written once per folder. The jailbreak plan deals layers and families evenly per `(category, entry_type, iteration)` group, and the natural kind adds mild variations of the same inputs.

---

## Reference Implementations

Four reference projects informed this architecture, in `../reference_files/` relative to this repo, in order of influence:

1. **Jailbreak library** (`../reference_files/BELLS_Jailbreak_Test/`) — LLM call wrappers, rate limiting, retry logic, regex utilities, dataset splitting getters, runner script structure.
2. **Content moderation output** (`../reference_files/output dataset/`) — the vLLM integration pattern; the paraphrasing / fingerprint removal call pattern (its training pipeline is out of scope here).
3. **Content moderation input** (`../reference_files/input dataset/`) — the original generation + filtering flow; its hosting code is obsolete.
4. **Constitutional classifier** (`../reference_files/constitutional_classifier/`) — `constitution_gen.ipynb` and `constitution.csv`, the origin of `constitution/`.

---

## Out of Scope (Separate Repositories)

- **Defingerprinting pipeline** — reverse paraphraser training, the fingerprint removal model, the finetuning loop. This library calls a paraphraser model (the `paraphraser` role) but does not train one.
- **Prompt finetuner / auto-optimizer** — automated prompt optimization (task-dataset generation, running the task, feedback), reading and writing prompt JSONs in this library's schema.

---

## Public Release Notes

Prompts and template files ship with the library (`prompts/`, `configs/`); there is no redaction step. To change a prompt, overlay it: `scripts/scaffold_prompts.py TARGET_DIR` writes an editable copy of `prompts/` (`--empty` for TODO skeletons), and `prompt_dir=TARGET_DIR` reads each prompt from there when present. Before publishing a release, clear `Datasets/` and `Data_cache/`.
