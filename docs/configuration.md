# Configuration reference

Every key a run reads: the recipe, the params file, the augmentation section,
the model registry, the environment, and the prompt overlay. For how these
fit together in a run, see [running.md](running.md).

## Recipe keys

A recipe is a JSON object. `recipe.load_recipe` refuses any other key; a key
that starts with `_` is a comment.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dataset_type` | `"eval"` or `"training"` | `"eval"` | `training` seeds the inputs from the constitution saved under `data_dir`; `eval` seeds them too when it lists `constitution` or a constitution is saved under `data_dir`, and otherwise runs the standalone inputs. Also the build's `mode` |
| `data_dir` | path | the default working root | The run's root; `Datasets/` and `Data_cache/` go under it. Relative to the current directory. `run_pipeline(data_dir=...)` and `--data-dir` override it |
| `models` | object | every role its default | A model per role; see [below](#the-models-object) |
| `taxonomy` | name or object | `"harmful"` | The taxonomy of the constitution and the standalone inputs, by file name or as a loaded object. The former name `"content_moderation_categories"` loads `harmful` with a `DeprecationWarning`, for one release |
| `taxonomy_dir` | path | `configs/taxonomy/` | Where a taxonomy name (`taxonomy`, `benign_taxonomy`) is looked up |
| `benign_taxonomy` | name or object | `"benign"` | The taxonomy whose subcategories the benign Q&A pairs of the `fsh` and `dap` techniques are generated for. Read at load, and checked to have a category, when the recipe runs `jailbreaks` and a kind draws a technique that needs benign data |
| `prompt_dir` | path | none | The prompt overlay; see [below](#the-prompt-overlay) |
| `stages` | list of stage names | `["inputs", "outputs", "jailbreaks", "build"]` | The stages to run, in order, each once. `[]` runs nothing |
| `augmentations` | object | the packaged default | The jailbreak stage's augmentation section; see [below](#the-augmentation-section) |
| `params_file` | path | none | The params file, relative to the recipe's folder. `--params` and `run_pipeline(params=...)` override it |
| `resume` | bool | `true` | `false` starts every listed stage over ([running.md](running.md#resume-fresh-runs-and-a-kill-partway-through)) |

A top-level `copies` key raises: write the count in the role's object
instead.

## The models object

Keys are recipe roles: `gen`, `check`, `translation`, `constitution`,
`paraphraser`, `paraphrase_check`. A value is a registry name, `null` (the
role's default), or an object:

| Key | Type | Meaning |
|---|---|---|
| `name` | registry name | Left out or `null`: the role's default |
| `backend_type` | `"api"`, `"vllm"`, `"introspect"` | The setup of the registry entry that serves the model |
| `copies` | int >= 1 | vLLM setups only: copies of the engine, one per card. Left out: a copy on each idle card |
| `workers` | int >= 1 | API setups only: requests in flight. Left out: `recommended_max_workers`. Must be 1 on Anthropic |

`paraphraser` may also be `"distribution"`, the pool of every model with the
`paraphraser` role; it takes no object. One model is served one way per run.
[running.md](running.md#choose-models) gives the roles and their defaults.

## Params keys per stage

The params file maps a stage name to keyword arguments of its front. A key
must be a parameter of the front and not one the run passes itself (the
models, `data_dir`, `prompt_dir`, `taxonomy`, `taxonomy_dir`,
`benign_taxonomy`, `resume`, `verbose`, the jailbreak stage's
`augmentation`, the build's `mode`, `read_back`). Parameters that take a DataFrame
(`inputs`, `outputs`, `constitution_df`) cannot be written in JSON.

### `constitution`

| Key | Default | Meaning |
|---|---|---|
| `num_categories` | `10` | Constitution categories asked for per call |
| `num_taxonomy_categories` | `null` | Only the first N taxonomy categories; `null` for all |
| `entry_types` | `null` | Entry types to generate; `null` for all four |
| `include_standalone_benign` | `false` | Add one call for benign entries with no taxonomy category |
| `standalone_benign_categories` | `10` | Categories asked for in that call |

### `inputs`, seeded (training)

These keys hold for an eval run whose inputs are seeded too (its recipe
lists `constitution`, or a constitution is saved under `data_dir`).

| Key | Default | Meaning |
|---|---|---|
| `style` | `"long"` | The prompt under `prompts/input/generation/from_constitution/<style>/`: `long` or `short` |
| `samples_per_entry` | `3` | Samples asked for per constitution entry |
| `entry_types` | `null` | Use only the constitution entries of these entry types |
| `batch_size` | `32` | Requests per grouped call; the stage keeps twice as many units live, or the widest async lane when larger |
| `max_attempts` | `3` | Generations of an entry before a reply with the wrong number of samples is given up |
| `check_max_tokens` | `null` | Reply limit of a check call; `null` for the checker's default |

### `inputs`, standalone (eval)

These keys hold for an eval run without a constitution.

| Key | Default | Meaning |
|---|---|---|
| `samples_per_category` | `15` | Accepted samples to reach per category |
| `samples_per_request` | `5` | Samples asked for per call |
| `num_categories` | `null` | Only the first N taxonomy categories |
| `use_metaprompt` | `true` | Generate each category's description and seeds with the model; `false` reads the taxonomy description and a seeds file |
| `seeds_name` | `"harmful"` | The seeds file under `configs/seeds/`, with `use_metaprompt: false`. The former name `"content_moderation_seeds"` loads `harmful` with a `DeprecationWarning`, for one release |
| `num_seeds` | `8` | Seed prompts to generate per category |

### `outputs`

| Key | Default | Meaning |
|---|---|---|
| `check_outputs` | `true` | Check each response; `false` accepts every response |
| `max_attempts` | `1` | Generations per input; above 1 a rejected response is generated again with the verdict as feedback |
| `max_per_category` | `null` | Keep the first N inputs per `(category, entry_type)` |
| `max_samples` | `null` | Keep the first N inputs overall, after `max_per_category` |
| `batch_size` | `32` | Requests per grouped call; units live as for `inputs` |
| `check_max_tokens` | `null` | Reply limit of a check call |
| `output_path` | `Datasets/output_responses.csv` | Where the responses go |

### `paraphrase`

| Key | Default | Meaning |
|---|---|---|
| `target` | `"both"` | `inputs`, `outputs` or `both` |
| `paraphrases_per_sample` | `1` | Paraphrase units per source row |
| `check` | `true` | Check each paraphrase with the `paraphrase_check` model |
| `batch_size` | `256` | Units per paraphrase batch and per check batch |
| `seed` | `42` | Seed of the assignment of units to paraphrasers |
| `inputs_path`, `outputs_path` | `Datasets/paraphrases_inputs.csv`, `Datasets/paraphrases_outputs.csv` | Where the paraphrases go |

### `jailbreaks`

| Key | Default | Meaning |
|---|---|---|
| `seed` | `42` | The run seed of the plan and of each technique's random choices |
| `batch_size` | `256` | Requests per grouped call; at least twice as many units are live |
| `auto_generate_benign` | `true` | When a pending unit needs benign data, generate the pairs `benign_path` lacks for the benign taxonomy's subcategories, then load it |
| `output_path` | `Datasets/jailbreaks.csv` | The adversarial kind's file |
| `variations_path` | beside `output_path`, else `Datasets/variations.csv` | The natural kind's file |
| `manifest_path` | `jailbreaks.manifest.jsonl` beside the output | The adversarial kind's plan |
| `benign_path` | `Data_cache/benign/benign_samples.csv` | The benign data, with its manifest and ledger beside it |

The flat keywords (`include_hacking`, ..., `entry_types`) are also accepted
here, with a `DeprecationWarning`, and applied over the adversarial kind (see
[deprecated keywords](#deprecated-flat-keywords)). A key that only the
augmentation section has, such as `family_weights`, raises here.

### `build`

| Key | Default | Meaning |
|---|---|---|
| `include_inputs`, `include_jailbreaks`, `include_variations`, `include_outputs`, `include_paraphrases` | `true` | Which parts to merge |
| `output_path` | `Datasets/complete_dataset.csv` | The merged file |
| `inputs_path` | `Datasets/constitution_inputs_merged.csv`, else the category files | The inputs part |
| `jailbreak_path`, `variations_path`, `responses_path` | the stages' default files | The other parts |
| `paraphrase_inputs_path`, `paraphrase_outputs_path` | the stage's default files | The paraphrase parts |

A path key moves one stage's file only. Later stages read the default paths
unless given their own: the build has a path key per part, and the paraphrase
stage reads `Datasets/output_responses.csv` from its default place.

## The augmentation section

The jailbreak stage runs two **kinds**:

- `adversarial`: jailbreak attacks, written to `jailbreaks.csv`;
- `natural`: natural variations of the inputs, written to `variations.csv`.

The section is `{"adversarial": {...}, "natural": {...}}`. Each kind given
overlays its default from `configs/jailbreak/augmentations.json`. A kind left
out is the default; a kind given as `null` is off. With both off the stage
writes nothing. `recipe.load_recipe` resolves the section, so every error
below stops a run before the first stage.

### Keys

Both kinds:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entry_types` | list or `null` | `null` | Augment only inputs of these entry types; `null` for all |
| `translation` | bool | `false` | Whether the kind draws the `translation` family. Decides alone, whatever the lists say |
| `layers` | `{"deny": [...]}` | `{"deny": []}` | Layers left out |
| `families` | `{"allow": [...] or null, "deny": [...]}` | adversarial: `{"allow": null, "deny": []}`; natural: the natural families | Families admitted and removed |
| `techniques` | `{"allow": [...] or null, "deny": [...]}` | adversarial: `{"allow": null, "deny": []}`; natural: three techniques allowed, three denied | Techniques admitted and removed |
| `pure_only` | bool | `false` | Drop every technique that calls a model or needs benign data |
| `family_weights` | `{family: number > 0}` | `{}` | A family's share within its layer when the plan deals families; 1 each by default |
| `iterations` | int >= 1 | `1` | Units per input, numbered by `iteration` |

Adversarial only:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rounds` | `null`, `"escalation"` or a list | `null` | The sampler settings per round; see [rounds](#rounds) |
| `max_complexity` | int >= 0 | `6` | Budget mode: the most total complexity (each technique scores 0 to 3) |
| `max_obfuscations` | int >= 0 | `2` | Budget mode: the most obfuscation families |
| `layer_rates` | `{layer: 0..1}` | `{}`, the spec's rates | Target share of units that take a `hacking`, `manipulation` or `requests` technique, and the chance of `request_obfuscation` |
| `allow_request_obfuscation` | bool | `false` | Allow one more technique after the request technique, at the `request_obfuscation` rate: a light `encode` technique, or a translation when the pool has one |

Natural only:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `exact_techniques` | int >= 1 | `1` | Techniques per variation |

The packaged natural kind allows the families `typos`, `casual_writing`,
`structural`, `answer_language`, `answer_format`, `asking`, `indirect`,
`temporal`, `continuation`; allows the techniques `to_leetspeak_basic`,
`to_adversarial_suffix_punctuation`, `to_adversarial_suffix_emoji`; and
denies `to_refusal_then_comply`, `to_apology_prevention`,
`to_prefix_injection`. A variation that leaves its input unchanged is
rejected.

### How a kind's pool is built

From every registered technique, in this order:

1. remove the layers in `layers.deny`;
2. when `families.allow` or `techniques.allow` is a list, keep only the
   techniques of an allowed family or with an allowed name (`translation`
   counts as allowed when the switch is on);
3. remove the families in `families.deny`, and `translation` when the switch
   is off;
4. remove the techniques in `techniques.deny`;
5. with `pure_only`, remove every technique tagged `requires_llm` or
   `requires_benign`.

[techniques.md](techniques.md) lists every layer, family and technique name.

### Overlay rules

- A value given replaces the default's whole value. A list replaces a list.
- `layers`, `families` and `techniques` are overlaid list by list. An `allow`
  or `deny` given replaces the default's list of that name and keeps the
  other.
- A `deny` given without an `allow` also takes its names out of the default's
  `allow`. `"natural": {"families": {"deny": ["typos"]}}` is the natural
  default without typos. A deny that empties the default's allow raises:
  write the allow as well.
- An `allow` list may not be empty: write `null` for all, or name what to
  leave out under `deny`. A name may not be under both.
- The natural kind needs an allow list.
- `natural_only_families`, at the top of `augmentations.json`
  (`casual_writing` in the packaged file), are added to the adversarial `families.deny` unless the
  adversarial `families.allow` names them.

### The translation switch

`"translation": false` keeps the `translation` family out of a kind, and
`true` puts it in: the attacks draw it as any family, and the natural kind
adds it to its allow list. No deny or allow list changes this. An allow list
that names `translation` or one of its techniques while the switch is
`false` raises.

Both kinds default to `false`. Each translation makes 2 to 8 calls, up to 4
rounds of two ([stages/jailbreaks.md](stages/jailbreaks.md#models) says which
model each goes to). The translation model is planned and loaded only when a
kind has the switch on.
Turning it on by default is one line per kind in `augmentations.json`.

### Rounds

`rounds` sets the adversarial kind's units per input and each round's sampler
settings. Each round is merged over the kind's budget keys.

| Value | Rounds |
|---|---|
| `null` | `iterations` rounds of the kind's budget settings |
| `"escalation"` | The four rounds of `default_escalation_schedule()`, below |
| a list of objects | One round per object. Keys: `exact_techniques`, `max_complexity`, `max_obfuscations`, `layer_rates`, `allow_request_obfuscation` |

With `rounds`, `iterations` is the number of rounds; giving both with
different values raises. A round with `exact_techniques` picks exactly that
many compatible techniques and ignores the complexity budget. A round
without it picks by budget.

The escalation schedule:

| Round | Settings |
|---|---|
| 1 | `exact_techniques: 1` |
| 2 | `exact_techniques: 2` |
| 3 | `max_complexity: 6`, `max_obfuscations: 2`, rates hacking 0.7, manipulation 0.5, requests 0.8, request_obfuscation 0.5, `allow_request_obfuscation: true` |
| 4 | `max_complexity: 9`, `max_obfuscations: 3`, rates hacking 0.9, manipulation 0.7, requests 0.9, request_obfuscation 0.7, `allow_request_obfuscation: true` |

### Rates and weights

The plan deals techniques per `(category, entry_type, iteration)` group
rather than drawing each unit alone. In budget mode each of the `hacking`,
`manipulation` and `requests` layers joins a share of the group's units equal
to its rate: `layer_rates` over the spec's (hacking 0.5, manipulation 0.33,
requests 0.7, request_obfuscation 0.5). Within a layer, families are dealt by
count over `family_weights`, so `{"persona": 2}` gives `persona` twice the
share of each other hacking family. `family_weights` may name only families
the kind's pool draws. [internals/jailbreak.md](internals/jailbreak.md#balanced-dealing)
explains the dealing.

### Examples

Escalate the attacks on harmful inputs only, weight one family, no
variations:

```json
"augmentations": {
  "adversarial": {
    "entry_types": ["harmful", "dual_use_harmful"],
    "families": {"deny": ["ascii_art"]},
    "rounds": "escalation",
    "family_weights": {"persona": 2}
  },
  "natural": null
}
```

Variations only, two per benign input, no attacks:

```json
"augmentations": {
  "adversarial": null,
  "natural": {"entry_types": ["benign", "dual_use_benign"], "iterations": 2}
}
```

### Deprecated flat keywords

`generate_jailbreaks` still takes twelve flat keywords, read into the
adversarial kind with a `DeprecationWarning`. The same keys are read, and
warn, at the top level of a recipe's `augmentations` (then the natural kind is
the default) and in `params["jailbreaks"]`.

| Flat keyword | Section key |
|---|---|
| `include_hacking`, `include_manipulation`, `include_obfuscation`, `include_requests` | `false` adds the layer to `layers.deny` |
| `include_translation` | `translation` |
| `pure_only`, `max_complexity`, `max_obfuscations`, `iterations`, `entry_types` | the same key |
| `sampling_probs` | `layer_rates` |
| `settings_per_iteration` | `rounds`, each round's `sampling_probs` as its `layer_rates` |

## The model registry

`configs/llm/models.json` holds one entry per model, keyed by its registry
name. It loads into `llms.model_config.MODEL_REGISTRY` at import, and every
entry is validated then; one bad entry stops the import with a list of every
problem. `register_model(...)` adds or replaces an entry for the running
process only.

### Entry fields

| Field | Default | Meaning |
|---|---|---|
| `backend_type` | the only setup present | The default setup: `api`, `vllm` or `introspect`. Needed when there are several |
| `roles` | `[]` | Registry roles the model fills. Each must be in `configs/llm/roles.json` |
| `is_uncensored` | `false` | Marks an uncensored generation model |
| `supports_system_prompt` | `true` | `false` folds system messages into the first user message |
| `default_max_tokens` | `2000` | Reply limit when a call gives none |
| `default_temperature` | `null` | 0.0 to 2.0; `null` leaves it to the backend (vLLM and introspection use 0.7) |
| `api`, `vllm`, `introspect` | `null` | The setups; at least one |
| `notes` | | Free text, ignored |

### `api` (APIConfig)

| Field | Default | Meaning |
|---|---|---|
| `backend_type` | required | `openai` (any OpenAI-compatible endpoint) or `anthropic` |
| `api_key_env` | required | The environment variable holding the key |
| `rpm` | required | Requests per minute, over a sliding 60-second window |
| `base_url` | required for `openai` | The endpoint URL |
| `api_model_id` | the registry name | The model id sent to the provider |
| `rate_limit_scope` | `"model"` | `model`: one window per model. `endpoint`: one window for every model with the same provider, URL and key variable; they must declare the same `rpm` |
| `recommended_max_workers` | `1` | Requests in flight. Must be 1 for `anthropic` |
| `default_extra_body` | `null` | Provider parameters sent with each call (OpenAI-compatible only) |
| `price_per_1m_input`, `price_per_1m_output` | `null` | USD per million tokens, for the cost roll-up |

### `vllm` (VLLMConfig)

| Field | Default | Meaning |
|---|---|---|
| `hf_model_id` | required | HuggingFace id or local path |
| `quantization` | `null` | vLLM quantization method, such as `awq`, `gptq`, `fp8` |
| `vllm_kwargs` | `null` | Engine arguments (vLLM's `AsyncEngineArgs`), such as `max_model_len`, `enforce_eager`, `tokenizer_mode`. `gpu_memory_utilization` and `data_parallel_size` are refused: the planner sets them |
| `sampling` | `null` | Sampling defaults (`top_p`, `top_k`, ...) over the backend's `top_p` 0.85 |
| `vram_gb` | `null` | The weights in GiB, total over the model's cards, when known better than the estimate |
| `min_gpus` | `1` | Cards per copy; above 1 it is the `tensor_parallel_size` |

### `introspect` (IntrospectConfig)

| Field | Default | Meaning |
|---|---|---|
| `hf_model_id` | required | HuggingFace id or local path |
| `log_dir` | required | Root folder of the captured internals |
| `capture` | `logprobs: true`, `hidden_states: "last"`, `attention: false` | What to save per captured call; `hidden_states` is `false`, `"last"` or `"all"` |
| `device_map`, `torch_dtype` | `"auto"` | Passed to `from_pretrained` |
| `sampling` | `null` | Sampling defaults |
| `vram_gb` | `null` | The weights in GiB |
| `extra_kwargs` | `null` | More `from_pretrained` kwargs. Capturing attention loads with eager attention unless this names another |

### Roles and shipped models

`configs/llm/roles.json` lists the registry roles: `uncensored_gen`,
`translation`, `constitution_gen`, `paraphraser`, `debug_local`. A role's
default model is the first entry in `models.json` that lists it.

| Model | Default setup | Other setup | Roles | Limits |
|---|---|---|---|---|
| `venice-uncensored` | `api` (Venice) | `vllm`, 24B, `vram_gb` 43.9 | `uncensored_gen` | 75 rpm, 3 workers |
| `deepseek-v3.2` | `api` (Venice) | | `translation` | 20 rpm, 2 workers |
| `olafangensan-glm-4.7-flash-heretic` | `api` (Venice) | | `uncensored_gen` | 20 rpm, 2 workers |
| `llama-3.2-3b-debug` | `vllm`, 3B, `max_model_len` 384 | | `debug_local` | |
| `claude-opus-4-6` | `api` (Anthropic) | | `constitution_gen` | 5 rpm, 1 worker, endpoint window |
| `claude-opus-5-5` | `api` (OpenRouter) | | `constitution_gen` | 600 rpm, 8 workers |
| `venice-paraphraser` | `vllm`, same weights as `venice-uncensored` | | `paraphraser` | |

`venice-paraphraser` stands in for a trained defingerprinting model, which
lives in another repository. Register the real one with the `paraphraser`
role.

## Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `VENICE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | the shipped API setups | API keys, named by `api_key_env` |
| `HF_HOME` | vLLM, transformers | The HuggingFace cache for weights, `~/.cache/huggingface` when unset. Give it room: the local weights of `venice-uncensored` take about 48 GB |
| `HF_TOKEN` | HuggingFace | Access to gated checkpoints and datasets |
| `REDACT_OUTPUT_DIR` | `paths.get_output_dir` | The default working root, and where `.env` is read |
| `REDACT_TRACE` | `telemetry.install` | `jsonl` (default), `wandb` or `off` |
| `WANDB_PROJECT` | the W&B sink | Project name, default `redact` |
| `REDACT_GPU_PROVIDER` | the cost roll-up | Provider key in `gpu_pricing.json`; default `local`, priced at 0 |
| `REDACT_GPU_HOURLY_USD` | the cost roll-up | Hourly price per GPU; wins over the table |
| `CUDA_VISIBLE_DEVICES` | the planner, vLLM | The cards the run may use; the plan numbers cards over them |

## The prompt overlay

Prompts are JSON files under `src/redact/prompts/<pipeline>/<category>/`,
loaded by `load_prompt(pipeline, category, prompt_dir)` or
`PromptTemplate.load(...)`. A folder holds one `.json` file, or else
`template.json`.

`prompt_dir` is an overlay. For each prompt, the loader looks in
`<prompt_dir>/<pipeline>/<category>/` first and falls back to the packaged
copy. A user folder holds only the prompts it changes, so the others stay
current when the library's improve. At the start of a run the driver logs how
many prompts are overridden, and warns about a folder that matches no
packaged prompt (a misspelt name) or that has several `.json` files and no
`template.json`.

Every stage passes `prompt_dir` to the loader, so each prompt a model call
is built from is read from the overlay when the overlay has it: the
constitution, the inputs (generation, format instruction, check, and the
standalone description and seed calls), the outputs, the paraphrases, every
technique of the jailbreak stage (translation included) and its benign data.
The conversation functions, which no recipe runs (`evaluate_conversations`,
`generate_attacks`, `score_attacks`, `optimize_attack`, `optimize`), take
`prompt_dir` as a keyword.

### Prompt files

```json
{
  "system_prompt": "... {Category} ...",
  "template": "... {sample} ...",
  "seed_fields": ["Category", "sample"],
  "few_shot_examples": [],
  "metadata": {"version": "1.0", "notes": ""}
}
```

- `system_prompt` and `template` are rendered with `str.format_map`. A
  `PromptTemplate` renders the system prompt once, when built, and the
  template on each call. A literal brace is doubled (`{{ }}`). A missing
  placeholder raises and names it, and the file when the prompt is loaded
  with `PromptTemplate.load`.
- `seed_fields` documents the placeholders; nothing reads it.
- `few_shot_examples` are message objects passed through as they are.
- A prompt built with a `format_style` (`numbered`, `structured_qa`) gets the
  matching instruction from `prompts/format_instructions/<style>/`, a file
  with one key, `instruction`, appended to the system prompt.
- Some prompts carry extra keys the code reads (`level_descriptions` for the
  typo rewrite, `answer_styles` for benign generation). Keep them in an
  override.

### The prompt tree

| Pipeline | Prompts |
|---|---|
| `constitution` | `harmful`, `dual_use_harmful`, `dual_use_benign`, `benign`, `general_benign` |
| `input` | `generation/from_constitution/long`, `generation/from_constitution/short`, `generation/standalone`, `quality_check`, `category_description`, `seed_generation` |
| `output` | `generation`, `quality_check` |
| `paraphrase` | `generation`, `check` |
| `jailbreak` | `scenario_generation`, `persona_scenario_generation`, `jailbreak_construction`, `category_selection`, `benign_generation`, `extract_harmful`, `synonym_substitution`, `rewrite_with_typos`, `translate`, `translate_retry`, `translate_check` |
| `conversation` | `judge`, `success_judge`, `pair_attacker` |
| `format_instructions` | `numbered`, `structured_qa` |

The tree also holds `input/category_check`, `input/metaprompt`,
`jailbreak/benign_check`, `jailbreak/scenario_check` and
`jailbreak/typo_check`, which no code loads.

Text the stages add at run time, the rejection feedback and the list of
samples not to repeat, lives in `content_moderation/feedback.py`, not in a
prompt file.

### Scaffolding an overlay

```bash
python scripts/scaffold_prompts.py ./my_prompts           # a copy of the real prompts
python scripts/scaffold_prompts.py ./my_prompts --empty   # skeletons with the same keys
```

`--empty` replaces `system_prompt`, `template` and `instruction` with a TODO
that keeps the field's placeholders, so a skeleton renders with the same
values. Delete every file you do not change, then set `prompt_dir`.

## Other configuration files

| File | Holds | Read by |
|---|---|---|
| `configs/llm/gpu_pricing.json` | Provider, then GPU name as `nvidia-smi` reports it, then hourly USD per GPU; `*` is a provider's fallback | the cost roll-up |
| `configs/jailbreak/combination_spec.json` | `version`, `layer_order`, `layer_rates`, `layer_max_picks`, `families` (layer, order within the layer, incompatibilities), `manipulation_blocked_obfuscation`, `request_obfuscation_allowed_families`, `techniques` (families, complexity, `requires_llm`, `requires_benign`, `encode_weight`, `max_tokens`) | `jailbreak/technique.py`, `rules.py`, `sampling.py`, `plan.py` |
| `configs/jailbreak/augmentations.json` | The two default kinds and `natural_only_families` | `jailbreak/augmentation.py` |
| `configs/jailbreak/<layer>/*_templates.json` | Template variants of the pure techniques | `jailbreak/directives.py: load_templates` |
| `configs/jailbreak/multi_turn/crescendo_followups.json` | Follow-ups of the crescendo attack | `multiturn_attacks/attacks.py` |
| `configs/jailbreak/hacking/cognitive_techniques.json`, `personas.json` | The definitions of the cognitive techniques and of the named personas | `jailbreak/hacking/cognitive.py`, `personas.py` |
| `configs/taxonomy/harmful.json` | The harmful categories: the default `taxonomy` | `dataset/taxonomy.py: load_taxonomy` |
| `configs/taxonomy/benign.json` | The categories and subcategories of the fsh and dap techniques' benign Q&A pairs: the default `benign_taxonomy` | `dataset/taxonomy.py: load_taxonomy` |
| `configs/seeds/harmful.json` | Hand-written seeds per category, for standalone inputs with `use_metaprompt: false` | `dataset/taxonomy.py: load_seeds` |
| `configs/datasets/content_moderation_input.json` | A HuggingFace dataset id, split, columns and filters | `dataset/loading.py: load_from_config` |
| `configs/runs/*.json` | The example recipes and params file | you |

A `taxonomy_dir` of your own that still holds `content_moderation_categories.json`
serves it for `harmful`, with a `DeprecationWarning`, for one release; rename
the file `harmful.json`.
