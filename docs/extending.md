# Extending REDACT

How to add a model, a prompt, a jailbreak technique, a stage, or a category
or taxonomy. Each section names the files to touch and the tests that check
the change. Run the tests with:

```bash
PYTHONPATH=src python -m pytest tests -q
```

## Add a model

**For good**, add an entry to `src/redact/configs/llm/models.json`. Every
field is in [configuration.md](configuration.md#the-model-registry). A local
model:

```json
"my-local-8b": {
  "backend_type": "vllm",
  "roles": ["uncensored_gen"],
  "is_uncensored": true,
  "default_max_tokens": 2000,
  "vllm": {
    "hf_model_id": "org/my-model-8b",
    "vllm_kwargs": {"max_model_len": 8192}
  }
}
```

A hosted one, on any OpenAI-compatible endpoint:

```json
"my-hosted": {
  "backend_type": "api",
  "roles": ["translation"],
  "api": {
    "backend_type": "openai",
    "api_key_env": "MY_PROVIDER_KEY",
    "base_url": "https://api.example.com/v1",
    "rpm": 60,
    "recommended_max_workers": 4,
    "price_per_1m_input": 0.2,
    "price_per_1m_output": 0.8
  }
}
```

- One entry may carry both an `api` and a `vllm` setup when the hosted model
  and the local weights are the same model; `backend_type` names the default,
  and a recipe picks the other with `"backend_type"` in the role's object.
- A role's default model is the first entry that lists the role. To make a
  new model the default, place it above the others with that role.
- A role must be in `configs/llm/roles.json`. A new registry role only
  matters when a stage asks for it (a `Stage`'s `roles`).
- Set `vram_gb` when you know the weights' size better than the estimate (an
  MoE checkpoint, a measured figure). Set `min_gpus` for a model larger than
  one card.
- `python scripts/download_models.py my-local-8b` fetches the weights ahead
  of a run.

The registry loads and validates every entry at import; a bad entry stops the
import with a message naming the field. `tests/llms/test_model_config.py`
covers the validation.

**For one process**, call `register_model`:

```python
from redact.llms import APIConfig, VLLMConfig, register_model

register_model(
    "my-model", backend_type="api", roles=["uncensored_gen"],
    api=APIConfig(backend_type="openai", api_key_env="MY_KEY",
                  base_url="https://api.example.com/v1", rpm=60),
    vllm=VLLMConfig(hf_model_id="org/my-model"),
)
```

A recipe run resolves names through the same registry, so a model registered
before `run_pipeline` can be named in the recipe.

**A new provider** is a new backend: a class in `llms/backends/<name>.py`
that subclasses `LLMBackend` and sets its `compute_config` flags, with
`from_config(config)` (reading its setup from the entry), `generate(...)`,
`agenerate(...)` when it has async calls, `backend_name` and `clear_cache`.
Emit one `call` event per transport call with `self._record_call(...)`, and
raise `PromptTooLongError` when the provider rejects a prompt as over the
context. Add the class to `BACKEND_TYPES` in `backends/capabilities.py`, and
to `API_BACKEND_TYPES` for a hosted API. Nothing above `llms/` changes:
callers read the flags. [internals/llms.md](internals/llms.md#backends)
describes the contract.

## Add a prompt

A prompt is a JSON file in its own folder:
`src/redact/prompts/<pipeline>/<category>/template.json`, with
`system_prompt`, `template`, `seed_fields`, `few_shot_examples` and
`metadata` ([configuration.md](configuration.md#prompt-files)). Load it in
code with the run's overlay:

```python
from redact.llms import PromptTemplate

template = PromptTemplate.load("my_pipeline", "my_step", prompt_dir=prompt_dir,
                               Category=category)          # system prompt values
messages = template(sample=text)                           # template values, per call
```

Pass `prompt_dir` down from the front, so a user's overlay reaches the new
prompt. A placeholder may not share a name with a parameter of
`PromptTemplate` (`prompt_config`, `few_shot`, `format_style`, `name`,
`source`, `is_override`, `prompt_dir`).

**A new input style** needs only a file:
`prompts/input/generation/from_constitution/<style>/template.json`, then
`"style": "<style>"` under `inputs` in the params file. The seeded inputs
stage renders it with `Category`, `sample_description`,
`constitution_subcategory`, `entry_type` and `num_samples`, and appends the
`numbered` format instruction.

**To change a prompt for your runs** without editing the library, use the
overlay: `scripts/scaffold_prompts.py ./my_prompts`, keep the files you
change, and set `prompt_dir`
([configuration.md](configuration.md#the-prompt-overlay)).

## Add a jailbreak technique

Read [internals/jailbreak.md](internals/jailbreak.md#techniques) for the
contract.

1. **Write the function** in the module of its family, under
   `jailbreak/hacking/`, `manipulation/`, `obfuscation/` or `requests/`.
   - A transform: `def to_name(prompt: str, *, rng=None) -> tuple[str, str]`.
     Draw every random choice from `rng`.
   - A technique that calls a model: a generator with the signature
     `(prompt, **kwargs)`, delegating to a helper that takes the keywords it
     needs (`gen_model`, `translate_model`, `benign_data`, `prompt_dir`,
     `rng`) and yields `LLMRequest(gen_model, messages)`. Load its prompt from
     `prompts/jailbreak/<name>/` with `prompt_dir`.
   - A template-driven transform keeps its variants in
     `configs/jailbreak/<layer>/<name>_templates.json` and applies one with
     `directives.apply_template_directive(..., rng=rng)`.
   - Return `(text, info)`, `info` being `""` or `key=value;...`.
2. **Register it**: return it from the module's `get_<family>_functions()`,
   which its layer's `get_all_<layer>_functions()` collects.
3. **Add its spec entry** under `techniques` in
   `configs/jailbreak/combination_spec.json`: `families` (the first sets the
   layer and the order), `complexity` (0 to 3), `requires_llm`,
   `requires_benign`, `encode_weight` for an `encode` technique, and
   `max_tokens` to cap its calls' replies. A new family also needs an entry
   under `families`: `layer`, `within_layer_order`, `max_picks`,
   `cross_incompatible_with`, `description`. Importing `redact.jailbreak`
   raises while the spec and the registry name different techniques.
4. **Decide which kinds draw it.** The adversarial kind draws every family
   unless denied. For variations, add the family to the natural kind's
   `families.allow`, or the name to its `techniques.allow`, in
   `configs/jailbreak/augmentations.json`. A family meant for variations only
   goes in `natural_only_families`.
5. **Test it.** Add tests beside the module's, under
   `tests/jailbreak/<layer>/`. `tests/jailbreak/test_registry.py` compares
   the registry with `tests/fixtures/technique_registry.json` (names, tags,
   signatures, and the combinations five seeds draw), so it fails on any new
   technique. Once the change is meant, regenerate the fixture from the
   repository root with `py -m tests.jailbreak.test_registry`.

## Add a stage

A stage is a core, a front, and a row in the stage table.
[internals/top-level.md](internals/top-level.md) describes all three.

1. **The core**, in the stage's package, follows
   [the seven steps](internals/top-level.md#the-seven-steps-of-a-stage). Use
   `Ledger`, `Manifest`, `resume_state` and `CommitBuffer` from
   `redact.dataset`; do not write JSONL or CSV by hand.
   - A unit is a generator that yields `LLMRequest`s; build it from
     `llm_pipeline.checked` or `extracted` where they fit.
   - Drive the units with `drive_units(..., pool_size=pool_size(batch_size,
     *clients), cap=batch_size)`. Isolate a unit's own errors in `on_error`
     (a `PromptTooLongError` becomes a rejected row or ack); let any other
     propagate.
   - Give every row a `sample_id`, the content id of its text, and an
     `input_id` when it derives from another row.
   - Take `resume`, `verbose` and `read_back`.
2. **A path** for its output in `paths.py`.
3. **The front** in `stages.py`: resolve the inputs (a frame, or the run's
   own), one model per role (`None` for the registry role's default, through
   `ModelClient.create`), the output path, `prompt_dir`; call the core; return
   what it returns. The front never touches a ledger. Export it from
   `redact/__init__.py`: `Stage.run` looks the front up there.
4. **A `Stage` row** in `STAGES`, at its place in the run order: `name`,
   `front`, `roles` (pairs of recipe role and registry role), `files` (the
   CSVs that hold its rows, for the counts), `recipe_kwargs` (what the run
   passes the front). A new recipe role also goes in `_DEFAULT_MODELS` in
   `recipe.py`.
5. **The build**: add a `Part` in `build_dataset` when its rows belong in the
   complete dataset.
6. **Tests.** Cover a stopped and resumed run: no unit twice, no unit lost.
   `tests/test_stage_files.py` pins each stage's files after a stopped and a
   resumed run against `tests/fixtures/stage_files.json`; add the new stage
   there. `tests/conftest.py` has a mock backend and client.
7. **Docs**: a page under `docs/stages/`.

## Add a category or a taxonomy

A taxonomy is a JSON file under `configs/taxonomy/`, or under the folder a
recipe names as `taxonomy_dir`:

```json
{
  "name": "my_taxonomy",
  "description": "...",
  "categories": {
    "Category Name": {
      "description": "Used in the prompts.",
      "subcategories": ["First", "Second"],
      "group_tag": "my_group"
    }
  }
}
```

- **A new category** is one more entry under `categories`. Its name is a
  folder name on disk, so it may not hold `/` or `\`.
- The constitution stage fills its prompts with the category's name,
  `description`, `subcategories` (a list, or a dict whose keys are used) and
  `group_tag` (default `general`, recorded as `source_group_tag`). The
  standalone inputs read the `description` when `use_metaprompt` is off.
- `aliases` and `groups` may be written (`create_taxonomy` writes both); no
  stage reads them.
- **A new taxonomy**: write the file, or call
  `redact.create_taxonomy(name, categories, taxonomy_dir=...)`, then set
  `"taxonomy": "<name>"` (and `taxonomy_dir`) in the recipe. A dict passed to
  a front as `taxonomy=` works too.
- For standalone inputs without meta-prompts, give the category's seeds in a
  seeds file, `{"seeds": {"Category Name": ["...", "..."]}}`, under
  `configs/seeds/`, and set `seeds_name`.

The category reaches every later stage through its rows' `category` column.
The benign categories of the `fsh` and `dap` data are a taxonomy of their
own, `configs/taxonomy/benign.json`, in the same format: the pairs are
generated per subcategory. Write another and name it as the recipe's
`benign_taxonomy`.
