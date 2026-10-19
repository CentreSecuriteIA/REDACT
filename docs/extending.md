# Extending REDACT

How to add a model, a prompt, a jailbreak technique, a stage, a category
or taxonomy, and how to change a dataset type. Each section names the files to touch and the tests that check
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

A local machine-translation model, which takes one user message in its own
format and follows no other instruction:

```json
"my-translator": {
  "backend_type": "vllm",
  "roles": ["translation", "machine_translation"],
  "supports_system_prompt": false,
  "default_temperature": 0.2,
  "vllm": {
    "hf_model_id": "org/my-translator",
    "hf_config": "my-translator",
    "vllm_kwargs": {"max_model_len": 4096}
  }
}
```

- `machine_translation` makes the translation technique send it
  `jailbreak/translate_mt` and check its output itself. Overlay that prompt
  when the model's format differs; the stage checks it renders to one user
  message.
- `hf_config` names a folder under `configs/llm/hf_configs/` holding a
  `config.json` read in place of the checkpoint's, when the installed
  transformers cannot read the checkpoint's own. vLLM also reads the config
  of the folder it loads the tokenizer from, so the engine is given a folder
  of the checkpoint's tokenizer files beside a copy of yours, built at load
  (a `tokenizer` in `vllm_kwargs` replaces it). Leave it out otherwise.
- Every language needs a code, accepted GlotLID labels and a median length
  ratio in `jailbreak/obfuscation/translation.py`; import fails when one is
  missing. Measure the median with the model on a parallel corpus such as
  FLORES-200: a guessed 1.0 rejected most Japanese translations, whose
  median is 0.47.

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
     `configs/jailbreak/<layer>/<module>_templates.json`, reads it with
     `directives.load_templates`, and applies one with
     `directives.apply_template_directive(..., rng=rng)`.
   - Return `(text, info)`, `info` being `""` or `key=value;...`.
2. **Register it**: return it from its module's getter
   (`get_encoding_functions`, `get_framing_functions`, ...), which its
   subpackage's `get_all_hacking_functions`, `get_all_manipulation_functions`,
   `get_all_obfuscation_functions` or `get_all_request_functions` collects. A
   new module's getter is added to that collector in the subpackage's
   `__init__.py`.
3. **Add its spec entry** under `techniques` in
   `configs/jailbreak/combination_spec.json`: `families` (the first sets the
   layer and the order), `complexity` (0 to 3), `requires_llm`,
   `requires_benign`, `encode_weight` for an `encode` technique,
   `max_tokens` to cap its calls' replies, `cross_incompatible_with` for
   families this technique, not its whole family, may not join, and
   `kinds` when it should differ from its first family's (below). A new
   family also needs an entry under `families`: `layer`,
   `within_layer_order` (its place in the obfuscation order; 0 elsewhere),
   `dimension`, `cross_incompatible_with` and, for one variations may
   draw, `kinds`, plus a `description`, which nothing reads.
   A combination takes one technique per family; a layer's cap is in
   `layer_max_picks`. A rule binds both ways, so one of the two families'
   lists is enough; a rule between two families of one capped layer is
   needless. Importing `redact.jailbreak` raises while the spec and the
   registry name different techniques.
4. **Decide which kinds draw it.** The spec's `kinds`, a list of
   `"adversarial"` and `"natural"`, say which kinds may draw a technique: a
   technique's own `kinds` replace its first family's, and a family without
   `kinds` is `["adversarial"]`. Give a technique `"natural"` only when a
   benign input it rewrites stays benign and reads as a person would write
   it. A family whose techniques are mostly natural carries both kinds,
   and its attack-only techniques say `["adversarial"]` (the high and insane
   typos, five answer formats); an attack family with one natural
   technique keeps no `kinds` and that technique says both
   (`to_leetspeak_basic`). So a new technique in an attack family is an
   attack only until its entry says otherwise, and a new technique in a
   family marked natural (`typos`, `casual_writing`, `translation`,
   `answer_language`, `answer_format`, `asking`, `indirect`) takes its
   family's kinds and joins the natural pool unless its entry says
   `["adversarial"]`. The adversarial kind draws every family unless
   denied. For variations, also add the family to the natural kind's
   `families.allow`, or the name to its `techniques.allow`, in
   `configs/jailbreak/augmentations.json`; an allow list naming what the
   spec keeps from the kind raises. A family meant for variations only goes
   in `natural_only_families`.
   - **The natural pool.** `tests/fixtures/natural_pool.json` pins the 46
     names the natural kind draws by default, and
     `tests/jailbreak/test_pool.py` fails until the owner decides: give the
     technique `"kinds": ["adversarial"]`, or add its name to the fixture.
     The same file checks the attack pool's 142 techniques in 22 families.
   - **The eval pool.** `configs/dataset_types/eval.json` allows whole
     families (`encode`, `fsh`, `dap`, `framing`, `structural`, `ascii_art`,
     `continuation`, `temporal`), so a technique added to one of them joins
     the eval attacks unless `eval.json` denies it under
     `techniques.deny`. `tests/fixtures/eval_pool.json` pins the 37 names the
     eval type draws, and `tests/jailbreak/test_pool.py` fails until the
     owner decides: deny the technique in `eval.json`, or add its name to the
     fixture. A user's copy of `eval.json` in a `config_dir` is not updated:
     a deny added to the packaged file does not reach it, so a technique new
     to a family the copy allows joins that user's eval pool.
5. **Test it.** Add tests beside the module's, under
   `tests/jailbreak/<layer>/`. `tests/jailbreak/test_registry.py` compares
   the registry with `tests/fixtures/technique_registry.json` (names, tags,
   signatures, and the combinations five seeds draw), so it fails on any new
   technique. Once the change is meant, regenerate the fixture from the
   repository root with `PYTHONPATH=src python -m tests.jailbreak.test_registry`.

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
     *clients), cap=batch_size)`. Isolate a unit's own errors in `on_error`;
     let any other propagate. On the async lane (a client with
     `supports_async_calls`: OpenAI-compatible, vLLM) a `PromptTooLongError`
     is raised inside the unit at its `yield`; handle it there or in
     `on_error` as a rejected row or ack. On the grouped path (Anthropic,
     introspection) it propagates from `generate()` and stops the stage, as
     any failed call does.
   - Give every row a `sample_id`, the content id of its text, and an
     `input_id` when it derives from another row.
   - Take `resume`, `verbose` and `read_back`.
2. **A path** for its output in `paths.py`.
3. **The front** in `stages.py`: resolve the inputs (a frame, or the run's
   own), one model per role (`None` for the registry role's default, through
   `ModelClient.create`), the output path, `prompt_dir`; call the core; return
   what it returns. The front never touches a ledger. When the inputs hold
   no accepted row, it warns, writes nothing, and with `resume=False` clears
   the stage's files with `dataset.clear_stage` (the core that would clear
   them does not run). Export it from `redact/__init__.py`: `Stage.run`
   looks the front up there.
4. **A `Stage` row** in `STAGES`, at its place in the run order: `name`,
   `front`, `roles` (pairs of recipe role and registry role), `files`
   (`ctx -> (paths, values)`: the CSVs that hold its rows and the column
   values a counted row holds), `recipe_kwargs` (`ctx -> dict`, what the run
   passes the front; a value an earlier stage saves is a `Saved`), `sources` (`ctx -> list[Path]`, the files it reads its
   inputs from, for its run record; `[]` when it reads none), `folded`
   when the run folds some params keys into a kwarg it passes itself, and
   `extra_counts` when its record counts more than its files' rows. A new
   recipe role also goes in `_DEFAULT_MODELS` in `recipe.py`.
5. **The build**: add a `Part` in `_stage_parts`, with `accepted_only=True`,
   when its rows belong in the complete dataset, with its `include_<part>`
   and path keys, and list its file in `_build_sources`; when its rows
   belong in an eval file too, map it onto that file's columns in
   `_eval_parts`. A row whose
   `sample_id` an earlier part wrote is left out. Update
   `tests/fixtures/build_files.json`.
6. **Tests.** Cover a stopped and resumed run: no unit twice, no unit lost.
   `tests/test_stage_files.py` pins each stage's files after a stopped and a
   resumed run against `tests/fixtures/stage_files.json`; add the new stage
   there. `tests/conftest.py` has a mock backend and client.
7. **Docs**: a page under `docs/stages/`.

## Add a category or a taxonomy

A taxonomy is a JSON file under `configs/taxonomy/`, under the `taxonomy/`
folder of a recipe's `config_dir`, or under the folder a recipe names as
`taxonomy_dir`:

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
  `group_tag` (default `general`, recorded as `source_group_tag`).
- **A new taxonomy**: write the file, or call
  `redact.create_taxonomy(name, categories, taxonomy_dir=...)`, then set
  `"taxonomy": "<name>"` (and `taxonomy_dir`) in the recipe. A dict passed to
  a front as `taxonomy=` works too.

The category reaches every later stage through its rows' `category` column.
The benign categories are a taxonomy of their own,
`configs/taxonomy/benign.json`, in the same format. Two stages read it: the
constitution builds its general-benign part from it, one call per category
with its subcategories, and the jailbreak stage generates the `fsh` and
`dap` benign pairs per subcategory. Write another and name it as the
recipe's `benign_taxonomy`.

## Change a dataset type

A dataset type is a JSON file with exactly three keys besides `_` comments:
`stages` (the default stage list), `params` (per stage, the front's defaults,
checked as a params file's are) and `augmentations` (an overlay of the
augmentation section, kind by kind). To change what `eval` or `training`
means for your runs, copy its file into `<config_dir>/dataset_types/` and
edit it; `scripts/scaffold_setup.py` writes both. There are only these two
types, as the build has only these two modes: a recipe naming another
`dataset_type` stops when it loads, and a file of another name in
`<config_dir>/dataset_types/` is not read, with a warning
([configuration.md](configuration.md#dataset-types)).
