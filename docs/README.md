# REDACT documentation

REDACT builds red-teaming datasets. You give it a taxonomy of categories and a
recipe; it generates samples, checks them with a second model, augments them,
and merges the accepted rows into one CSV for training, or into three flat
files for evaluation. Every stage saves as it goes and resumes where it
stopped.

These pages explain what each part of the library is for, and how a run goes
from recipe to finished dataset. They are written for developers who use or
extend REDACT.

## Where to start

| You want to | Read |
|---|---|
| See what REDACT does and follow one run end to end | [overview.md](overview.md) |
| Run a pipeline: recipe, models, GPUs, resume, cost | [running.md](running.md) |
| Look up a recipe key, a params key or a registry field | [configuration.md](configuration.md) |
| Know what one stage reads, writes and resumes by | [stages/](#stages) |
| Add a model, a prompt, a technique, a stage or a taxonomy | [extending.md](extending.md) |
| Change the library itself | [internals/](#internals) |

## All pages

- [overview.md](overview.md): the stages, a diagram of the files they pass
  on, and one training run end to end.
- [running.md](running.md): installing, running and writing a recipe,
  choosing models, the GPU planner, resume, the files on disk, telemetry and
  cost, common failures.
- [configuration.md](configuration.md): every recipe key, the dataset
  types, the params keys per stage, the generation axes, the augmentation
  section and its release notes (what an upgrade asks of a run you already
  have), the model registry, the environment variables, your own config
  folder, the prompt overlay.
- [extending.md](extending.md): how to add a model, a prompt, a jailbreak
  technique, a stage or a taxonomy, and how to change a dataset type.
- [techniques.md](techniques.md): every jailbreak technique by layer and
  family, for allow and deny lists, and the eval and natural pools.
- [api/index.md](api/index.md): the API reference, generated from the
  docstrings.

### Stages

One page per stage. Each gives the purpose, inputs, files and how it
resumes; the generation stages also give their models, settings, unit and
ledger key, and what happens to a rejected or failed unit.

- [stages/constitution.md](stages/constitution.md)
- [stages/inputs.md](stages/inputs.md): seeded from the constitution
- [stages/outputs.md](stages/outputs.md)
- [stages/paraphrase.md](stages/paraphrase.md)
- [stages/jailbreaks.md](stages/jailbreaks.md): the adversarial and natural
  kinds, `jailbreaks.csv` and `variations.csv`
- [stages/conversations.md](stages/conversations.md): conversations and
  scoring, outside the recipe; a planned extension, not finished
- [stages/build.md](stages/build.md)

### Internals

One page per layer. Each says what the layer is responsible for, what it must
not do, its main types and functions, and how the stages use it.

- [internals/top-level.md](internals/top-level.md): `stages.py`,
  `recipe.py`, `driver.py`, `paths.py`, `config_overlay.py`, and the seven
  steps of a stage
- [internals/llms.md](internals/llms.md): the model registry, backends,
  clients and prompts
- [internals/llm_pipeline.md](internals/llm_pipeline.md): generators of model
  calls, the unit driver and the async lane
- [internals/dataset.md](internals/dataset.md): rows on disk, the ledger and
  manifest, the save and resume model
- [internals/jailbreak.md](internals/jailbreak.md): the jailbreak engine,
  plan then drive
- [internals/residency.md](internals/residency.md): the GPU planner
- [internals/telemetry.md](internals/telemetry.md): the trace and the cost
  roll-up

## Conventions

- A source path such as `stages.py` is under `src/redact/`. Scripts are under
  `scripts/`, tests under `tests/`.
- `<data_dir>` is the run's working root. `Datasets/` and `Data_cache/` are
  under it.
- Names in `code` are the names the code uses.

## Terms

| Term | Meaning |
|---|---|
| Stage | One step of a run that a recipe can name: `constitution`, `inputs`, `outputs`, `paraphrase`, `jailbreaks`, `build` |
| Dataset type | What a run makes, `training` or `eval`: a file of default stages, params and augmentations, and the build's mode |
| Front | A stage's public function in `stages.py`, such as `generate_outputs`. It resolves models and paths, then calls the core |
| Core | The function in the stage's package that does the work, such as `content_moderation.run_output_generation` |
| Unit | The piece of work a stage saves and resumes by, such as one input row, or one `(input_id, iteration)` |
| Ledger | `<output>.state.jsonl`: one record per finished unit |
| Manifest | `<output>.manifest.jsonl`: the plan, one row per unit |
| Role | A job a model does in a run (`gen`, `check`, `translation`, ...). A role resolves to a registered model |
| Setup | One way to serve a registered model: `api`, `vllm` or `introspect` |
| Kind | One half of the augmentation section: `adversarial` (attacks) or `natural` (variations) |
| Entry type | The severity of a sample: `harmful`, `dual_use_harmful`, `dual_use_benign`, `benign` |
| Phase | The set of local models resident while one stage runs |
| Copy | One more instance of a local vLLM model on another card |
| Async lane | The path in `drive_units` that sends each request alone, the moment its unit yields it |
