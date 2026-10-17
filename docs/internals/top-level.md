# The top level: stages, recipe, driver, paths

`stages.py`, `recipe.py`, `driver.py` and `paths.py` turn the layers below
into stages and run a recipe. This page also gives the shape every stage
core follows: [the seven steps](#the-seven-steps-of-a-stage).

## Package layout

Under `src/redact/`:

| Path | Holds |
|---|---|
| `llms/` | The model registry, one backend per provider, `ModelClient`, rate limits, prompt loading and reply parsing ([llms.md](llms.md)) |
| `llm_pipeline/` | `LLMRequest`, the drivers of units of model calls, checks and retry steps ([llm_pipeline.md](llm_pipeline.md)) |
| `residency/` | The GPU planner: which local models fit, per stage; preload and unload ([residency.md](residency.md)) |
| `constitution/` | The constitution stage: `generation.py` (`ConstitutionPipeline`, `read_constitution`), `units.py` (one call per taxonomy category and entry type), `parse.py` (the `[END]` marker and the three-level markdown parser) |
| `content_moderation/` | The inputs, outputs and paraphrase stages: `inputs.py` (`InputPipeline`), `standalone.py` (the deprecated standalone inputs), `outputs.py`, `paraphrase.py`, `checker.py` (the checkers), `feedback.py` (rejection feedback, the do-not-repeat note), `metaprompt.py` (category description and seeds), `results.py` (result classes) |
| `jailbreak/` | The technique library, one subpackage per layer, and the augmentation engine ([jailbreak.md](jailbreak.md)) |
| `multi_turn/` | Conversation datasets: settings, actors, `generate_conversations`, `evaluate_conversations` |
| `multiturn_attacks/`, `optimization/` | Crescendo and PAIR attacks against a safety-trained target; a beam search over conversation moves ([conversations](../stages/conversations.md#attacks-and-search)) |
| `dataset/` | Rows on disk: identity, the ledger and manifest, resume, the per-category tree ([dataset.md](dataset.md)) |
| `configs/` | Package data: `llm/` (registry, roles, GPU prices), `jailbreak/` (the spec, the augmentation defaults, technique templates), `taxonomy/`, `seeds/`, `datasets/`, `runs/` (example recipes) |
| `prompts/` | One prompt JSON per pipeline and step |
| `stages.py`, `recipe.py`, `driver.py`, `paths.py` | This page |
| `telemetry.py` | The run trace and the cost roll-up ([telemetry.md](telemetry.md)) |
| `exceptions.py`, `types.py` | `RedactError`, `PromptTooLongError`; `EntryType` |

Beside the package: `scripts/` (`run.py`, `inspect_run.py`,
`scaffold_prompts.py`, `download_models.py`), `notebooks/` (walkthroughs) and
`tests/`.

## Responsibilities and boundaries

| Module | Responsible for | Must not |
|---|---|---|
| `stages.py` | The fronts (`generate_*`, `build_dataset`) and the `STAGES` table | Touch a ledger, a manifest, `resume_state` or `CommitBuffer`: that is the core's job |
| `recipe.py` | Reading and checking a recipe and its params file; how a role resolves; how a recipe serves its models | Run anything |
| `driver.py` | One run: serving, residency, telemetry, the stages in order, a record each | Read a stage's rows back: counts come from the files |
| `paths.py` | Every file and folder name under a working root, the default root, and the package's own folders | Hold stage logic: a stage asks it for a path and nothing else |

The library attaches only a `NullHandler` to the `redact` logger and sets no
level; the application configures logging (`scripts/run.py` does).

## Fronts

A front is the public function of a stage. It resolves what the stage needs
and calls one core, then returns what the core returns.

| Front | Core | Returns |
|---|---|---|
| `generate_constitution` | `ConstitutionPipeline.run` | the saved constitution |
| `generate_inputs` | `InputPipeline.run_from_constitution`, or `run_standalone_inputs` | the accepted inputs (of the current style, on the seeded path) |
| `generate_outputs` | `content_moderation.run_output_generation` | the rows of `output_responses.csv` |
| `generate_paraphrases` | `content_moderation.paraphrase.run_paraphrase_target`, per target | `{"inputs": ..., "outputs": ...}` |
| `generate_jailbreaks` | `jailbreak.pipeline.run_augmentation`, per kind | the rows of `jailbreaks.csv` |
| `build_dataset` | none; `dataset.write_concat` | the rows of the merged file |
| `generate_conversations`, `evaluate_conversations` | themselves, in `multi_turn/` | their file's rows |

Every front takes, where they apply:

- its inputs, as a frame or `None` for the run's own (`dataset.merge_inputs`);
- `data_dir`, the working root;
- one model name per role, `None` meaning the registry role's default;
- `prompt_dir`, the prompt overlay;
- its tunables, which are the params keys;
- `resume=True`, `verbose=True` and `read_back=True` (`False` returns an
  empty frame and reads nothing back);
- path overrides (`output_path`, ...).

A front that reads the run's inputs raises when no category file exists
(`No input samples found`), and warns and returns an empty frame when none
is accepted (`<stage> skipped: ...`).

A core can be called without its front, with clients and paths of your own.
The seeded inputs, for example:

```python
from redact.content_moderation import InputPipeline
from redact.llms import ModelClient, load_prompt

client = ModelClient.create("venice-uncensored")
pipeline = InputPipeline(gen=client, check=client,
                         dataset_dir="./runs/eval/Datasets/constitution_inputs")
result = pipeline.run_from_constitution(
    constitution_df=constitution,                     # from generate_constitution()
    prompt_config=load_prompt("input", "generation/from_constitution/long"),
    samples_per_entry=3, style="long",
)
print(f"{result.total_prompts_accepted} accepted samples")
```

## The stage table

`stages.STAGES` maps each recipe stage name to a `Stage`, in the order a run
takes them.

| Field | Meaning |
|---|---|
| `name` | The recipe's name for the stage |
| `front` | The front's name; `Stage.run` looks it up on `redact` when the stage runs |
| `roles` | `(recipe role, registry role)` pairs the stage calls |
| `files` | `ctx -> (csv paths, keep)`: the files that hold the stage's rows, and the values a row must hold to be counted |
| `recipe_kwargs` | `ctx -> kwargs` the run passes the front itself. A value an earlier stage saves is a `Saved`, read when the stage starts |
| `sources` | `ctx -> paths`: the files the stage reads its inputs from; those that exist go in its run record |
| `folded` | Params keys the run applies to a kwarg it passes itself (the jailbreak stage's flat keywords, folded into `augmentation`) |

`Stage.params(ctx)` returns the params file's entry for the stage and raises
on a key that is not a parameter of the front. `Stage.check(ctx)` also
raises on a key the run passes itself. `Stage.kwargs(ctx)` returns the
recipe's kwargs updated with the params file's.

| Stage | Roles | Files counted | Sources |
|---|---|---|---|
| `constitution` | `constitution` -> `constitution_gen` | `Data_cache/constitution/merged.csv` | the taxonomy file |
| `inputs` | `gen`, `check` -> `uncensored_gen` | the category files of the run's tree; accepted rows of the current style | seeded: `merged.csv`; standalone: the taxonomy file, and the seeds file with `use_metaprompt: false` |
| `outputs` | `gen`, `check` -> `uncensored_gen` | `output_responses.csv`, or its `output_path` | the input category files (`dataset.input_files`) |
| `paraphrase` | `paraphraser` -> `paraphraser`, `paraphrase_check` -> `uncensored_gen` | the files of the target(s) | the input category files and `output_responses.csv`, per target |
| `jailbreaks` | `gen` -> `uncensored_gen`, `translation` -> `translation` | `jailbreaks.csv` and `variations.csv`, each when its kind is on | the input category files |
| `build` | none | `complete_dataset.csv`, or its `output_path` | the part files, where the params put them |

`RunContext` is what one run hands the stages: `data_dir`, `dataset_type`,
`models` (role to name or `None`), `prompt_dir`, `taxonomy`,
`taxonomy_dir`, `benign_taxonomy`, `params`, `augmentations` (resolved),
`resume`, `verbose`, `stages` (the recipe's).

## recipe.py

| Function | Does |
|---|---|
| `load_recipe(path or dict)` | Reads and checks a recipe ([running.md](../running.md#write-a-recipe)); resolves `augmentations`; reads the former taxonomy name `content_moderation_categories` as `harmful`, with a `DeprecationWarning` |
| `benign_taxonomy_of(recipe, stages)` | The recipe's `benign_taxonomy` (`"benign"` by default); read, and checked to have a category, when the jailbreak stage runs a kind that draws a technique needing benign data |
| `load_params(path or dict, base_dir)` | Reads the params file; refuses a `jailbreaks` key that belongs in the augmentation section, warns for a flat keyword |
| `augmentations_of(recipe, jailbreak_params)` | The section, both kinds resolved; reads the flat form; applies the params' flat keywords over the adversarial kind |
| `serving_of(models)` | Checks the role objects; returns the keys given per model name |
| `resolve_role(models, role, augmentations)` | The models a role stands for, as its stage resolves them |

`resolve_role` has three rules beyond "the name, else the registry role's
default": `paraphraser` may be a pool (`"distribution"`); `check` left out is
the `gen` model; `translation` is no model when no kind's pool holds a
translation technique.

## driver.py

`run_pipeline(recipe, params=None, verbose=True, data_dir=None)`:

1. `load_recipe`, `load_params`, `serving_of`; build the `RunContext`.
2. `Stage.check` for every stage.
3. `telemetry.install(data_dir)`; `report_prompt_sources(prompt_dir)`.
4. `capabilities.set_run_serving(...)`: each model's `backend_type` and
   `workers` for every client built during the run.
5. `_plan_phases`: `residency.plan_phases` over each stage's models
   (`_stage_models` resolves them as the stages will), `residency.report` per
   distinct plan, `residency.apply_plan`. A planning error is logged, never
   raised.
6. `_check_cards`: one `nvidia-smi` read (`measure.card_use`) before any
   model loads; a `RuntimeError` naming the cards, processes and PIDs when
   another process holds memory on a card the plan uses. A process on an
   unplanned card, or 1 GiB or more in use on a planned card with no process
   listed, only warns.
7. `_served`: per role, the name, setup, and copies or workers, for the
   summary and the records.
8. Per stage: `_enter_phase` (unload, apply the stage's plan, preload this
   stage's models and the next stage's when they fit), then inside
   `telemetry.stage(name)`: the front with `read_back=False`, `file_counts`,
   `write_record` with `sources`. Leaving `telemetry.stage` logs the
   stage's replies cut at the token limit, per model.
9. `finally`: clear the run's serving, join the preload threads, unload every
   local engine, clear the planned shares and cards, `telemetry.summary()`.

`scripts/run.py` makes the first Ctrl-C or SIGTERM exit through that
`finally`; a second sends SIGTERM to the run's child processes (SIGKILL
after 2 seconds) and leaves with `os._exit`.

| Function | Does |
|---|---|
| `file_counts(files, keep)` | Counts `rows`, and `accepted` when a file has the column, reading in chunks |
| `write_record`, `read_record`, `record_path` | The run record `Datasets/<stage>.run.json` |

`write_manifest`, `read_manifest` and `manifest_path` are aliases of the
three record functions, kept for one release. `pipelines.py` re-exports
`stages.py` and `runconfig.py` re-exports `recipe.py` and `driver.py`, also
for one release.

## paths.py

Run files take a `root`; `None` means `get_output_dir()`: `REDACT_OUTPUT_DIR`,
else the first folder above the package holding `pyproject.toml` or `.git`,
else the current directory.

| Function | Path |
|---|---|
| `datasets(root)`, `data_cache(root)` | `<root>/Datasets`, `<root>/Data_cache` |
| `category_csv(category, root)` | `<root or Datasets>/<category>/samples.csv`; a category with a path separator raises |
| `constitution_dir`, `constitution_merged_csv` | `Data_cache/constitution/`, `.../merged.csv` |
| `constitution_inputs_dir`, `constitution_inputs_merged_csv` | `Datasets/constitution_inputs/`, `Datasets/constitution_inputs_merged.csv` |
| `output_responses_csv`, `jailbreaks_csv`, `variations_csv` | under `Datasets/` |
| `paraphrases_inputs_csv`, `paraphrases_outputs_csv`, `paraphrased_csv` | under `Datasets/` |
| `conversations_csv`, `conversations_scored_csv`, `complete_dataset_csv` | under `Datasets/` |
| `benign_csv` | `Data_cache/benign/benign_samples.csv` |

Package folders take no root: `package_dir`, `prompts_dir`, `configs_dir`,
`llm_configs_dir`, `models_json`, `roles_json`, `gpu_pricing_json`,
`taxonomy_dir` (`harmful.json`, `benign.json`), `seeds_dir` (`harmful.json`),
`dataset_configs_dir` (`configs/datasets/`), `jailbreak_configs_dir`.

## The seven steps of a stage

A core owns its files. Every core with units follows the same seven steps,
built from `dataset/` and `llm_pipeline/`. `content_moderation/outputs.py`
is the plainest example.

```python
ids = row_ids(inputs, text_col)                                          # 1
ledger = Ledger.sidecar(out, key_fields=("input_id",), casters={"input_id": str})
manifest = Manifest.sidecar(out)                                         # 2
done = resume_state(ledger, resume=resume, artifact=out, manifest=manifest)  # 3
manifest.write({"input_id": i, "status": "planned"} for i in ids)       # 4
with CommitBuffer(append=lambda rows: append_rows(out, rows), ledger=ledger) as buf:
    drive_units(units_not_in(done), on_done=lambda k, v: buf.add(rows_of(v), record_of(k)),
                on_error=..., pool_size=pool_size(batch_size, *clients),
                cap=batch_size)                                          # 5, 6
return read_rows(out) if read_back else pd.DataFrame()                   # 7
```

1. **Identity.** Each input row gets its id: its `sample_id`, else the
   content id of its text (`dataset.row_ids`).
2. **The two sidecars.** The ledger and the manifest beside the output.
3. **Fresh or resume.** `resume_state` runs before anything is written. It
   takes the stage's lock, clears everything on a fresh run, or checks and
   cuts back the files on a resume, and returns the finished unit keys.
4. **The plan.** The manifest is written whole, every run. The jailbreak
   kinds keep an existing plan on resume, since theirs is sampled.
5. **The units.** Each pending unit is a generator of model calls, driven by
   `llm_pipeline.drive_units`. Its key in the driver is its position among
   the pending units, so two rows with one id stay two units; the ledger key
   goes in the record `on_done` adds.
6. **The save.** `on_done` hands a unit's rows and ack record to the
   `CommitBuffer`, which appends rows and then acks every 32 units and on any
   exit. A failed call propagates and is never turned into rows.
7. **Read back.** Only when `read_back`. The driver passes `False`.

A stage whose resume finds nothing pending opens no buffer, and calls
`dataset.release_lock(ledger)` itself.

How each stage departs from this:

| Stage | Difference |
|---|---|
| constitution | A plain ledger: `resume_state` gets no artifact, so no lock and no sizes. It clears its own CSVs on a fresh run, saves after every unit (`every=1`), runs one unit at a time (`pool_size=1`), and reads back through `read_constitution` |
| seeded inputs | A tree of files: `artifact=folder, pattern="*/samples.csv"`, sidecars named `constitution_inputs.*` in the folder, `drop_units` for a fresh run of one style. Returns `ConstitutionInputResult`; the front reads the tree |
| standalone inputs | No sidecars and no units |
| paraphrase | Chunked batch calls instead of `drive_units`, one save per chunk |
| jailbreaks | Steps 2 to 7 run once per kind; the plan is reused on resume |
| scoring | No plan |
| build | No units |
