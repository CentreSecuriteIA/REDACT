# Stage: jailbreaks

Front `generate_jailbreaks` (`stages.py`); core
`jailbreak.pipeline.run_augmentation`, called once per kind.

## Purpose

Augment the accepted inputs with technique combinations, in two kinds:

| Kind | Writes | Default |
|---|---|---|
| `adversarial` | `Datasets/jailbreaks.csv`: jailbreak attacks | one round per input, by budget, from every technique but `translation` and `casual_writing` |
| `natural` | `Datasets/variations.csv`: natural variations | one technique per input, from the natural allow lists |

Both kinds are on by default. The recipe's `augmentations` section, or the
front's `augmentation=` argument, says what each kind draws, and switches a
kind off with `null` ([configuration.md](../configuration.md#the-augmentation-section)).
The front returns the rows of `jailbreaks.csv`; read `variations.csv` with
`dataset.read_rows`.

How a combination is drawn, planned and run is in
[internals/jailbreak.md](../internals/jailbreak.md). The technique names are
in [techniques.md](../techniques.md).

## Inputs

- The accepted inputs of the run, or `inputs=`. A `sample` column is renamed
  `prompt`; the frame must have one of the two. Each row's id is its
  `sample_id`, else the content id of its text.
- Each kind keeps the rows of its `entry_types` (all by default).
- With no input file the front raises `No input samples found`; with no
  accepted input it warns `jailbreaks skipped: ...` and writes nothing.

## Models

| Recipe role | Registry role | Default | Calls |
|---|---|---|---|
| `gen` | `uncensored_gen` | `venice-uncensored` | Every technique that calls a model, `translation` aside, and the benign data |
| `translation` | `translation` | `deepseek-v3.2` | The `translation` family; resolved, planned and loaded only when a kind has `"translation": true` |

Each translation is up to 4 rounds of two calls, a translation and its check,
both to the `translation` model.

## Settings

The augmentation section holds what each kind draws and how. The params
file holds the execution settings: `seed`, `batch_size`,
`auto_generate_benign`, `output_path`, `variations_path`, `manifest_path`,
`benign_path` ([configuration.md](../configuration.md#jailbreaks)).

## Unit and ledger key

One unit is one input at one iteration, for one kind: `(input_id,
iteration)`. A kind with `rounds` has one iteration per round. Each kind has
its own plan, ledger and coverage table.

| | Value |
|---|---|
| Ledger key | `(input_id, iteration)` |
| Ledger record | `{"input_id": ..., "iteration": ...}` |
| Manifest row | `sample_id` (the input's id), `iteration`, `combination` (technique names in applied order), `settings` (the round's sampler settings and the `seed`, with `family_weights` when set), `category`, `entry_type`, `status` |

## Files

| File | Content |
|---|---|
| `Datasets/jailbreaks.csv` | The adversarial rows |
| `Datasets/jailbreaks.manifest.jsonl` | Its plan |
| `Datasets/jailbreaks.state.jsonl` | Its ledger |
| `Datasets/jailbreaks.coverage.csv` | What its plan drew |
| `Datasets/variations.csv` and its three sidecars | The same for the natural kind |
| `Data_cache/benign/benign_samples.csv` | The benign data, shared by both kinds |

`variations.csv` goes to `variations_path` when given, else beside
`output_path` when that is given, else to `Datasets/variations.csv`.

Both CSVs have the same columns: the input's columns, then the fields of the
row. The input's `sample_id` and `accepted` columns keep their place and hold
the row's own values.

| Column | Meaning |
|---|---|
| input columns | `prompt` (the input text), `category`, `entry_type`, ... as the input had them |
| `sample_id` | Content id of `jailbreak`; for an error row, of `<input_id>:<iteration>` |
| `input_id` | The input's id |
| `jailbreak` | The augmented text |
| `technique` | The combination: technique names joined by `+`, `identity` when empty |
| `technique_info` | Each step's info joined by `;`; `noop=<name>` for a step that left the text unchanged |
| `complexity` | Sum of the techniques' complexities |
| `num_techniques` | Techniques in the combination |
| `is_noop` | Whether `jailbreak` equals the input |
| `accepted` | Whether the row is usable |
| `reasoning` | Why not, when `accepted` is false |
| `iteration` | The unit's iteration |
| `combination_spec_version` | The spec's `version` |

Columns of the coverage table: `level` (`layer`, `family` or `technique`),
`name`, `category`, `entry_type`, `iteration`, `count`. It counts what the
plan drew per group, and adds a zero row for each technique of the pool
never drawn. The planner logs a WARNING when a layer falls two or more picks
short of its rate in a group.

## The benign data

Techniques tagged `requires_benign` (the `fsh` and `dap` families) need
benign question-and-answer pairs. When a pending unit's combination needs
them and `auto_generate_benign` is on, the stage loads
`Data_cache/benign/benign_samples.csv`, or generates it with the `gen` model
when it is missing: 58 benign categories, two calls each (8 pairs with short
answers, 8 with long), one category at a time, the file saved after each.
Columns: `main_category`, `sub_category`, `prompt`, `answer`, `answer_type`.

In the GPU check the generation takes about 40 minutes on a 24B model, about
35 seconds per category: most of the stage's time. The file counts as complete
whenever it exists: a run stopped half way leaves a partial file that later
runs load, and a reply with the wrong number of pairs adds no rows. Delete
the file to generate it again. The generation reads the packaged prompt only.

## How it resumes

Each kind resumes on its own, from its ledger, with the lock and cut-back of
[internals/dataset.md](../internals/dataset.md).

The plan is frozen. A resumed kind reuses its manifest and runs the planned
units its ledger does not record. A changed augmentation section, a changed
`seed`, and inputs added since the plan take effect only with
`resume: false`, which removes that kind's CSV, `.cut` file, ledger, plan and
coverage table, then plans again. A kind that is off is not touched.

## Rejected and failed units

| Case | Row |
|---|---|
| A translation fails its check in all 4 rounds | `jailbreak` holds the text before the translation; `accepted` false; `reasoning` and `technique_info` start with `DISCARDED; technique=<name>` |
| A natural variation leaves the text unchanged | `accepted` false; `reasoning` `unchanged: the text is its input's`. An adversarial row that changes nothing stays accepted, with `is_noop` true |
| A technique raises (a bad reply, a prompt too long) | An error row: the input text, `accepted` false, `technique_info` `ERROR: <message>` |
| A call fails (provider error) | The stage stops; saved units stay saved |

Every row, rejected or not, is acked, so a resume does not retry it.

## Running one prompt by hand

`jailbreak.apply_combination(fn, prompt, client=...)` runs one technique, or
one `combine_techniques(...)` result, on one prompt with blocking calls, and
returns `(text, info, accepted, reasoning)`. It loads the benign data when a
technique needs it. `combine_techniques` sorts the techniques into layer
order but does not check the compatibility rules.
