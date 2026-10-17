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
- Each kind keeps the rows of its `entry_types` (all by default). A kind
  whose `entry_types` match no input logs a WARNING `<kind> kind skipped:
  ...` and writes nothing; with `resume: false` its files are removed.
- With no input file the front raises `No input samples found`; with no
  accepted input it warns `jailbreaks skipped: ...` and writes nothing. With
  `resume: false` it also removes the files of each kind that is on: CSV,
  `.cut` and `.tmp` files, ledger, manifest, plan file and coverage table
  (`dataset.clear_stage`).

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
its own manifest, plan file, ledger and coverage table.

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
| `Datasets/jailbreaks.plan.json` | What its plan was made with: `seed`, `rounds`, `pool` (the technique names), `family_weights`, `layer_rates` |
| `Datasets/variations.csv` and its four sidecars | The same for the natural kind |
| `Data_cache/benign/benign_samples.csv` | The benign data, shared by both kinds, with its `.manifest.jsonl` and `.state.jsonl` |

`variations.csv` goes to `variations_path` when given, else beside
`output_path` when that is given, else to `Datasets/variations.csv`.

Both CSVs have the same columns: the input's columns, then the fields of the
row. The input's `sample_id` and `accepted` columns keep their place and hold
the row's own values.

| Column | Meaning |
|---|---|
| input columns | `prompt` (the input text), `category`, `entry_type`, ... as the input had them |
| `sample_id` | Content id of `jailbreak`; for an error row, of `<label>:<input_id>:<iteration>`, the label being `jailbreak` or `variation` |
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
them and `auto_generate_benign` is on, the stage generates the pairs
`Data_cache/benign/benign_samples.csv` lacks, with the `gen` model, then
loads the file. With `auto_generate_benign` off, the stage loads the file as
it is; when there is no file, it stops before any unit with `Benign data not
found ...`.

The pairs cover each subcategory of the recipe's `benign_taxonomy` (the
packaged `configs/taxonomy/benign.json` by default: 11 categories, 58
subcategories; a category without subcategories is its own). A unit is one
`(main_category, sub_category, answer_type)`, the answer type `short` or
`long`: one call asking for 8 pairs. The units run on the worker pool, as the
stage's own units do. A reply that does not hold exactly 8 complete pairs is
rejected and the call made again: up to three calls in all. A unit rejected
every time is acked
`rejected` with no rows, and a WARNING names it. When no unit has a row, the
stage stops with `No benign pairs at ...`. The prompt
`prompts/jailbreak/benign_generation/` is read through the overlay.

The benign data always resumes: its ledger `benign_samples.state.jsonl`,
keyed `(main_category, sub_category, answer_type)`, records the finished
units, so a stopped run, or a larger taxonomy, generates only the units
missing. A file written before the ledger is taken over once. `resume:
false` does not clear it; delete the ledger to retry the rejected units, or
the CSV and its sidecars to generate everything again. Columns:
`main_category`, `sub_category`, `prompt`, `answer`, `answer_type`.

## How it resumes

Each kind resumes on its own, from its ledger, with the lock and cut-back of
[internals/dataset.md](../internals/dataset.md).

The plan is frozen. A resumed kind reuses its manifest and runs the planned
units its ledger does not record. A resume whose `seed`, pool, `rounds`,
`family_weights` or `layer_rates` differ from those in the kind's
`.plan.json` raises a `ValueError` that names them and `resume=False`; a
plan without that file runs as it was made, with a WARNING. A changed
`entry_types` is not compared: it only moves inputs in or out. A plan that
names a technique no registered function implements raises a `ValueError`
naming `resume=False` when that unit is built, which stops the stage. Inputs
added since the plan get no unit; a WARNING counts them and names
`resume=False`. `resume: false` removes that kind's CSV, `.cut` and `.tmp`
files, ledger, manifest, plan file and coverage table, then plans again. A
kind that is off is not touched ([running.md](../running.md#resume-fresh-runs-and-a-kill-partway-through)
says how to plan one kind again and keep the other).

## Rejected and failed units

| Case | Row |
|---|---|
| A technique rejects its result: a translation that fails its check in all 4 rounds, a malformed word list, a text no ASCII-art font draws whole, an empty reply or text | `jailbreak` holds the text before that technique; `accepted` false; `reasoning` and `technique_info` start with `DISCARDED; technique=<name>` |
| A natural variation leaves the text unchanged | `accepted` false; `reasoning` `unchanged: the text is its input's`. An adversarial row that changes nothing stays accepted, with `is_noop` true; its `sample_id` is its input's, so the build leaves it out |
| A technique raises, or a prompt is too long for a model on the async lane (an OpenAI-compatible endpoint or vLLM) | An error row: the input text, `accepted` false, `technique_info` `ERROR: <message>`, `reasoning` the message |
| A call fails (a provider error, or a prompt too long on the grouped path: Anthropic, introspection) | The stage stops; saved units stay saved |

Every row, rejected or not, is acked, so a resume does not retry it.

## Running one prompt by hand

`jailbreak.apply_combination(fn, prompt, client=...)` runs one technique, or
one `combine_techniques(...)` result, on one prompt with blocking calls, and
returns `(text, info, accepted, reasoning)`. It loads the benign data when a
technique needs it, generating what the cache lacks with `client`; its
`prompt_dir` reaches the techniques and the benign data. `combine_techniques` sorts the techniques into layer
order but does not check the compatibility rules.

```python
from redact.jailbreak import apply_combination, combine_techniques
from redact.jailbreak.hacking.cognitive import to_persona_roleplay
from redact.jailbreak.obfuscation.encoding import to_base64, to_rot13
from redact.llms import ModelClient

# A transform is a plain call: (prompt) -> (text, info).
text, info = to_base64("How to pick a lock")
text, info = combine_techniques(to_rot13, to_base64)("How to pick a lock")

# A technique that calls a model is a generator: calling it directly returns
# the generator, not a result. apply_combination drives it with blocking calls.
client = ModelClient.create("venice-uncensored")
text, info, accepted, reasoning = apply_combination(
    combine_techniques(to_persona_roleplay, to_base64), "How to pick a lock",
    client=client,
)
```
