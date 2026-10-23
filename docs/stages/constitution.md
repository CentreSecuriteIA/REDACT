# Stage: constitution

Front `generate_constitution` (`stages.py`); core
`ConstitutionPipeline.run` (`constitution/generation.py`).

## Purpose

For each taxonomy category, ask a strong model for a hierarchy of
constitution categories, subcategories and sample descriptions, at each of
the four entry types: `harmful`, `dual_use_harmful`, `dual_use_benign`,
`benign`. Each sample description later seeds input samples
([inputs.md](inputs.md)). The general-benign part adds benign entries for
each category of the benign taxonomy. Every run seeds its inputs from a
constitution, in `eval` and `training` runs alike: both dataset types list
this stage in their default `stages`; a recipe that lists its own keeps it,
or runs on a `data_dir` where an earlier run saved `merged.csv`. An `eval`
run takes the `harmful` and `benign` entry types only, with the
general-benign part; a `training` run all four, with the general-benign
part ([configuration.md](../configuration.md#dataset-types)).

The four entry types are the severities a classifier is trained to tell
apart:

| Entry type | Meaning | File |
|---|---|---|
| `harmful` | Absolutely harmful: always flag | `harmful.csv` |
| `dual_use_harmful` | Borderline, with a harmful framing: debatable | `dual_use_harmful.csv` |
| `dual_use_benign` | Borderline, with a benign framing: could look harmful | `dual_use_benign.csv` |
| `benign` | Absolutely benign: never flag (hard negatives) | `benign.csv` |

## Inputs

- The taxonomy: a file under `taxonomy_dir` (default
  `configs/taxonomy/harmful.json`) or a dict. Only the
  first `num_taxonomy_categories` categories are used. Each category's
  `description`, `subcategories` and `group_tag` fill the prompt.
- With `include_general_benign`, the benign taxonomy: the recipe's
  `benign_taxonomy` (default `configs/taxonomy/benign.json`, 11 categories,
  58 subcategories), the same file the `fsh` and `dap` techniques' benign data
  covers. Its first `num_taxonomy_categories` categories are used, each
  filling the `general_benign` prompt as a taxonomy category fills its own:
  `Category`, `description`, `subcategories` (joined by `, `) and
  `num_categories`. The call passes `group_tag` too (`general` when the file
  names none), which the packaged `general_benign` prompt has no placeholder
  for; it is the row's `source_group_tag`.
- The prompts `prompts/constitution/<entry_type>/`, plus `general_benign`.
  The recipe's `prompt_dir` overlays them.

## Models

| Recipe role | Registry role | Default |
|---|---|---|
| `constitution` | `constitution_gen` | `claude-opus-4-6`: Anthropic API, 5 requests a minute, one at a time |

Each call asks for at most 10,000 reply tokens. `claude-opus-5-5`, through
OpenRouter, is registered for the same role; name it in the recipe to use it.

## Settings

Params keys: `num_categories`, `num_taxonomy_categories`, `entry_types`,
`include_general_benign`, `general_benign_categories`
([configuration.md](../configuration.md#constitution)). The former names
`include_standalone_benign` and `standalone_benign_categories` are read for
this release, with a `DeprecationWarning`; given, they win. The run passes
`benign_taxonomy` from the recipe.

## Unit and ledger key

One unit is one call: a taxonomy category at one entry type, through the
entry type's prompt. With `include_general_benign`, each benign-taxonomy
category is one more unit, through the `general_benign` prompt, asking for
`general_benign_categories` constitution categories; its rows are `benign`
entries with the benign category as `source_category` and `benign` as
`source_taxonomy`.

| | Value |
|---|---|
| Ledger key | `unit` = `<source_category>::<prompt>`: `<category>::<entry_type>` for a taxonomy unit, `<benign category>::general_benign` for a general-benign one, so a benign category named as a harmful one is a unit of its own |
| Ledger record | `{"unit": ...}`; a unit rejected on every attempt: `{"unit": ..., "status": "rejected", "attempts": 3}` |
| Manifest row | `source_category`, `entry_type`, `prompt`, `status` (`planned`) |

The units run one at a time, in plan order: taxonomy categories in file
order, entry types in the order above (or in the order `entry_types`
gives them), then the general-benign units in the benign taxonomy's order.
At the defaults, 11 harmful categories and 11 benign ones, a training run
makes 44 + 11 calls and an eval run 22 + 11. A ledger written before the
general-benign part was built from the benign taxonomy holds a
`general::general_benign` ack that matches no unit now; its rows stay in the
CSVs until a fresh run.

## Files

All under `Data_cache/constitution/`:

| File | Content |
|---|---|
| `harmful.csv`, `dual_use_harmful.csv`, `dual_use_benign.csv`, `benign.csv` | The entries of one entry type |
| `general_benign.csv` | The general-benign entries, with `include_general_benign` |
| `merged.csv` | Every entry. The inputs stage reads this file |
| `constitution.manifest.jsonl` | The plan |
| `constitution.state.jsonl` | The ledger |
| `constitution.state.jsonl.lock` | The stage's lock, held while it runs |
| `<file>.csv.cut` | What a resume cut off a CSV: the tail of a save that did not finish |

Every CSV has the same columns:

| Column | Meaning |
|---|---|
| `sample_id` | Content id of `sample_description` |
| `constitution_category` | The top heading of the reply's hierarchy |
| `constitution_subcategory` | The second heading |
| `sample_description` | One sample description; seeds input samples |
| `entry_type` | The unit's entry type; `benign` for a general-benign unit |
| `source_category` | The taxonomy category, or the benign taxonomy's category for a general-benign unit |
| `source_taxonomy` | `harmful` for a taxonomy unit, `benign` for a general-benign one: the general-benign part is one filter away |
| `source_group_tag` | The taxonomy category's `group_tag`, else `general` |
| `model` | The model that wrote the entry |

`constitution.read_constitution(dir)` reads `merged.csv` back with every cell
as text and an empty cell as `""`. The driver uses it to seed a run's
inputs.

## How it resumes

The stage resumes as every stage does
([internals/dataset.md](../internals/dataset.md#the-save-and-resume-model)):
it names its folder and the pattern `*.csv` to the resume, so every CSV in
it is checked and cut back together. Each unit's rows are appended to its
own file and to `merged.csv`, then acked, one save per unit, the ack
recording each file's size.

- `resume: true` takes the stage's lock, checks each CSV against the size
  and checksum the last save recorded, cuts off what a save that did not
  finish left (a crash in the middle of a row, or between a unit's rows and
  its ack: the bytes go to `<file>.csv.cut`), skips the units the ledger
  records, and appends to the CSVs. The unit of the unfinished save runs
  again, so no row is torn or written twice.
- `resume: false` removes every `*.csv` in the folder, their `.cut` and
  `.tmp` files, the ledger and the manifest first.
- `merged.csv` keeps every row a finished save wrote. A resume with fewer
  categories or entry types plans only those, and keeps the rows of the
  others: the inputs stage seeds from every row of `merged.csv`. Start over
  (`resume: false`) to drop them.
- A CSV that is missing, shorter than the ledger recorded or changed before
  its recorded end (deleted or edited by hand) stops the resume before any
  call, naming the remedies: restore the file, delete the ledger, or start
  over.
- Without its ledger (deleted by hand), or with the plain ledger an earlier
  version wrote, which records no sizes, a resume takes the folder over
  once: a unit whose rows are in `merged.csv` counts as done, and the other
  units run. A CSV of such a folder that ends inside a row stops the
  takeover, since nothing recorded says where its whole rows end: repair
  the file, or start over.
- A resume of a `merged.csv` saved without a column the rows now have (one
  written before `source_taxonomy`) raises before any call, naming the
  column and `resume=False`.

## Rejected and failed units

A reply is a three-level markdown hierarchy: a constitution category, its
subcategories, and a sample description per bullet.
`constitution.parse.parse_constitution(text)` parses it into `ParsedEntry`
objects (`category`, `subcategory`, `sample`):

```python
from redact.constitution.parse import parse_constitution

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

A reply is rejected whole when it does not end with the `[END]` marker (it
stopped early) or when no entry parses from it. A rejected reply is generated
again, up to 3 attempts. A unit rejected every time is acked `rejected` with
no rows, skipped on resume, and named in a warning at the end of the stage.
To try it again, start the stage over, or delete its line from the ledger.

A failed call (a provider error after the SDK's own retries) stops the
stage. The units saved before it stay acked. A prompt too long for the model
stops the stage too, on every backend: the stage does not handle it.

In the GPU check of this release, one taxonomy category gave four units and
72 entries, 18 per entry type.
