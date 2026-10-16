# Stage: constitution

Front `generate_constitution` (`stages.py`); core
`ConstitutionPipeline.run` (`constitution/generation.py`).

## Purpose

For each taxonomy category, ask a strong model for a hierarchy of
constitution categories, subcategories and sample descriptions, at each of
the four entry types: `harmful`, `dual_use_harmful`, `dual_use_benign`,
`benign`. Each sample description later seeds input samples
([inputs.md](inputs.md)). Only a `training` run lists this stage.

## Inputs

- The taxonomy: a file under `taxonomy_dir` (default
  `configs/taxonomy/content_moderation_categories.json`) or a dict. Only the
  first `num_taxonomy_categories` categories are used. Each category's
  `description`, `subcategories` and `group_tag` fill the prompt.
- The prompts `prompts/constitution/<entry_type>/`, plus `general_benign`.
  This stage reads the packaged prompts only; `prompt_dir` does not reach it.

## Models

| Recipe role | Registry role | Default |
|---|---|---|
| `constitution` | `constitution_gen` | `claude-opus-4-6`: Anthropic API, 5 requests a minute, one at a time |

Each call asks for at most 10,000 reply tokens. `claude-opus-5-5`, through
OpenRouter, is registered for the same role; name it in the recipe to use it.

## Settings

Params keys: `num_categories`, `num_taxonomy_categories`, `entry_types`,
`include_standalone_benign`, `standalone_benign_categories`
([configuration.md](../configuration.md#constitution)).

## Unit and ledger key

One unit is one call: a taxonomy category at one entry type. With
`include_standalone_benign` one more unit asks for benign entries with no
taxonomy category; its source category is `general` and its entry type
`general_benign`.

| | Value |
|---|---|
| Ledger key | `unit` = `<source_category>::<entry_type>` |
| Ledger record | `{"unit": ...}`; a unit rejected on every attempt: `{"unit": ..., "status": "rejected", "attempts": 3}` |
| Manifest row | `source_category`, `entry_type`, `status` (`planned`) |

The units run one at a time, in plan order: taxonomy categories in file
order, entry types in the order above, the general-benign unit last.

## Files

All under `Data_cache/constitution/`:

| File | Content |
|---|---|
| `harmful.csv`, `dual_use_harmful.csv`, `dual_use_benign.csv`, `benign.csv` | The entries of one entry type |
| `general_benign.csv` | The general-benign entries, with `include_standalone_benign` |
| `merged.csv` | Every entry. The inputs stage reads this file |
| `constitution.manifest.jsonl` | The plan |
| `constitution.state.jsonl` | The ledger |

Every CSV has the same columns:

| Column | Meaning |
|---|---|
| `sample_id` | Content id of `sample_description` |
| `constitution_category` | The top heading of the reply's hierarchy |
| `constitution_subcategory` | The second heading |
| `sample_description` | One sample description; seeds input samples |
| `entry_type` | The unit's entry type |
| `source_category` | The taxonomy category, or `general` |
| `source_group_tag` | The taxonomy category's `group_tag`, else `general` |
| `model` | The model that wrote the entry |

`constitution.read_constitution(dir)` reads `merged.csv` back with every cell
as text and an empty cell as `""`. The driver uses it to seed a training
run's inputs.

## How it resumes

This stage keeps a plain ledger: no lock, no recorded sizes, nothing cut
back. Each unit's rows are appended to its entry type's file and to
`merged.csv`, then acked, one unit at a time.

- `resume: true` skips the units the ledger records and appends to the CSVs.
- `resume: false` removes every `*.csv` in the folder, the ledger and the
  manifest first.
- A crash after a unit's rows are appended and before its ack leaves the
  unit un-acked. The next run generates it again and appends its rows again.

## Rejected and failed units

A reply is rejected whole when it does not end with the `[END]` marker (it
stopped early) or when no entry parses from it. A rejected reply is generated
again, up to 3 attempts. A unit rejected every time is acked `rejected` with
no rows, skipped on resume, and named in a warning at the end of the stage.
To try it again, start the stage over, or delete its line from the ledger.

A failed call (a provider error after the SDK's own retries) stops the
stage. The units saved before it stay acked.

In the GPU check of this release, one taxonomy category gave four units and
72 entries, 18 per entry type.
