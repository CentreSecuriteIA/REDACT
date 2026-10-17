# Stage: inputs

Front `generate_inputs` (`stages.py`), core
`InputPipeline.run_from_constitution` (`content_moderation/inputs.py`). Every
run's inputs are seeded from a constitution, in an `eval` run as in a
`training` one; an `eval` run keeps the `harmful` and `benign` entries only
(its dataset type sets `entry_types`).

The stage saves every sample, accepted or not, with its verdict, to
`Datasets/constitution_inputs/<category>/samples.csv`. The later stages read
only the accepted ones: outputs, paraphrase and jailbreaks through
`dataset.merge_inputs`, the build through
`Datasets/constitution_inputs_merged.csv`.

## Purpose

Turn each constitution entry into `samples_per_entry` input samples, each
checked against its category, entry type and subcategory. An entry of the
general-benign part is an input of its benign-taxonomy category: generated
and checked with that category, entry type `benign` and its constitution
subcategory.

## Inputs

- The constitution frame. In a recipe run the driver reads
  `Data_cache/constitution/merged.csv` when the stage starts; a run without
  it stops with `No constitution at ...`, which names both ways to have one:
  the `constitution` stage in the recipe, or one saved by an earlier run on
  the same `data_dir`. Called directly, pass `constitution_df=`; without it
  the front reads the saved one the same way.
- Columns read per entry: `sample_description`, `entry_type`,
  `source_category`, `constitution_subcategory`, `constitution_category`,
  `source_group_tag`, and `source_taxonomy` when the constitution has it.
  `entry_types` keeps only some entry types.
- The prompt `prompts/input/generation/from_constitution/<style>/` with the
  `numbered` format instruction, and the check prompt
  `prompts/input/quality_check/`, each read through the overlay.

## Models

| Recipe role | Registry role | Default | Calls |
|---|---|---|---|
| `gen` | `uncensored_gen` | `venice-uncensored` | One generation per entry, again on a bad reply |
| `check` | `uncensored_gen` | the `gen` model | One check per sample |

Pick a checker that can judge. In the GPU check, a 3B checker accepted 2 of
144 samples and a 24B checker 144 of 144.

## Settings

Params keys: `style`, `samples_per_entry`, `entry_types`, `batch_size`,
`max_attempts`, `check_max_tokens`
([configuration.md](../configuration.md#inputs)).

## Unit and ledger key

One unit is one constitution entry.

1. One call asks for `samples_per_entry` samples as a numbered list. The
   prompt also shows the first 20 texts this run has saved, and counts the
   rest, as samples not to repeat.
2. A reply with another number of samples is rejected whole and asked again,
   up to `max_attempts` generations.
3. Samples whose text is already in the folder (by content id, any category
   and any style) are dropped.
4. Each remaining sample gets one check call.

| | Value |
|---|---|
| Ledger key | `(sample_description, entry_type, style)` |
| Ledger record | the key, and `ids`: the `sample_id`s of the rows it wrote. Or the key with `status` `rejected` or `no_samples` and `attempts` |
| Manifest row | `sample_description`, `entry_type`, `style`, `status` |

## Files

Under `Datasets/constitution_inputs/`:

| File | Content |
|---|---|
| `<category>/samples.csv` | One per category (`source_category`): a harmful-taxonomy category, or a benign one for the general-benign part; a benign category named as a harmful one shares its file, its rows told apart by `entry_type` and `source_taxonomy`. Accepted and rejected rows |
| `constitution_inputs.manifest.jsonl` | The plan of the current style |
| `constitution_inputs.state.jsonl` | The ledger, for every style |

And `Datasets/constitution_inputs_merged.csv`: the rows of every category
and every style whose `accepted` reads true (`dataset.accepted_mask`),
copied by the front after the core returns. The build reads it. It is
removed when no category file holds an accepted row, so the build never
reads rows the category files no longer hold.

Columns of `samples.csv`:

| Column | Meaning |
|---|---|
| `sample_id` | Content id of `sample` |
| `sample` | The generated text |
| `category` | The entry's `source_category`: the taxonomy category, or the benign-taxonomy category |
| `subcategory` | The entry's `constitution_subcategory` |
| `entry_type` | The entry's entry type |
| `turn` | `0` |
| `accepted` | The checker's verdict |
| `rejection_reason` | The checker's reply when rejected, else empty |
| `source` | `constitution` |
| `source_sample_description` | The entry it came from |
| `constitution_category` | The entry's constitution category |
| `source_group_tag` | The entry's group tag |
| `source_taxonomy` | `harmful` or `benign`, copied from the entry: `benign` marks the general-benign part. Only when the constitution has the column |
| `template_style` | The `style` that generated it |

## How it resumes

The stage names its folder and the pattern `*/samples.csv` to the resume, so
all category files are checked and cut back together
([internals/dataset.md](../internals/dataset.md#a-stage-with-several-files)).
Entries the ledger records are skipped. The ids on the ledger records keep a
text from being written twice.

A category file written from a constitution without `source_taxonomy`
cannot take rows that have it: the save raises `Cannot append to ...: its
header has no column 'source_taxonomy'`. Start the style over
(`resume: false`) after regenerating the constitution.

`resume: false` starts over the current `style` only. Its rows are removed
from every category file and its acks from the ledger (`dataset.drop_units`);
the rows of other styles stay. Until that finishes, a
`constitution_inputs.state.jsonl.dropping` file names the style; a later run
stops until the same fresh run is repeated.

## Rejected and failed units

- A sample the checker rejects is saved with `accepted` false and the
  checker's reply in `rejection_reason`.
- An entry with every reply rejected is acked `rejected`, with no rows.
- An entry with only repeated samples is acked `no_samples`.
- On the async lane (an OpenAI-compatible endpoint or vLLM), an entry whose
  generation or check prompt is too long for its model is acked `rejected`
  after one attempt, with no rows, and logged `Entry ... rejected, its
  prompt is too long`. On an Anthropic or introspection binding the
  requests go out in grouped calls, and a prompt too long there fails the
  call and stops the stage.
- These acked entries are skipped on resume; a warning at the end counts the
  rejected ones. Start the style over to try them again.
- Any other error inside a unit, and any failed call, stops the stage. Saved
  units stay saved.

When under half of the generated samples are accepted, the front warns:
`inputs: only N of M generated samples were accepted (under half)`.

With a model that captures internals, the stage requires
`samples_per_entry: 1`, since one call's internals cannot be split between
samples.
