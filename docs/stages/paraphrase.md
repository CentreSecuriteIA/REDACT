# Stage: paraphrase

Front `generate_paraphrases` (`stages.py`); core `run_paraphrase_target`
(`content_moderation/paraphrase.py`), once per target.

## Purpose

Add reworded copies of the inputs and of the accepted outputs, to remove the
generating model's fingerprint from the text. A separate model checks that
each paraphrase keeps its original's meaning. Paraphrases are extra rows,
never replacements.

The shipped paraphraser, `venice-paraphraser`, is a stand-in: it loads the
same 24B weights as the local setup of `venice-uncensored`. The trained
defingerprinting model lives in another repository; register it with the
`paraphraser` role when you have it.

## Inputs

| Target | Source rows | Text column | Id column |
|---|---|---|---|
| `inputs` | the accepted inputs of the run, or `inputs=` | `sample`, else `prompt` | `sample_id` |
| `outputs` | the accepted rows of `Datasets/output_responses.csv`, or `outputs=` | `output_response`, else `sample` | `input_id` |

`target` is `inputs`, `outputs` or `both`. A target whose source has rows but
none accepted is skipped with a warning; one with no rows at all is skipped
without one.

The prompts are `prompts/paraphrase/generation/` and
`prompts/paraphrase/check/`, both through the overlay.

## Models

| Recipe role | Registry role | Default |
|---|---|---|
| `paraphraser` | `paraphraser` | `venice-paraphraser` (local vLLM) |
| `paraphrase_check` | `uncensored_gen` | `venice-uncensored` (hosted) |

`paraphraser` may name one model, or `"distribution"` for every model with
the `paraphraser` role. With several, each unit goes to one of them, picked
from a hash of `(seed, input_id, iteration)`. The front warns when the check
model is also a paraphraser, and when `paraphrases_per_sample` is above 1
with a single paraphraser: identical paraphrases are dropped, so fewer may
remain.

## Settings

Params keys: `target`, `paraphrases_per_sample`, `check`, `batch_size`,
`seed`, `inputs_path`, `outputs_path`
([configuration.md](../configuration.md#paraphrase)).

## Unit and ledger key

One unit is one paraphrase of one source row: `(input_id, iteration)`, with
`iteration` from 0 to `paraphrases_per_sample - 1`. For the `outputs` target,
`input_id` is the output row's `input_id`, the input it answers.

| | Value |
|---|---|
| Ledger key | `(input_id, iteration)` |
| Ledger record | the key, `paraphrase_model`, and `status`: `accepted`, `rejected` or `dropped_dedup` |
| Manifest row | `input_id`, `iteration`, `paraphrase_model`, `category`, `entry_type`, `status` |

The stage runs one paraphraser at a time. For each it sends the pending units
in chunks of `batch_size`: one batch of paraphrase calls, then one batch of
check calls, then one save of the whole chunk. It does not use the unit pool
or the async lane. When the paraphrasers cannot be resident together on the
GPUs, the stage unloads each one before the next, and keeps the check model
and the engines other stages hold.

## Files

Per target, under `Datasets/`:

| File | Content |
|---|---|
| `paraphrases_inputs.csv`, `paraphrases_outputs.csv` | The paraphrases |
| `paraphrases_<target>.manifest.jsonl` | The plan, written again every run |
| `paraphrases_<target>.state.jsonl` | The ledger |

| Column | Meaning |
|---|---|
| `sample_id` | Content id of the paraphrase |
| `input_id` | The source row's id (see above) |
| `iteration` | Which paraphrase of the row |
| `sample` | The paraphrase |
| `category`, `entry_type` | Copied from the source row |
| `paraphrase_model` | The model that wrote it |
| `accepted` | The checker's verdict; `true` when `check` is off |
| `reasoning` | The checker's reply when rejected |
| `source` | `paraphrase_inputs` or `paraphrase_outputs` |

The build merges the accepted paraphrases into the complete dataset in
`training` mode, and writes them to `Datasets/paraphrased.csv` in `eval`
mode ([build.md](build.md)).

## How it resumes

Every attempted unit is acked, whatever its outcome, so a resume runs only
units no save recorded. A resume reads the whole artifact once for the texts
already in it. `resume: false` removes the CSV, its `.cut` file, the ledger
and the manifest of each target first. The lock and the cut-back work as in
[internals/dataset.md](../internals/dataset.md).

## Rejected and failed units

- A paraphrase the checker rejects is saved with `accepted` false. The build
  leaves it out.
- A paraphrase that is empty, equal to its original, or equal to a text
  already in the file is dropped: no row, acked `dropped_dedup`.
- Neither is tried again on resume.
- A failed call stops the stage. A chunk that fails part way saves none of
  its units, so they run again on resume.
