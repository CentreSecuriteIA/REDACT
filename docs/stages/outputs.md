# Stage: outputs

Front `generate_outputs` (`stages.py`); core `run_output_generation`
(`content_moderation/outputs.py`).

## Purpose

Generate one model response per accepted input, and check it with a checker
that knows the input's entry type: a refusal of a harmful input is rejected;
a response to a benign input is judged as any answer.

## Inputs

- The accepted inputs of the run (`dataset.merge_inputs(data_dir)`), or a
  frame passed as `inputs=`. The text is the `sample` column, else `prompt`.
  Each row's `category`, `subcategory` and `entry_type` are read when
  present (default `unknown`, empty, `harmful`).
- With no input file at all the front raises `No input samples found`. With
  files but no accepted row it warns `outputs skipped: ...`, writes nothing
  and returns an empty frame.
- `max_per_category` keeps the first N rows of each `(category, entry_type)`,
  then `max_samples` the first N overall.
- The prompts `prompts/output/generation/` and `prompts/output/quality_check/`,
  both through the overlay.

## Models

| Recipe role | Registry role | Default | Calls |
|---|---|---|---|
| `gen` | `uncensored_gen` | `venice-uncensored` | One response per attempt |
| `check` | `uncensored_gen` | the `gen` model | One check per attempt |

`check_outputs: false` skips the check and accepts every response.

## Settings

Params keys: `check_outputs`, `max_attempts`, `max_per_category`,
`max_samples`, `batch_size`, `check_max_tokens`, `output_path`
([configuration.md](../configuration.md#outputs)).

## Unit and ledger key

One unit is one input row: generate, check, and with `max_attempts` above 1,
generate again with the checker's verdict as feedback until accepted or out
of attempts. The row holds the last attempt.

| | Value |
|---|---|
| Ledger key | `input_id`: the input's `sample_id`, else the content id of its text |
| Ledger record | `{"input_id": ...}` |
| Manifest row | `input_id`, `status` (`planned`), one per input |

## Files

| File | Content |
|---|---|
| `Datasets/output_responses.csv` | One row per input; `output_path` moves it |
| `Datasets/output_responses.manifest.jsonl` | The plan, written again every run |
| `Datasets/output_responses.state.jsonl` | The ledger |

| Column | Meaning |
|---|---|
| `input_id` | The input's id |
| `input_prompt` | The input text |
| `category`, `subcategory`, `entry_type` | Copied from the input |
| `model` | The generation model |
| `source` | The input's `source` |
| `sample_id` | Content id of `output_response` |
| `output_response` | The response |
| `accepted` | The checker's verdict on the last attempt |
| `rejection_reason` | The checker's reply when rejected, else empty |

## How it resumes

The ledger alone decides: inputs it records are skipped, the rest run.
Inputs added since the last run are planned and run. Rows are saved every 32
units. `resume: false` removes the CSV, its `.cut` file, the ledger and the
manifest first. See [internals/dataset.md](../internals/dataset.md) for the
lock and the cut-back.

## Rejected and failed units

- A rejected response is saved with `accepted` false and the reason, and its
  input is acked: a resume does not try it again.
- An input whose prompt is too long for the model gets a row with an empty
  `output_response`, `accepted` false and `rejection_reason`
  `prompt too long: ...`.
- Any other error inside a unit, and any failed call, stops the stage. Saved
  units stay saved; resume to go on.
