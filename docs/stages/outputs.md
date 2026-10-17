# Stage: outputs

Front `generate_outputs` (`stages.py`); core `run_output_generation`
(`content_moderation/outputs.py`).

## Purpose

Generate one model response per accepted input, written to a spec of the
response axes (length, format, tone), and check it with a checker that knows
the input's entry type: a refusal of a harmful input is rejected; a response
to a benign input is judged as any answer.

## Inputs

- The accepted inputs of the run (`dataset.merge_inputs(data_dir)`), or a
  frame passed as `inputs=`. The text is the `sample` column, else `prompt`.
  Each row's `category`, `subcategory` and `entry_type` are read when
  present (default `unknown`, empty, `harmful`).
- With no input file at all the front raises `No input samples found`. With
  files but no accepted row, or an empty `inputs=` frame, it warns
  `outputs skipped: no input sample was accepted, ...`, writes nothing and
  returns an empty frame. With `resume: false` it first removes what earlier
  runs left, as a fresh run does: the CSV, its `.cut` and `.tmp` files, the
  ledger, the manifest, the plan file and the coverage table
  (`dataset.clear_stage`).
- `max_per_category` keeps the first N rows of each `(category, entry_type)`,
  then `max_samples` the first N overall.
- The prompts `prompts/output/generation/` and `prompts/output/quality_check/`,
  both through the overlay. The generation template's `{response_spec}`
  shows the row's spec; a prompt without it is warned about. The checker is
  not shown the spec: its messages are those of a run without axes.
- The `outputs` section of `generation_axes.json`, a recipe's `config_dir`'s
  when it has one ([configuration.md](../configuration.md#the-generation-axes)).

## Models

| Recipe role | Registry role | Default | Calls |
|---|---|---|---|
| `gen` | `uncensored_gen` | `venice-uncensored` | One response per attempt |
| `check` | `uncensored_gen` | the `gen` model | One check per attempt |

`check_outputs: false` skips the check and accepts every response.

## Settings

Params keys: `check_outputs`, `max_attempts`, `max_per_category`,
`max_samples`, `axes`, `seed`, `batch_size`, `check_max_tokens`,
`output_path` ([configuration.md](../configuration.md#outputs)).

## Unit and ledger key

One unit is one input row: generate, check, and with `max_attempts` above 1,
generate again with the checker's verdict as feedback until accepted or out
of attempts. The row holds the last attempt.

Each row is dealt one spec, as the inputs stage deals its units
([inputs](inputs.md#the-generation-axes)): per `(category, entry_type)`, in
`input_id` order from one dealer, each row's rng seeded by `(seed, category,
entry_type, input_id, 0)`, and never a value beside one it names in
`not_with` (`steps` and a `brief` answer, for example). The deal is made
again every run.

| | Value |
|---|---|
| Ledger key | `input_id`: the input's `sample_id`, else the content id of its text |
| Ledger record | `{"input_id": ..., "accepted": ...}`, the verdict on the last attempt |
| Manifest row | `input_id`, `category`, `entry_type`, `spec` (the dealt value of each axis; none without axes), `status` (`planned`), one per input |

## Files

| File | Content |
|---|---|
| `Datasets/output_responses.csv` | One row per input; `output_path` moves it |
| `Datasets/output_responses.manifest.jsonl` | The plan, written again every run |
| `Datasets/output_responses.state.jsonl` | The ledger |
| `Datasets/output_responses.plan.json` | What the deal was made with: `seed` and each axis's weights (`null` without axes) |
| `Datasets/output_responses.coverage.csv` | Per axis value and group: `axis`, `value`, `category`, `entry_type`, `planned`, `accepted`, as the inputs stage's. Written when the stage ends; removed without axes |

| Column | Meaning |
|---|---|
| `input_id` | The input's id |
| `input_prompt` | The input text |
| `category`, `subcategory`, `entry_type` | Copied from the input |
| `model` | The generation model |
| `source` | The input's `source` |
| `sample_id` | Content id of `output_response`; for a prompt too long, of `{input_id}:prompt too long` |
| `output_response` | The response |
| `accepted` | The checker's verdict on the last attempt |
| `rejection_reason` | The checker's reply when rejected, else empty |
| `response_length`, `response_format`, `response_register` | The row's dealt value of each response axis. Only with axes |

## How it resumes

The ledger alone decides: inputs it records are skipped, the rest run.
Inputs added since the last run are planned, dealt into their groups and
run; the rows done keep the specs they record. Rows are saved every 32
units. `resume: false` removes the CSV, its `.cut` and `.tmp` files, the
ledger and the manifest first, also when the stage is skipped (the plan file
and the coverage table too). See
[internals/dataset.md](../internals/dataset.md) for the lock and the
cut-back.

A resume compares the run's `seed` and axes' weights with
`output_responses.plan.json`, and raises `Outputs: this run's [...] differ
from those the responses in ... were dealt with`, naming the keys and
`resume=False`. A file without the response columns (written before the
axes, or without them) stops a resume before any call, naming the remedies:
`resume: false`, a new `data_dir`, or `outputs.axes: false`.

## Rejected and failed units

- A rejected response is saved with `accepted` false and the reason, and its
  input is acked: a resume does not try it again.
- On the async lane (an OpenAI-compatible endpoint or vLLM), an input whose
  generation or check prompt is too long for its model gets a row with an
  empty `output_response`, `accepted` false and `rejection_reason`
  `prompt too long: ...`, and is acked. On an Anthropic or introspection
  binding the requests go out in grouped calls, and a prompt too long there
  fails the call and stops the stage.
- Any other error inside a unit, and any failed call, stops the stage. Saved
  units stay saved; resume to go on.
