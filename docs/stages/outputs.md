# Stage: outputs

Front `generate_outputs` (`stages.py`); core `run_output_generation`
(`content_moderation/outputs.py`).

## Purpose

Generate one model response per accepted input and check it. Each response
is dealt an answer mode, `harmful` or `helpful`, from those its input's
entry type allows (see [Answer modes](#answer-modes)), then a spec of the
response axes (length, format, tone). The mode picks the generation prompt
and the checker, and gives the response its label: the input's entry type
and the mode's output label, as `dual_use_benign_harmful`.

## Inputs

- The accepted inputs of the run (`dataset.merge_inputs(data_dir)`), or a
  frame passed as `inputs=`. The text is the `sample` column, else `prompt`.
  Each row's `category`, `subcategory` and `entry_type` are read when
  present (default `unknown`, empty, `harmful`; a missing, NaN or empty
  `entry_type` cell is `harmful` too). An entry type that `outputs.modes`
  does not map raises before anything is cleared, naming it and
  `outputs.modes`.
- With no input file at all the front raises `No input samples found`. With
  files but no accepted row, or an empty `inputs=` frame, it warns
  `outputs skipped: no input sample was accepted, ...`, writes nothing and
  returns an empty frame. With `resume: false` it first removes what earlier
  runs left, as a fresh run does: the CSV, its `.cut` and `.tmp` files, the
  ledger, the manifest, the plan file and the coverage table
  (`dataset.clear_stage`).
- `max_per_category` keeps the first N rows of each `(category, entry_type)`,
  then `max_samples` the first N overall.
- The prompts of each mode dealt: `prompts/output/generation/<mode>/` and
  `prompts/output/quality_check/<mode>/`, each through the overlay, read
  before anything is cleared. The generation template's `{response_spec}`
  shows the row's spec; a prompt without it is warned about, naming its
  folder. The checker is not shown the spec: its messages are those of a
  run without axes. An overlay written before the modes,
  `<prompt_dir>/output/generation/template.json`, overrides nothing, and
  `report_prompt_sources` lists it as an unused override: copy it to
  `generation/harmful/`.
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
`max_samples`, `modes`, `axes`, `axes_weighting`, `seed`, `batch_size`,
`check_max_tokens`, `output_path`
([configuration.md](../configuration.md#outputs)).

## Answer modes

The mode says how a response answers its input:

| Mode | Generation prompt | Checker | Output label |
|---|---|---|---|
| `harmful` | `output/generation/harmful/` | `output/quality_check/harmful/` | `harmful` |
| `helpful` | `output/generation/helpful/` | `output/quality_check/helpful/` | `benign` |

The modes each entry type allows are `outputs.modes`, laid over this default
entry type by entry type and checked when the params load:

| Entry type | Modes | Labels |
|---|---|---|
| `harmful` | `harmful` | `harmful_harmful` |
| `dual_use_harmful` | `harmful`, `helpful` | `dual_use_harmful_harmful`, `dual_use_harmful_benign` |
| `dual_use_benign` | `harmful`, `helpful` | `dual_use_benign_harmful`, `dual_use_benign_benign` |
| `benign` | `helpful` | `benign_benign` |

A custom mapping, as `{"benign": ["harmful"]}`, gives the other two labels
(`benign_harmful`, `harmful_benign`). The label names what was asked and
how it was answered, and judges nothing.

Each checker keeps the verdict wording, the reply shapes and criteria 2 to
4 of the single checker it replaces, and differs in its semantics line and
criterion 1: the harmful checker holds the harmful entry type's line of
that checker, the helpful one asks for a normal, helpful, non-refusal
answer to the request as it was asked, not a safer reinterpretation, with
no moralising or unrequested disclaimers. A refusal is rejected in both. The
checker is shown the category, not the input's entry type.

The mode is not a generation axis: it picks the prompt, the checker and the
label rather than adding a line to the spec, it holds with `axes: false`,
and no augmentation family varies it, so `check_dimensions` has nothing to
say about it.

## Unit and ledger key

One unit is one input row: generate, check, and with `max_attempts` above 1,
generate again with the checker's verdict as feedback until accepted or out
of attempts. The row holds the last attempt.

Each row is dealt a mode, then a spec, both per `(category, entry_type)` in
`input_id` order from one dealer each, so the deal does not depend on the
input order. The mode is the allowed one with the lowest count in the group
so far, ties broken by the row's rng, seeded by `(seed, category,
entry_type, input_id, 0, "mode")`: an entry type with one mode gives every row that
mode, and a dual-use group alternates, half each. The spec is dealt as the
inputs stage deals its units ([inputs](inputs.md#the-generation-axes)): each
row's rng seeded by `(seed, category, entry_type, input_id, 0)`, never a
value beside one it names in `not_with` (`steps` and a `brief` answer, for
example). The deal is made again every run; inputs that join a dual-use
group can change the mode the manifest plans for a done input, whose row
keeps the mode it was written with.

| | Value |
|---|---|
| Ledger key | `input_id`: the input's `sample_id`, else the content id of its text |
| Ledger record | `{"input_id": ..., "accepted": ...}`, the verdict on the last attempt |
| Manifest row | `input_id`, `category`, `entry_type`, `mode`, `spec` (the dealt value of each axis; none without axes), `status` (`planned`), one per input |

## Files

| File | Content |
|---|---|
| `Datasets/output_responses.csv` | One row per input; `output_path` moves it |
| `Datasets/output_responses.manifest.jsonl` | The plan, written again every run |
| `Datasets/output_responses.state.jsonl` | The ledger |
| `Datasets/output_responses.plan.json` | What the deal was made with: `seed`, `modes` (the whole mapping, every entry type) and each axis's weights (`null` without axes) |
| `Datasets/output_responses.coverage.csv` | Per axis value and group: `axis`, `value`, `category`, `entry_type`, `planned`, `accepted`, as the inputs stage's, then a row `axis` `mode` per mode a group was dealt. Written when the stage ends, also without axes (the `mode` rows alone) |

| Column | Meaning |
|---|---|
| `input_id` | The input's id |
| `input_prompt` | The input text |
| `category`, `subcategory`, `entry_type` | Copied from the input; `entry_type` is the input's, not the response's |
| `model` | The generation model |
| `source` | The input's `source` |
| `sample_id` | Content id of `output_response`; for a prompt too long, of `{input_id}:prompt too long` |
| `output_response` | The response |
| `accepted` | The checker's verdict on the last attempt |
| `rejection_reason` | The checker's reply when rejected, else empty |
| `response_mode` | `harmful` or `helpful`: the prompt and checker the row was made with |
| `output_label` | The mode's label: `harmful` or `benign` |
| `response_label` | `<entry_type>_<output_label>`, as `dual_use_benign_harmful`. The parts hold underscores: split it on its last one only, or read `entry_type` and `output_label` |
| `response_length`, `response_format`, `response_register` | The row's dealt value of each response axis. Only with axes |

## How it resumes

The ledger alone decides: inputs it records are skipped, the rest run.
Inputs added since the last run are planned, dealt into their groups and
run; the rows done keep the modes and specs they record. Rows are saved
every 32 units. `resume: false` removes the CSV, its `.cut` and `.tmp`
files, the ledger and the manifest first, also when the stage is skipped
(the plan file and the coverage table too). See
[internals/dataset.md](../internals/dataset.md) for the lock and the
cut-back.

Before any call, a resume onto a file of an earlier shape stops: one
without `response_mode`, `output_label` and `response_label` (written
before the answer modes), naming `resume: false` and a new `data_dir`; one
without the response columns (written before the axes, or without them),
naming those and `outputs.axes: false` too. This check comes before the
plan file's, whose keys an old folder lacks. A resume then compares the
run's `seed`, `modes` and axes' weights with `output_responses.plan.json`,
and raises `Outputs: this run's [...] differ from those the responses in
... were dealt with`, naming the keys and `resume=False`.

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
