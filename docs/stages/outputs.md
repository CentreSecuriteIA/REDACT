# Stage: outputs

Front `generate_outputs` (`stages.py`); core `run_output_generation`
(`content_moderation/outputs.py`).

## Purpose

Generate `outputs_per_input` model responses per accepted input (1 by
default, 2 in a `training` run) and check each. Each response is dealt an
answer mode, `harmful` or `helpful`, from those its input's entry type
allows (see [Answer modes](#answer-modes)), then a spec of the response axes
(length, format, tone); an input's responses differ on the axes, and from
two per input a borderline input gets one answer of each mode. The mode
picks the generation prompt and the checker, and gives the response its
label: the input's entry type and the mode's output label, as
`dual_use_benign_harmful`.

## Inputs

- The accepted inputs of the run (`dataset.merge_inputs(data_dir)`), or a
  frame passed as `inputs=`. The text is the `sample` column, else `prompt`.
  Each row's `category`, `subcategory` and `entry_type` are read when
  present (default `unknown`, empty, `harmful`). A frame without an
  `entry_type` column is answered as `harmful` inputs, with a WARNING. A
  missing, NaN or empty cell in the column raises before anything is
  cleared, naming the count and the first ids: the entry type picks the
  modes and labels the responses, so it is never guessed. So does an entry
  type that `outputs.modes` does not map, naming it and `outputs.modes`.
- With no input file at all the front raises `No input samples found`. With
  files but no accepted row, or an empty `inputs=` frame, it warns
  `outputs skipped: no input sample was accepted, ...`, writes nothing and
  returns an empty frame. With `resume: false` it first removes what earlier
  runs left, as a fresh run does: the CSV, its `.cut` and `.tmp` files, the
  ledger, the manifest, the plan file and the coverage table
  (`dataset.clear_stage`).
- `max_per_category` keeps the first N rows of each `(category, entry_type)`,
  then `max_samples` the first N overall. Both count inputs, before each is
  expanded to `outputs_per_input` units: N inputs per group give N × k
  responses.
- The prompts of every mode the inputs' entry types allow:
  `prompts/output/generation/<mode>/` and
  `prompts/output/quality_check/<mode>/`, each through the overlay. Before
  anything is cleared each is read and rendered once with empty values, and
  the checker of each category and mode is built (and kept for the units),
  so an overlay that does not render, such as one with an unknown
  placeholder, stops the run there and not at its first unit. The generation template's `{response_spec}`
  shows the row's spec; a prompt without it is warned about, naming its
  folder. The checker is not shown the spec: its messages are those of a
  run without axes. An overlay written before the modes, a prompt file
  directly in `<prompt_dir>/output/generation/` or
  `<prompt_dir>/output/quality_check/`, overrides nothing:
  `report_prompt_sources` warns `moved override: output/generation` (or
  `output/quality_check`) and names its new place. Move it to
  `generation/harmful/` (or `quality_check/harmful/`), and write a
  `helpful/` one if wanted.
- The `outputs` section of `generation_axes.json`, a recipe's `config_dir`'s
  when it has one ([configuration.md](../configuration.md#the-generation-axes)).

## Models

| Recipe role | Registry role | Default | Calls |
|---|---|---|---|
| `gen` | `uncensored_gen` | `venice-uncensored` | One response per attempt |
| `check` | `uncensored_gen` | the `gen` model | At most one check per attempt: none for a reply the length floor rejects |

`check_outputs: false` skips the check and accepts every response.

The check calls run at temperature 0.15 (`llm_pipeline.CHECK_TEMPERATURE`),
the generations at the model's default: in the checker test a checker at
0.15 repeated its own verdict on 99.5 percent of harmful responses, against
94.0 at the stages' 0.7, at the same acceptance.

## Settings

Params keys: `check_outputs`, `max_attempts`, `max_per_category`,
`max_samples`, `outputs_per_input`, `modes`, `axes`, `axes_weighting`,
`seed`, `batch_size`, `check_max_tokens`, `output_path`
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

Each input row is `outputs_per_input` units `(input_id, iteration)`,
`iteration` from 0 to k - 1, each a row. A unit generates, checks, and with
`max_attempts` above 1 generates again with the checker's verdict as
feedback until accepted or out of attempts. The row holds the last attempt.

Each unit is dealt a mode, then a spec, both per `(category, entry_type)`
in `(input_id, iteration)` order from one dealer each, so the deal does not
depend on the input order. The mode is the allowed one with room and the
lowest count in the group so far, ties broken by the unit's rng, seeded by
`(seed, category, entry_type, input_id, iteration, "mode")`; an input's k
units take a mode at most ceil(k / modes) times. So an entry type with one
mode gives every unit that mode; for a dual-use input, at k = 2 one unit of
each, which one first being its own rng's draw, at k = 3 two and one, and at
k = 1 the group alternates, half each. The spec is dealt as the inputs
stage deals its units ([inputs](inputs.md#the-generation-axes)): each
unit's rng seeded by `(seed, category, entry_type, input_id, iteration)`, an
input's k units taking a value at most ceil(share × k) times, so they differ
on every axis none of whose values weighs over 1/k (at k = 2 every packaged
value: the largest is `neutral` at .40), and never a value beside one it
names in `not_with` (`steps` and a `brief` answer, for example).

The deal is made again every run. At an even k each input takes each of two
modes k / 2 times, so the group's counts are level at every input's first
unit and an input's modes depend on its own key alone: inputs that join the
group leave them as they were. At k = 1 and an odd k the group alternates,
so inputs that join a dual-use group would deal the units before them
again. A done unit keeps the mode its ack records instead: the deal counts
it as dealt and deals the other units around it, so the manifest and the
coverage table agree with the rows, and the new inputs balance the modes
the rows have. An ack written without a mode (on this branch before the
fix) leaves its unit to the deal. The specs of done units are dealt again
as before; the rows keep theirs.

| | Value |
|---|---|
| Ledger key | `(input_id, iteration)`: the input's `sample_id`, else the content id of its text, and which of its responses |
| Ledger record | `{"input_id": ..., "iteration": ..., "mode": ..., "accepted": ...}`: the mode the row was written with, and the verdict on the last attempt |
| Manifest row | `input_id`, `iteration`, `category`, `entry_type`, `mode`, `spec` (the dealt value of each axis; none without axes), `status` (`planned`), one per unit |

## Files

| File | Content |
|---|---|
| `Datasets/output_responses.csv` | One row per unit, `outputs_per_input` per input; `output_path` moves it |
| `Datasets/output_responses.manifest.jsonl` | The plan, written again every run |
| `Datasets/output_responses.state.jsonl` | The ledger |
| `Datasets/output_responses.plan.json` | What the deal was made with: `seed`, `outputs_per_input`, `modes` (the whole mapping, every entry type) and each axis's weights (`null` without axes) |
| `Datasets/output_responses.coverage.csv` | Per axis value and group: `axis`, `value`, `category`, `entry_type`, `planned`, `accepted`, as the inputs stage's, then a row `axis` `mode` per mode a group was dealt. Written when the stage ends, also without axes (the `mode` rows alone) |

| Column | Meaning |
|---|---|
| `input_id` | The input's id |
| `input_prompt` | The input text |
| `category`, `subcategory`, `entry_type` | Copied from the input; `entry_type` is the input's, not the response's |
| `model` | The generation model |
| `source` | The input's `source` |
| `sample_id` | Content id of `output_response`; for a prompt too long, of `{input_id}:{iteration}:prompt too long` |
| `output_response` | The response |
| `accepted` | The checker's verdict on the last attempt |
| `rejection_reason` | The checker's reply when rejected, else empty |
| `iteration` | Which of the input's responses the row is, 0 to `outputs_per_input` - 1 |
| `response_mode` | `harmful` or `helpful`: the prompt and checker the row was made with |
| `output_label` | The mode's label: `harmful` or `benign` |
| `response_label` | `<entry_type>_<output_label>`, as `dual_use_benign_harmful`. The parts hold underscores: split it on its last one only, or read `entry_type` and `output_label` |
| `response_length`, `response_format`, `response_register` | The row's dealt value of each response axis. Only with axes |

Two responses of one input can come back with one text, hence one
`sample_id` twice in the file; the stage does not drop the repeat. The
build compares a row only with the rows of earlier parts, so both reach the
dataset, each with its own `response_label`; the siblings' different specs
make it improbable.

## How it resumes

The ledger alone decides: units it records are skipped, the rest run.
Inputs added since the last run are planned, dealt into their groups and
run; the rows done keep the modes and specs they record. Rows are saved
every 32 units. `resume: false` removes the CSV, its `.cut` and `.tmp`
files, the ledger and the manifest first, also when the stage is skipped
(the plan file and the coverage table too). See
[internals/dataset.md](../internals/dataset.md) for the lock and the
cut-back.

Before any call, a resume onto a folder of an earlier shape stops, naming
`resume: false` and a new `data_dir`: a ledger with an ack without
`iteration` (written when a unit was an input; read through a ledger keyed
on `input_id`, since the stage's own skips a record without its key and
would run every unit again onto the rows), or a file without `iteration`,
`response_mode`, `output_label` and `response_label` (written before an
input had several responses, each in an answer mode). A file without the
response columns (written before the axes, or without them) stops it too,
naming `outputs.axes: false` besides. These checks come before the plan
file's, whose keys an old folder lacks. A resume then compares the run's
`seed`, `outputs_per_input`, `modes` and axes' weights with
`output_responses.plan.json`, and raises `Outputs: this run's [...] differ
from those the responses in ... were dealt with`, naming the keys and
`resume=False`.

## Rejected and failed units

- A rejected response is saved with `accepted` false and the reason, and its
  unit is acked: a resume does not try it again.
- A response with almost no text is rejected before its check, with no check
  call: one with fewer than 20 characters (`MIN_RESPONSE_CHARS`) once its
  code-fence marker lines (a line that is three backticks, with an optional
  language) are removed and its ends stripped. So `Here is a response:` (19)
  and an empty code block are rejected, and a fenced code answer keeps its
  content. Its `rejection_reason` is `too short: the response has almost no
  content.`; with `max_attempts` above 1 it is generated again with that
  reason as feedback. With `check_outputs: false` there is no floor: every
  response is accepted.
- On the async lane (an OpenAI-compatible endpoint or vLLM), an input whose
  generation or check prompt is too long for its model gets a row with an
  empty `output_response`, `accepted` false and `rejection_reason`
  `prompt too long: ...` for each of its units, which are acked. On an Anthropic or introspection
  binding the requests go out in grouped calls, and a prompt too long there
  fails the call and stops the stage.
- Any other error inside a unit, and any failed call, stops the stage. Saved
  units stay saved; resume to go on.
