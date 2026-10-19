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
| `outputs` | the rows of `Datasets/output_responses.csv`, or of `outputs=`, whose `accepted` reads true (`dataset.accepted_mask`); a frame without the column is taken whole | `output_response`, else `sample` | `sample_id`, the response's own id, as `response_id`; `input_id`, the input answered, is kept |

`target` is `inputs`, `outputs` or `both`. A target whose source has rows but
none accepted, or that is given an empty frame, is skipped with a warning
(`paraphrase (<target>) skipped: ...`). A target with no rows at all goes to
`run_paraphrase_target`, which takes the lock, clears the target on a fresh
run, logs `[paraphrase:<target>] no source rows; skipping.` and returns an
empty frame. Either way, with `resume: false` the target's files are removed
as a fresh run removes them.

The prompts are `prompts/paraphrase/generation/` and
`prompts/paraphrase/check/`, both through the overlay.

## Models

| Recipe role | Registry role | Default |
|---|---|---|
| `paraphraser` | `paraphraser` | `venice-paraphraser` (local vLLM) |
| `paraphrase_check` | `uncensored_gen` | `venice-uncensored` (hosted) |

`paraphraser` may name one model, or `"distribution"` for every model with
the `paraphraser` role. With several, each unit goes to one of them, picked
from a hash of `(seed, id, iteration)`, the id being the unit's: `input_id`
for the inputs, `response_id` for the outputs. The front warns when the check
model is also a paraphraser, and when `paraphrases_per_sample` is above 1
with a single paraphraser: identical paraphrases are dropped, so fewer may
remain.

## Settings

Params keys: `target`, `paraphrases_per_sample`, `check`, `batch_size`,
`seed`, `inputs_path`, `outputs_path`
([configuration.md](../configuration.md#paraphrase)).

## Unit and ledger key

One unit is one paraphrase of one source row, with `iteration` from 0 to
`paraphrases_per_sample - 1`. For the `inputs` target the row is an input,
and the unit `(input_id, iteration)`, `input_id` the input's `sample_id`
(else the content id of its text). For the `outputs` target the row is a
response, and the unit `(response_id, iteration)`, `response_id` the
response's `sample_id` (else the content id of its text): an input may have
several responses, each paraphrased on its own. The unit keeps `input_id`,
the response's own `input_id`, the input it answers.

| | `inputs` | `outputs` |
|---|---|---|
| Ledger key | `(input_id, iteration)` | `(response_id, iteration)` |
| Ledger record | the key, `paraphrase_model`, and `status`: `accepted`, `rejected` or `dropped_dedup` | the key, `input_id`, `paraphrase_model`, `status` |
| Manifest row | `input_id`, `iteration`, `paraphrase_model`, `category`, `entry_type`, `status` | `response_id`, `input_id`, `iteration`, `paraphrase_model`, `category`, `entry_type`, `status` |

Two accepted responses with one text (siblings of one input that a model
answered alike) have one `response_id`: the target plans two units under
one key, which a resume skips alike once acked, and the second paraphrase,
equal to a text already written, is dropped.

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
| `response_id` | `outputs` only: the response it rewords |
| `input_id` | The input: the one reworded, or the one the response answers |
| `iteration` | Which paraphrase of the row |
| `sample` | The paraphrase |
| `category`, `entry_type` | Copied from the source row: for a response, its input's |
| `response_mode`, `output_label`, `response_label` | `outputs` only: copied from the response, so a paraphrase carries its response's label ([outputs](outputs.md#answer-modes)). Always written, empty when the source lacks the column |
| `paraphrase_model` | The model that wrote it |
| `accepted` | The checker's verdict; `true` when `check` is off |
| `reasoning` | The checker's reply when rejected |
| `source` | `paraphrase_inputs` or `paraphrase_outputs` |

A paraphrase row carries no axis columns, by design: the spec of the
generation axes stays on the row it rewords, which its `input_id` (an
input) or `response_id` (a response) names. A response's answer-mode
columns are copied, since they are its label.

The build merges the accepted paraphrases into the complete dataset in
`training` mode; in `eval` mode it writes them into `eval_inputs.csv` and
`eval_outputs.csv`, after the rows they reword, marked `paraphrased`
([build.md](build.md)).

## How it resumes

Every attempted unit is acked, whatever its outcome, so a resume runs only
units no save recorded. A resume reads the whole artifact once for the texts
already in it. `resume: false` removes the CSV, its `.cut` and `.tmp`
files, the ledger and the manifest of each target the run names, a skipped
target too. The lock and the cut-back work as in
[internals/dataset.md](../internals/dataset.md).

A resume of the `outputs` target onto a folder written before its units were
keyed on the response stops before any call: a ledger with an ack without
`response_id` (read through a ledger keyed on `iteration`, since the stage's
own skips a record without its key), or a `paraphrases_outputs.csv` whose
header lacks `response_id`, `response_mode`, `output_label` or
`response_label`. The message names `resume: false` (with `target:
"outputs"`, which leaves the inputs' paraphrases as they are) and a new
`data_dir`. The `inputs` target of the same folder resumes as before.

## Rejected and failed units

- A paraphrase the checker rejects is saved with `accepted` false. The build
  leaves it out.
- A paraphrase that is empty, equal to its original, or equal to a text
  already in the file is dropped: no row, acked `dropped_dedup`.
- Neither is tried again on resume.
- A failed call stops the stage. A chunk that fails part way saves none of
  its units, so they run again on resume.
- The stage does not use the async lane, so a prompt too long for the
  paraphraser or the checker fails its batch call and stops the stage, on
  every backend.
