# Stage: paraphrase

Front `generate_paraphrases` (`stages.py`); core `run_paraphrase_target`
(`content_moderation/paraphrase.py`), once per target.

## Purpose

Add reworded copies of the inputs, to remove the generating model's
fingerprint from the text; as an opt-in, of the accepted outputs too
(`target: "outputs"` or `"both"`). A separate model checks that each
paraphrase keeps its original's meaning. Paraphrases are extra rows, never
replacements.

The shipped paraphraser is `humanizer-12b`, self-hosted on vLLM: Gemma 4 12B
fine-tuned to rewrite model drafts as human text, keeping every fact.
`venice-paraphraser`, the generator's own 24B weights quantized to FP8 on
load, stood in before it and stays callable by name, for comparison. The
trained defingerprinting model lives in another repository; register it with
the `paraphraser` role when you have it.

## Inputs

| Target | Source rows | Text column | Id column |
|---|---|---|---|
| `inputs` | the accepted inputs of the run, or `inputs=` | `sample`, else `prompt` | `sample_id` |
| `outputs` | the rows of `Datasets/output_responses.csv`, or of `outputs=`, whose `accepted` reads true (`dataset.accepted_mask`); a frame without the column is taken whole | `output_response`, else `sample` | `sample_id`, the response's own id, as `response_id`; `input_id`, the input answered, is kept |

`target` is `inputs` (the default), `outputs` or `both`: a call or a recipe
that names none paraphrases the inputs alone. Another value, a
`paraphrases_per_sample` that is not a whole number of at least 1, or a
`min_words` that is not one of at least 0, raises before anything runs
(`paraphrase.check_settings`); a recipe's params are checked when they load,
before the run's first stage. A target whose source has rows but
none accepted, or that is given an empty frame, is skipped with a warning
(`paraphrase (<target>) skipped: ...`). A target with no rows at all goes to
`run_paraphrase_target`, which takes the lock, clears the target on a fresh
run, logs `[paraphrase:<target>] no source rows; skipping.` and returns an
empty frame. Either way, with `resume: false` the target's files are removed
as a fresh run removes them.

The prompts are `prompts/paraphrase/generation/`, or
`prompts/paraphrase/generation/<model>/` for a paraphraser that has one (the
directory is its registry name), and `prompts/paraphrase/check/`, all
through the overlay. A model's own prompt is looked for first in the
overlay, then in the packaged tree, before the generic one: a generic
override does not reach a model that has a prompt of its own. The sample is
stripped before it is rendered.

## Models

| Recipe role | Registry role | Default |
|---|---|---|
| `paraphraser` | `paraphraser` | `humanizer-12b` (local vLLM, 12B) |
| `paraphrase_check` | `uncensored_gen` | `venice-uncensored` (hosted) |

`humanizer-12b` is the one model with the `paraphraser` role, so it is the
default and the whole `"distribution"` pool; `venice-paraphraser` holds no
role and runs only when a recipe names it. The humanizer is a
text-completion model: its prompt, `prompts/paraphrase/generation/humanizer-12b/`,
is the model card's format, the card's instruction, a blank line, the
stripped draft, then `\n\n### Rewritten:\n\n`, after `<bos>`. The entry
folds the instruction into the user message (`supports_system_prompt`
false) and its `chat_template` adds `<bos>` and refuses anything but one
user message.

A paraphrase prompt with a `fingerprint` key (`{"sample": ..., "sha256_16":
...}`) is checked at the stage's start, before any client is built: the
prompt rendered for the sample, under the model's system-prompt policy, must
hash to the key (`check_paraphrase_prompt`). A mismatch raises, naming the
file and both hashes. An override without the key is not checked, and a
WARNING says so when the packaged prompt it replaces has one; remove the key
to run another wording.

`paraphraser` may name one model, or `"distribution"` for every model with
the `paraphraser` role. Each must be registered: an unregistered one stops
the front, with a `KeyError`, before a fresh run clears anything. With several, each unit goes to one of them, picked
from a hash of `(seed, id, iteration)`, the id being the unit's: `input_id`
for the inputs, `response_id` for the outputs. The front warns when the check
model is also a paraphraser, and when `paraphrases_per_sample` is above 1
with a single paraphraser: identical paraphrases are dropped, so fewer may
remain.

## Settings

Params keys: `target`, `paraphrases_per_sample`, `check`, `min_words`,
`batch_size`, `seed`, `inputs_path`, `outputs_path`
([configuration.md](../configuration.md#paraphrase)).

## Unit and ledger key

One unit is one paraphrase of one source row, with `iteration` from 0 to
`paraphrases_per_sample - 1`. For the `inputs` target the row is an input,
and the unit `(input_id, iteration)`, `input_id` the input's `sample_id`
(else the content id of its text). For the `outputs` target the row is a
response, and the unit `(response_id, iteration)`, `response_id` the
response's `sample_id` (else the content id of its text): an input may have
several responses, each paraphrased on its own. The unit keeps `input_id`,
the response's own `input_id`, the input it answers. When the outputs
target runs, with the outputs stage's `outputs_per_input` at k (2 in a
training run), it paraphrases each of an input's k responses: its units are
`paraphrases_per_sample` per accepted response, k times those per input.

| | `inputs` | `outputs` |
|---|---|---|
| Ledger key | `(input_id, iteration)` | `(response_id, iteration)` |
| Ledger record | the key, `paraphrase_model`, and `status`: `accepted`, `rejected`, `dropped_dedup`, `prompt_too_long` or `skipped` (with `reason` and `min_words`) | the key, `input_id`, `paraphrase_model`, `status` (and for `skipped`, `reason` and `min_words`) |
| Manifest row | `input_id`, `iteration`, `paraphrase_model`, `category`, `entry_type` (each empty where the source has none, never `nan`), `status` (`planned`, or `skipped` for a text under `min_words` words) | `response_id`, `input_id`, `iteration`, `paraphrase_model`, `category`, `entry_type`, `status` |

Two accepted responses with one text (siblings of one input that a model
answered alike) have one `response_id`: the target plans two units under
one key, and both are paraphrased (each paraphrase is its own sample, so
the dedup keeps both unless they come out the same), giving two rows and
two acks under that key; a resume skips both.

The stage runs one paraphraser at a time. Its pending units go to the unit
pool (`llm_pipeline.drive_units`), each its own `llm_pipeline.checked`
generator: one paraphrase request, the gates, then, for a paraphrase no gate
rejected, one check request. The paraphrase and check requests of different
units overlap, on the async lane for an OpenAI-compatible or vLLM model.
Each paraphrase request has its own reply cap, `reply_cap(text, ceiling)`:
2.5 times the original's tokens (3 characters a token) plus 32, rounded up to
a power of two from 64 (so a grouped call meets few distinct caps), at most
the paraphraser's `default_max_tokens`. With the check on, a reply cut at its
cap usually fails the length gate. Not always: the gate counts characters, so
a cut reply in a script of few characters a token (Chinese, Japanese) can
pass it; and with `check: false` no gate runs. The check request's reply
limit is the check model's default (its temperature is `CHECK_TEMPERATURE`,
0.15). A finished unit is saved without waiting for the
others: every 32 units, after each paraphraser's last unit (before the next
paraphraser is loaded), and on any exit. When the paraphrasers cannot be resident together on the GPUs, the
stage unloads each one before the next, and keeps the check model and the
engines other stages hold.

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
| `category`, `entry_type` | Copied from the source row: for a response, its input's. Empty where the source has no such column or cell, never `nan` |
| `response_mode`, `output_label`, `response_label` | `outputs` only: copied from the response, so a paraphrase carries its response's label ([outputs](outputs.md#answer-modes)). Always written, empty when the source lacks the column |
| `paraphrase_model` | The model that wrote it |
| `accepted` | The gates' and the checker's verdict; `true` when `check` is off |
| `reasoning` | The gate's reason or the checker's reply when rejected |
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
already in it. Rows written before keep their verdicts, made with the check
of their day; `resume: false` checks every paraphrase again.
`resume: false` removes the CSV, its `.cut` and `.tmp`
files, the ledger and the manifest of each target the run names, a skipped
target too. The lock and the cut-back work as in
[internals/dataset.md](../internals/dataset.md).

A run that does not paraphrase the outputs leaves `paraphrases_outputs.csv`
as an earlier run of that target wrote it, and the build still merges it.
When the file holds rows, the front warns once (`paraphrase: <file> holds
paraphrases of responses from an earlier run, and the build merges them;
...`): delete the file, or run `target: "outputs"` with `resume: false` to
replace them.

A resume of the `outputs` target onto a folder written before its units were
keyed on the response stops before any call: a ledger with an ack without
`response_id` (read through a ledger keyed on `iteration`, since the stage's
own skips a record without its key), or a `paraphrases_outputs.csv` whose
header lacks `response_id`, `response_mode`, `output_label` or
`response_label`. The message names `resume: false` (with `target:
"outputs"`, which leaves the inputs' paraphrases as they are) and a new
`data_dir`. The `inputs` target of the same folder resumes as before.

## Skipped texts

A text under `min_words` words (12 by default, split on whitespace) gets no
paraphrase from any paraphraser. Its unit is acked with `status` `skipped`,
`reason` `short` and the `min_words` it was judged by, with no call and no
row; the manifest marks it `skipped` too, on a resume as its ack says. Below 12 words the humanizer
paraphraser read a text as a headline and wrote a post under it (on the
benign test, 0 to 50% of its rewrites of 4 to 11 words were good, against
92% at 12 to 15 words), and every input under 12 words was a fragment. In a
default training run that is about one input in five, the `fragment`
length value (0.19 with the training type's log weighting; 0.15, one in
six or seven, at the table's weights). `min_words: 0` paraphrases every
text.

The run record counts the skipped units per reason, over the targets the run
named: `counts.skipped`, as `{"short": n}`; a run that skipped none has no
`skipped` key.

The acks win on resume: a unit acked `skipped` stays skipped whatever the
current `min_words`. When the current threshold would send some of them, the
stage warns once (`[paraphrase:<target>] N units acked under an earlier
min_words stay unparaphrased on resume; run with resume: false to paraphrase
them.`). A target whose texts are all short writes acks only and logs
`N units under 12 words skipped; nothing to paraphrase.`

## The check

With `check` on, each paraphrase passes five gates before the check model
sees it. They are code, not model calls, and are tried in this order; the
first that fails rejects the paraphrase:

| Gate | Fails when | Reason saved |
|---|---|---|
| length | the paraphrase, squeezed (table rule lines, lines of only `-=_*\|:`, pipes and runs of whitespace taken out), is over 1.5 times the squeezed original plus 20 characters, or under 0.5 times it | `length ratio <r>` |
| split | the original is one line of at most 35 words and the paraphrase has two or more non-empty lines | `split into lines` |
| script | 10% or more of the paraphrase's letters are in scripts that make up under 1% of the original's | `script changed: <script> -> <script>`: the original's main script, then the main one of those it lacks (`Latin -> Han`) |
| language | GlotLID labels the two texts differently, both with a probability of 0.9 or above | `language <label> -> <label>` |
| artifact | the paraphrase holds an HTML tag (`p`, `br`, `div`, `span`, `b`, `i`, `ul`, `li`, `h1` to `h6`) or a rule line (`====`, `------`, `______`) and the original holds none | `artifact <match>` |

A gated paraphrase is saved as a rejected row, its `reasoning` the gate's
reason, and is not sent to the check model. The rest go to the check model,
one request a unit, at temperature 0.15
(`llm_pipeline.CHECK_TEMPERATURE`, as the inputs and outputs checks), with the
strict prompt `prompts/paraphrase/check/`. It judges content, not wording: it
lists the rewordings that are fine, and answers No to an answer in place of a
rewrite, added or missing content, repetition, added profanity, another
language, extra text (a preamble, a tag, a note about the task) and a change
of meaning or severity. Its verdict is the reply's first word.

The thresholds were measured on 1,638 benign rewrites by two paraphrasers,
set so that no rewrite read as good fails a gate; together with the strict
prompt they caught 85% of the failures read, and rejected 0.6% of the
1,389 rewrites not labelled a failure. Both were measured on benign text
only.

`check: false` runs neither the gates nor the check model: every paraphrase
is accepted. Without the `langid` extra, or when GlotLID cannot be loaded,
the language gate passes every text and the front warns once (the script
gate still runs).

A paraphrase that loses its Markdown formatting (headings, code fences,
lists, tables) but keeps the information is accepted: the gates squeeze the
layout out before they compare lengths, and the check judges content.

## Rejected and failed units

- A paraphrase a gate rejects is saved with `accepted` false and the gate's
  reason as `reasoning`, without a check call.
- A paraphrase the checker rejects is saved with `accepted` false. The build
  leaves it out. The check reply is read as in the inputs stage
  ([inputs](inputs.md#rejected-and-failed-units)): its first word, `Yes` or
  `No`, is the verdict.
- A paraphrase that is empty, equal to its original, or equal to a text
  already in the file is dropped: no row, acked `dropped_dedup`. With the
  check on, the unit finds it before its check call, and makes none. Of two
  units with one paraphrase whose checks ran at once, the one that finishes
  first keeps its row.
- On the async lane (an OpenAI-compatible or vLLM model), a prompt too long
  for the paraphraser or the checker acks its unit `prompt_too_long`, with
  no row and a WARNING; the other units go on.
- None of these is tried again on resume.
- A failed call stops the stage, and so does a too-long prompt on a grouped
  call (an Anthropic or introspection model), since which item was too
  long is not known. The units that finished before it are saved; the
  others run again on resume.
