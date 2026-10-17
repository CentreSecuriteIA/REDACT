# Stage: build

Front `build_dataset` (`stages.py`). It has no core and no units: it copies
the rows of the other stages' files into the dataset of the run's type with
`dataset.write_concat`.

## Purpose

A training build merges the accepted rows of everything a run generated into
`Datasets/complete_dataset.csv`, one part after another, with a
`dataset_type` column per part. An eval build writes the same rows, less the
variations, into three files of one flat schema, each with one `text` and
one `label` column ([eval mode](#eval-mode)).

## Inputs

| Part | `dataset_type` | Read from |
|---|---|---|
| Inputs | `content_moderation_input` | `Datasets/constitution_inputs_merged.csv`; if it does not exist, the accepted rows of the category files under `Datasets/constitution_inputs/` (`merge_inputs`) |
| Jailbreaks | `jailbreak` | `Datasets/jailbreaks.csv` |
| Variations | `natural_variation` | `Datasets/variations.csv`; training only |
| Outputs | `content_moderation_output` | `Datasets/output_responses.csv` |
| Paraphrases | `content_moderation_paraphrase` | `Datasets/paraphrases_inputs.csv`, `Datasets/paraphrases_outputs.csv` |

A part with no file, or no row, is left out, and with `verbose` the log
says so: `Inputs: none found` (no merged file and no category file),
`Jailbreaks: none found` or `Jailbreaks: file not found (<path>)`,
`Variations: none found` or `Variations: file not found (<path>)`,
`Outputs: none found` (no file), `Paraphrases: none found` (neither file has
a row). An outputs file or a merged inputs file without a row is a part of 0
rows. The `include_*` keys leave a part out on purpose, and the `*_path`
keys read another file ([configuration.md](../configuration.md#build)).

Every part adds only its rows whose `accepted` reads true, in any case
(`dataset.accepted_mask`); an empty cell, `yes` or `1` is not accepted. A
rejected row, and a jailbreak or variation error row, keeps its
input's `entry_type`, so in the dataset it would carry the wrong label; it
stays in its stage file, where `rejection_reason` or `reasoning` says why. A
file without an `accepted` column is copied whole.

Each row is written once by `sample_id` in a file. A row whose `sample_id`
an earlier part of the file wrote is left out, and the log counts them per
part (`Jailbreaks: left out N rows whose sample_id an earlier part
wrote.`). A row without a `sample_id` stays. Rows of one part are not
compared with each other.

## Mode

The recipe's `dataset_type` is the build's `mode`.

| Mode | Writes | Paraphrases | Variations |
|---|---|---|---|
| `training` | `Datasets/complete_dataset.csv` | merged in | merged in |
| `eval` | `Datasets/eval_inputs.csv`, `eval_outputs.csv`, `eval_jailbreaks.csv` | rows of the input and output files, marked `paraphrased` | left out |

## Training mode

`complete_dataset.csv` holds the parts in the order of the table above;
`output_path` moves it. So an accepted adversarial no-op (`is_noop` true),
whose text and id are its input's, is not in the complete dataset, and
neither is a variation whose text a jailbreak row has.

The columns are those of every part, in the order they first appear, plus
`dataset_type`. A row is empty in the columns its part lacks. Each part keeps
its text in its own columns:

| `dataset_type` | Text | Origin |
|---|---|---|
| `content_moderation_input` | `sample` | the row is the origin; `source_sample_description` names its constitution entry |
| `jailbreak`, `natural_variation` | `jailbreak`; the input in `prompt` | `input_id` |
| `content_moderation_output` | `output_response`; the input in `input_prompt` | `input_id` |
| `content_moderation_paraphrase` | `sample` | `input_id`; `source` says which target |

## Eval mode

An eval build writes no `complete_dataset.csv` (an `output_path` raises)
and no separate paraphrase file. It writes three files, each with a fixed
set of columns, whatever columns the stage files have:

| File | Rows, in order | Read by |
|---|---|---|
| `eval_inputs.csv` | the accepted inputs, then the accepted paraphrases of inputs | content classifiers on the input side; jailbreak detectors, as the bare set they must stay silent on |
| `eval_outputs.csv` | the accepted responses, then the accepted paraphrases of responses | content classifiers on the output side |
| `eval_jailbreaks.csv` | the accepted attacks whose text is not their input's (`is_noop` false), then the accepted inputs of entry type `benign`, bare | jailbreak detectors (`is_jailbreak`); content classifiers under attack (`label`) |

`eval_inputs.csv`:

| Column | Value |
|---|---|
| `sample_id` | the row's id, the content id of `text` |
| `text` | the input, or its paraphrase |
| `label` | the row's entry type; a paraphrase's is its input's |
| `paraphrased` | `False` for an input, `True` for a paraphrase |
| `input_id` | for a paraphrase, the input it rewords; empty for an input |
| `paraphrase_model` | the paraphraser, for a paraphrase |
| `entry_type`, `category`, `subcategory`, `source_taxonomy` | the input's |
| `dataset_type` | `content_moderation_input` or `content_moderation_paraphrase` |

`eval_outputs.csv`:

| Column | Value |
|---|---|
| `sample_id`, `text` | the response or its paraphrase, and its id |
| `label` | the entry type of the input answered: an accepted response to a harmful input complied (the checker rejects refusals) |
| `paraphrased` | `False` for a response, `True` for a paraphrase |
| `input_id` | the input answered, on every row |
| `model` | the responding model; empty for a paraphrase |
| `paraphrase_model` | the paraphraser, for a paraphrase |
| `entry_type`, `category`, `subcategory`, `source_taxonomy` | the input's |
| `dataset_type` | `content_moderation_output` or `content_moderation_paraphrase` |

`eval_jailbreaks.csv`:

| Column | Value |
|---|---|
| `sample_id`, `text` | the attack or the bare benign input, and its id |
| `label` | the entry type of the content: `harmful` for an attack on a harmful input, `benign` for a negative |
| `is_jailbreak` | `True` for an attack, `False` for a bare input |
| `technique` | the combination, `+`-joined; empty for a negative |
| `complexity`, `num_techniques`, `iteration` | the attack's; empty for a negative |
| `input_id` | for an attack, the input attacked; empty for a negative |
| `entry_type`, `category`, `subcategory`, `source_taxonomy` | the input's |
| `dataset_type` | `jailbreak` or `content_moderation_input` |

`label` and `is_jailbreak` are independent: an attack on a benign input,
which the eval run's jailbreak kind does not make by default, is a `benign`
row with `is_jailbreak` `True`. A row's `subcategory` and `source_taxonomy`
are its own when its stage file has the column, else those of the input its
`input_id` names; `source_taxonomy` is empty when the inputs have none. The
provenance columns (`technique_info`, `reasoning`, `rejection_reason`, the
spec version) stay in the stage files, where a row is found by `sample_id`.
The input's text is not repeated on derived rows: `input_id` finds it in
`eval_inputs.csv`.

`include_inputs` false leaves out both the inputs of `eval_inputs.csv` and
the negatives of `eval_jailbreaks.csv`. When the build finds no part at all
it writes nothing; otherwise each file is written, with its header when it
has no row.

## Files

A file's cells are copied as text, in chunks of 50,000 rows, so no part file
is held in memory whole; only the fallback inputs part is read whole, and
the `sample_id`s written are held in memory (an eval build also holds the
inputs' `subcategory` and `source_taxonomy` by id). The rows go to
`<target>.tmp`, synced to disk, which then replaces the target, so the
target holds its old rows or all the new ones. With no part at all the
build writes nothing and logs `No data found`.

The front returns the rows of the complete dataset, or in eval mode
`{"inputs": ..., "outputs": ..., "jailbreaks": ...}`; with `read_back=False`
empty frames.

## Resume

There is nothing to resume: every build writes its files again from the
files on disk. The run record counts the rows of the files of the run's
dataset type, and the accepted rows of a training build, which are the same
number unless a part has no `accepted` column.
