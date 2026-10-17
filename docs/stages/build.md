# Stage: build

Front `build_dataset` (`stages.py`). It has no core and no units: it copies
the rows of the other stages' files into one CSV with
`dataset.write_concat`.

## Purpose

Merge the accepted rows of everything a run generated into
`Datasets/complete_dataset.csv`, one part after another, with a
`dataset_type` column per part.

## Inputs

| Part | `dataset_type` | Read from |
|---|---|---|
| Inputs | `content_moderation_input` | `Datasets/constitution_inputs_merged.csv`; if it does not exist, the accepted rows of the category files under `Datasets/constitution_inputs/` (`merge_inputs`) |
| Jailbreaks | `jailbreak` | `Datasets/jailbreaks.csv` |
| Variations | `natural_variation` | `Datasets/variations.csv` |
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

Each row is written once by `sample_id`. A row whose `sample_id` an earlier
part wrote is left out, and the log counts them per part (`Jailbreaks: left
out N rows whose sample_id an earlier part wrote.`). The parts go in the
order of the table above. So an accepted adversarial no-op (`is_noop`
true), whose text and id are its input's, is not in the complete dataset,
and neither is a variation whose text a jailbreak row has. A row without a
`sample_id` stays. Rows of one part are not compared with each other.

## Mode

The recipe's `dataset_type` is the build's `mode`.

| Mode | Paraphrases |
|---|---|
| `training` | merged into `complete_dataset.csv` |
| `eval` | written to `Datasets/paraphrased.csv`, left out of the complete dataset |

## Files and columns

| File | Content |
|---|---|
| `Datasets/complete_dataset.csv` | The accepted rows of every part; `output_path` moves it |
| `Datasets/paraphrased.csv` | The accepted paraphrases in `eval` mode, each `sample_id` once. Written only when a paraphrase file has a row |

The columns are those of every part, in the order they first appear, plus
`dataset_type`. A row is empty in the columns its part lacks. Each part keeps
its text in its own columns:

| `dataset_type` | Text | Origin |
|---|---|---|
| `content_moderation_input` | `sample` | the row is the origin; `source_sample_description` names its constitution entry |
| `jailbreak`, `natural_variation` | `jailbreak`; the input in `prompt` | `input_id` |
| `content_moderation_output` | `output_response`; the input in `input_prompt` | `input_id` |
| `content_moderation_paraphrase` | `sample` | `input_id`; `source` says which target |

A file's cells are copied as text, in chunks of 50,000 rows, so no part file
is held in memory whole; only the fallback inputs part is read whole, and
the `sample_id`s written are held in memory. The rows go to `<target>.tmp`,
synced to disk, which then replaces the target, so the target holds its old
rows or all the new ones. With no part at all the build writes
nothing and logs `No data found`.

## Resume

There is nothing to resume: every build writes the whole file again from the
files on disk. The run record counts its rows and accepted rows, which are
the same number unless a part has no `accepted` column.
