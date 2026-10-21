# dataset/: rows on disk, the ledger, and resume

`dataset/` is everything a stage does with files: the identity of a row,
appending and reading CSV rows, the two sidecar files, saving finished units,
resuming, a fresh run of part of a stage, and the per-category tree of input
samples.

## Responsibilities and boundaries

It is responsible for:

- what a row's id is (`identity.py`);
- the rows of one CSV, and one CSV made from several (`rows.py`);
- the ledger and the manifest (`sidecar.py`);
- a save: rows first, acks after (`save.py`);
- what a run skips, and what a resume puts back (`resume.py`);
- a fresh run of part of a stage (`drop.py`);
- the per-category tree of inputs (`categories.py`);
- what an eval build leaves out because another run holds it
  (`exclusion.py`: `exclusion_files`, `Exclusion`);
- balanced dealing and the plan file beside a manifest (`dealing.py`,
  `planfile.py`);
- writing and loading the taxonomy files (`taxonomy.py`) and published datasets
  (`loading.py`).

It never sees a prompt, a client or a model; it imports `redact.paths`,
pandas and `filelock`, and `datasets` inside `load_hf_dataset` (the `hf`
extra). `resume.py` holds the policy (what is skipped, what
is cut back); `rows.py` holds the mechanism, so a change to how rows are
stored stays out of the resume logic.

## Identity

| Function | Does |
|---|---|
| `content_id(text)` | The MD5 of the text, cut to 16 hex characters |
| `row_id(row, text_col, id_col="sample_id")` | The id a row carries in `id_col`, else the content id of its text. `None`, NaN, empty or `"nan"` count as missing |
| `row_ids(frame, text_col, id_col="sample_id")` | `row_id` of every row, reading two columns only |
| `normalised_id(text)` | The content id of the text in lower case, without punctuation, each run of whitespace one space |

Every row a stage writes has a `sample_id`: the content id of its own text,
or of its unit key where the row has no text of its own. A conversation uses
`<input_id>:<iteration>`; a jailbreak or variation error row uses
`<label>:<input_id>:<iteration>`, where `label` is `jailbreak` or
`variation`; an outputs row of a prompt too long
`<input_id>:<iteration>:prompt too long`. A row derived from another has an `input_id`, that row's id.
A paraphrase of a response keeps the input answered as `input_id` and names
the response in `response_id`, the response's `sample_id`.

## Rows

| Function | Does |
|---|---|
| `append_rows(path, rows)` | Appends in one write, synced to disk, in the column order of the file's header (a file without a header takes the rows' order and gets the header). Raises, writing nothing, on a column the header lacks. No rows write nothing and create no file. Returns `{path: size}` |
| `check_columns(path, rows)` | Raises as `append_rows` would, without writing |
| `read_rows(path)` | The whole file. `sample_id` and `input_id` are read as text. A missing cell of a text column, or of a column with no value at all, reads as `""`; numeric columns keep their gaps |
| `iter_rows(path, columns=None)` | The rows as dicts, read in chunks of 50,000; ids as text, other values typed per chunk |
| `count_rows(path)`, `has_rows(path)` | The row count (one column, read in chunks); whether a row follows the header (one row read) |
| `keys_in(path, key_fields, casters)` | The unit keys of a file's rows, every cell read as text and converted by `casters` (a field without a caster gives a string), in chunks. A file that lacks a key column gives no keys |
| `accepted_mask(verdicts)` | Which cells of an `accepted` column read true, in any case, booleans or text; an empty cell, `yes` or `1` does not. The tree merge, the build and the paraphrase front use it |
| `write_concat(target, parts, *, keep_empty=True, unique=None)` | One CSV from several `Part`s (`source`, a file or a frame; `dataset_type`; `accepted_only`; `label`; `where`; `columns`), one after another |

A file that is missing, or has 0 bytes (a save killed before its first
byte), reads as no rows in every reader.

`write_concat` takes the columns of every part, in the order they first
appear; a row is empty in the columns its part lacks. A file part is read
as text, in chunks, and `accepted_only` keeps the rows `accepted_mask`
passes (a file without the column is copied whole); a frame part is not
filtered. `where`, a function of a chunk, keeps the rows it gives true, in
a file or a frame part. `columns` maps each column written, in order, to
where it comes from: a string names a column of the part (empty in a part
without it), a function of a chunk gives the cells, any other value is
every row's cell; the part's own columns are then not written, and a part
with `columns` whose file is missing still gives the header. In a frame part, a float column of whole numbers that has gaps is
written as `Int64`, so it prints `1`, not `1.0`. The rows go to
`<target>.tmp`, synced to disk, which is then renamed onto the target.
`keep_empty=False` leaves the target as it is when no part has a row.
`unique` names a column: a row whose value in it an earlier part wrote is
left out (an empty value stays, and rows of one part are not compared with
each other), the count left out of each part is logged at INFO, and the
values written are held in memory. It returns the rows written per part.

On Windows a file another program holds (a virus scanner, a sync client, a
spreadsheet) is tried again for a few seconds before a write, cut or rename
fails.

## The two sidecars

Both are JSON Lines files beside the stage's output, built on
`JsonlSidecar`, whose read skips a line it cannot parse.
`Ledger.sidecar(artifact, key_fields, casters, name=None)` and
`Manifest.sidecar(artifact, name=None)` give the file beside an artifact;
`name` replaces the file name.

**The manifest**, by default `<output stem>.manifest.jsonl` (`Manifest`), is
the plan:
one row per unit, written whole before the units run. `write` goes through
`<manifest name>.tmp`, synced to disk, and a rename onto the manifest; an
empty plan writes nothing and leaves an existing manifest as it is. Its rows
carry whatever the stage needs.

**The ledger**, by default `<output stem>.state.jsonl` (`Ledger`), records
finished
units, one record each. `Ledger(path, key_fields, casters)` names the fields
that make a unit's key; `casters` convert them on read so they compare with
the keys the stage builds. A record may carry more fields (a `status`, the
`ids` of its rows); they are kept and ignored on read, except `ids`.

The last record of each save carries `_saved`: the size and a checksum of
every file of the stage after that save. A new ledger opens with an empty
`_saved` line, written before the first rows:

```
{"_saved": {}}
{"input_id": "3f2a", "iteration": 0}
{"input_id": "91bc", "iteration": 1, "_saved": {"jailbreaks.csv": [5284, "5d41402a"]}}
```

A file is named relative to the ledger's folder, with `/`, so a folder copied
to another machine resumes there. The checksum is the CRC32 of the last 4 kB
before the recorded size.

`Ledger.read()` returns the finished keys, the last `_saved`, and the byte
offset where the counted lines end. A line counts only if it ends with a
newline, and lines after the last `_saved` belong to a save that did not
finish, so they do not count. A ledger with no `_saved` line at all, a plain
one, counts every line. `completed()` is the keys alone; `ids` the
content ids of every finished unit's rows.

## The save and resume model

### A save

A stage saves through a `CommitBuffer`:

```python
done = resume_state(ledger, resume=resume, artifact=out, manifest=manifest)
with CommitBuffer(append=lambda rows: append_rows(out, rows), ledger=ledger) as buf:
    ...
    buf.add(rows_of_unit, ack_record)      # on each finished unit
```

`add` queues a unit's rows and its ack together. Every `every` units (32 by
default) and on leaving the block, whatever ends it, `commit` makes one save:

1. check that every file still has the size the ledger last recorded; if
   not, another writer touched it, and the save stops with a `RuntimeError`;
2. on a new ledger, write the opening `{"_saved": {}}` line;
3. append the rows with one synced write per file;
4. append the acks with one synced write, the last one carrying `_saved`.

A unit is acked only once its rows are on disk, and a save counts only once
its `_saved` line is. A unit that produced no rows is still acked, so it is
not retried forever. Leaving the block also releases the stage's lock.

### A resume

`resume_state(ledger, resume=True, artifact=out)`:

1. takes the stage's lock (below);
2. stops if a `drop_units` did not finish;
3. reads the ledger;
4. checks every file the last `_saved` names: a file that is missing,
   shorter than recorded, or different in the last 4 kB before the recorded
   size stops the resume with a `RuntimeError` naming the remedies. Every
   file is checked before any is changed;
5. cuts each file back to its recorded size. The bytes cut off, the tail of a
   save that did not finish, are appended to `<file>.cut`;
6. cuts the ledger back to the end of its last `_saved` line;
7. returns the finished keys. It reads no row of the output.

The units of the unfinished save run again, so no unit ends with two sets of
rows. `tests/dataset/test_crash_matrix.py` crashes a save at every write and
checks five guarantees: a unit is finished only if its rows are on disk; no
row is on disk that a rerun would write again; what is acked before the
crash stays finished; each file is exactly what the ledger records; the resume reads no
row.

A folder whose ledger records no sizes, such as one written before the
ledger did, is taken over once: each output file is read for the unit keys
of its rows, and one ledger write acks them and records the sizes.

`resume_state(ledger, resume=False, artifact=out, manifest=manifest)` takes
the stage's lock and removes the output, its `.cut` and `.tmp` files, the
`.dropping` marker, the ledger and the manifest, and returns no key. The
lock stays held until the `CommitBuffer` exits or `release_lock` is called.
Called with no `artifact`, as the constitution stage does, the ledger is
plain: read as it is, with no lock, no sizes, nothing cut, and
`resume=False` removes only the ledger and the manifest.

`clear_stage(artifact, *, manifest_path=None, also=())` does what a fresh
run's `resume_state` does, for a stage that starts over with nothing to run:
it takes the stage's lock, removes the output, its `.cut` and `.tmp` files,
the `.dropping` marker, the ledger beside it and the manifest (beside it
unless `manifest_path` names it), gives the lock back, then removes the
files in `also`. The fronts call it with `resume=False` for a stage skipped
for want of accepted rows: outputs, each paraphrase target, and each
jailbreak kind. A jailbreak kind is also cleared when its `entry_types`
match no input. `also` holds the `.coverage.csv` and `.plan.json` of the
outputs stage and of a kind.

### The lock

The lock is `<ledger name>.lock` beside the ledger (`filelock`, no waiting).
It makes one process the writer of a stage from `resume_state` until the
`CommitBuffer` exits, or until `release_lock(ledger)` for a stage that finds
nothing to save. A second run on the same output stops with `... is locked`.
The operating system holds the lock, so a run that died holds nothing. The
size check before each save also stops a second writer on a volume that
ignores locks.

The limits, as `resume.py` states them: a change to an output that keeps its
size and lies before its last 4 kB is not noticed; on a volume that ignores
the lock, two runs that save within the same millisecond are not stopped;
a reader other than the stage sees a file as it is, with what an unfinished
save left in it.

### A stage with several files

A stage that writes a tree names the folder and a pattern:
`resume_state(ledger, resume=..., artifact=folder, pattern="*/samples.csv")`.
Every matching file is checked and cut back together. A matching file that
no finished save recorded is cut whole: a save that did not finish created
it. The seeded inputs stage checks the columns of every file of a save
(`check_columns`) before it writes the first, so a save never stops half way
over a bad column.

### A fresh run of part of a stage

`drop_units(ledger, *, artifact, name, rows, acks, pattern=None,
units_in=None)` removes some units' rows and acks and leaves the rest. A
folder whose ledger records no sizes is taken over first. It puts the files
back at the last save, writes each output again without the picked rows,
then the ledger without the picked acks, each through `<file>.tmp`, synced,
and a rename. A file left with no row is removed, so the next save writes
its header. The CSV reader's field limit is raised to 2^31 - 1, so a cell as
long as pandas writes it is read; a file that does not parse as CSV stops it
with a `RuntimeError`. Until it is done, `<ledger name>.dropping` holds
`name`: a resume stops, and only the same call, made again, finishes the
job. The inputs stage uses it for a fresh run of one style.

## The per-category tree

Input samples live in one `samples.csv` per category folder, under
`Datasets/constitution_inputs/`; the functions below default to that folder.

| Function | Does |
|---|---|
| `categories.SAMPLE_COLUMNS` | `sample_id`, `sample`, `category`, `subcategory`, `entry_type`, `turn`, `accepted`, `rejection_reason`, `source`; more columns are kept |
| `sample_row(text, category, turn, accepted, source, extra)` | One input row in that column order |
| `categories.read_category_csv(category, dataset_dir=None, filename="samples.csv")` | A category's file: ids as text, missing text as `""`; an empty `SAMPLE_COLUMNS` frame when there is no file or it has 0 bytes |
| `discover_categories(dir=None)` | The sorted names of the category folders whose `samples.csv` exists and has a byte |
| `merge_all(dir=None, accepted_only=True)` | The rows of every category in one frame, the accepted ones only by default (`accepted_mask`); an empty frame when there are none |
| `merge_inputs(data_dir)` | The run's accepted inputs, from `Datasets/constitution_inputs/` |
| `input_files(data_dir)` | The category files `merge_inputs` reads |
| `take_per_group(frame, n, group_by=("category", "entry_type"))` | The first `n` rows of each group, in the frame's order. Rows with a missing group value form a group of their own. Group columns the frame lacks are left out; with none present, the first `n` rows. `n=None` returns the frame |

## What an eval build leaves out

| Function | Does |
|---|---|
| `exclusion_files(paths)` | The CSVs the paths name: a file as it is, a run's folder as its `Datasets/*.csv` and its `Data_cache/constitution/merged.csv` |
| `Exclusion(files)` | Reads those files in chunks, their accepted rows only (a file without `accepted`, all), into three sets: `sample_id`s, normalised ids of the text columns (`sample`, `jailbreak`, `output_response`, `text`), and content ids of the descriptions (`sample_description`, `source_sample_description`) |
| `Exclusion.read_inputs(inputs)` | Notes which eval inputs are left out, and why, and which are kept from a shared constitution entry, for the rows derived from them |
| `Exclusion.keeps(text, origin)` | A `Part.where` filter: a row is left out for its own `sample_id` (`exact`) or text (`normalised`), else for its input's reason through the `origin` column; it counts each, and the kept rows of a shared entry (`descriptions`), in `counts` |

The build applies it to every part of the three eval files
([stages/build.md](../stages/build.md#leaving-out-what-a-training-run-holds)).

## Dealing and plan files

Shared by the planners: the jailbreak plan, and the generation axes of the
inputs and outputs stages.

| Function | Does |
|---|---|
| `dealing.Dealer(weights, key)` | Running counts of one group: `pick(rng, names)` takes the name with the lowest count over weight, ties broken by `rng`; `pick_member(rng, names)` the name with the lowest count in its own `members` counter (a family's techniques), ties broken by `rng`, kept apart from `counts` so `pick` deals the same with or without it; `include(part, rate)` deals an optional part at its rate with an offset seeded per group and part |
| `dealing.stable_seed(*parts)` | A seed from the parts, the same in every process: the first 16 hex digits of the SHA-256 of the parts joined by `::` |
| `planfile.plan_path(manifest)`, `coverage_path(manifest)` | `<stem>.plan.json` and `<stem>.coverage.csv` beside a manifest |
| `planfile.plan_record(**made_with)` | What a plan is made with, as JSON reads it back |
| `planfile.write_plan(manifest, record)` | Writes the plan file |
| `planfile.read_plan(manifest)` | The record the plan file holds; `None` without a plan file |
| `planfile.changed_plan(manifest, current)` | The keys whose values differ from the plan file's; `None` without a plan file |

## Taxonomy and published datasets

| Function | Does |
|---|---|
| `load_taxonomy(name, config_dir=None)` | A taxonomy file, from the folder `config_dir`, by default `configs/taxonomy/` (the folder of the files, not a recipe's `config_dir`, whose `taxonomy/` a run passes here through `config_overlay.taxonomy_folder`). The name is taken with or without `.json`; a name with another suffix gets `.json` added (`harmful.v2` reads `harmful.v2.json`). A missing file raises `FileNotFoundError`. The former name `content_moderation_categories` loads `harmful`, and a folder holding only `content_moderation_categories.json` is read for `harmful`, each with a `DeprecationWarning` |
| `create_taxonomy(name, categories, description="", taxonomy_dir=None, verbose=True)` | Writes `<name>.json` under `taxonomy_dir` (default the packaged `configs/taxonomy/`) and returns the taxonomy; a bare description is written as `{"description": ...}`. It adds `.json` unless the name ends in it |
| `iter_categories(taxonomy)` | `(name, info)` per category, a bare description as `{"description": ...}` |
| `load_hf_dataset(dataset_id, split, columns, filters, exclude, ...)`, `load_from_config(name)` | A HuggingFace dataset, filtered, directly or from a config under `configs/datasets/`; needs the `hf` extra |

[extending.md](../extending.md#add-a-category-or-a-taxonomy) shows the
taxonomy format.

## How the stages use it

| Stage | `resume_state` call | Ledger key | Records also carry |
|---|---|---|---|
| constitution | plain: no artifact | `unit` | `status`, `attempts` when rejected |
| inputs | `artifact=folder, pattern="*/samples.csv"`, `units_in` | `sample_description`, `entry_type`, `style`, `iteration` | `ids`, `accepted`; or `status`, `attempts` |
| outputs | `artifact=output_responses.csv` | `input_id`, `iteration` | `accepted` |
| paraphrase | `artifact=paraphrases_<target>.csv` | `input_id`, `iteration` (inputs); `response_id`, `iteration` (outputs) | `paraphrase_model`, `status`; `input_id` (outputs); `reason`, `min_words` when `skipped` |
| jailbreaks, per kind | `artifact=jailbreaks.csv` or `variations.csv` | `input_id`, `iteration` | |
| benign data (jailbreaks) | `artifact=benign_samples.csv`, always `resume=True` | `main_category`, `sub_category`, `answer_type` | `status`, `attempts` when rejected |
| conversations | `artifact=conversations.csv` | `input_id`, `iteration` | `stop_reason` |
| scoring | `artifact=conversations_scored.csv`, no manifest | `sample_id` | |

The build uses no ledger.
