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
- loading the taxonomy and seed files (`taxonomy.py`) and published datasets
  (`loading.py`).

It never sees a prompt, a client or a model; it imports only `redact.paths`,
pandas and `filelock`. `resume.py` holds the policy (what is skipped, what
is cut back); `rows.py` holds the mechanism, so a change to how rows are
stored stays out of the resume logic.

## Identity

| Function | Does |
|---|---|
| `content_id(text)` | The MD5 of the text, cut to 16 hex characters |
| `row_id(row, text_col, id_col="sample_id")` | The id a row carries in `id_col`, else the content id of its text. `None`, NaN, empty or `"nan"` count as missing |
| `row_ids(frame, text_col)` | `row_id` of every row, reading two columns only |

Every row a stage writes has a `sample_id`, the content id of its own text.
A row derived from another has an `input_id`, that row's id.

## Rows

| Function | Does |
|---|---|
| `append_rows(path, rows)` | Appends in one write, synced to disk, in the column order of the file's header (a new file takes the rows' order and gets the header). Raises, writing nothing, on a column the header lacks. Returns `{path: size}` |
| `check_columns(path, rows)` | Raises as `append_rows` would, without writing |
| `read_rows(path)` | The whole file, or an empty frame |
| `iter_rows(path, columns)`, `count_rows(path)`, `has_rows(path)` | Read in chunks of 50,000 rows |
| `keys_in(path, key_fields, casters)` | The unit keys of a file's rows |
| `write_concat(target, parts)` | One CSV from several `Part`s (a file or a frame, a `dataset_type`, `accepted_only`), copied in chunks through a temporary file that replaces the target |

On Windows a file another program holds (a virus scanner, a sync client, a
spreadsheet) is tried again for a few seconds before a write, cut or rename
fails.

## The two sidecars

Both are JSON Lines files beside the stage's output, built on
`JsonlSidecar`, whose read skips a line it cannot parse.

**The manifest**, `<output stem>.manifest.jsonl` (`Manifest`), is the plan:
one row per unit, written whole before the units run. `write` goes through a
temporary file and a rename; an empty plan writes nothing. Its rows carry
whatever the stage needs.

**The ledger**, `<output stem>.state.jsonl` (`Ledger`), records finished
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
finish, so they do not count. `completed()` is the keys alone; `ids` the
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
row is on disk that a rerun would write again; what was acked stays
finished; each file is exactly what the ledger records; the resume reads no
row.

A folder whose ledger records no sizes, such as one written before the
ledger did, is taken over once: each output file is read for the unit keys
of its rows, and one ledger write acks them and records the sizes.

`resume_state(ledger, resume=False, artifact=out, manifest=manifest)`
removes the output, its `.cut` files, the ledger and the manifest, and
returns no key. Called with no `artifact`, as the constitution stage does,
the ledger is plain: read as it is, with no lock, no sizes, nothing cut.

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

`drop_units(ledger, artifact=, name=, rows=, acks=, pattern=)` removes some
units' rows and acks and leaves the rest. It puts the files back at the last
save, writes each output again without the picked rows, then the ledger
without the picked acks, each through a temporary file and a rename. Until it
is done, `<ledger name>.dropping` holds `name`: a resume stops, and only the
same call, made again, finishes the job. The seeded inputs stage uses it for
a fresh run of one style.

## The per-category tree

Input samples live in one `samples.csv` per category folder.

| Function | Does |
|---|---|
| `SAMPLE_COLUMNS` | `sample_id`, `sample`, `category`, `subcategory`, `entry_type`, `turn`, `accepted`, `rejection_reason`, `source`; more columns are kept |
| `sample_row(text, category, turn, accepted, source, extra)` | One input row in that column order |
| `append_samples(samples, category, turn, ...)` | Adds samples to a category's file, skipping ids already in it. Reads and rewrites the whole file; only the standalone inputs use it |
| `discover_categories(dir)` | The category folders that hold a `samples.csv` |
| `merge_all(dir, accepted_only=True)` | Every category's rows in one frame |
| `merge_inputs(data_dir)` | The run's accepted inputs: the seeded tree when it has a category file, else the categories under `Datasets/` |
| `take_per_group(frame, n, group_by=("category", "entry_type"))` | The first `n` rows of each group |

## Taxonomy, seeds and published datasets

| Function | Does |
|---|---|
| `load_taxonomy(name, config_dir=None)` | A taxonomy file, from `configs/taxonomy/` by default |
| `create_taxonomy(name, categories, ...)` | Writes a taxonomy file and returns it |
| `iter_categories(taxonomy)` | `(name, info)` per category, a bare description as `{"description": ...}` |
| `load_seeds(name)`, `get_seed_prompts(seeds, category)` | The seeds file, and one category's seeds as a numbered list |
| `load_hf_dataset(dataset_id, split, columns, filters, exclude, ...)`, `load_from_config(name)` | A HuggingFace dataset, filtered; needs the `hf` extra |

[extending.md](../extending.md#add-a-category-or-a-taxonomy) shows the
taxonomy format.

## How the stages use it

| Stage | `resume_state` call | Ledger key | Records also carry |
|---|---|---|---|
| constitution | plain: no artifact | `unit` | `status`, `attempts` when rejected |
| seeded inputs | `artifact=folder, pattern="*/samples.csv"`, `units_in` | `sample_description`, `entry_type`, `style` | `ids`; or `status`, `attempts` |
| outputs | `artifact=output_responses.csv` | `input_id` | |
| paraphrase | `artifact=paraphrases_<target>.csv` | `input_id`, `iteration` | `paraphrase_model`, `status` |
| jailbreaks, per kind | `artifact=jailbreaks.csv` or `variations.csv` | `input_id`, `iteration` | |
| conversations | `artifact=conversations.csv` | `input_id`, `iteration` | `stop_reason` |
| scoring | `artifact=conversations_scored.csv`, no manifest | `sample_id` | |

The standalone inputs and the build use no ledger.
