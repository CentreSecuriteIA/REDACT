# Stage: inputs

Front `generate_inputs` (`stages.py`), core
`InputPipeline.run_from_constitution` (`content_moderation/inputs.py`). Every
run's inputs are seeded from a constitution, in an `eval` run as in a
`training` one; an `eval` run keeps the `harmful` and `benign` entries only
(its dataset type sets `entry_types`).

The stage saves every sample, accepted or not, with its verdict, to
`Datasets/constitution_inputs/<category>/samples.csv`. The later stages read
only the accepted ones: outputs, paraphrase and jailbreaks through
`dataset.merge_inputs`, the build through
`Datasets/constitution_inputs_merged.csv`.

## Purpose

Turn each constitution entry into `samples_per_entry` input samples, each
written to its own spec of the generation axes (length, register, writer,
framing, directness, focus) and checked against its category, entry type and
subcategory. An entry of the general-benign part is an input of its
benign-taxonomy category: generated and checked with that category, entry
type `benign` and its constitution subcategory.

## Inputs

- The constitution frame. In a recipe run the driver reads
  `Data_cache/constitution/merged.csv` when the stage starts; a run without
  it stops with `No constitution at ...`, which names both ways to have one:
  the `constitution` stage in the recipe, or one saved by an earlier run on
  the same `data_dir`. Called directly, pass `constitution_df=`; without it
  the front reads the saved one the same way.
- Columns read per entry: `sample_description`, `entry_type`,
  `source_category`, `constitution_subcategory`, `constitution_category`,
  `source_group_tag`, and `source_taxonomy` when the constitution has it.
  `entry_types` keeps only some entry types.
- The prompt `prompts/input/generation/from_constitution/<style>/` with the
  `numbered` format instruction, and the check prompt
  `prompts/input/quality_check/`, each read through the overlay. The
  template's `{sample_spec}` shows the unit's spec; a prompt without it is
  warned about (`The prompt ... has no {sample_spec}`), and its units are
  dealt and recorded a spec the model is not shown.
- The `inputs` section of `generation_axes.json`, a recipe's `config_dir`'s
  when it has one ([configuration.md](../configuration.md#the-generation-axes)).

## Models

| Recipe role | Registry role | Default | Calls |
|---|---|---|---|
| `gen` | `uncensored_gen` | `venice-uncensored` | One generation per unit, again on a bad reply |
| `check` | `uncensored_gen` | the `gen` model | One check per sample |

Pick a checker that can judge. In the GPU check, a 3B checker accepted 2 of
144 samples and a 24B checker 144 of 144.

The check calls run at temperature 0.15 (`llm_pipeline.CHECK_TEMPERATURE`),
the generations at the model's default: in the checker test the input
checker repeated its own verdict on 95.2 percent of harmful samples at
0.15, against 83.3 at the stages' 0.7, at the same acceptance.

## Settings

Params keys: `style`, `samples_per_entry`, `entry_types`, `axes`,
`axes_weighting`, `seed`, `batch_size`, `max_attempts`, `check_max_tokens`
([configuration.md](../configuration.md#inputs)). A `samples_per_entry`
that is not a whole number of at least 1 raises before the stage writes
anything. The `eval` type pins the
`directness` axis at `explicit`; the `training` type deals the weights
log-scaled (`axes_weighting: "log"`).

## Unit and ledger key

Each entry is `samples_per_entry` units, one per `iteration` (0 to k - 1).
Every unit is its own generator on the `drive_units` pool, so its generation
and check requests take the async lane and units finish apart.

1. One call asks for one sample written to the unit's spec. No earlier
   sample is shown in the prompt, so a unit's prompt is a function of its
   entry and its spec alone.
2. A reply without its `[END]` marker at the end (cut at the token limit, or
   written without it), or holding another number of samples than one, is
   rejected whole and asked again, up to `max_attempts` generations. The
   replies without the marker are counted in a WARNING at the end.
3. A sample whose text is already in the folder (by content id, any category
   and any style) is dropped.
4. A sample left gets one check call.

The key holds no category. So an entry whose `sample_description` and
`entry_type` an earlier entry of the frame has (one description in two
categories, or a constitution row saved twice by an earlier version) is
left out, the first kept, with a WARNING (`Inputs: N constitution entries
repeat ...`): the two would share their units' keys, both run, and a resume
would skip both once one was done.

| | Value |
|---|---|
| Ledger key | `(sample_description, entry_type, style, iteration)` |
| Ledger record | the key, `ids` (the `sample_id` of the row it wrote) and `accepted` (the verdict). Or the key with `status` `rejected` or `no_sample` and `attempts`. Each also holds `spec`, the unit's dealt spec (none without axes) |
| Manifest row | `sample_description`, `entry_type`, `style`, `iteration`, `category`, `spec` (the dealt value of each axis; none without axes), `status` |

## The generation axes

Each unit is dealt one spec: a value of each axis of the `inputs` section,
at the table's weights with the `axes` param over them, log-scaled when
`axes_weighting` is `"log"`, as in the `training` type
([configuration.md](../configuration.md#log-weighting)): each value of
weight w above 0 weighted 1 + ln(w / m), m the axis's least weight above 0,
normalised per axis. The `domain` axis is not dealt: it shows the entry's
subcategory.

- **Groups.** Units are dealt per `(source_category, entry_type)`, the
  coverage table's `category` and `entry_type` columns (the input tree has
  one folder per category, its entry types together in one file). A group's units go
  in `(sample_description, iteration)` order, so an entry's units are dealt
  one after another, from one dealer (`dataset/dealing.py`, the jailbreak
  plan's): each value goes to the unit when its count so far over its weight
  is the lowest of the axis, ties broken by the unit's rng, seeded by
  `(seed, category, entry_type, sample_description, iteration)`.
- **Siblings.** An entry's units take a value at most `ceil(share x k)`
  times, `share` its share of the axis's weight. So an entry's samples
  differ on every axis none of whose values has more than 1/k of the weight,
  and no prompt sentence asks for it. In the packaged table at k = 3 those
  axes are `length` and `writer`, and with `axes_weighting` `"log"` also
  `register` and `framing`; `directness` (`explicit` holds over a third) and
  `focus` (one dealt value) can repeat a value among an entry's samples.
- **`not_with`.** A value is not dealt beside a value of another axis that
  either names in `not_with` (`advice` and a `fragment`, for example).
- **Shown.** `{sample_spec}` renders the spec as one line per axis in the
  table's order: `Label: value label — hint`, then `Domain: <subcategory>`.
  Without axes (`axes: false`), no spec is dealt or recorded and
  `{sample_spec}` is empty.
- **Deterministic.** The deal is a function of the seed, the weights, the
  units of each group and `samples_per_entry`. It is made again on every
  run, and the plan file stops a run that would deal another.
- **Done units keep their spec.** A done unit keeps the spec its ack
  records, the one its row was written with: the deal counts it as dealt
  and deals the other units around it, so the manifest and the coverage
  table agree with the rows. Entries that join a group already dealt (a
  frame passed with entries added) are dealt around the done units, and
  the specs of units not done may move; a new category is a group of its
  own. An ack written by an earlier version, without a spec, leaves its
  unit to the deal: its row keeps its spec, and the manifest may show
  another.
- **Entries that leave.** The units of entries no longer in the frame (a
  constitution regenerated with `resume: false`, or a subset passed) are
  not planned: their rows stay in the category files and reach the later
  stages, and the manifest and the coverage table no longer count them.

## Files

Under `Datasets/constitution_inputs/`:

| File | Content |
|---|---|
| `<category>/samples.csv` | One per category (`source_category`): a harmful-taxonomy category, or a benign one for the general-benign part; a benign category named as a harmful one shares its file. Its general-benign rows and the harmful taxonomy's `benign` entry rows are both `benign`: only `source_taxonomy` tells them apart. Accepted and rejected rows |
| `constitution_inputs.manifest.jsonl` | The plan of the current style, written again every run |
| `constitution_inputs.state.jsonl` | The ledger, for every style |
| `constitution_inputs.plan.json` | What each style's deal was made with, `{"styles": {<style>: {...}}}`: `seed`, `samples_per_entry`, and each dealt axis's weights as dealt, log-scaled with `axes_weighting` `"log"` (`null` without axes). Labels and hints are not in it: rewording them changes no deal |
| `constitution_inputs.coverage.csv` | Per axis value and group: `axis`, `value`, `category`, `entry_type`, `planned` (from the manifest), `accepted` (from the acks of this style, each counting the spec it records). Every value of the table has a row in every group seen, `0` when never dealt. Written when the stage ends, and when it has nothing to run; removed without axes |

And `Datasets/constitution_inputs_merged.csv`: the rows of every category
and every style whose `accepted` reads true (`dataset.accepted_mask`),
copied by the front after the core returns. The build reads it. It is
removed when no category file holds an accepted row, so the build never
reads rows the category files no longer hold.

Columns of `samples.csv`:

| Column | Meaning |
|---|---|
| `sample_id` | Content id of `sample` |
| `sample` | The generated text |
| `category` | The entry's `source_category`: the taxonomy category, or the benign-taxonomy category |
| `subcategory` | The entry's `constitution_subcategory` |
| `entry_type` | The entry's entry type |
| `turn` | `0` |
| `accepted` | The checker's verdict |
| `rejection_reason` | The checker's reply when rejected, else empty |
| `source` | `constitution` |
| `source_sample_description` | The entry it came from |
| `constitution_category` | The entry's constitution category |
| `source_group_tag` | The entry's group tag |
| `source_taxonomy` | `harmful` or `benign`, copied from the entry: `benign` marks the general-benign part. Only when the constitution has the column |
| `template_style` | The `style` that generated it |
| `iteration` | The unit's iteration: 0 for an entry's first sample |
| `length`, `register`, `writer`, `framing`, `directness`, `focus` | The unit's dealt value of each axis. Only with axes |

## How it resumes

The stage names its folder and the pattern `*/samples.csv` to the resume, so
all category files are checked and cut back together
([internals/dataset.md](../internals/dataset.md#a-stage-with-several-files)).
Units the ledger records are skipped. The ids on the ledger records keep a
text from being written twice.

The deal is made again, and compared with the style's record in
`constitution_inputs.plan.json`: a resume whose `seed`, `samples_per_entry`
or axes' weights differ (another `axes` or `axes_weighting`) from those its
style's done units were dealt with raises `Inputs: this run's [...] differ
from those the units of style ... were dealt with`, naming the keys and
`resume=False`. A `generate_inputs` call on a training run's folder passes
`axes_weighting="log"`, as the run does. The file holds one record per
style: a fresh run of one style replaces its record and keeps the others',
as it keeps their rows. A style with no done unit is not compared (its
first run, or a fresh run that stopped before its first save). A file an
earlier version wrote, with one record, is taken as the record of every
style with done units. A style with finished units and no record runs
unchecked, with a WARNING. A plan file that holds no record (one an earlier version cut off
as it wrote it) stops the resume before anything is cut, naming the file:
delete it to resume unchecked, or start the style over.

A folder written before units had an iteration stops before any call: a
ledger that acks a unit without `iteration` (an entry was one call then), or
a category file whose header lacks `iteration` or an axis column. The
message names the remedies: start the style over (`resume: false`), run in
a new `data_dir`, or, for a folder without the axis columns,
`inputs.axes: false`.

A category file written from a constitution without `source_taxonomy`
cannot take rows that have it: with a constitution that has the column, the
stage stops before any call with `Cannot append to ...: its header has no
column 'source_taxonomy'`. Start the style over (`resume: false`) after
regenerating the constitution, or run in a new `data_dir`.

`resume: false` starts over the current `style` only. Its rows are removed
from every category file and its acks from the ledger (`dataset.drop_units`);
the rows of other styles stay. An empty constitution frame (every entry
filtered out by `entry_types`, say) still starts the style over, and leaves
no manifest; on a resume it changes nothing. Until that finishes, a
`constitution_inputs.state.jsonl.dropping` file names the style; a later run
stops until the same fresh run is repeated.

## Checked before any call

Before anything is cleared (a fresh run's `drop_units`) or any call made:

- each entry's `source_category` must name a folder
  (`dataset.check_category_name`; see
  [constitution](constitution.md#inputs)), else a `ValueError` names it;
- the generation prompt, with its format instruction, and the check prompt
  are rendered once with empty values, so an overlay with a placeholder the
  stage does not fill stops the run here rather than inside its first units;
- the folder of an earlier layout stops it, as above.

## Rejected and failed units

- A sample the checker rejects is saved with `accepted` false and the
  checker's reply in `rejection_reason`. The check prompt asks for `Yes` or
  `No` as the reply's first word; `is_accepted`
  ([llm_pipeline](../internals/llm_pipeline.md)) accepts a first word of
  `yes`, `ok`, `okay`, `accept`, `accepted`, `pass` or `passed`, in any case
  and after leading markdown or quote marks, and rejects any other reply, one
  that opens with a preamble included.
- A unit with every reply rejected is acked `rejected`, with no row.
- A unit whose sample the folder holds already is acked `no_sample`.
- On the async lane (an OpenAI-compatible endpoint or vLLM), a unit whose
  generation or check prompt is too long for its model is acked `rejected`
  after one attempt, with no row, and logged `Entry ... rejected, its
  prompt is too long`. On an Anthropic or introspection binding the
  requests go out in grouped calls, and a prompt too long there fails the
  call and stops the stage.
- These acked units are skipped on resume; a warning at the end counts the
  rejected ones. Start the style over to try them again.
- Any other error inside a unit, and any failed call, stops the stage. Saved
  units stay saved.

When under half of the generated samples are accepted, the front warns:
`inputs: only N of M generated samples were accepted (under half)`.

Every call asks for one sample, so a model that captures internals works at
any `samples_per_entry`: a generation is captured under
`<entry id>/<iteration>/input_<style>`, its check under
`<sample_id>/val_in_<style>`.
