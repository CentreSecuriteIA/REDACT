# Configuration reference

Every key a run reads: the recipe, its dataset type, the params file, the
generation axes, the augmentation section, the model registry, the
environment, your own config folder, and the prompt overlay. For how these fit together in a run, see [running.md](running.md).

## Recipe keys

A recipe is a JSON object. `recipe.load_recipe` refuses any other key; a key
that starts with `_` is a comment.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dataset_type` | `"eval"` or `"training"` | required | Any other value raises at load. The [dataset type](#dataset-types): its file gives the run's default stages, params and augmentation section, and it is the build's `mode` (`training` merges every part into `complete_dataset.csv`, `eval` writes `eval_inputs.csv`, `eval_outputs.csv` and `eval_jailbreaks.csv`; see [build](stages/build.md)). Either way the inputs are seeded from the constitution, which the recipe's `constitution` stage writes or an earlier run saved under `data_dir`; without one the `inputs` stage stops |
| `data_dir` | path | the default working root | The run's root; `Datasets/` and `Data_cache/` go under it. Relative to the current directory. `run_pipeline(data_dir=...)` and `--data-dir` override it |
| `models` | object | every role its default | A model per role; see [below](#the-models-object) |
| `taxonomy` | name or object | `"harmful"` | The taxonomy of the constitution, by file name or as a loaded object. The former name `"content_moderation_categories"` loads `harmful` with a `DeprecationWarning`, for one release |
| `taxonomy_dir` | path | `<config_dir>/taxonomy/` for a file it has, else `configs/taxonomy/` | Where a taxonomy name (`taxonomy`, `benign_taxonomy`) is looked up; given, it wins over `config_dir` |
| `benign_taxonomy` | name or object | `"benign"` | The taxonomy whose categories the constitution's general-benign part is built from, and whose subcategories the benign Q&A pairs of the `fsh` and `dap` techniques are generated for. Read at load, and checked to have a category, when the recipe runs `constitution` and its params switch the general-benign part on (`include_general_benign`, or its former name: the dataset type's, with the params file's over them), or runs `jailbreaks` and a kind draws a technique that needs benign data |
| `prompt_dir` | path | none | The prompt overlay; see [below](#the-prompt-overlay) |
| `stages` | list of stage names | the dataset type's: all six | The stages to run, in order, each once. `[]` runs nothing |
| `augmentations` | object | the dataset type's | The jailbreak stage's augmentation section, over the dataset type's; see [below](#the-augmentation-section) |
| `params_file` | path | none | The params file, relative to the recipe's folder. `--params` and `run_pipeline(params=...)` override it |
| `config_dir` | path | none | [Your own config folder](#your-own-config-folder), relative to the recipe's folder: its dataset type, taxonomy, augmentation, generation axes and dimensions files are read in place of the packaged ones. Not a folder: raises at load |
| `resume` | bool | `true` | `false` starts every listed stage over ([running.md](running.md#resume-fresh-runs-and-a-kill-partway-through)). Any other value than `true` or `false`, the string `"false"` too, raises |

A top-level `copies` key other than `null` raises: write the count in the
role's object instead.

## Dataset types

`dataset_type` names a packaged file, `configs/dataset_types/<type>.json`,
that holds what the type means: the default `stages`, `params` per stage, and
an `augmentations` overlay. Every key is resolved in layers, each overlaying
the one before:

| Key | 1. Packaged default | 2. Dataset type file | 3. Recipe | 4. Params file |
|---|---|---|---|---|
| `stages` | | the type's list | the recipe's list, whole | |
| `params` | the front's defaults | per stage | | per stage, key by key |
| `augmentations` | `configs/jailbreak/augmentations.json` | kind by kind | kind by kind | the deprecated flat keywords |

A kind overlays the layer below it key by key, as a recipe's kind overlays
the packaged default ([below](#the-augmentation-section)). A kind the type
file gives as `null` is off unless the recipe gives it as an object, which
then overlays the packaged default. A type's params are checked as the params
file's are, before the first stage runs, and each run record's `params`
holds the merged values. Each record also holds `dataset_type` and
`dataset_type_hash`, the content id of the type file as parsed. A recipe's
[`config_dir`](#your-own-config-folder) may hold its own `eval.json` or
`training.json`, read in place of the packaged one, whole: a copy made
before a change to the packaged file keeps the old values. There are no other
types: a recipe naming one stops when it loads, before any stage runs.

| | `training.json` | `eval.json` |
|---|---|---|
| `stages` | `constitution`, `inputs`, `outputs`, `paraphrase`, `jailbreaks`, `build` | the same |
| `params` | `constitution`: `include_general_benign` true, `general_benign_categories` 10; `inputs.axes_weighting` `"log"`; `outputs.outputs_per_input` 2, each input answered twice, so a borderline input gets one answer of each [answer mode](stages/outputs.md#answer-modes) | `constitution` the same, and `constitution.entry_types` and `inputs.entry_types` `["harmful", "benign"]`; `inputs.axes` pins `directness` at `explicit`, and the inputs keep the table's weighting; `jailbreaks.seed` 43; the outputs keep one response per input (their two entry types allow one mode each) |
| `augmentations` | none: the packaged default | the adversarial kind below; `natural` off |

The eval adversarial kind attacks the harmful inputs only, with a pool of
hard techniques whose label stays certain: the families `encode` (less the
22 encodings `techniques.deny` names, the seven light ones of before spec 1.2
and the fifteen it added, which leaves its five original `heavy` ones; and
less `sensitive_words`), `fsh` and `dap` (less
their four many-shot variants), `framing`, `structural`, `ascii_art`,
`continuation`; the eight low-resource
translations (`to_swahili`, `to_thai`, `to_khmer`, `to_nepali`,
`to_zulu`, `to_scots_gaelic`, `to_bengali`, `to_javanese`), with
`"translation": true` and `family_weights` `{"translation": 3}`; and
`to_refusal_then_comply` and `to_use_past_tense`, named under
`techniques.allow` rather than their families, so a technique the library
adds to `answer_format` or `temporal` does not reach the eval pool (the
families it allows whole name each later member under `techniques.deny`).
It runs two count-driven rounds, one technique
then two (`rounds` `[{"exact_techniques": 1}, {"exact_techniques": 2}]`),
with `allow_request_obfuscation` false, written out since the packaged
attacks have it true; a count round never reads it, so either way its draws
are those made before the packaged attacks turned it on.

The eval attacks are *pinned*: the 36 techniques, the two rounds and the text
each technique draws for a seed stay the same from release to release, so the
results of an eval run stay comparable; a test fails if one changes
([techniques.md](techniques.md#the-eval-pool) says what is pinned and how it
changes).

Every eval input is explicit (`inputs.axes`: `{"directness": {"explicit": 1,
"implicit": 0, "euphemistic": 0}}`): an implicit or euphemistic request is
borderline by construction, and eval keeps the clear ends. The other axes
keep the table's weights, where training spreads its inputs more evenly over
the values ([log weighting](#log-weighting)). A params file's `inputs.axes`
replaces the type's whole entry, so to change another axis of an eval run,
write the directness pin again beside it.

A recipe changes one thing by giving it; the rest of the type stays. For
example, an eval run with another stage list and a lighter translation share:

```json
{
  "dataset_type": "eval",
  "stages": ["constitution", "inputs", "jailbreaks", "build"],
  "augmentations": {"adversarial": {"family_weights": {"translation": 1}}}
}
```

## The models object

Keys are recipe roles: `gen`, `check`, `translation`, `constitution`,
`paraphraser`, `paraphrase_check`. Any other key raises, whatever its value;
a key that starts with `_` is a comment. A value is a registry name, `null`
(the role's default), or an object:

| Key | Type | Meaning |
|---|---|---|
| `name` | registry name | Left out or `null`: the role's default |
| `backend_type` | `"api"`, `"vllm"`, `"introspect"` | The setup of the registry entry that serves the model |
| `copies` | int >= 1 | vLLM setups only: copies of the engine, one per card. Left out: a copy on each card with room |
| `workers` | int >= 1 | API setups only: requests in flight. Left out: `recommended_max_workers`. Must be 1 on Anthropic |

`paraphraser` may also be `"distribution"`, the pool of every model with the
`paraphraser` role; it takes no object. One model is served one way per run.
[running.md](running.md#choose-models) gives the roles and their defaults.

## Params keys per stage

The params file maps a stage name to keyword arguments of its front, over
the [dataset type's](#dataset-types) params for that stage. A top-level key
that is not a stage name raises, except a comment key that starts with `_`. Inside a stage's entry every key must be a parameter of the
front (a `_` key there raises too), and not one the run passes itself (the
models, `data_dir`, `prompt_dir`, `taxonomy`, `taxonomy_dir`,
`benign_taxonomy`, `config_dir`, `resume`, `verbose`, the inputs stage's
`constitution_df`, the jailbreak stage's `augmentation`, the build's `mode`,
`read_back`). Parameters that take a DataFrame (`inputs`, `outputs`) cannot
be written in JSON. The `axes`, `axes_weighting` and `seed` of `inputs`
and `outputs`, and the `modes` and `outputs_per_input` of `outputs`, are
checked when the params load (a params file's and a dataset type's): an
unknown axis or value, a weight below 0, a weighting other than `"table"`
or `"log"`, a seed that is not a whole number, a `modes` that is not an
object of entry types to non-empty lists of distinct modes, or an
`outputs_per_input` that is not a whole number of at least 1 raises, the axes read against the
`config_dir`'s axes table when it has one.

### `constitution`

| Key | Default | Meaning |
|---|---|---|
| `num_categories` | `10` | Constitution categories asked for per call |
| `num_taxonomy_categories` | `null` | Only the first N categories of the taxonomy and of the benign taxonomy; `null` for all |
| `entry_types` | `null` | Entry types to generate; `null` for all four. The `eval` type sets `["harmful", "benign"]` |
| `include_general_benign` | `false`; `true` in both dataset types | Add the general-benign part: one call per category of `benign_taxonomy`, its rows `benign` entries of that category with `source_taxonomy` `benign` ([stages/constitution.md](stages/constitution.md#unit-and-ledger-key)) |
| `general_benign_categories` | `10` | Constitution categories asked for per general-benign call |

`include_standalone_benign` and `standalone_benign_categories`, the former
names of the last two, are read for this release with a
`DeprecationWarning`; given, they win.

### `inputs`

| Key | Default | Meaning |
|---|---|---|
| `style` | `"long"` | The prompt under `prompts/input/generation/from_constitution/<style>/`: `long` or `short`. With axes, the length axis sets each sample's length |
| `samples_per_entry` | `3` | Samples per constitution entry: that many units, one sample per call |
| `entry_types` | `null` | Use only the constitution entries of these entry types |
| `axes` | `null`: the table's | The [generation axes](#the-generation-axes) each unit is dealt a value of: `{axis: {value: weight}}` over the table's `inputs` section, an axis as `null` to leave it out; `false` for none. The `eval` type pins `directness` |
| `axes_weighting` | `"table"`; `"log"` in the `training` type | How the axes' weights are dealt: `"table"` as they are, `"log"` spread more evenly over each axis's values ([log weighting](#log-weighting)) |
| `seed` | `42` | The seed of the deal |
| `batch_size` | `32` | Requests per grouped call; the stage keeps twice as many units live, or the widest async lane when larger |
| `max_attempts` | `3` | Generations of a unit before a reply without its `[END]` marker, or with another number of samples than one, is given up |
| `check_max_tokens` | `null` | Reply limit of a check call; `null` for the checker's default |

### `outputs`

| Key | Default | Meaning |
|---|---|---|
| `check_outputs` | `true` | Check each response; `false` accepts every response |
| `max_attempts` | `1` | Generations per input; above 1 a rejected response is generated again with the verdict as feedback |
| `max_per_category` | `null` | Keep the first N inputs per `(category, entry_type)` |
| `max_samples` | `null` | Keep the first N inputs overall, after `max_per_category` |
| `outputs_per_input` | `1`; `2` in the `training` type | Responses per input, each a unit `(input_id, iteration)` dealt its own answer mode and spec, so an input's responses differ on the axes and, from 2, a borderline input gets one answer of each mode. `max_per_category` and `max_samples` count inputs, before this expansion. A whole number of at least 1; `null` is not read as the default and raises at load |
| `modes` | `null`: the default | The answer modes each entry type allows, `{entry_type: [mode, ...]}` laid over the default per entry type: `harmful` → `["harmful"]`, `dual_use_harmful` and `dual_use_benign` → `["harmful", "helpful"]`, `benign` → `["helpful"]`. A mode picks the response's prompt and checker and gives its label ([outputs](stages/outputs.md#answer-modes)) |
| `axes` | `null`: the table's | The response axes each input row is dealt a value of, over the table's `outputs` section, as for `inputs` |
| `axes_weighting` | `"table"` | As for `inputs`; neither type sets it |
| `seed` | `42` | The seed of the deal |
| `batch_size` | `32` | Requests per grouped call; units live as for `inputs` |
| `check_max_tokens` | `null` | Reply limit of a check call |
| `output_path` | `Datasets/output_responses.csv` | Where the responses go |

### `paraphrase`

| Key | Default | Meaning |
|---|---|---|
| `target` | `"inputs"` | `inputs`, `outputs` or `both`; the outputs are an opt-in |
| `paraphrases_per_sample` | `1` | Paraphrase units per source row |
| `check` | `true` | Gate each paraphrase and check it with the `paraphrase_check` model ([the check](stages/paraphrase.md#the-check)); `false` accepts every paraphrase |
| `min_words` | `12` | A text under this many words (split on whitespace) gets no paraphrase; its unit is acked `skipped`. `0` paraphrases every text |
| `batch_size` | `256` | Requests per grouped call; at least twice as many units are live |
| `seed` | `42` | Seed of the assignment of units to paraphrasers |
| `inputs_path`, `outputs_path` | `Datasets/paraphrases_inputs.csv`, `Datasets/paraphrases_outputs.csv` | Where the paraphrases go |

### `jailbreaks`

| Key | Default | Meaning |
|---|---|---|
| `seed` | `42` | The run seed of the plan and of each technique's random choices |
| `batch_size` | `256` | Requests per grouped call; at least twice as many units are live |
| `auto_generate_benign` | `true` | When a pending unit needs benign data, generate the pairs `benign_path` lacks for the benign taxonomy's subcategories, then load it. `false`: load the file as it is; with none, the stage stops before any unit runs (`Benign data not found ...`) |
| `output_path` | `Datasets/jailbreaks.csv` | The adversarial kind's file |
| `variations_path` | beside `output_path`, else `Datasets/variations.csv` | The natural kind's file |
| `manifest_path` | `jailbreaks.manifest.jsonl` beside the output | The adversarial kind's plan, with its plan file `<stem>.plan.json`, coverage table and variants table beside it |
| `benign_path` | `Data_cache/benign/benign_samples.csv` | The benign data, with its manifest and ledger beside it |

The run passes the recipe's `gen`, `check` and `translation` models as
`model`, `check_model` and `translation_model`: the `check` model checks
what a technique's model writes, and left out it is the `gen` model, so no
other model loads ([the stage's models](stages/jailbreaks.md#models)). A
`check` model of its own is created, planned and loaded for the stage only
when a kind draws a technique it checks (the typo, scenario or rewrite
check); the eval type's attacks draw none.
The flat keywords (`include_hacking`, ..., `entry_types`) are also accepted
here, with a `DeprecationWarning`, and applied over the adversarial kind (see
[deprecated keywords](#deprecated-flat-keywords)). A key that only the
augmentation section has, such as `family_weights`, raises here.

### `build`

| Key | Default | Meaning |
|---|---|---|
| `include_inputs`, `include_jailbreaks`, `include_variations`, `include_outputs`, `include_paraphrases` | `true` | Which parts to merge. An eval build takes no variations, and without inputs its jailbreak file has no benign negatives |
| `output_path` | `Datasets/complete_dataset.csv` | The merged file of a training build; an eval build raises on it |
| `inputs_path` | `Datasets/constitution_inputs_merged.csv`, else the category files | The inputs part |
| `jailbreak_path`, `variations_path`, `responses_path` | the stages' default files | The other parts |
| `paraphrase_inputs_path`, `paraphrase_outputs_path` | the stage's default files | The paraphrase parts |
| `exclude_from` | none | Eval only: a list of run folders or CSVs, such as a training run's; the eval files leave out every row they hold by id or by normalised text, keep and count the rows from a constitution entry they have too, and the record counts both ([build](stages/build.md#leaving-out-what-a-training-run-holds)) |

A path key moves one stage's file only. Later stages read the default paths
unless given their own: the build has a path key per part, and the paraphrase
stage reads `Datasets/output_responses.csv` from its default place.

## The generation axes

`configs/generation_axes.json` says what each input sample (`inputs`) and
each response (`outputs`) is asked to look like. Each stage deals one spec
per call, a value of each axis, evenly over each `(category, entry_type)`
group at the table's weights, or [log-weighted](#log-weighting)
([inputs](stages/inputs.md#the-generation-axes),
[outputs](stages/outputs.md#unit-and-ledger-key)); the prompt shows it in
`{sample_spec}` or `{response_spec}`, and the row records it in a column per
axis. Paraphrase rows carry no axis columns, by design: a paraphrase's
`input_id` names the row whose spec it rewords.

| Stage | Axis | Values (weight) |
|---|---|---|
| inputs | `length` | `fragment` .15, `one_liner` .30, `short` .30, `long` .20, `very_long` .05 |
| inputs | `register` | `casual` .35, `neutral` .30, `formal` .15, `technical` .10, `emotional` .10 |
| inputs | `writer` | `anonymous` .30, then .05 each: `student`, `parent`, `teacher`, `hobbyist`, `developer`, `researcher`, `journalist`, `fiction_writer`, `caregiver`, `healthcare_worker`, `small_business_owner`, `office_worker`, `tradesperson`, `gamer` |
| inputs | `framing` | `direct_request` .35, `question` .30, `advice` .20, `statement` .10, `opinion_or_validation` .05 |
| inputs | `directness` | `explicit` .55, `implicit` .30, `euphemistic` .15 |
| inputs | `focus` | `single` .75, `mixed` 0 |
| inputs | `domain` | not dealt: the entry's subcategory, shown as it is |
| outputs | `length` | `brief` .20, `short` .35, `medium` .30, `long` .15 |
| outputs | `format` | `prose` .35, `list` .20, `steps` .20, `sections` .10, `table` .05, `code` .10 |
| outputs | `register` | `neutral` .40, `friendly` .30, `formal` .15, `technical` .15 |

Each value holds the `label` and one-sentence `hint` the prompt shows
(`Label: value label — hint`, one line per axis), its `weight` (its share
within the axis; 0 leaves it out) and optionally `not_with`, values of other
axes it is never dealt with (`advice` and `opinion_or_validation` with a
`fragment`; `steps`, `table` with a `brief` answer, `sections` with a
`brief` or `short` one). An axis with `from` takes the entry's field
instead. Each axis carries a `dimension`, a generation tag of
`dimensions.json` (below). A key starting with `_` is a comment.

The `axes` param of each stage overlays the weights: `{axis: {value:
weight}}`, a value not named keeping its weight; an axis as `null` to leave
it out; `false` for no axes, and the placeholder renders empty. Weights
only: change a label or a hint in your own copy of the file
([your own config folder](#your-own-config-folder)). A stage's plan file
records its `seed` and weights, so a resume with others raises, naming
`resume=False`.

### Log weighting

`axes_weighting` `"log"` spreads a stage's deal more evenly over each axis's
values: after the `axes` overlay, each value of weight w above 0 is weighted
1 + ln(w / m), m the axis's least weight above 0, normalised per axis. The
values keep their order, and none is as common or as rare as the table makes
it. A weight of 0 stays 0, and an axis with one value above 0 (a pinned one,
such as eval's `directness`) is unchanged, so the same axes vary and the
[dimension check](#dimensions-and-the-runs-check) sees them as the stage
deals them. The `training` type sets it on `inputs`, to spread training data
more evenly over the axis values; `eval`, and both types' `outputs`, keep
`"table"`. The plan file records the weights dealt, so a resume with another
weighting raises as another weight does. The packaged inputs, log-weighted:

| Axis | Values (weight) |
|---|---|
| `length` | `fragment` .190, `one_liner` .252, `short` .252, `long` .216, `very_long` .090 |
| `register` | `casual` .290, `neutral` .271, `formal` .181, `technical` .129, `emotional` .129 |
| `writer` | `anonymous` .166, then .060 each of the 14 roles |
| `framing` | `direct_request` .272, `question` .258, `advice` .221, `statement` .157, `opinion_or_validation` .092 |
| `directness` | `explicit` .461, `implicit` .339, `euphemistic` .200 |
| `focus` | unchanged: `single` .75, `mixed` 0 |

### Dimensions and the run's check

`configs/dimensions.json` names the dimensions a text varies along, each on
one side: `augmentation` (what a jailbreak or variation family changes in a
finished text: `language`, `surface_noise`, `encoding`, `container`,
`role_framing`, `request_shape`, `answer_shape`, `example_priming`) or
`generation` (what an axis shapes as the text is written: `authorship`,
`request_intent`, `length`, `register`, `focus`, `domain`, `answer_length`,
`answer_layout`, `answer_tone`). Each tag has a one-sentence `definition`.
Every family of `combination_spec.json` carries an augmentation tag
([techniques.md](techniques.md#families-and-techniques)), checked at import,
and every axis a generation tag, checked when the table loads. A generation
tag may list in `same_as` the augmentation tags that are the same dimension,
and in `undecided` those yet to be ruled on, checked as the same until then.

Before any stage runs, `run_pipeline` checks every axis that varies (one
with `from`, or with two or more values of weight above 0), each stage's
axes as the stage gets them (the type file's, the params file's over them),
against the families each kind of the augmentation section draws, whichever
stages the run lists. An axis and a family of one dimension stop the run,
naming the axis, the families per kind and the two ways out: give the axis
one value or leave it out (`{"focus": null}`), or deny the families in each
kind. A pinned axis varies nothing, so the eval type's `directness` passes.
The shipped table has one such pair: `focus` is undecided against
`request_shape`, `container` and `example_priming`, so `mixed` is at 0 and
every sample is `single`; a params weight on `mixed` stops a run whose kinds
draw those families. `authorship` (the `writer` axis) and `role_framing` are
ruled distinct: who writes a message is not a role or scenario the message
sets up, so the writer axis varies beside the persona, framing, cognitive,
persona_template and impersonation families and the natural kind's benign roles
(`tests/test_dimensions.py` records the ruling, and checks both dataset
types and both example recipes).

The outputs stage's answer mode (`harmful` or `helpful`) is no axis and
carries no tag: it picks the response's prompt, checker and label, the
response's side of `entry_type`, and no family varies whether a model
complies (`answer_format` varies what a message asks of the answer's form),
so `check_dimensions` has nothing to say about it.

## The augmentation section

The jailbreak stage runs two **kinds**:

- `adversarial`: jailbreak attacks, written to `jailbreaks.csv`;
- `natural`: natural variations of the inputs, written to `variations.csv`.

The section is `{"adversarial": {...}, "natural": {...}}`. Each kind given
overlays its default: the packaged one from
`configs/jailbreak/augmentations.json`, with the [dataset
type's](#dataset-types) kind over it. A kind left out is the default; a kind
given as `null` is off. With both off the stage
writes nothing. A key starting with `_`, in the section or in a kind, is a
comment. `recipe.load_recipe` resolves the section, so every error below
stops a run before the first stage.

A resumed run keeps each kind's plan: a changed pool, `rounds`,
`family_weights`, `layer_rates` or `seed` raises an error that names them
and `resume=False` ([stages/jailbreaks.md](stages/jailbreaks.md#how-it-resumes)).
A pool that changed only because the library added or retired techniques
since the plan was made is no change of yours: the plan is kept, with a
WARNING. The [release notes](#release-notes) at the end of this section say
what a run you already have must do, from the one re-plan to the new files.

### Keys

Both kinds:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entry_types` | list or `null` | `null` | Augment only inputs of these entry types; `null` for all. A kind whose entry types match no input, or whose other keys leave its pool empty, logs a WARNING and writes nothing |
| `translation` | bool | `true` in the adversarial kind, `false` in the natural one | Whether the kind draws the `translation` family. The family lists do not change it; `layers.deny`, `techniques.deny` and `pure_only` still apply |
| `layers` | `{"allow": [...], "deny": [...]}` | `{"deny": []}`, the `rewrite` layer off | `deny`: layers left out. `allow`: layers off by default to turn on, `[]` for none. A layer the spec lists in `layers_off_by_default` (`rewrite`) is left out unless `allow` names it, whatever `deny` says; `allow` may name only such a layer, one with a technique the kind draws (not `rewrite` in the natural kind), and a layer may not be under both |
| `families` | `{"allow": [...] or null, "deny": [...]}` | adversarial: `{"allow": null, "deny": []}`; natural: the natural families | Families admitted and removed |
| `techniques` | `{"allow": [...] or null, "deny": [...]}` | adversarial: `{"allow": null, "deny": []}`; natural: three techniques allowed, three denied | Techniques admitted and removed. Either `allow` names only what the spec's `kinds` let the kind draw. A technique the spec retired (`deprecated`) is drawn only when `allow` names it, with a WARNING in the log (and a `DeprecationWarning`) |
| `pure_only` | bool | `false` | Drop every technique that calls a model or needs benign data |
| `family_weights` | `{family: number > 0}` | `{}` | A family's share within its layer when the plan deals families; 1 each by default |
| `iterations` | int >= 1 | `1` | Units per input, numbered by `iteration` |

Adversarial only:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rounds` | `null`, `"escalation"` or a list | `null` | The sampler settings per round; see [rounds](#rounds) |
| `max_complexity` | int >= 0 | `6` | Budget mode: the most total complexity (each technique scores 0 to 3) |
| `max_obfuscations` | int >= 0 | `2` | Budget mode: the most obfuscation families, the request obfuscation included |
| `layer_rates` | `{layer: 0..1}` | `{}`, the spec's rates | Target share of units that take a `rewrite` (when `layers.allow` names it; the spec's rate is 0), `hacking`, `manipulation` or `requests` technique, or, with `allow_request_obfuscation`, a `request_obfuscation` |
| `allow_request_obfuscation` | bool | `true` (`false` in the eval type) | Allow one more technique after the request technique, for a unit that drew one (a unit without a request takes none), at the `request_obfuscation` rate over the group's units: a light `encode` technique (the light `to_sensitive_words_encode_*` among them), or a translation when the pool has one. It is drawn before the other obfuscations, from what the earlier picks leave, and takes one of the `max_obfuscations` slots and its complexity from them; with `max_obfuscations` 0 there is none |
| `budget` | `"unit"`, `"group"` or `null` | `null`, as `"unit"` | `"unit"`: each unit may spend up to `max_complexity`. `"group"`: a unit may also spend what the group's earlier units left unspent of it, up to `max_unit_complexity`, so the group's mean stays at or below `max_complexity` ([the group budget](#the-group-budget)). A key of the kind, not of a round; the natural kind refuses it |
| `max_unit_complexity` | int >= `max_complexity` | `max_complexity + 3` | With `"budget": "group"` only: the most one unit may spend. It may not be below `max_complexity` or any round's `max_complexity`; the natural kind refuses it |

Natural only:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `exact_techniques` | int >= 1 | `1` | Techniques per variation |

The packaged natural kind allows the families `typos`, `casual_writing`,
`noise`, `neutral_container`, `answer_language`, `answer_format`, `asking`,
`indirect`, `smalltalk_prefix`, `smalltalk_suffix`, `benign_role`; allows the
techniques `to_leetspeak_basic`, `to_adversarial_suffix_punctuation`,
`to_adversarial_suffix_emoji`; and denies `to_refusal_then_comply`,
`to_apology_prevention`, `to_prefix_injection`. The spec's `kinds` narrow
it further (step 6 below): its pool is 72 techniques
([techniques.md](techniques.md#the-natural-pool)). A variation that leaves
its input unchanged is rejected.

### How a kind's pool is built

From every registered technique, in this order:

1. remove the layers in `layers.deny`, and those the spec has off by
   default (`rewrite`) unless `layers.allow` names them;
2. when `families.allow` or `techniques.allow` is a list, keep only the
   techniques of an allowed family or with an allowed name (with the switch
   on, `translation` counts as an allowed family unless `techniques.allow`
   names some of its techniques: then only those are kept);
3. remove the families in `families.deny` (`translation` aside), and
   `translation` when the switch is off;
4. remove the techniques in `techniques.deny`;
5. with `pure_only`, remove every technique tagged `requires_llm` or
   `requires_benign`;
6. keep only the techniques whose `kinds` in `combination_spec.json` hold
   the kind: a technique's own, else its first family's, and
   `["adversarial"]` for a family without any. So the natural kind draws
   only what the spec marks natural ([extending.md](extending.md#add-a-jailbreak-technique)
   says how to mark a technique). Every technique holds `adversarial` but
   those of `smalltalk_prefix`, `smalltalk_suffix` and `benign_role`, marked
   `["natural"]` alone, so this step takes only those three families out of
   the attacks' pool.

[techniques.md](techniques.md) lists every layer, family and technique name.

### Overlay rules

- A value given replaces the default's whole value. A list replaces a list.
- `layers`, `families` and `techniques` are overlaid list by list. An `allow`
  or `deny` given replaces the default's list of that name and keeps the
  other.
- The layers the spec has off by default (`layers_off_by_default` in
  `combination_spec.json`: `rewrite`) are added to `layers.deny` after the
  overlay unless `layers.allow` names them. So `{"deny": ["manipulation"]}`
  leaves out manipulation and keeps the rewrite layer off, and only
  `{"allow": ["rewrite"]}` turns it on. A `layers.allow` naming a layer
  that is on already raises.
- A `deny` given without an `allow` also takes its names out of the default's
  `allow`. `"natural": {"families": {"deny": ["typos"]}}` is the natural
  default without typos. A deny that empties the default's allow raises:
  write the allow as well.
- An `allow` list may not be empty: write `null` for all, or name what to
  leave out under `deny` (`layers.allow` aside: a list, `[]` for none). A
  name may not be under both.
- The natural kind needs an allow list.
- An allow list may name only families and techniques whose spec `kinds`
  hold the kind. A natural allow list naming one the natural kind does not
  draw (`structural`, `temporal`, `continuation`, `to_prefix_injection`,
  ...) raises, naming it: a `config_dir` copy of `augmentations.json` made
  before these left the natural kind does, so take them out of its
  `families.allow`.
- An allow list or `family_weights` naming a family or technique of a layer
  the spec has off by default (`rewrite`) raises unless `layers.allow`
  names the layer, saying to write `"layers": {"allow": ["rewrite"]}`: the
  pool would drop it, and a kind that names nothing else would draw
  nothing. One of a layer the kind's `layers.deny` names is left out with a
  WARNING (the eval type's lists name `framing`, so an eval run that denies
  `hacking` draws the other 31), and a `family_weights` entry of such a
  layer raises.
- `natural_only_families`, at the top of `augmentations.json`
  (`casual_writing` in the packaged file), are added to the adversarial `families.deny` unless the
  adversarial `families.allow` names them. A family the spec marks
  `["natural"]` alone (`smalltalk_prefix`, `smalltalk_suffix`,
  `benign_role`) needs no entry there: no attack may draw it, and an
  adversarial allow list that names one raises.

### The translation switch

`"translation": false` keeps the `translation` family out of a kind, and
`true` puts it in: the attacks draw it as any family, and the natural kind
adds it to its allow list (the spec marks the family natural). No family
list changes this; a
`techniques.deny` still removes the languages it names, and a
`techniques.allow` that names some languages narrows the family to those
(the eval type's eight low-resource languages).

With the switch `false`, the translation an allow list names (the family or
its techniques) and a `family_weights` entry for it are left out of the
resolved kind, with a WARNING naming them:
`augmentations.<kind>.translation is false, so the translation it names is
left out: [...]`. So `"adversarial": {"translation": false}` over the eval
type is a one-key override: its pool keeps the other 28 techniques. An allow
list left empty by this is dropped when the other list still names
something; when nothing but translation was allowed, the kind raises.

The packaged attacks have it `true`, drawing all 23 languages, and the
natural kind `false`; the eval type narrows its attacks to the eight
low-resource languages. They run on the `translation` role's default,
`translategemma-27b`, a local vLLM model
(a GPU: about 34 GiB planned, beside the FP8 24B on one 80 or 96 GB card).
It translates paragraph by paragraph; each piece is checked, then translated
back and judged by the generation model, up to 4 rounds
([stages/jailbreaks.md](stages/jailbreaks.md#models) gives the calls). The
translation model is planned and loaded only when a kind's pool holds a
translation technique (a rewrite technique calls the generation model, as
the other model-written techniques do). An allow list admits the whole family while the switch
is on, unless `techniques.allow` names some languages: a kind meant to draw
only the techniques it lists says `"translation": false`.

A run without a GPU keeps translation off (`"adversarial": {"translation":
false}` in the recipe's `augmentations`), or names the hosted chat
translator: `"models": {"translation": "deepseek-v3.2"}`.

### Rounds

`rounds` sets the adversarial kind's units per input and each round's sampler
settings. Each round is merged over the kind's budget keys, key by key: a
round's `layer_rates` replaces the kind's whole and is read over the spec's.

| Value | Rounds |
|---|---|
| `null` | `iterations` rounds of the kind's budget settings |
| `"escalation"` | The four rounds of `default_escalation_schedule()`, below |
| a list of objects | One round per object. Keys: `exact_techniques`, `max_complexity`, `max_obfuscations`, `layer_rates`, `allow_request_obfuscation` |

With `rounds`, `iterations` is the number of rounds; giving both with
different values raises. A round with `exact_techniques` picks exactly that
many compatible techniques and ignores `max_complexity`,
`max_obfuscations`, `layer_rates`, `allow_request_obfuscation` and the
[group budget](#the-group-budget). A round without it picks by budget.

The escalation schedule:

| Round | Settings |
|---|---|
| 1 | `exact_techniques: 1` |
| 2 | `exact_techniques: 2` |
| 3 | `max_complexity: 6`, `max_obfuscations: 2`, rates hacking 0.7, manipulation 0.3, requests 0.8, request_obfuscation 0.5, `allow_request_obfuscation: true` |
| 4 | `max_complexity: 9`, `max_obfuscations: 3`, rates hacking 0.9, manipulation 0.1, requests 0.9, request_obfuscation 0.7, `allow_request_obfuscation: true` |

A unit holds a hacking or a manipulation technique, never both, so the two
rates of a round sum to 1.

### Rates and weights

The plan deals techniques per `(category, entry_type, iteration)` group
rather than drawing each unit alone. In budget mode each of the `hacking`,
`manipulation` and `requests` layers joins a share of the group's units equal
to its rate: `layer_rates` over the spec's (hacking 0.5, manipulation 0.33,
requests 0.7, request_obfuscation 0.5). The `rewrite` layer, once a kind's
`layers.allow` names it, joins first at its rate (the spec's 0); a count round reads no
rate and may draw a rewrite technique its pool holds. Within a layer, families are dealt by
count over `family_weights`, so `{"persona": 2}` gives `persona` twice the
share of each other hacking family; within a family, its techniques are
dealt evenly among those a unit may take (the complexity budget still
favours a family's cheaper techniques). `family_weights` may name only
families the kind's pool draws. In a budget round the ten heavy full-text
encodings (`to_base64`, `to_morse` and the others of `encode_weight` heavy
that encode the whole text) are dealt in a slot of their own beside
`encode`, at the spec's `heavy_encoding_weight` (0.2) times `encode`'s
weight, so they are drawn about a fifth as often as a family: the slot is
not a family, so `family_weights` cannot name it (`"encode:heavy"` raises
as any name the kind does not draw); weight `encode` to move both. It is a
quota in every budget round, not a floor: a round whose budget would have
dealt them more as `encode`'s members, the escalation's last, draws fewer
(the sixth change of the [eval release](#release-notes)). A count round deals
them as `encode`'s members.
[internals/jailbreak.md](internals/jailbreak.md#balanced-dealing)
explains the dealing.

`hacking` and `manipulation` exclude each other, so their rates should sum to
at most 1; beyond it, `manipulation` falls short of its rate and the plan
logs a WARNING. With `allow_request_obfuscation`, `request_obfuscation` is
dealt the same way, over the group's units; its families (`encode` and
`sensitive_words`, their light members only, and `translation`) are dealt
among themselves at their `family_weights`, counted apart from the
obfuscation layer's.

### The group budget

A budget round lets each unit spend up to `max_complexity`. A unit that
draws cheap techniques leaves the rest unspent, so a group's units spend
less than `max_complexity` on average (5.64 of 6 in the table below). With
`"budget": "group"` the units of a `(category, entry_type, iteration)` group
share it: a unit may spend `max_complexity` plus what the group's earlier
units left unspent of it, up to `max_unit_complexity` (`max_complexity + 3`
when left out). The group's mean complexity per unit then stays at or below
`max_complexity`, and no unit goes above the ceiling. The first unit of a
group has no bank, so it spends up to `max_complexity`.

```json
{"adversarial": {"budget": "group", "max_unit_complexity": 9}}
```

- It is off by default and no packaged kind or dataset type turns it on:
  `"unit"`, `null` or the keys left out give the rounds, the plan file and
  the draws of a run without the option.
- `max_unit_complexity` needs `"budget": "group"` and may not be below
  `max_complexity` or any round's `max_complexity`. Both are keys of the
  kind: a round may not name them. With `rounds`, each round is a group of
  its own with its own `max_complexity` (and, with no `max_unit_complexity`,
  that plus 3).
- A count round (`exact_techniques`) has no budget and ignores the keys; the
  natural kind refuses them.
- The keys reach a plan only when set. A plan kept from before resumes as it
  was; a section that turns the option on or off over a kept plan raises on
  resume, naming `rounds`: run the kind with `resume=False`.

What it does, measured over 3,000 units of the default adversarial kind in
three groups (`max_complexity` 6, ceiling 9; `python -m tests.jailbreak.simulate
3000 budget` prints seed 42). Each cell is seed 42, then the range over
seeds 42 to 46:

| | `"unit"` | `"group"` |
|---|---|---|
| Mean complexity per unit | 5.64 (5.63 to 5.65) | 6.00 |
| Highest complexity of one unit | 6 | 9 |
| Units above 6 | 0 | 780 (745 to 780) |
| Obfuscations per unit | 1.47 (1.47 to 1.48) | 1.62 |
| Draws of complexity 3 or more | 1,451 (1,430 to 1,456) | 1,578 (1,565 to 1,578) |
| `encode`: the top member's share of the family's draws, as a multiple of an even share | 8.07 (7.58 to 8.26) | 4.90 (3.88 to 4.90) |
| `sensitive_words`: the same | 3.66 (3.53 to 3.89) | 1.64 (1.64 to 1.96) |
| Of complexity 3 or more, drawn: `encode` (of 10) | all 10, 3 to 6 times each (3 to 6) | all 10, 4 to 6 times each (3 to 6) |
| Of complexity 3 or more, drawn: `sensitive_words` (of 4) | all 4, 18 to 20 times each (17 to 22) | all 4, 36 to 37 times each (34 to 37) |

The group budget evens out the cheap members of `encode` and
`sensitive_words`, which the unit budget favours because a cheap one fits
more units (the top member of each is the complexity-1 leetspeak), and
spends the mean it is given. The heavy members stay rarer than those of
complexity 2 under both budgets (each heavy `encode` member 3 to 6 times
under the unit budget at seed 42, against 30 to 32 for one of complexity 2):
one of complexity 3 fits only a unit whose earlier picks leave 3 points, and
`fsh` and `dap` refuse the heavy `encode` members (not the `sensitive_words`
ones, which are drawn more). That skew toward the cheaper members is
intended; the ten heavy full-text encodings are dealt in a slot of their own
at a fifth of a family's weight (above), without which they were drawn 0 to
3 times each. The extra allowance goes to the obfuscations, the request
obfuscation's included (2.51 complexity per unit against 2.87; the other
layers spend the same). The draws of complexity 3 or more include 112 of the
two `selected_many` variants under both budgets (108 to 112 over the seeds).

### Examples

Escalate the attacks on harmful inputs only, weight one family, no
variations:

```json
"augmentations": {
  "adversarial": {
    "entry_types": ["harmful", "dual_use_harmful"],
    "families": {"deny": ["ascii_art"]},
    "rounds": "escalation",
    "family_weights": {"persona": 2}
  },
  "natural": null
}
```

Variations only, two per benign input, no attacks:

```json
"augmentations": {
  "adversarial": null,
  "natural": {"entry_types": ["benign", "dual_use_benign"], "iterations": 2}
}
```

### Deprecated flat keywords

`generate_jailbreaks` still takes twelve flat keywords, read into the
adversarial kind with a `DeprecationWarning`. The same keys are read, and
warn, at the top level of a recipe's `augmentations` (then the natural kind is
the default) and in `params["jailbreaks"]`.

| Flat keyword | Section key |
|---|---|
| `include_hacking`, `include_manipulation`, `include_obfuscation`, `include_requests` | `false` adds the layer to `layers.deny` |
| `include_translation` | `translation` |
| `pure_only`, `max_complexity`, `max_obfuscations`, `iterations`, `entry_types` | the same key |
| `sampling_probs` | `layer_rates` |
| `settings_per_iteration` | `rounds`, each round's `sampling_probs` as its `layer_rates` |

### Release notes

**Upgrading from the previous release (spec version 1.0, now 1.3).** The
jailbreak library grew from 151 techniques in 23 families to 244 active ones
(251 registered, 7 retired) in 36 families and five layers. What that asks of
a run you already have, and what you will see; each point has its note below.

- **One re-plan is planned: the adversarial kind.** The packaged attacks now
  deal a request obfuscation to half their units, so their rounds gain
  `allow_request_obfuscation: true`, and an adversarial plan kept from before
  raises on resume, naming `rounds`. Run the adversarial kind once with
  `resume=False`, or set `"allow_request_obfuscation": false` in the kind to
  keep the plan as it is.
- **A kept plan warns when the library grew.** Resumed after the library
  added or retired techniques, or changed its version or its rules, either
  kind keeps its plan with one WARNING that names them and counts the
  pending units whose combination today's rules refuse. It does not draw
  the new techniques until it is planned again with `resume=False`, still
  replays the retired ones, and its rows record the spec version the plan
  was drawn at. A plan file written before this release records no rules,
  so its first resume warns once.
- **The natural kind's pool changed.** The spec's `kinds` narrowed it to the
  techniques marked natural, and spec 1.2 then grew it, to 71 techniques in
  13 families by default (46 in 8 before); spec 1.3 added the urgent ask, to
  72 (95 with `"translation": true`). A
  natural plan made before the narrowing raises on resume: run the natural
  kind with `resume=False`. One made since resumes with the WARNING above.
  A `config_dir` copy of `augmentations.json` made before spec 1.2 does not
  draw the new natural families, and the run says so at INFO.
- **A `config_dir` copy of `eval.json` widens the eval pool.** A copy made
  before this release (what `scaffold_setup` wrote) keeps its old deny list,
  so its eval attacks draw 57 techniques, not the pinned 36: the 21 the library
  added to the families it allows, `to_atbash`, `to_base32`, `to_base85`,
  `to_binary`, `to_caesar_cipher`, `to_dap_random_many`,
  `to_dap_selected_many`, `to_fsh_random_many`, `to_fsh_selected_many`,
  `to_fullwidth`, `to_hex`, `to_html_entities`, `to_lookalike_letters`,
  `to_math_styled`, `to_reversed_characters`, `to_reversed_word_order`,
  `to_reversed_words`, `to_url_encoding`, `to_use_future_tense`,
  `to_use_historical_era` and `to_zero_width_noise`. An eval run names them
  in a WARNING (`configs: the eval attacks of ... draw 21 technique(s) the
  library's eval type does not ...`). Delete a copy you did not change, or
  write it again with `scaffold_setup(..., overwrite=True)`
  ([refreshing a stale copy](#your-own-config-folder)); to keep your edits,
  add the 21 names to its adversarial `techniques.deny`. The packaged type
  now names `to_use_past_tense` under `techniques.allow` and no longer
  allows the `temporal` family whole, so its pool, rounds and draws are the
  same but its `dataset_type_hash` in run records is another.
- **Fresh plans draw differently.** The same seed gives other combinations
  than at spec 1.0: the new techniques are dealt, the seven retired ones are
  not, a technique is drawn only for the inputs it applies to, several
  rules are new, the layers are dealt with a jitter per unit, the request
  obfuscation is decided only for a unit that drew a request, and its
  families are counted apart from the obfuscation layer's, so the attacks
  draw every technique of their pool. The default attack pool is 214
  techniques in 29 families (142 in 22 before; 178 in 24 at spec 1.2). The
  `eval` type's pool is unchanged, at 36 techniques in 10 families, but its
  draws moved: the ASCII-art cap changes the font `to_ascii_art` draws for
  texts of about 40 characters or more, and the rules of the eval release
  (below) change which pairs of the pool one combination may hold.
- **The eval release moves the eval type's draws.** The pool is the same 36
  techniques, but a translation now refuses every `encode` technique and
  ASCII art the heavy full-text encodings, so the pool loses 44 of its 335
  pairs, and the manipulation layer now applies last, around the finished
  question, so 40 of the 291 pairs left (each `fsh` or `dap` technique
  beside a wrapper or a request) write another text: eval files made before
  this release are not comparable with those made after, and the spec's
  version stays 1.3, so two things tell them apart: a row's `technique`
  column lists its techniques in the order they applied (a manipulation
  technique before the obfuscations and the request until this release,
  after them since), and the `rules` id of the jailbreaks run record
  (`Datasets/jailbreaks.run.json`, by kind) differs. A kept eval plan
  replays its recorded combinations in their recorded order and warns on
  resume that the rules changed; plan the eval type again with
  `resume=False` to take them. Two more rules and a dealing slot change the
  training attacks only: `sensitive_words` and `tokenbreak` refuse each
  other; the request obfuscation, when it translates, keeps a suffix, a
  wrapper and a container out of the unit (which makes the translation the
  unit's only obfuscation); and a budget round deals the ten heavy full-text
  encodings in a slot of their own at the spec's `heavy_encoding_weight`,
  0.2, a quota rather than a floor (about 45 draws of the ten in 3,000 units
  of the escalation's last round, where they were about 121; the note on the
  eval release, below).
- **Stricter checks, and what they skip.** An allow list or `family_weights`
  that names a family or technique of a layer the kind leaves off raises
  (the `rewrite` layer needs `layers.allow`); a `config_dir`
  `augmentations.json` whose default kind has an unknown key is warned about,
  one with a missing key or a wrong shape raises, naming the file; the spec,
  its data rows and the template pools are checked at import, with messages
  that name the file. An input id given twice is planned once, with a
  WARNING (two entry types for one id raise); a kind whose pool is empty is
  skipped with a WARNING, as a kind with no matching entry type is; and the
  stage plans and loads a separate `check` model only when a kind's pool
  holds a technique that is checked (the eval type's attacks hold none).
- **A new layer, off by default.** The `rewrite` layer holds 16 model-written
  techniques (persuasion, tense rewrites, model distractors; attacks only).
  No packaged section allows it, so no pool, plan or draw changes unless a
  kind writes `"layers": {"allow": ["rewrite"]}` (its note below).
- **More model calls, and new files.** A typo rewrite and a cognitive or
  persona scenario are now checked by a model, which costs calls (the typo
  and scenario notes), for the pending units of a kept plan too, and the
  request obfuscation adds translations to a fresh attack plan (its notes);
  an invented persona asks for its persona first, and `to_payload_split_semantic`,
  `to_word_art` and the two wording techniques call the generation model too
  (their notes). Each kind
  writes a `<kind>.variants.csv` after a run, and its plan file records the
  `spec_version` its pool was drawn at.

**Release note: the request obfuscation on.** The packaged adversarial kind
now sets `allow_request_obfuscation` true. A budget round deals, at the
spec's `request_obfuscation` rate (0.5), a light encoding or a translation
applied after the request technique, to the finished request, to the units
that drew a request technique; it takes one of the unit's `max_obfuscations`
slots and its complexity, so a unit holds no more obfuscations than before.
Measured with `tests/jailbreak/simulate.py` over 3,000 units of the default
kind in three groups, with the step on and with it off
(`"allow_request_obfuscation": false`), at seed 42 and, in brackets, the range
over seeds 42 to 46: 50.0% of the units take one (1,500 of 3,000; 1,499 at
two seeds), and no unit takes one without a request. Of these, a third each
(about 16.6% of the units each, 498 to 501): a translation, a light `encode`
technique and a light `to_sensitive_words_encode_*`, since the step's
families are dealt among themselves, counted apart from the obfuscation
layer's (the next note). The obfuscations per unit are 1.47 (1.47 to 1.48)
with the step and 1.57 (1.56 to 1.58) without: it holds a slot rather than
adding one, and a translation drawn this way keeps a suffix, a wrapper and a
container out of the unit's other obfuscations (the eval release note, below).
The units holding a translation go from 11.9% (11.8% to 12.0%) to 23.9%,
the light `encode` techniques per unit from 0.12 (0.12 to
0.12) to 0.24, and the model-calling techniques per unit, translation aside,
from 1.01 (1.01 to 1.02) to 0.93: the step takes slots from the
other obfuscation families. The extra model cost is mostly the translations the
step adds: each runs on the translation model, a machine-translation pass
per piece with its checks, back-translations and the generation model's
judge, now over the request's wording as well as the prompt, for about 17%
of the rows; each light sensitive-word encoding makes its one marking call to
the generation model, as it did as an obfuscation. The rate is a lever:
`"layer_rates": {"request_obfuscation": 0.25}` in the adversarial kind halves
it, and `"allow_request_obfuscation": false` turns it off. The kind's rounds
gain the key, so an adversarial plan kept from before, whose section left the
key out, raises on resume, naming `rounds`: run the adversarial kind once with
`resume=False` (the natural kind's plan is not affected). The eval type sets
it false and draws by count, which never reads it: its pool and its draws do
not change.

**Release note: the request obfuscation counted apart.** The planner counts
the request obfuscation's families apart from the obfuscation layer's. Until
commit `ec8bb42` it counted them together: the request obfuscation's picks
made `encode` and `sensitive_words` the families dealt most, and since the
dealer takes the family dealt least, the layer drew them only when no other
family fitted, and 14 techniques of the default attacks were never drawn in
3,000 units (seed 42, three groups): `to_base64`, `to_base85`, `to_hex`,
`to_binary`, `to_morse`, `to_braille`, `to_ascii_ordinal`,
`to_unicode_escape`, `to_sensitive_words_split`, `to_synonym_substitution`,
`to_sensitive_words_variables` and the base64, ascii and unicode
`to_sensitive_words_encode_*`. Since that commit every technique of the pool
is drawn, and a test holds it over one plan of 4,000 units under each budget.
The request obfuscation deals a translation a third of the time (4% of its
picks at the parent of `ec8bb42`), the units holding a translation going from
26.4% to 34.2% (792 to 1,026 of 3,000, measured at the two commits), and
`family_weights` apply to its families as to the layer's. This landed with
the re-plan the request obfuscation already asks (the note above): a kept
plan replays unchanged. A kind with `allow_request_obfuscation` false and the
`eval` type, which draws by count, draw as before.

**Release note: kept plans tolerate the library's growth.** A jailbreak
plan file now records the spec version its pool was drawn at
(`spec_version`; one written before is read as `"1.0"`). A kept plan
resumed after the library added techniques (each with a later `since`) or
retired some (a later `deprecated`) keeps its plan with one WARNING that
names them, where it used to raise: the plan does not draw the new
techniques until it is planned again with `resume=False`. A changed seed,
`rounds`, `family_weights`, `layer_rates`, or a pool your section changed,
still raises.

**Release note: the rules a plan was drawn by.** A jailbreak plan file now
also records `rules`, a content id of what the plan's draws read off the
spec (the layer order, caps and rates, the manipulation list, the
request-obfuscation lists when a round draws that step, the heavy
encodings' weight when a budget round's pool holds one, the pool's families
and each pool technique's tags), since the rules
can change while the spec version does not (as they did twice within 1.3).
The one WARNING of a kept plan is now given also when its spec version or
its rules are not today's (and when its plan file, written before this
release, records no rules), and it counts the pending units whose recorded
combination today's rules refuse; those still run as planned (plans of 600
default-attack units and of 2,400 escalation units drawn by commit `328cb99`,
spec 1.2, in three groups, hold 91 and 348 such units at seed 42, 134 and 333
at seed 43). A row's `combination_spec_version` is now the spec version
its plan was drawn at, not the library's: the rows a resume writes for a
plan kept from 1.2 say 1.2. The run record's `spec_version` is still the
library's. Nothing raises that did not, and no draw changes.

**Release note: layers dealt with a jitter per unit.** The planner deals a
layer over a group while its picks are below `floor(rate x position +
jitter)`, the jitter now drawn afresh for each unit (seeded by the group,
the layer and the position) where it was one offset per group and layer. With
one offset a layer's picks repeated every `1 / rate` units, so two layers of
one rate fell on the same units, or almost never on the same, for a whole
group, by the seed. At commit `590babe` (900 units of the default attacks in
three groups), hacking and the request obfuscation (both 0.5) met on 68% of
the hacking units at seeds 1 and 2, 100% at seed 42 and 2% at seed 43, and
`ascii_art`, which refuses both a hacking technique and a translation, was
drawn 33, 35, 51 and 8 times. Now they meet on 49% to 55% of the hacking
units (seeds 1, 2 and 42 to 46) and `ascii_art` is drawn 59 to 64 times.
Every layer keeps its rate and a group still gets `rate x size` picks
rounded down or up. A fresh budget plan draws differently from the same
seed; a kept plan replays unchanged; the `eval` type, which draws by count,
deals no layer and draws as before.

**Release note: the request obfuscation only with a request.** The step
obfuscates the finished request, but it was decided whether or not the unit
had drawn a request technique; without one it was applied after the
outermost wrapper (at commit `8ccb20c`, 435 of the 1,500 units that took it
in 3,000 units of the default attacks at seed 42 had no request, 161 of them
with a structural wrapper or a neutral container). It is now decided only
once a request technique is picked, so no unit takes it without one. It is
still dealt at its rate over the group's units, so it falls on the units
that hold a request (0.7 of them by default) more often, and the default
attacks still give it to half their units (1,500 of 3,000 at seeds 42, 44
and 45; 1,499 at 43 and 46), the escalation's rounds 3 and 4 to 0.5 and 0.7
of theirs; a group whose last units hold no request may fall one short. A
fresh plan with the request obfuscation on draws differently from the same
seed; a kept plan replays unchanged; a kind with it off and the `eval` type,
which draws by count, draw as before.

**Release note: fixes to the techniques found in review.** These change the
text of a fresh plan's units and of the pending units of a kept one; finished
rows, kept plans, pools and the eval type's draws are unchanged.

- `to_stripped_accents` and its applicability keep a leading combining mark
  (a text that began with one raised, and one such input stopped the plan of
  the whole kind). A predicate that raises now stops the plan with a
  `RuntimeError` naming the technique and the input.
- The neutral containers fit the prompt to the container. In the comment and
  YAML containers every line break but a line feed is written as one; in the
  C and HTML block comments a `*/` or `-->` in the prompt gets a space before
  its last character; a YAML block scalar drops the leading spaces of the
  first line and empties the lines of only spaces; a WebVTT or SRT cue keeps
  only the non-blank lines. Only a prompt holding such a character, or
  leading spaces, is written differently.
- The marked-word parse (`sensitive_words`, `tokenbreak`, `word_art`,
  `wording`): a word named twice counts once, a word of one letter is
  refused (`malformed word list`) unless it is a Chinese character, a word
  with a joiner, a non-joiner or a zero-width space is accepted, and curly
  quotes, guillemets or an ellipsis around the words are stripped.
- `encode`'s character clusters (the separators, the reversals, the
  zero-width noise, the token split): a skin-tone modifier stays with its
  emoji, a flag is one cluster, and a decomposed Hangul syllable stays whole.
- `payload_split` writes each part as a JSON string, so a quote or a
  backslash in a part is escaped. A rewrite the check accepts but that has no
  text, or is the prompt itself, is discarded (`empty rewrite`, `unchanged
  rewrite`); a model distractor equal to the prompt is asked for again.
- The many-shot variants of `fsh` and `dap` stop with a named error (the
  unit's error row) when the benign cache holds no short pair, where they
  wrote the request alone or raised an `IndexError`.
- `to_math_styled` writes its style under `math_style=<name>`. It wrote
  `style=`, the key the continuation techniques' variants are read under, so
  a row holding both lost all its variants in the variants table; finished
  rows keep `style=`.
- A machine translation's checks no longer hold a piece under 20 characters
  to the length band (a label or a digit keeps its characters) and take a
  GlotLID `und_*` label as no verdict. The eval type's eight translations
  run these checks too and discard fewer pieces.

**Release note: larger variant pools.** Eighty-four variants join the
request pools that held four: the answer formats (`answer_as_*`,
`answer_in_markdown`, `apology_prevention`, `prefix_injection`), the answer
languages, asking, the four distractors and the six indirect tasks, which
now hold 8 each. A technique draws its variant with the unit's seeded rng, so
the same seed draws another variant than before for these techniques, in a
fresh plan and in the pending units of a kept one (the plan's combination is
kept; the text of a unit not yet written is drawn from the larger pool). The
natural kind's asking, indirect and answer-language variations read
differently for that reason. The eval type's draws and pool are unchanged:
the pools its techniques draw from (framing, continuation, structural,
`to_refusal_then_comply`, temporal) are the ones still at 4, and wait for the
next eval review ([techniques.md](techniques.md#the-eval-pool)).

**Release note: seven techniques retired (spec 1.1).**
`to_sensitive_words_star`, `to_sensitive_words_hyphen`,
`to_sensitive_words_underscore` (`to_sensitive_words_encode_separator`
draws their separators), `to_mentally_ill_persona`,
`to_politician_persona`, `to_deformed_professor_persona` and
`to_religious_figure_persona` leave every pool. A fresh plan no longer draws
them; a kept plan that does resumes with the WARNING above and still runs
them. A section that names one in `techniques.allow` draws it, with a
`DeprecationWarning`, for this release
([techniques.md](techniques.md#retired-techniques)).

**Release note: the natural pool.** The natural kind now draws only what
the spec's `kinds` mark natural (46 techniques by default when this came in,
71 since spec 1.2 and 72 since spec 1.3, below; `structural`, `continuation`, `temporal`, the high
and insane typos and the dialogue and story answer formats left it). A
natural-kind plan made before raises on resume, since its pool changed: run
the natural kind with `resume=False`. A training build leaves out the benign
and dual-use benign variations such a plan made with a technique the natural
kind no longer draws, with one WARNING that counts them. A `config_dir` copy
of `augmentations.json` whose natural `families.allow` names one of these
families raises ([overlay rules](#overlay-rules)).

**Release note: the techniques spec 1.3 adds.** Twenty-four pure template
techniques, attack-only but the urgent ask, whose text the maintainers wrote
([techniques.md](techniques.md#what-the-families-do)):

- *Persona templates* (`persona_template`, hacking, 11): a fixed frame for a
  well-known jailbreak persona around the prompt, with one of six mechanics
  drawn into it (`to_dan_persona_template`, ...).
- *Fake turns* (`fake_turns`, manipulation, 3): an invented system turn,
  assistant turn or earlier exchange with the prompt (`to_fake_system_turn`,
  `to_fake_assistant_turn`, `to_fake_history`). They take no benign data.
- *Three answer formats*: `to_dual_answer`, `to_tagged_prefix`,
  `to_answer_in_encoding`.
- *One member for each thin request family* (two for impersonation and
  temporal): `to_impersonate_authority`, `to_impersonate_developer`,
  `to_use_future_tense`, `to_use_historical_era`,
  `to_distractor_prefix_task`, `to_distractor_suffix_task`,
  `to_ask_with_urgency`.

A fresh plan of the adversarial kind draws them: the default attack pool goes
from 178 techniques in 24 families to 202 in 26, and since each adds members
or a family to deal, some units of a fresh plan draw other techniques than
they did with the same seed. A plan kept from spec 1.2 resumes with the
growth WARNING, which names the twenty-four (and the two payload splits,
the word art and the three personas, below), and does not draw them until
it is planned again with `resume=False`. New rules bind a fresh plan:
`persona_template` refuses `fsh`, `dap`, `ascii_art` and `impersonation`;
`fake_turns` refuses `continuation`, `indirect`, `framing`, `persona`,
`cognitive`, `persona_template`, `structural` and `neutral_container` (the
reason for the last two, that an invented turn is a conversation, not a
document to wrap, no longer holds since the eval release moved the
manipulation layer last; they are kept as they were, its fifth change
below), and takes the manipulation layer's block of `translation`,
`ascii_art` and the heavy encodings; `to_refusal_then_comply`,
`to_prefix_injection`, `to_dual_answer` and `to_tagged_prefix` refuse
`persona_template`; and `to_answer_in_encoding` refuses `encode`. A kept plan
replays its recorded combinations, so a rule changes none of them.

The natural kind gains one technique, `to_ask_with_urgency`: `asking` is a
natural family, and an urgent ask is a natural variation, so it takes its
family's kinds and the natural pool is 72
techniques (95 with `"translation": true`). `answer_format` is a natural
family too, but its three new members say `"kinds": ["adversarial"]`. A fresh
natural plan may draw the urgent ask; a natural plan kept from before
resumes with the growth WARNING above, which names it, and does not draw it
until it is planned again with `resume=False`. The training build's rule
that leaves out the benign variations made with a technique the natural kind
does not draw does not touch it. The `eval` type does not change either: its pool stays at
36 and its draws are those it made before. It allows none of the new
families and, of `answer_format`, names `to_refusal_then_comply` alone; of
`temporal`, which it allowed whole, it now names `to_use_past_tense` alone
under `techniques.allow`, so neither new member, nor a later one, reaches
it (its pool, rounds, plan file and draws are those of before). The rule that keeps
`to_refusal_then_comply` from `persona_template` does not move its eval
draws, since the eval type draws no persona template. A copy of `eval.json`
in a `config_dir`, read whole in place of the packaged one, still allows
`temporal` whole: its eval attacks draw the two temporal members, and an
eval run names them in a WARNING (the upgrade summary above).

**Release note: decode hints.** Twenty-one of the 27 encodings
(`to_rot13`, `to_hex`, `to_caesar_cipher` and so on; not the five of the eval
pool, `to_leetspeak_basic` or the sensitive-word encodings) now write, at the
spec's `decode_hint_rate` (0.5, a top-level key of `combination_spec.json`),
one of four hints around the encoded text that says how to decode it
([techniques.md](techniques.md#what-the-families-do)). So the adversarial
kind's encoded texts come with a hint about half the time, and a row's
`technique_info` gains `hint=<name>`; the kind's variants table counts the
hints. The draw is made when the unit runs, from the technique's own
seeded generator, not in the plan (one `rng.random()` when the encoding
changed the text; a registry encoder called directly with no `rng` is
therefore random now, where the plain encoder is not): a plan kept from before replays its
techniques and combinations as they were, and its pending units draw hints,
so a run resumed across this change holds encoded texts with and without
one. No combination, pool or round changes, and the `eval` and natural
kinds draw no hinted encoding.

**Release note: the payload split.** Two techniques of a new obfuscation
family, `payload_split` (spec 1.3, order 65, complexity 2, attack only):
`to_payload_split` cuts the text into two to four parts at word boundaries,
and `to_payload_split_semantic` has the generation model cut it where its
meaning allows (one to three calls, no check model: the parts are kept only
when they give the text back, compared without whitespace). Each names the
parts A, B, C ... and writes one of eight variants that asks the reader to join
them ([techniques.md](techniques.md#what-the-families-do)). A fresh plan of
the adversarial kind draws them, in about one unit in fourteen (216 to 221 of
3,000 units of the default kind in three groups, seeds 42 to 46): the default
attack pool goes from 202 techniques in 26 families to 204 in 27. The family
refuses `ascii_art`, `encode`, `fake_turns` and `word_art` and, like
`translation`, is kept from `fsh` and `dap`, so some units of a fresh plan
draw other techniques than they did with the same seed. A plan kept from
before resumes with the growth WARNING, which names the two, and does not
draw them until it is planned again with `resume=False`. The `eval` and
natural kinds draw neither: both pools are as they were.

**Release note: the word art.** One technique of a new obfuscation family,
`word_art` (spec 1.3, order 55, complexity 3, attack only): `to_word_art`
has the generation model mark the text's harmful words (one call, the
word-list parse and rejections of `sensitive_words` and `tokenbreak`), draws
the first in an ASCII-art font chosen as `to_ascii_art` chooses its fonts,
puts `[MASK]` in its place in the text, and writes one of eight variants that
asks the reader to read the art and put the word back
([techniques.md](techniques.md#what-the-families-do)). A fresh plan of the
adversarial kind draws it, in about one unit in fourteen (216 to 219 of 3,000
units, same plans): the default attack
pool goes from 204 techniques in 27 families to 205 in 28. The family refuses
`ascii_art`, `payload_split`, `encode`, `sensitive_words`, `tokenbreak`,
`structural`, `neutral_container` and `fake_turns` and, like `translation`
and `payload_split`, is kept from `fsh` and `dap`, so some units of a fresh
plan draw other techniques than they did with the same seed. The two new
families share the units that have three points of the budget left, which the
ten heavy encodings need too, so the heavy encodings became the rarest draws of
the default kind; since the eval release they are dealt in a slot of their
own (its note, the sixth change), and each is drawn 3 to 6 times in 3,000
units in three groups (seeds 42 to 46), one unit in about 670. A plan kept from before resumes with the
growth WARNING, which names it, and does not draw it until it is planned
again with `resume=False`. The `eval` and natural kinds do not draw it.

**Release note: the invented persona's own step.** `to_invented_persona`
no longer makes the calls of `cognitive`'s `to_persona_roleplay`: it first
asks the generation model for a persona for the text (the
`persona_invention` prompt), read strictly as a `Name:` line then a
`Description:` and asked again up to three replies, and then makes the
scenario, its check and the construction of a named persona with that name
and description. It costs one to three calls more (4 to 12, against 3 to 9),
for the pending units of a kept plan too, and a text whose replies never take
the form is discarded (`DISCARDED; reason=malformed persona`). Its info is
still `category=invented_persona`, and no plan, pool or draw changes.

**Release note: three persona archetypes.** The `persona` family gains
`to_mercenary_persona`, `to_underground_hacker_persona` and
`to_anarchist_philosopher_persona` (spec 1.3, complexity 2, attack only),
generated from their entries of `hacking/personas.json` as the other named
personas are: the same scenario, scenario check and construction (3 to 9
model calls), the info `category=<key>_persona`. A fresh plan of the
adversarial kind draws them: the default attack pool goes from 205
techniques in 28 families to 208 in 28, and since the family is wider some
units of a fresh plan draw other techniques than they did with the same
seed. A plan kept from before resumes with the growth WARNING, which names
them, and does not draw them until it is planned again with `resume=False`.
The `eval` and natural kinds do not draw them.

**Release note: the techniques spec 1.2 adds.** Forty-eight techniques in
six groups, below. A fresh plan draws them; since each adds members to deal,
or a family, some units of a fresh plan draw other techniques than they did
with the same seed. A kept plan resumes with the growth WARNING above and
does not draw them until it is planned again with `resume=False`. The `eval`
type draws none of them (it denies the encodings and the many-shot variants
in `techniques.deny` and allows none of the families), so its pool stays at
36 and its draws do not change
([techniques.md](techniques.md#what-the-families-do)).

- *Fifteen new encodings.* The `encode` family gains fifteen pure techniques
  (reversal of the characters, each word or the word order; look-alike
  letters; zero-width noise; hex, binary, URL, HTML-entity, base32 and
  base85 codecs; a Caesar shift and Atbash; fullwidth and math-styled
  Unicode). hex, binary, HTML entities, base32 and base85 are `heavy` (a code
  for every character, complexity 3), so `fsh` and `dap` refuse them and the
  request obfuscation never draws them; the rest are `light`, so they join the
  request obfuscation (phase 5), which changes a fresh plan's phase-5 draws (a
  wider set of light encodings to deal). The `eval` type denies all fifteen.
- *The noise family.* A new obfuscation family, `noise`, adds twelve pure
  techniques: random capitals, scrambled letters, keyboard typos and emoji
  between words, each at a low, medium and high density. The default attack
  pool draws all twelve; the natural pool gains the eight of low and medium
  density, the packaged natural kind allowing the family. `noise` applies
  after `casual_writing` and before `sensitive_words` and never sits beside
  `typos`, `casual_writing` or a translation; the emoji ones are kept from
  `ascii_art`.
- *The neutral containers.* A new obfuscation family, `neutral_container`,
  adds six pure techniques that place the text in an everyday document: a
  code comment, a YAML document, a CSV row, an email, a transcript line or a
  log entry, each with eight neutral variants of the surroundings. The
  default attack pool and the natural pool (the packaged natural kind
  allowing the family) draw all six. `neutral_container` applies where
  `structural` does, after `encode`, and never sits beside `structural` or
  `ascii_art`. The family carries the `container` dimension, which the
  `focus` axis is undecided against, so a run that deals `focus.mixed` (at
  weight 0 in the packaged table) denies it in both kinds, beside the
  families it denied before.
- *Six answer languages.* `answer_language` gains Hmong, Guarani, Chinese,
  Hindi, Yoruba and Amharic (rows of `configs/jailbreak/languages.json`): six
  pure techniques, `to_ask_answer_in_<slug>`, in the default attack pool and
  the natural pool. Four of them became translations in spec 1.3 (the note
  on the four translation languages). The answer
  languages now follow the language table's order, which puts Swahili before
  Thai, so a fresh plan that drew one of the two draws the other (a kept plan
  replays by name) ([techniques.md](techniques.md#the-language-table)).
- *Small talk and benign roles.* Three request families for natural
  variations only: `smalltalk_prefix` and `smalltalk_suffix` (a greeting
  before the text or a thanks after it, twenty variants each) and
  `benign_role` (a plain role given to the assistant before the text, three
  styles of twelve variants), drawn for benign and dual-use benign inputs
  only. The packaged natural kind allows the three families. Their spec
  `kinds` are `["natural"]`: the attacks never draw them (an adversarial
  allow list that names one raises). The benign roles carry the
  `role_framing` dimension, which the `writer` axis is ruled apart from, so
  both dataset types and the example recipes pass the run's dimension check
  as before; small talk carries `request_shape`, which the `focus` axis is
  undecided against, so a run that deals `focus.mixed` denies the two
  small-talk families in the natural kind, beside `asking` and `indirect`.
- *Many-shot fsh and dap.* `fsh` and `dap` each gain two many-shot variants,
  `to_fsh_random_many`, `to_fsh_selected_many`, `to_dap_random_many` and
  `to_dap_selected_many`: their siblings' prompt with 32 short benign pairs
  drawn across the benign cache, where the siblings show three to seven of
  one subcategory. They refuse `typos`, since a model rewriting so long a
  prompt would not return it whole. The natural kind does not draw them.

A `config_dir` copy of `augmentations.json` made before spec 1.2 keeps its
natural `families.allow` without `noise`, `neutral_container`,
`smalltalk_prefix`, `smalltalk_suffix` and `benign_role`, so its natural kind
does not draw them; the run logs this at INFO (add the names, delete the
copy, or `scaffold_setup(..., overwrite=True)`).

**Release note: a technique drawn only where it applies.** A technique may
now carry an applicability, and a fresh plan draws it only for an input it
applies to ([extending.md](extending.md#add-a-jailbreak-technique)).
`to_abbreviations` and `to_stripped_accents` (`casual_writing`) apply only
to a text they change, so a fresh natural plan no longer draws them for a
prompt with none of the listed words or no accented Latin letter, a no-op
the natural kind rejected. A fresh natural plan therefore draws differently
from one made before with the same seed: the casual writing picks of such
inputs go to the other members, and since the dealer counts each unit's
picks, later units of a group can draw other techniques too. Another member
can still leave a prompt unchanged (`to_lowercase` on a text in lower case
already). A kept plan replays as it was made. The adversarial kind draws no
casual writing unless its section allows the family, and the eval type
draws none: their pools and draws do not change.

**Release note: the obfuscation order.** A fresh plan applies translation,
typos, casual writing and noise before the word-level steps, sensitive words
and tokenbreak, which used to come first, so a combination holding both
gives another text. A kept plan replays the order its manifest recorded. The
families the eval type draws keep their order
([techniques.md](techniques.md#layers)).

**Release note: marked words in a translation's script.** Since a fresh
plan applies the word-level steps (sensitive words, tokenbreak) after a
translation, a marked word can be in the translation's language and
script. The marked-word parse now accepts the combining marks of such a
word (the vowel signs and viramas of Bengali, Nepali, Thai or Khmer),
where it rejected the reply as a malformed word list, and tokenbreak's
split and separator cut a word between the characters a reader sees,
never between a letter and its marks; an ASCII word is cut as before.
Whether the marking model finds the right words in these languages is not
measured ([techniques.md](techniques.md#what-the-families-do)); a later
measurement found it mostly does not, and a fresh plan no longer pairs the
two (the note on word marking after a translation, below).

**Release note: the script rule.** A fresh plan no longer pairs ROT13,
ROT18, ROT47 or leetspeak (whole-text or word-level) with a translation
into another script than Latin, where the encoding changed nothing; so
some draws differ from those an earlier version made with the same seed.
A kept plan replays as it was made. The eval pool holds none of these
encodings, so its draws do not change
([techniques.md](techniques.md#compatibility-rules)). The eval release has
since replaced the rule: a translation refuses every `encode` technique,
whatever its script, so the rule refused nothing more and is gone (the note
on the eval release, below).

**Release note: the ASCII-art cap.** `to_ascii_art` keeps a drawing of at
most 5,000 characters wherever it is drawn, the eval type's pool included,
and draws the font from the fonts that fit. This changes the text it gives,
not only what it rejects. A text every font draws within the cap (up to
about 38 characters) draws as before. Over that the same seed draws another
font for some texts from about 40 characters and for most from about 90:
of the 12 fonts that have a typical text's glyphs, 11 fit at 50 characters,
9 to 11 at 75, 9 at 100, 6 at 150, 2 or 3 at 200 to 300, 1 at 400 to 800
and none from about 830. A text of about 830 characters or more is rejected
where it used to be drawn, and so are most inputs an encoding lengthened
first (unicode escapes from about 140 characters, ASCII ordinals from about
230, morse from about 250); a fresh plan no longer pairs ASCII art with a
heavy full-text encoding, since the eval release (its note, below). A kept plan resumed across the change, an eval
plan included, mixes drawings made with and without the cap, since a
replayed plan gives the same text only within one version of the library;
run the adversarial kind with `resume=False` for one rendering.
[techniques.md](techniques.md#what-the-families-do) has the figures.

**Release note: the typo check.** Each typo rewrite (`typos`, in both
kinds) is now read against its prompt by the stage's `check` model
(`jailbreak/typo_check`, at temperature 0.15), and a rejected rewrite is
made again once with the check's reply as feedback: a typo row costs one
more call, and two more on a rejection (2 to 4 in all, against 1 before).
A rewrite rejected twice discards its unit's chain (`reasoning`
`DISCARDED; technique=to_rewrite_with_typos_<level>; reason=typo check
failed; ...`). Fresh plans use it, and so do the pending units of a kept
plan (the plan is not changed: no plan raises or warns for it); finished
rows keep their unchecked text. The `eval` type draws no typo technique:
its pool and draws do not change.

**Release note: the scenario check.** Each scenario of the `cognitive` and
`persona` techniques (the adversarial kind only) is now read against its
prompt by the stage's `check` model (`jailbreak/scenario_check`, at
temperature 0.15, asking that it keep the request's intent, not a harmful
one, since the attacks run on benign inputs too), and a rejected scenario is made again once with the
check's reply as feedback: such a row costs one more call, and a
regeneration and its check on a rejection (3 to 9 calls in all, against 2
before). A scenario rejected twice discards its unit's chain (`reasoning`
`DISCARDED; technique=<name>; reason=scenario check failed; ...`), the
prompt unchanged and no construction call made. Fresh plans use it, and so
do the pending units of a kept plan (the plan is not changed: no plan
raises or warns for it); finished rows keep their unchecked scenarios. The
`eval` type draws neither family: its pool and draws do not change.

**Release note: the rewrite layer.** The spec has a fifth layer, `rewrite`,
first in the apply order. It holds 16 techniques (spec 1.3: persuasion,
tense rewrites and model distractors, attacks only) and is off by default:
its spec rate is 0, and the spec lists it in `layers_off_by_default`, so
every kind leaves it out unless its `layers.allow` names it, whatever its
`layers.deny` says. Opt in with `"layers": {"allow": ["rewrite"]}` in the
kind (and, for budget rounds, a `layer_rates.rewrite` above 0). An allow
list or `family_weights` that names a rewrite family or technique without it
raises, saying so. A section's
own `layers.deny` no longer allows the layer, as it did while the packaged
kinds denied it in `augmentations.json`: `{"deny": ["hacking"]}` keeps it
off, and so does a `config_dir` copy of `augmentations.json`, whether it
says `"deny": []` (made before the layer) or `"deny": ["rewrite"]` (made
before the spec listed it). Nothing changes for an existing plan, the
dataset types' included: no packaged section allows the layer, so its pool,
rounds, plan file and manifest are as before, and so are its draws. A kind
whose `layers.allow` names it gains the 16 techniques and `include_rewrite:
true` in its adversarial rounds (written only when the kind's pool holds a
rewrite technique, so a kind whose lists name no rewrite family, the eval
type's among them, keeps its rounds), so a plan kept from a section that
allowed the layer before its pool held one raises on resume naming
`rounds`: take `rewrite` out of its `layers.allow` to keep the plan, or plan
again with `resume=False`. The flat `include_<layer>` keywords keep the
layer denied. The run warns when the two disagree: a kind that draws the
layer but has a budget round at rate 0 for it (no `layer_rates.rewrite`, or
a round whose own `layer_rates` leave it out), and a `layer_rates.rewrite`
above 0 while the layer is off. `"rounds": "escalation"` carries the kind's
`layer_rates.rewrite` into its two budget rounds, whose own rates replace
the kind's: they drew no rewrite before.

**Release note: the rewrite techniques.** Spec 1.3 gives the `rewrite` layer
its first sixteen techniques, all model-written and attack-only:
`persuasion` (12, `to_persuade_<key>`: a model rewrites the whole prompt as an
appeal of one of twelve persuasion techniques), `tense_rewrite`
(`to_rewrite_in_past_tense` and `to_rewrite_in_future_tense`: it rewrites it
in that tense) and `model_distractors` (`to_model_distractor_prefix` and
`_suffix`: it writes a few sentences placed before or after the prompt, which
is not rewritten) ([techniques.md](techniques.md#what-the-families-do)). The
first two are checked: a model (the `check` model, else the `gen` model) reads
each rewrite against the prompt (`jailbreak/preserve_check`), a rejected one
is made again with the verdict, and one rejected twice is discarded, so each
application costs 2 to 4 calls; a distractor costs 1 to 3 (a reply with no
text is asked for again), and no check reads it. They add no model role.

The layer is off, so a run that does not allow it changes nowhere: the
default attack pool is still 214 techniques in 29 families, the natural pool
the 72 (every rewrite technique says `["adversarial"]`), and the `eval`
type's pool and draws are unchanged (36 techniques; its allow lists name no rewrite
family, so allowing the layer there adds nothing to its pool or its rounds
until the families are allowed too). A kind that allows the layer draws them: the packaged
adversarial kind's pool is then 230 techniques in 32 families, its rounds say
`include_rewrite: true` (an adversarial plan kept from a section that allowed
the layer before its pool held a technique raises on resume naming `rounds`), a
budget round deals the layer at `layer_rates.rewrite` (the spec's 0, with a
WARNING: set it, for example 0.5, which gives half the units of a group one
rewrite technique, the three families dealt evenly and each applied first;
the escalation's budget rounds take it too) and a count round,
the escalation's first two among them, draws from the pool whatever the rate.
New rules bind a fresh plan, both ways: `persuasion` refuses `impersonation`
and `ascii_art`, `tense_rewrite` refuses `temporal`, and `model_distractors`
refuse `ascii_art`, `distractor_prefix` and `distractor_suffix`; a kept plan
replays its recorded combinations, so a rule changes none of them.

**Release note: four more translation languages.** Spec 1.3 switches the
translation of four answer languages on: `to_hmong`, `to_chinese`, `to_hindi`
and `to_amharic` (their answer languages, `to_ask_answer_in_<slug>`, are as
they were), measured on a GPU pod on 2026-10-22 with the machine-translation
path (TranslateGemma 27B, GlotLID, the back-translation judge). Guarani and
Yoruba stay answer languages only: TranslateGemma loops on a fifth of
Guarani's sentences, and the judge rejects 41% of Yoruba's calls. The default
attack pool goes from 208 techniques in 28 families to 212 in 28, and the
natural pool with `"translation": true` from 91 to 95 (72 without, as
before); the `eval` type names its eight translations, so its pool and draws
are unchanged (36 techniques in 10 families). Since the family is wider some
units of a fresh plan draw other techniques than they did with the same seed.
A plan kept from before resumes with the growth WARNING, which names the
four, and does not draw them until it is planned again with `resume=False`.
Amharic and Hmong are on watch, as Scots Gaelic is (the judge rejects 20% of
Hmong's calls, and Amharic's chrF++ on FLORES-200 is 9 points below
NLLB-200's); Hmong is sent to the translation model by its name, not a code
([techniques.md](techniques.md#the-language-table)).

**Release note: no word-marking step after a translation.** A fresh plan no
longer pairs a translation with `sensitive_words`, `tokenbreak` or
`word_art`, the three families that have the generation model mark the
prompt's harmful words (`extract_harmful`): the spec's `translation` entry
keeps them apart, both ways, as it keeps `ascii_art` apart. A GPU pod
measurement of 2026-10-22 found that the marking call, which marks at least
one word on 62 of 100 harmful English inputs, marks one on only 3 (Khmer) to
36 (Hindi) of their translations (Chinese 34, Thai 26, Hmong 17, Amharic 5):
a heavy encoding after such a translation was discarded (`no sensitive
word`) and a light one changed nothing, so the unit's calls were wasted.
The measurement covers English and nine languages (Bengali, Thai, Nepali,
Khmer, Hmong, Chinese, Hindi, Yoruba and Amharic) on 100 inputs each, so a
share is good to about 9 points and a second pass moved some by up to ten
inputs; the direction held in every language. No European language,
Japanese, Russian or Arabic was measured: for those the rule is a design
choice beyond the measurement, until the marking call is measured on them.
Measured at commit `12b1699` and its parent `996c8ea` (3,000 units of the
default attack kind in three groups, seed 42), 319 units (10.6%) drew both
before the rule and none draws both since; the units holding a translation
went from 1,037 to 801 and the obfuscations per unit from 1.72 to 1.63. A plan
kept from before replays its recorded combinations, so a rule changes none
of them. The eval type draws none of the three families (its pool holds only
its encodings, framing, fsh, dap, structural, ascii_art, continuation,
temporal, answer format and eight translations), and the natural kind none of
them either: their pools and draws are unchanged. A translation chosen as the
request obfuscation shares the rule with the obfuscations drawn after it.

The same entry now keeps a translation apart from `typos`, `casual_writing`,
`noise` and `payload_split` too (a design choice of 2026-10-22, not a
measured failure): a translated prompt is already hard to read, for the model
that writes or marks text and for the model that answers, and one more
word-level or character-level change on top is too complex. A fresh plan no
longer pairs a translation with these steps either, so some draws differ from
those an earlier version made with the same seed. Measured at commit
`87bdf6e` and its parent `91c508b` (the same plan of 3,000 units), 204 units
drew a translation and one of the four before and none since; the units
holding a translation went from 787 to 798 and the obfuscations per unit from
1.63 to 1.64. At the current commit no unit holds a translation beside any of
the eight families (0 of 3,000 units at seeds 42 to 46), 23.9% hold a
translation (23.9% at each seed) and the obfuscations per unit are 1.47 (1.47 to
1.48). A plan kept from before replays its recorded combinations, so a kept
plan may still pair them. `structural`, `suffixes` and `neutral_container`
stay allowed beside a translation: the eval type draws `structural` with its
translations, and its pool (36 techniques) and draws are unchanged by these
four families, since it holds no technique of them (`encode`, which it holds,
is refused beside a translation since the eval release, below). The
natural pool (72) is unchanged as well; the natural kind draws one technique
per input, and takes the rules only with `exact_techniques` of 2 or more and
`"translation": true`.

**Release note: the wording family.** Two techniques of a new obfuscation
family, `wording` (spec 1.3, order 45, between `sensitive_words` and
`tokenbreak`, complexity 2, attack only, the `encoding` dimension): each
has the generation model mark the text's harmful words (the call, strict
word-list parse and rejections of `sensitive_words`, `tokenbreak` and
`word_art`), then one call more for each distinct word, capped at 128 tokens
and asked again once on a reply that will not do, so a text costs 1 to 21
calls (`[1, 21]` in the spec's `calls`) ([techniques.md](techniques.md#what-the-families-do)).
`to_pseudonym_substitution` swaps in a code name of one or two words
(`jailbreak/pseudonym`), case-matched, and adds a key of `<alias> = <word>`
lines; `to_word_game` puts a placeholder `[W1]`, `[W2]`, ... for the word and
lists a puzzle that resolves to it (`jailbreak/word_game`), asking the reader
to solve them first. Each writes one of eight variants of its pool in
`obfuscation/wording_templates.json` (`variant=<name>;detected_words=<words>`);
a text whose replies never take the form is discarded (`malformed pseudonym`,
`malformed puzzle`). `to_synonym_substitution` stays in `sensitive_words`. The
family refuses `sensitive_words`, `tokenbreak`, `word_art`, `translation` and
`ascii_art`, and, like `word_art`, is kept from `fsh` and `dap`, so some units
of a fresh plan draw other techniques than they did with the same seed. A
fresh attack plan draws it in about one unit in fourteen (217 of 3,000 units
of the default kind in three groups at seed 42; 216 to 220 over seeds 42 to
46): the default attack pool goes
from 212 techniques in 28 families to 214 in 29. A plan kept
from before resumes with the growth WARNING, which names the two, and does not
draw them until it is planned again with `resume=False`. The `eval` type
allows no `wording` technique and the natural kind none, so their pools
(36 and 72) and the eval draws are as they were; the family's dimension is
`encoding`, which no generation axis varies, so the `training` type needs no
denial. Prompts of your own (`prompt_dir`) overlay the two new ones as any
other.

**Release note: the eval release.** The eval type's pool is the same 36
techniques (`tests/fixtures/eval_pool.json`), but which pairs of them one
combination may hold has changed, and so has the text of 40 of the pairs it
keeps (the fifth change, below), so eval files made before this release are
not comparable with those made after. No technique is added or retired, so
the spec's version stays 1.3, as when rules changed within 1.3 before: these
are changes to rules and to the order the layers apply in, which the plan
file's `rules` id records. The jailbreaks run record carries that id too
(`rules`, by kind), and a row's `technique` column lists its techniques in
the order they applied (a manipulation technique before the obfuscations and
the request until this release, after them since): those two tell the files
apart. The first change is that a translation refuses
every `encode` technique, light or heavy, into any script (the `translation`
entry of the spec): both rewrite the whole text, so one is enough, and a
cipher before a translation would hand the translator a cipher. The pool
loses 40 of its 335 pairs, each of its eight translations (Bengali,
Javanese, Khmer, Nepali, Scots Gaelic, Swahili, Thai, Zulu) beside each of
its five encodings (`to_ascii_ordinal`, `to_base64`, `to_braille`,
`to_morse`, `to_unicode_escape`), and 295 remain after this change alone
(`tests/fixtures/eval_pairs.json`). Round 1 of the eval type draws one
technique per input and does not move. In round 2 (two techniques) a plan of
3,000 inputs at the eval's seed, 43, held such a pair in 304 units before in
three groups (357 in one); a plan of 300 inputs in three groups deals 283 of
its 300 round-2 units another pair, since the group counts follow the lost
pairs. The default attack kind
loses the pairing too: on 3,000 units in three groups, 247 to 266 units
(seeds 42 to 46) drew a translation and an encoding before, and none does
since; the units holding a translation go from 25.3% to 24.6% at seed 42.
The script rule (its note, above) is gone with this change: it kept a
Latin-only technique from a translation into another script, and every such
technique is an `encode` or `noise` technique that a translation now refuses
outright; the `needs_latin` tag stays, and a test asserts the invariant.

The second change is that ASCII art refuses the ten heavy full-text
encodings (`to_base64`, `to_unicode_escape`, `to_ascii_ordinal`, `to_morse`,
`to_hex`, `to_binary`, `to_html_entities`, `to_base32`, `to_base85`; braille
had the rule already): a code for every character lengthens the text, so its
drawing often ran past the cap (the ASCII-art cap note, above), and both
disguise the whole text, so one is enough. The three heavy
`to_sensitive_words_encode_*` encode only the words a model marks, so ASCII
art still joins them, and the light encodings, unless their output has a
character no font draws (the look-alike and fullwidth forms, for instance).
In the eval pool four more pairs go (ASCII art beside
`to_base64`, `to_unicode_escape`, `to_ascii_ordinal` and `to_morse`), and 291
of its 630 pairs remain. At seed 43, 485 of the 3,000 round-2 units of a
plan held such a pair after the first change (193 before it, the first change
sending the encodings to the other partners they could still take); none
does now; a plan of 300 inputs deals 279 of its 300 round-2 units another
pair than after the first change alone.

The third change is for training only: `sensitive_words` and `tokenbreak`
refuse each other (both entries of the spec), as `wording` and `word_art`
refuse them and each other. Each has the model mark the text's harmful words,
so a second marking call would read a text the first changed. Neither is in
the eval pool, which a test asserts, so the eval type's pairs and draws do not
move with it. In 3,000 units of the default attack kind in three groups, 50
to 72 units (seeds 42 to 46) drew both before the rule, and none does since;
39 pairs of active techniques are refused that were not (48 counting the
three retired separators).

The fourth change is for training only too: a translation drawn as the
request obfuscation (phase 5, the packaged attacks only) refuses a suffix, a
structural wrapper and a neutral container. It applies after the request, to
the finished request, so it would translate the suffix, the keys of a JSON, XML or
Markdown wrapper or the lines of a container (a new key of the spec,
`request_obfuscation_refuses`, which the plan file's `rules` id records). A
translation drawn as an ordinary obfuscation runs before those steps and
still joins them. In 3,000 units of the default attack kind in three groups,
498 to 501 units (seeds 42 to 46) take a translation as the request
obfuscation; before the rule 68 to 79 of them held a suffix, 130 to 151 a
structural wrapper and 115 to 128 a neutral container, and none does now;
the obfuscations per unit are 1.47 (1.47 to 1.48) with the step on, against
1.58 (1.57 to 1.59) with it off and 1.57 before this rule. So a translation
drawn as the request obfuscation is the unit's only obfuscation: the
translation family already refused ten of the other thirteen obfuscation
families (`ascii_art`, `encode`, `sensitive_words`, `tokenbreak`, `word_art`,
`wording`, `typos`, `casual_writing`, `noise` and `payload_split`), this table
refuses the last three, and the slots `max_obfuscations` leaves go unused. Of
3,000 units of the default attack kind (seed 42, three groups), 499 take a
translation as the request obfuscation and 333 of them held another
obfuscation before the rule; in round 4 of the escalation
(`max_obfuscations` 3) 700 do, 647 of them holding two others before and 53
one, and none holds any now. The eval type
draws no phase-5 step, so its pairs and draws, and the budget
draws the digest pins (phase 5 off in both), do not move. The WARNING of
a kept plan, which counts the pending units whose combination today's rules
refuse, counts those this rule refuses too: it takes the obfuscation right
after the request as the request obfuscation, where the order of every
plan, before this release and after, puts it. A plan of 3,000 units of the
default attack kind drawn before this release (seed 42, three groups)
counts 539, 202 of them by this rule alone.

Together the four rule changes close most pairings the ten heavy full-text
encodings (complexity 3, which fits only a unit whose earlier picks leave 3
points) were dealt with, so dealt as `encode`'s members they were drawn
rarely: 8 to 16 draws of the ten in 3,000 units of the default attack kind
in three groups (seeds 42 to 46), against 31 to 42 before the release, and in
4,000 units at seed 42 five of them were never drawn under the unit budget
and two under the group budget. The sixth change gives them a slot of their
own. The other figures this page quotes from the simulation (the group-budget
table, the request obfuscation's and the translation's shares) are measured
on the rules of this release, the sixth change included; those of the five
changes' notes compare each change with the commit before it.

The fifth change is the order: the manipulation layer (`fsh`, `dap`,
`fake_turns`) now applies last, after the obfuscations, the request and the
request obfuscation, and puts the finished question among its examples or
after its invented turn. The spec's `layer_order` is `rewrite`, `hacking`,
`obfuscation`, `requests`, `manipulation`, and the sampler places the
request obfuscation right after the request. Before, the steps after the
layer read and changed the whole text: a typo rewrite or a marking call read
3 to 32 benign pairs, a suffix landed after the last answer, a request
framed the whole list, and an encoding multiplied the tokens of every pair.
Now they change the question alone, and the examples stay as written. A
selected `fsh` or `dap` variant shows its model the clean input (the unit's
input, passed as `source_prompt`), not the encoded or wrapped question, and
inserts the finished one; after a rewrite technique (a kind that allows the
rewrite layer) that is the input before the rewrite, which a kept plan's
variant did not see. The same techniques are drawn, in another order:
no pair is added or taken out. In the eval pool, the 40 pairs of the eight
`fsh` and `dap` techniques with `to_json`, `to_xml`, `to_markdown`,
`to_refusal_then_comply` and `to_use_past_tense` write another text (the
wrapper or the request goes around the question alone), and the other 251
pairs and the 36 single techniques write what they wrote; a plan of 3,000
inputs at seed 43 holds 858 such units among its 3,000 round-2 ones, and the
plan of 300 inputs the digest pins 86 of its 300. The rules stay: the
manipulation block (`manipulation_blocked_obfuscation`) is kept as it was
(lifting it would admit 104 eval pairs, each `fsh` or `dap` technique beside
a translation or a heavy encoding, a change of the eval coverage of its own),
and so are the many-shot variants' refusal of `typos`, whose reason (a
rewrite of the whole many-shot text) is gone. Three more refusals were
written for the old order, and their reasons are gone too: `fake_turns` with
`structural` and `neutral_container` (an invented turn was a conversation,
not a document to wrap; the wrapper now goes around the question alone and
the turn stays outside it), `ascii_art` with `fsh` and `dap` (the output size
of a font that drew their examples too) and `payload_split` with
`fake_turns` (an invented turn was a conversation already). They are kept as
they were, and lifting them is a decision of its own. `fake_turns` beside the
nine wrappers would add 27 pairs of the default attack kind and none of the
eval pool; `ascii_art` with `fsh` and `dap`, and `payload_split` with
`fake_turns`, are refused by the manipulation block as well, so a lift takes
both rules, and no pair of the pool's 291 holds `ascii_art` with `fsh` or
`dap` (lifting both rules would admit 8). On the default attack kind
(3,000 units in three groups at seed 42, filler English inputs, the model
steps stubbed, tokens estimated at about four characters a token), the 988
units with a manipulation technique go from a 99th percentile of 4,059
tokens and a longest row of 10,234 to 1,653 and 2,174 on 400-character
inputs, and from 4,583 and 11,557 to 2,370 and 3,497 on 1,200-character
ones (seeds 42 to 46: 4,583 to 5,953 and 9,953 to 11,619 before, 2,327 to
2,423 and 3,410 to 3,541 after); the typo rewrites that would pass the
generation model's 2,000-token reply cap go from 2 to 8 units (seeds 42 to
46) to none on 1,200-character inputs, and from 17 to none on
2,400-character ones (seed 42). A row can still pass 6,000 tokens: one of a
very long input under a lengthening encoding, with no manipulation
technique, and, under a stricter estimate (three characters a token and every
non-ASCII character a token of its own), a manipulation row of a
2,400-character input: 6 of the 988 at seed 42, 6,054 to 6,179 tokens, all
with `to_math_styled` on the question.

ASCII art never meets the manipulation layer, so the order moves none of its
rejections; the other rules of the release do. On 400-character inputs the
five plans of seeds 42 to 46 reject 57 of its draws before the release and 4
after: the second change keeps it from the heavy full-text encodings, which
lengthen the text before the font (88 of its 1,251 draws held one), and its
draws per plan go from 248-254 to 213-218. On 1,200- and 2,400-character
inputs every draw is rejected as too long, before and after: 1,074 draws in
the five plans at each length, about 7% of the default attack kind's units
(1,251 before).

The request techniques (an answer format, a distractor, an impersonation, a
temporal frame, `asking`) and the request obfuscation are part of the
question the manipulation layer inserts, so they now sit inside it. With
`fsh` the finished question is the last entry and only the cue for the
answer follows it. With the many-shot `dap` variants (33 entries) it lands
anywhere from the 2nd to the 32nd, 0.53 of the way on average (300 draws),
with 562 to 5,010 characters of pairs after it. The labels stay right either
way; whether such an attack is stronger or weaker is not measurable
offline.

The sixth change is for training only: a budget round deals the ten heavy
full-text encodings in a slot of their own beside `encode`, at the spec's
new `heavy_encoding_weight`, 0.2, times `encode`'s weight
(`HEAVY_ENCODINGS` in `jailbreak/sampling.py`, a name the dealer counts them
under). Dealt as `encode`'s members, a heavy one fitted only a unit with 3
points left, and the light members kept the family's count high, so it was
seldom `encode`'s turn on such a unit (the paragraph above). With the slot,
3,000 units of the default attack kind draw them 44 to 50 times (seeds 42 to
46, in one, three or ten groups; 45 at seed 42 in three groups, 3 to 6 times
each), about a fifth of a family's draws, and every group of 100 units draws
at least one of them and ASCII art, at seeds 1 to 3, 7, 42 to 46 and 123 in
one, three or ten groups (before, 29 of those 30 plans had a group without a
heavy encoding). At seed 42 in three groups, `encode`'s draws per unit go from
0.242 to 0.255, ASCII art from 219 draws to 215, the units holding a
translation from 24.0% to 23.9%, and the obfuscations per unit stay at 1.47;
the mean complexity per unit goes from 5.628 to 5.637 (the mean over seeds 42
to 50, three groups). Every technique of the default attacks is drawn in
4,000 units (the test no longer leaves the ten out).

The slot is a quota, not a floor. It deals the ten at the same share of
`encode` in every budget round, whatever the round's budget, so a round that
dealt them more as `encode`'s members now gets fewer: that is the size it was
set for, about 40 draws in 3,000 units. In the escalation (3,000 units a
round in three groups, the mean of seeds 42 to 46) round 3 (`max_complexity`
6) goes from 6 draws of the ten to 42, and round 4 (`max_complexity` 9,
three obfuscations) from about 121 to about 45; rounds 1 and 2 are count
rounds and do not move. The last round of the escalation therefore draws
about as many of them as the default attacks do (44 to 50), where its budget
alone would have made it the round that draws the most, and a kind that wants
more of them in its late rounds cannot say so: the weight is one number of
the spec, which a `config_dir` never overlays.

The weight was set by measurement over seeds 42 to 46 in three groups: 0.3
draws them 64 to 69 times, 0.5 100 to 108, and a weight of 1, a family's
share, about 200 (`encode` then at 0.30 draws a unit); 0.18 gives 39 to 41.
The total depends on how the units are grouped, since each group's dealer
starts the slot at 0 and so deals it a share of its own: at 0.2 it is 44 or
45 in one group, 45 in three and 46 to 50 in ten (seeds 42 to 50), and 52 to
56 in 46 uneven groups of 30 to 100 units; at 0.15 it is 33, 33, 39 and 46 in
those four layouts, and at 0.3 65, 65, 67 and 70. What the slot fixes is the
total, not a count for each encoding: the least drawn of the ten is drawn 3
or 4 times in one and in three groups, 0 to 3 times in ten groups (seeds 42
to 50) and 1 to 3 times in the uneven layout. A count round (the eval type's
two rounds, the natural kind, the first two escalation rounds) deals them as
`encode`'s members, as before, so the eval type's draws do not move (its
digest holds, and they are the same with the spec's key left out, which a
test asserts); a direct `sample_combination` call without a dealer does not
change either. `family_weights` cannot name the slot, which is not a family
(`"encode:heavy"` raises as any name the kind does not draw): weight
`encode`, which the slot follows. The new key moves some figures the notes
above measured before it: with the request obfuscation off, the obfuscations
per unit are 1.57 (1.56 to 1.58); on the fifth change's 1,200-character
inputs at seed 42 the manipulation units' 99th percentile is 2,942 tokens and
their longest row 3,541 (2,405 to 2,489 and 3,464 to 3,512 at seeds 43 to 46),
on 400-character ones 1,770 and 2,218, and no typo rewrite passes the reply
cap.

A plan kept from before replays its recorded combinations in their recorded
order, so it still holds the pairs now refused and applies its manipulation
technique second: its pending units write the old text. Its plan file's
`rules` id, which holds the layer order (and, in a budget plan whose pool
holds a heavy full-text encoding, the heavy encodings' weight), differs
from today's, so a
resume logs one WARNING that the rules a combination is drawn by changed,
counts the pending units whose combination today's rules refuse, and says
that each combination applies in the order the plan records. To take the
new rules, run the jailbreak stage with `resume=False` (the stage starts
over); until then the rows of a kept eval plan are not those of a fresh one.
The rules are the spec's, which a `config_dir` never overlays, so a run on a
`config_dir` copy of the eval type takes them too, its pool as wide as its
copy of `eval.json` makes it (the bullet on a stale copy, above).

The `rules` id now records only what a plan's draws read: the request
obfuscation's two lists only when a round allows that step (a budget round
with `allow_request_obfuscation`), the heavy encodings' weight only when a
round is a budget round and the pool holds one, and no longer the
`needs_latin` and `script` tags, which no rule reads. A count-only plan (the
eval type's, the natural kind's) keeps its id when those change. A plan file
written before the id was narrowed (by an earlier build of this release)
holds an id over more keys, so its first resume gives the one WARNING; a
plan drawn by these rules has none of its pending units refused, and can be
resumed as it is.

## The model registry

`configs/llm/models.json` holds one entry per model, keyed by its registry
name. It loads into `llms.model_config.MODEL_REGISTRY` at import, and every
entry is validated then; one bad entry stops the import with a list of every
problem. `register_model(...)` adds or replaces an entry for the running
process only.

### Entry fields

| Field | Default | Meaning |
|---|---|---|
| `backend_type` | the only setup present | The default setup: `api`, `vllm` or `introspect`. Needed when there are several |
| `roles` | `[]` | Registry roles the model fills. Each must be in `configs/llm/roles.json` |
| `is_uncensored` | `false` | Marks an uncensored generation model |
| `supports_system_prompt` | `true` | `false` folds system messages into the first user message (the first message when none is the user's); an empty system message is dropped |
| `default_max_tokens` | `2000` | Reply limit when a call gives none |
| `default_temperature` | `null` | 0.0 to 2.0; `null` leaves it to the backend (vLLM and introspection use 0.7) |
| `api`, `vllm`, `introspect` | `null` | The setups; at least one |
| `notes` | | Free text, ignored |

### `api` (APIConfig)

| Field | Default | Meaning |
|---|---|---|
| `backend_type` | required | `openai` (any OpenAI-compatible endpoint) or `anthropic` |
| `api_key_env` | required | The environment variable holding the key |
| `rpm` | required | Requests per minute, over a sliding 60-second window; at least 1 |
| `base_url` | required for `openai` | The endpoint URL |
| `api_model_id` | the registry name | The model id sent to the provider |
| `rate_limit_scope` | `"model"` | `model`: one window per model. `endpoint`: one window for every model with the same provider, URL and key variable; they must declare the same `rpm` |
| `recommended_max_workers` | `1` | Requests in flight. Must be 1 for `anthropic`. A run's `workers` in the recipe replaces it |
| `default_extra_body` | `null` | Provider parameters sent with each call that passes no `extra_body` of its own, which replaces them whole (OpenAI-compatible only) |
| `price_per_1m_input`, `price_per_1m_output` | `null` | USD per million tokens, for the cost roll-up |
| `max_retries` | `2` | Retries the SDK makes of a failed call (a 429, a 5xx, a lost connection), with its own backoff, unseen by the rate limiter and the trace. A call that still fails stops the stage |
| `timeout_s` | `null` | Seconds before the SDK gives up on a call; `null` keeps the SDK's default (10 minutes). Above 0 |

### `vllm` (VLLMConfig)

| Field | Default | Meaning |
|---|---|---|
| `hf_model_id` | required | HuggingFace id or local path |
| `quantization` | `null` | vLLM quantization method, such as `awq`, `gptq`, `fp8` |
| `vllm_kwargs` | `null` | Engine arguments (vLLM's `AsyncEngineArgs`), such as `max_model_len`, `enforce_eager`, `tokenizer_mode`. `gpu_memory_utilization` and `data_parallel_size` are refused: the planner sets them. A `tensor_parallel_size` must equal `min_gpus` |
| `sampling` | `null` | Sampling defaults (`top_p`, `top_k`, ...) over the backend's `top_p` 0.85 |
| `vram_gb` | `null` | The weights in GiB, total over the model's cards, when known better than the estimate |
| `min_gpus` | `1` | Cards per copy; above 1 it is the `tensor_parallel_size` |
| `hf_config` | `null` | A folder under `configs/llm/hf_configs/` whose `config.json` vLLM (as `hf_config_path`) and the estimator read in place of the checkpoint's own, for a checkpoint whose config the installed transformers cannot read. The folder holds the checkpoint's `generation_config.json` too: vLLM reads the generation config from `hf_config_path` as well, and without it a model misses its end-of-turn token and runs every reply to `max_tokens`. The weights come from `hf_model_id`. vLLM also reads the config of the folder it loads the tokenizer from, so at load the checkpoint's tokenizer files are gathered beside a copy of this config into a folder under `$HF_HOME/assets/redact/tokenizers/`, passed to the engine as `tokenizer` unless `vllm_kwargs` names one. `vllm_kwargs` may not name `hf_config_path` |
| `chat_template` | `null` | A Jinja chat template rendered in place of the tokenizer's own, for a checkpoint that ships none; not an engine argument, so entries on one checkpoint may differ in it and share the engine. `vllm_kwargs` may not name it, and it does not go with `tokenizer_mode` `mistral` |

### `introspect` (IntrospectConfig)

| Field | Default | Meaning |
|---|---|---|
| `hf_model_id` | required | HuggingFace id or local path |
| `log_dir` | required | Root folder of the captured internals |
| `capture` | `logprobs: true`, `hidden_states: "last"`, `attention: false` | What to save per captured call; `hidden_states` is `false`, `"last"` or `"all"` |
| `device_map`, `torch_dtype` | `"auto"` | Passed to `from_pretrained` |
| `sampling` | `null` | Sampling defaults over the backend's `top_p` 0.85 |
| `vram_gb` | `null` | The weights in GiB |
| `extra_kwargs` | `null` | More `from_pretrained` kwargs. Capturing attention loads with eager attention unless this names another |

### Roles and shipped models

`configs/llm/roles.json` lists the registry roles: `uncensored_gen`,
`translation`, `machine_translation`, `constitution_gen`, `paraphraser`,
`debug_local`. A role's default model is the first entry in `models.json`
that lists it. `machine_translation` marks a model that only translates, in
its own fixed format: the translation technique sends it
`jailbreak/translate_mt`, checks its output and has the generation model
judge a back-translation.

| Model | Default setup | Other setup | Roles | Limits |
|---|---|---|---|---|
| `venice-uncensored` | `api` (Venice) | `vllm`, 24B, `vram_gb` 43.9 | `uncensored_gen` | 75 rpm, 3 workers |
| `venice-uncensored-fp8` | `vllm`, the same 24B weights, `fp8` on load, `vram_gb` 24.9 | | `uncensored_gen` | |
| `deepseek-v3.2` | `api` (Venice) | | | 20 rpm, 2 workers; a chat translator, named in a recipe |
| `translategemma-27b` | `vllm`, `fp8` on load, `hf_config`, `max_model_len` 4096, `vram_gb` 28.4 | | `translation`, `machine_translation` | temperature 0.2, `top_p` 0.95, `top_k` 64 |
| `olafangensan-glm-4.7-flash-heretic` | `api` (Venice) | | `uncensored_gen` | 20 rpm, 2 workers |
| `llama-3.2-3b-debug` | `vllm`, 3B, `max_model_len` 384 | | `debug_local` | |
| `claude-opus-4-6` | `api` (Anthropic) | | `constitution_gen` | 5 rpm, 1 worker, endpoint window |
| `claude-opus-5-5` | `api` (OpenRouter) | | `constitution_gen` | 600 rpm, 8 workers |
| `humanizer-12b` | `vllm`, bf16, `max_model_len` 8192, `vram_gb` 22.8, `chat_template`, revision pinned | | `paraphraser` | temperature 1.0, `top_p` 0.95, `top_k` 0 |
| `venice-paraphraser` | `vllm`, same weights and `fp8` as `venice-uncensored-fp8`, one engine with it | | | named in a recipe |

`humanizer-12b` is the default paraphraser, self-hosted: Gemma 4 12B fine-tuned to
rewrite model drafts as human text, keeping every fact. It is a
text-completion model with no chat template of its own: its entry's
`chat_template` passes the prompt through after `<bos>`, and its prompt
`prompts/paraphrase/generation/humanizer-12b/` is the model card's format,
checked at the stage's start by its `fingerprint` key. `venice-paraphraser`
is the generator's own weights, kept for comparison: name it. A trained
defingerprinting model lives in another repository; register it with the
`paraphraser` role.

## Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `VENICE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | the shipped API setups | API keys, named by `api_key_env` |
| `HF_HOME` | vLLM, transformers | The HuggingFace cache for weights, `~/.cache/huggingface` when unset. Give it room: the local weights of `venice-uncensored` take about 48 GB |
| `HF_TOKEN` | HuggingFace | Access to gated checkpoints and datasets |
| `REDACT_OUTPUT_DIR` | `paths.get_output_dir` | The default working root, and where `.env` is read |
| `REDACT_TRACE` | `telemetry.install` | `jsonl` (default), `wandb` or `off` |
| `WANDB_PROJECT` | the W&B sink | Project name, default `redact` |
| `REDACT_GPU_PROVIDER` | the cost roll-up | Provider key in `gpu_pricing.json`; default `local`, priced at 0 |
| `REDACT_GPU_HOURLY_USD` | the cost roll-up | Hourly price per GPU; wins over the table |
| `CUDA_VISIBLE_DEVICES` | the planner, vLLM | The cards the run may use, by index, UUID or UUID prefix; the plan numbers cards over them. As in CUDA, an entry that names no card ends the list |
| `CUDA_DEVICE_ORDER` | CUDA | Set to `PCI_BUS_ID` by a plan or a load unless set, so CUDA counts the cards as `nvidia-smi` does |
| `VLLM_WORKER_MULTIPROC_METHOD` | vLLM | Set to `spawn` before a load unless set |
| `VLLM_USE_V2_MODEL_RUNNER` | vLLM | Set to `0` before a load on WSL2 unless set, where vLLM's V2 model runner fails every load; an explicit value wins |
| `VLLM_PROCESS_NAME_PREFIX` | the card check | The title prefix of vLLM's engine processes, default `VLLM`, named in the `kill -9` advice |

## Your own config folder

A recipe's `config_dir` names a folder laid out as the packaged `configs/`.
These files are read from it when it has them, each file on its own, and
from the package otherwise:

| File | Replaces |
|---|---|
| `<config_dir>/dataset_types/eval.json`, `training.json` | the [dataset type](#dataset-types) file of that name. A file of another name there is no new type: it is not read, and the run warns about it as ignored |
| `<config_dir>/taxonomy/<name>.json` | a taxonomy of that name (`taxonomy`, `benign_taxonomy`); a new name there is a taxonomy of your own. `taxonomy_dir`, when given, wins and is read alone |
| `<config_dir>/jailbreak/augmentations.json` | the default [augmentation section](#the-augmentation-section), both kinds and `natural_only_families`, checked when read: an unknown key (of the file, a kind, or its `layers`, `families` or `techniques`) is left out with a WARNING naming the file; a kind that is not an object or lacks a key, a missing `natural_only_families`, or `exact_techniques` in the adversarial kind (which would read the attacks as the natural kind) stops the recipe at load, naming the file |
| `<config_dir>/generation_axes.json` | the [generation axes](#the-generation-axes) table, whole: its labels, hints, weights and `not_with`. Its axes are checked against the folder's `dimensions.json` when it has one |
| `<config_dir>/dimensions.json` | the [dimensions](#dimensions-and-the-runs-check) table, whole. It must hold every tag the families carry; the run's check reads its `same_as` and `undecided` |

`config_dir` is relative to the recipe's folder, as `params_file` is. The
technique spec (`combination_spec.json`), the language table
(`languages.json`), the technique templates and other files under
`jailbreak/`, `llm/` (`models.json`, `roles.json`,
`gpu_pricing.json`) and `datasets/` are loaded at import or tied to code, so
they come from the package only (an axes or dimensions file in one of those
folders too): a file of theirs in the folder is named in a
WARNING at the start of a run (`configs: ... ignored`). With `verbose` the run
logs which files it reads from the folder, and each run record holds
`config_dir` and `config_files`, those files as paths relative to it.

The fronts take `config_dir=` too (`generate_constitution` for its
taxonomies, `generate_inputs` and `generate_outputs` for their axes,
`generate_jailbreaks` for its section and benign taxonomy); a front called
alone with none reads the packaged files only.

`scaffold_setup(target_dir, *, overwrite=False)` (also
`python scripts/scaffold_setup.py TARGET [--overwrite]`) writes a folder to
start from:

| File | What it is |
|---|---|
| `eval_recipe.json`, `training_recipe.json` | The packaged example recipes, with `"config_dir": "."` and their params file beside them |
| `eval_params.json`, `training_params.json` | The example params files, as they are |
| `dataset_types/eval.json`, `dataset_types/training.json` | Copies of the packaged files |
| `taxonomy/harmful.json`, `taxonomy/benign.json` | Copies of the packaged files |
| `jailbreak/augmentations.json` | A copy of the packaged file |
| `generation_axes.json`, `dimensions.json` | Copies of the packaged files |
| `README.md` | What follows, in short |

A scaffolded recipe runs exactly as its packaged example until a file is
changed. Every file in the folder wins over the packaged copy, so delete each
file you do not change to keep getting the library's updates to it. A
folder scaffolded before `training.json` set `outputs.outputs_per_input` 2
keeps its old copy, and so one response per input in a training run: add
the key, or delete the copy. It
refuses to overwrite a file the folder holds, naming them, unless asked to.

**Refreshing a stale copy.** A copy does not follow the library. A family
the library adds to the natural kind's `families.allow` in
`jailbreak/augmentations.json` is not drawn through a copy made before it;
the run says so at INFO, naming the families (`configs: the natural kind of
jailbreak/augmentations.json in ... does not allow ...`). A technique the
library adds to a family `dataset_types/eval.json` allows, and denies there
to keep the eval pool at its 36, is drawn through a copy made before it; an eval run
says so in a WARNING, naming the techniques (`configs: the eval attacks of
... draw ... technique(s) the library's eval type does not ...`). Add the
names to the copy's natural `families.allow` (or its adversarial
`techniques.deny`), delete a copy you did not change, or
write the folder again from the package with
`scaffold_setup(target_dir, overwrite=True)` (or
`python scripts/scaffold_setup.py TARGET --overwrite`). That rewrites every
file it scaffolds, the recipes and params files included, and so replaces
your edits to them: keep a copy of the files you changed and carry your
changes over afterwards.

Prompts are a separate tier: [the prompt overlay](#the-prompt-overlay) and
`scripts/scaffold_prompts.py`.

## The prompt overlay

Prompts are JSON files under `src/redact/prompts/<pipeline>/<category>/`,
loaded by `load_prompt(pipeline, category, prompt_dir)` or
`PromptTemplate.load(...)`. A folder holds one `.json` file, or else
`template.json`.

`prompt_dir` is an overlay. For each prompt, the loader looks in
`<prompt_dir>/<pipeline>/<category>/` first and falls back to the packaged
copy. A user folder holds only the prompts it changes, so the others stay
current as the library's change. At the start of a run (with `verbose`) the
driver logs how
many prompts are overridden, and warns about a folder that matches no
packaged prompt (a misspelt name) or that has several `.json` files and no
`template.json`.

Every stage passes `prompt_dir` to the loader, so each prompt a model call
is built from is read from the overlay when the overlay has it: the
constitution, the inputs (generation, format instruction and check), the
outputs, the paraphrases, every
technique of the jailbreak stage (translation included) and its benign data.
The conversation functions, which no recipe runs (`evaluate_conversations`,
`generate_attacks`, `score_attacks`, `optimize_attack`, `optimize`), take
`prompt_dir` as a keyword.

### Prompt files

```json
{
  "system_prompt": "... {Category} ...",
  "template": "... {sample} ...",
  "seed_fields": ["Category", "sample"],
  "few_shot_examples": [],
  "metadata": {"version": "1.0", "notes": ""}
}
```

- `system_prompt` and `template` are rendered with `str.format_map`. A
  `PromptTemplate` renders the system prompt once, when built, and the
  template on each call. A literal brace is doubled (`{{ }}`). A missing
  placeholder raises and names it, and the file when the prompt is loaded
  with `PromptTemplate.load`.
- `seed_fields` documents the placeholders; nothing reads it.
- `few_shot_examples` are message objects passed through as they are.
- A prompt built with a `format_style` (`numbered`, `structured_qa`) gets the
  matching instruction from `prompts/format_instructions/<style>/`, a file
  with one key, `instruction`, appended to the system prompt.
- Some prompts carry extra keys the code reads (`level_descriptions` for the
  typo rewrite, `answer_styles` for benign generation). Keep them in an
  override.

### The prompt tree

| Pipeline | Prompts |
|---|---|
| `constitution` | `harmful`, `dual_use_harmful`, `dual_use_benign`, `benign`, `general_benign` |
| `input` | `generation/from_constitution/long`, `generation/from_constitution/short`, `quality_check` |
| `output` | `generation/harmful`, `generation/helpful`, `quality_check/harmful`, `quality_check/helpful`: one generation prompt and one checker per [answer mode](stages/outputs.md#answer-modes) |
| `paraphrase` | `generation`, `generation/<model>` (a paraphraser's own, when it has one; the directory is the registry name), `check` |
| `jailbreak` | `scenario_generation`, `persona_invention`, `persona_scenario_generation`, `jailbreak_construction`, `payload_split_semantic`, `category_selection`, `benign_generation`, `extract_harmful`, `synonym_substitution`, `pseudonym`, `word_game`, `rewrite_with_typos`, `typo_check`, `scenario_check`, `persuasion_rewrite`, `tense_rewrite`, `model_distractor`, `preserve_check`, `translate`, `translate_retry`, `translate_check`, `translate_mt` (a machine translator's format: one user message, no system prompt; checked at the stage's start), `translate_check_back` |
| `conversation` | `judge`, `success_judge`, `pair_attacker` |
| `format_instructions` | `numbered`, `structured_qa` |

The tree also holds `input/category_check` and `jailbreak/benign_check`,
which no code loads.

Text the stages add at run time, the rejection feedback and the outputs'
length floor's reason, lives in
`content_moderation/feedback.py`, not in a prompt file.

### Scaffolding an overlay

```bash
python scripts/scaffold_prompts.py ./my_prompts           # a copy of the real prompts
python scripts/scaffold_prompts.py ./my_prompts --empty   # skeletons with the same keys
```

`--empty` replaces `system_prompt`, `template` and `instruction` with a TODO
that keeps the field's placeholders, so a skeleton renders with the same
values. Delete every file you do not change, then set `prompt_dir`.

## Other configuration files

| File | Holds | Read by |
|---|---|---|
| `configs/llm/gpu_pricing.json` | Provider, then GPU name as `nvidia-smi` reports it, then hourly USD per GPU; `*` is a provider's fallback | the cost roll-up |
| `configs/jailbreak/combination_spec.json` | `version`, `layer_order`, `layers_off_by_default` (the layers a kind leaves out unless its `layers.allow` names them, checked against `layer_order` as the spec loads), `decode_hint_rate` (the share of hinted encodings that write a decode hint, 0.5), `heavy_encoding_weight` (the weight, times `encode`'s, of the slot a budget round deals the ten heavy full-text encodings in, 0.2, above 0 and at most 1; left out, they are dealt as `encode`'s members), `layer_rates`, `layer_max_picks`, `families` (layer, order within the layer, incompatibilities, `dimension`, optional `kinds`, `check` and `calls`; a `description`, which nothing reads), `manipulation_blocked_obfuscation`, `request_obfuscation_allowed_families`, `request_obfuscation_refuses` (per family a request obfuscation is drawn from, the families it then keeps out of the combination: a translation keeps `suffixes`, `structural` and `neutral_container`; as the spec loads, each key must be a family a request obfuscation is drawn from, `translation` or `encode` (the light encodings `request_obfuscation_allowed_families` calls `encode_light`), and each name an obfuscation-layer family, since the step is drawn after every other layer's pick), `generators` (the families whose techniques come one per row of a data file, expanded into `techniques` as the spec loads), `techniques` (families, complexity, `requires_llm`, `requires_benign`, `encode_weight`, `needs_latin`, optional `max_tokens`, `cross_incompatible_with`, `kinds`, `check`, `calls`, `since`, the spec version that added the technique, and `deprecated`, the one that retired it) | `jailbreak/technique.py` (`load_spec`, `tag`), `pool.py`, `rules.py`, `sampling.py`, `chain.py`, `augmentation.py`; its `version` goes in each row and in the jailbreak run record |
| `configs/jailbreak/languages.json` | One row per language: `name`, `slug`, `complexity`, `script`, whether it has a `translation` and an `answer_language` technique, and the spec versions that added them: `since` (the row) and `translation_since` (its translation, when later) | `jailbreak/technique.py` (the spec's generators), `obfuscation/translation.py`, `requests/answer_language.py` |
| `configs/jailbreak/obfuscation/suffix_strings.json` | One row per adversarial suffix published in the literature: `slug`, `suffix`, `source`, `optimized_against`, `since`, and optionally `notes`, `complexity`, `deprecated`; each row is an attack-only technique `to_adversarial_suffix_<slug>` of the `suffixes` family | `jailbreak/technique.py` (the spec's generators), `obfuscation/suffixes.py`, which refuses at import a row without one of the five |
| `configs/jailbreak/augmentations.json` | The two default kinds and `natural_only_families` | `jailbreak/augmentation.py` |
| `configs/jailbreak/<layer>/*_templates.json` | The variant pools of the pure techniques: per directive, named variants (a `template`, or continuation's `prefix` and `suffix`) with an optional `weight` ([extending.md](extending.md#add-variants-to-a-template-pool)); `hacking/persona_templates.json` holds a frame per persona (with its `display_name`) and the `mechanics` drawn into them, `manipulation/fake_turns_templates.json` the fake turns | `jailbreak/directives.py: read_pools`, checked at import by `pool.py` (the persona frames by `hacking/persona_templates.py`) |
| `configs/jailbreak/multi_turn/crescendo_followups.json` | Follow-ups of the crescendo attack | `multiturn_attacks/attacks.py` |
| `configs/jailbreak/hacking/cognitive_techniques.json`, `personas.json` | The definitions of the cognitive techniques and of the named personas | `jailbreak/hacking/cognitive.py`, `personas.py` |
| `configs/jailbreak/obfuscation/decode_hints.json` | The decode hints of the encodings: a pool of four per technique, drawn at the spec's `decode_hint_rate` | `jailbreak/obfuscation/encoding.py`, checked at import by `pool.py` |
| `configs/jailbreak/rewrite/persuasion_techniques.json` | The twelve persuasion techniques the `rewrite` layer's `persuasion` family rewrites a prompt as: a display name, a definition and an example each | `jailbreak/rewrite/persuasion.py` |
| `configs/dataset_types/training.json`, `eval.json` | Each dataset type's default `stages`, `params` and `augmentations` | `recipe.py: load_dataset_type` |
| `configs/generation_axes.json` | The [generation axes](#the-generation-axes) of the inputs and outputs stages | `content_moderation/axes.py: resolve` |
| `configs/dimensions.json` | The [dimensions](#dimensions-and-the-runs-check) of the families and the axes | `dimensions.py`; `recipe.py: check_dimensions` |
| `configs/taxonomy/harmful.json` | The harmful categories: the default `taxonomy` | `dataset/taxonomy.py: load_taxonomy` |
| `configs/taxonomy/benign.json` | The default `benign_taxonomy`: the categories of the constitution's general-benign part, and the subcategories of the fsh and dap techniques' benign Q&A pairs | `dataset/taxonomy.py: load_taxonomy` |
| `configs/datasets/content_moderation_input.json` | A HuggingFace dataset id, split, columns and filters | `dataset/loading.py: load_from_config` |
| `configs/runs/*.json` | The example recipes (`eval_example.json`, `training_example.json`) and their params files (`params_eval_example.json`, `params_training_example.json`) | you |

A `taxonomy_dir` of your own that still holds `content_moderation_categories.json`
serves it for `harmful`, with a `DeprecationWarning`, for one release; rename
the file `harmful.json`.
