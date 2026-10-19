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
`training.json`, read in place of the packaged one. There are no other
types: a recipe naming one stops when it loads, before any stage runs.

| | `training.json` | `eval.json` |
|---|---|---|
| `stages` | `constitution`, `inputs`, `outputs`, `paraphrase`, `jailbreaks`, `build` | the same |
| `params` | `constitution`: `include_general_benign` true, `general_benign_categories` 10; `inputs.axes_weighting` `"log"` | `constitution` the same, and `constitution.entry_types` and `inputs.entry_types` `["harmful", "benign"]`; `inputs.axes` pins `directness` at `explicit`, and the inputs keep the table's weighting; `jailbreaks.seed` 43 |
| `augmentations` | none: the packaged default | the adversarial kind below; `natural` off |

The eval adversarial kind attacks the harmful inputs only, with a pool of
hard techniques whose label stays certain: the families `encode` (less the
seven light encodings, and less `sensitive_words`), `fsh`, `dap`, `framing`,
`structural`, `ascii_art`, `continuation`, `temporal`; the eight low-resource
translations (`to_swahili`, `to_thai`, `to_khmer`, `to_nepali`,
`to_zulu`, `to_scots_gaelic`, `to_bengali`, `to_javanese`), with
`"translation": true` and `family_weights` `{"translation": 3}`; and
`to_refusal_then_comply`. It runs two count-driven rounds, one technique
then two (`rounds` `[{"exact_techniques": 1}, {"exact_techniques": 2}]`),
with `allow_request_obfuscation` false.

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
and `outputs`, and the `modes` of `outputs`, are checked when the params
load (a params file's and a dataset type's): an unknown axis or value, a
weight below 0, a weighting other than `"table"` or `"log"`, a seed that is
not a whole number, or a `modes` that is not an object of entry types to
non-empty lists of distinct modes raises, the axes read against the
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
| `target` | `"both"` | `inputs`, `outputs` or `both` |
| `paraphrases_per_sample` | `1` | Paraphrase units per source row |
| `check` | `true` | Check each paraphrase with the `paraphrase_check` model |
| `batch_size` | `256` | Units per paraphrase batch and per check batch |
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
| `manifest_path` | `jailbreaks.manifest.jsonl` beside the output | The adversarial kind's plan, with its plan file `<stem>.plan.json` and coverage table beside it |
| `benign_path` | `Data_cache/benign/benign_samples.csv` | The benign data, with its manifest and ledger beside it |

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
sets up, so the writer axis varies beside the persona, framing, cognitive and
impersonation families (`tests/test_dimensions.py` records the ruling).

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

**Release note: the natural pool.** The natural kind now draws only what
the spec's `kinds` mark natural (46 techniques by default; `structural`,
`continuation`, `temporal`, the high and insane typos and the dialogue and
story answer formats left it). A natural-kind plan made before raises on
resume, since its pool changed: run the natural kind with `resume=False`.
A training build leaves out the benign and dual-use benign variations such
a plan made with a technique the natural kind no longer draws, with one
WARNING that counts them.

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
230, morse from about 250). A kept plan resumed across the change, an eval
plan included, mixes drawings made with and without the cap, since a
replayed plan gives the same text only within one version of the library;
run the adversarial kind with `resume=False` for one rendering.
[techniques.md](techniques.md#what-the-families-do) has the figures.

### Keys

Both kinds:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `entry_types` | list or `null` | `null` | Augment only inputs of these entry types; `null` for all. A kind whose entry types match no input logs a WARNING and writes nothing |
| `translation` | bool | `true` in the adversarial kind, `false` in the natural one | Whether the kind draws the `translation` family. The family lists do not change it; `layers.deny`, `techniques.deny` and `pure_only` still apply |
| `layers` | `{"deny": [...]}` | `{"deny": []}` | Layers left out |
| `families` | `{"allow": [...] or null, "deny": [...]}` | adversarial: `{"allow": null, "deny": []}`; natural: the natural families | Families admitted and removed |
| `techniques` | `{"allow": [...] or null, "deny": [...]}` | adversarial: `{"allow": null, "deny": []}`; natural: three techniques allowed, three denied | Techniques admitted and removed. Either `allow` names only what the spec's `kinds` let the kind draw |
| `pure_only` | bool | `false` | Drop every technique that calls a model or needs benign data |
| `family_weights` | `{family: number > 0}` | `{}` | A family's share within its layer when the plan deals families; 1 each by default |
| `iterations` | int >= 1 | `1` | Units per input, numbered by `iteration` |

Adversarial only:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rounds` | `null`, `"escalation"` or a list | `null` | The sampler settings per round; see [rounds](#rounds) |
| `max_complexity` | int >= 0 | `6` | Budget mode: the most total complexity (each technique scores 0 to 3) |
| `max_obfuscations` | int >= 0 | `2` | Budget mode: the most obfuscation families, the request obfuscation included |
| `layer_rates` | `{layer: 0..1}` | `{}`, the spec's rates | Target share of units that take a `hacking`, `manipulation` or `requests` technique, or, with `allow_request_obfuscation`, a `request_obfuscation` |
| `allow_request_obfuscation` | bool | `false` | Allow one more technique after the request technique, at the `request_obfuscation` rate: a light `encode` technique (the light `to_sensitive_words_encode_*` among them), or a translation when the pool has one. It is drawn before the other obfuscations, from what the earlier picks leave, and takes one of the `max_obfuscations` slots and its complexity from them; with `max_obfuscations` 0 there is none |

Natural only:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `exact_techniques` | int >= 1 | `1` | Techniques per variation |

The packaged natural kind allows the families `typos`, `casual_writing`,
`answer_language`, `answer_format`, `asking`, `indirect`; allows the
techniques `to_leetspeak_basic`, `to_adversarial_suffix_punctuation`,
`to_adversarial_suffix_emoji`; and denies `to_refusal_then_comply`,
`to_apology_prevention`, `to_prefix_injection`. The spec's `kinds` narrow
it further (step 6 below): its pool is 46 techniques
([techniques.md](techniques.md#the-natural-pool)). A variation that leaves
its input unchanged is rejected.

### How a kind's pool is built

From every registered technique, in this order:

1. remove the layers in `layers.deny`;
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
   says how to mark a technique). Every technique holds `adversarial`
   today, so this step leaves the attacks' pool as it is.

[techniques.md](techniques.md) lists every layer, family and technique name.

### Overlay rules

- A value given replaces the default's whole value. A list replaces a list.
- `layers`, `families` and `techniques` are overlaid list by list. An `allow`
  or `deny` given replaces the default's list of that name and keeps the
  other.
- A `deny` given without an `allow` also takes its names out of the default's
  `allow`. `"natural": {"families": {"deny": ["typos"]}}` is the natural
  default without typos. A deny that empties the default's allow raises:
  write the allow as well.
- An `allow` list may not be empty: write `null` for all, or name what to
  leave out under `deny`. A name may not be under both.
- The natural kind needs an allow list.
- An allow list may name only families and techniques whose spec `kinds`
  hold the kind. A natural allow list naming one the natural kind does not
  draw (`structural`, `temporal`, `continuation`, `to_prefix_injection`,
  ...) raises, naming it: a `config_dir` copy of `augmentations.json` made
  before these left the natural kind does, so take them out of its
  `families.allow`.
- `natural_only_families`, at the top of `augmentations.json`
  (`casual_writing` in the packaged file), are added to the adversarial `families.deny` unless the
  adversarial `families.allow` names them.

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

The packaged attacks have it `true`, drawing all 19 languages, and the
natural kind `false`; the eval type narrows its attacks to the eight
low-resource languages. They run on the `translation` role's default,
`translategemma-27b`, a local vLLM model
(a GPU: about 34 GiB planned, beside the FP8 24B on one 80 or 96 GB card).
It translates paragraph by paragraph; each piece is checked, then translated
back and judged by the generation model, up to 4 rounds
([stages/jailbreaks.md](stages/jailbreaks.md#models) gives the calls). The
translation model is planned and loaded only when a kind's pool holds a
translation technique. An allow list admits the whole family while the switch
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
`max_obfuscations`, `layer_rates` and `allow_request_obfuscation`. A round
without it picks by budget.

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
requests 0.7, request_obfuscation 0.5). Within a layer, families are dealt by
count over `family_weights`, so `{"persona": 2}` gives `persona` twice the
share of each other hacking family; within a family, its techniques are
dealt evenly among those a unit may take (the complexity budget still
favours a family's cheaper techniques). `family_weights` may name only
families the kind's pool draws. [internals/jailbreak.md](internals/jailbreak.md#balanced-dealing)
explains the dealing.

`hacking` and `manipulation` exclude each other, so their rates should sum to
at most 1; beyond it, `manipulation` falls short of its rate and the plan
logs a WARNING. With `allow_request_obfuscation`, `request_obfuscation` is
dealt the same way, over the group's units.

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
| `default_extra_body` | `null` | Provider parameters sent with each call (OpenAI-compatible only) |
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
| `venice-paraphraser` | `vllm`, same weights and `fp8` as `venice-uncensored-fp8`, one engine with it | | `paraphraser` | |

`venice-paraphraser` stands in for a trained defingerprinting model, which
lives in another repository. Register the real one with the `paraphraser`
role.

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
| `VLLM_PROCESS_NAME_PREFIX` | the card check | The title prefix of vLLM's engine processes, default `VLLM`, named in the `kill -9` advice |

## Your own config folder

A recipe's `config_dir` names a folder laid out as the packaged `configs/`.
These files are read from it when it has them, each file on its own, and
from the package otherwise:

| File | Replaces |
|---|---|
| `<config_dir>/dataset_types/eval.json`, `training.json` | the [dataset type](#dataset-types) file of that name. A file of another name there is no new type: it is not read, and the run warns about it as ignored |
| `<config_dir>/taxonomy/<name>.json` | a taxonomy of that name (`taxonomy`, `benign_taxonomy`); a new name there is a taxonomy of your own. `taxonomy_dir`, when given, wins and is read alone |
| `<config_dir>/jailbreak/augmentations.json` | the default [augmentation section](#the-augmentation-section), both kinds and `natural_only_families` |
| `<config_dir>/generation_axes.json` | the [generation axes](#the-generation-axes) table, whole: its labels, hints, weights and `not_with`. Its axes are checked against the folder's `dimensions.json` when it has one |
| `<config_dir>/dimensions.json` | the [dimensions](#dimensions-and-the-runs-check) table, whole. It must hold every tag the families carry; the run's check reads its `same_as` and `undecided` |

`config_dir` is relative to the recipe's folder, as `params_file` is. The
technique spec (`combination_spec.json`), the technique templates and other
files under `jailbreak/`, `llm/` (`models.json`, `roles.json`,
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

`scaffold_setup(target_dir, overwrite=False)` (also
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
file you do not change to keep getting the library's updates to it. It
refuses to overwrite a file the folder holds, naming them, unless asked to.
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
| `paraphrase` | `generation`, `check` |
| `jailbreak` | `scenario_generation`, `persona_scenario_generation`, `jailbreak_construction`, `category_selection`, `benign_generation`, `extract_harmful`, `synonym_substitution`, `rewrite_with_typos`, `translate`, `translate_retry`, `translate_check`, `translate_mt` (a machine translator's format: one user message, no system prompt; checked at the stage's start), `translate_check_back` |
| `conversation` | `judge`, `success_judge`, `pair_attacker` |
| `format_instructions` | `numbered`, `structured_qa` |

The tree also holds `input/category_check`, `jailbreak/benign_check`,
`jailbreak/scenario_check` and `jailbreak/typo_check`, which no code loads.

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
| `configs/jailbreak/combination_spec.json` | `version`, `layer_order`, `layer_rates`, `layer_max_picks`, `families` (layer, order within the layer, incompatibilities, `dimension`, optional `kinds`; a `description`, which nothing reads), `manipulation_blocked_obfuscation`, `request_obfuscation_allowed_families`, `techniques` (families, complexity, `requires_llm`, `requires_benign`, `encode_weight`, optional `max_tokens`, `cross_incompatible_with`, `kinds`) | `jailbreak/technique.py` (`load_spec`, `tag`), `pool.py`, `rules.py`, `sampling.py`, `chain.py`, `augmentation.py`; its `version` goes in each row and in the jailbreak run record |
| `configs/jailbreak/augmentations.json` | The two default kinds and `natural_only_families` | `jailbreak/augmentation.py` |
| `configs/jailbreak/<layer>/*_templates.json` | Template variants of the pure techniques | `jailbreak/directives.py: load_templates` |
| `configs/jailbreak/multi_turn/crescendo_followups.json` | Follow-ups of the crescendo attack | `multiturn_attacks/attacks.py` |
| `configs/jailbreak/hacking/cognitive_techniques.json`, `personas.json` | The definitions of the cognitive techniques and of the named personas | `jailbreak/hacking/cognitive.py`, `personas.py` |
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
