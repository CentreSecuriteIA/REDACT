# Stage: jailbreaks

Front `generate_jailbreaks` (`stages.py`); core
`jailbreak.pipeline.run_augmentation`, called once per kind.

## Purpose

Augment the accepted inputs with technique combinations, in two kinds:

| Kind | Writes | Default |
|---|---|---|
| `adversarial` | `Datasets/jailbreaks.csv`: jailbreak attacks | one round per input, by budget, from every technique but `translation` and `casual_writing` |
| `natural` | `Datasets/variations.csv`: natural variations | one technique per input, from the natural allow lists, of the 46 techniques the spec marks natural ([techniques.md](../techniques.md#the-natural-pool)) |

These are the packaged defaults, which a `training` run takes: both kinds
are on. An `eval` run's dataset type attacks the harmful inputs only, with
37 hard techniques in two count-driven rounds, one technique then two, and
switches the natural kind off
([configuration.md](../configuration.md#dataset-types)). The recipe's
`augmentations` section, over the dataset type's, or the front's
`augmentation=` argument, over the packaged one, says what each kind draws,
and switches a kind off with `null`
([configuration.md](../configuration.md#the-augmentation-section)).
The front returns the rows of `jailbreaks.csv`; read `variations.csv` with
`dataset.read_rows`.

How a combination is drawn, planned and run is in
[internals/jailbreak.md](../internals/jailbreak.md). The technique names are
in [techniques.md](../techniques.md).

## Inputs

- The accepted inputs of the run, or `inputs=`. A `sample` column is renamed
  `prompt`; the frame must have one of the two. Each row's id is its
  `sample_id`, else the content id of its text.
- Each kind keeps the rows of its `entry_types` (all by default). A kind
  whose `entry_types` match no input logs a WARNING `<kind> kind skipped:
  ...` and writes nothing; with `resume: false` its files are removed.
- With no input file the front raises `No input samples found`; with no
  accepted input it warns `jailbreaks skipped: ...` and writes nothing. With
  `resume: false` it also removes the files of each kind that is on: CSV,
  `.cut` and `.tmp` files, ledger, manifest, plan file and coverage table
  (`dataset.clear_stage`).

## Models

| Recipe role | Registry role | Default | Calls |
|---|---|---|---|
| `gen` | `uncensored_gen` | `venice-uncensored` | Every technique that calls a model, the benign data, and the judge of a machine translation |
| `translation` | `translation` | `translategemma-27b` | The `translation` family; resolved, planned and loaded only when a kind's pool holds a translation technique (`"translation": true`, as in the eval type's attacks) |

A chat translator (`deepseek-v3.2`, named in the recipe) makes up to 4
rounds of two calls, a translation and its check, both to it.

A machine-translation model (the `machine_translation` role, `translategemma-27b`) gets the text piece by piece, one request after another inside its unit: a paragraph, or a paragraph over 1,200 characters cut at sentence ends, each request capped at about three times its piece's tokens (1,536 at most). Each piece is checked at no model cost (empty; its GlotLID label among the language's accepted ones; its length ratio, over the language's median, within 0.5 to 2.0, which a reply cut at its cap is not), and a failing piece is resampled alone. When every piece passes, each is translated back to English and the generation model judges the original against the joined back-translation (prompt `jailbreak/translate_check_back`, 128 tokens), a rejection resampling every piece. Up to 4 rounds; the `DISCARDED` info names the piece (`piece=2/3`, `piece=all` for the judge) and why. A round is one call per piece when a piece fails, else
two per piece and one judge call: three for a one-piece text, twelve at
most over 4 rounds. Every request is its unit's own, on the async lane, so
other units keep both engines busy meanwhile. Before any call the stage
renders `translate_mt` once and stops, naming the file, unless it is one
user message with the model's three markers in order; without GlotLID
(`pip install -e ".[langid]"`, in `[vllm]` too) it warns once and skips the
language check.

## Settings

The augmentation section holds what each kind draws and how. The params
file holds the execution settings: `seed`, `batch_size`,
`auto_generate_benign`, `output_path`, `variations_path`, `manifest_path`,
`benign_path` ([configuration.md](../configuration.md#jailbreaks)). The
`eval` type sets `seed` 43 against the default 42, so an input that is in a
training run and an eval run does not get the same draws. The run passes
`benign_taxonomy`, `taxonomy_dir` and `config_dir` from the recipe.

## Unit and ledger key

One unit is one input at one iteration, for one kind: `(input_id,
iteration)`. A kind with `rounds` has one iteration per round. Each kind has
its own manifest, plan file, ledger and coverage table.

| | Value |
|---|---|
| Ledger key | `(input_id, iteration)` |
| Ledger record | `{"input_id": ..., "iteration": ...}` |
| Manifest row | `sample_id` (the input's id), `iteration`, `combination` (technique names in applied order), `settings` (the round's sampler settings and the `seed`, with `family_weights` when set), `category`, `entry_type`, `status` |

## Files

| File | Content |
|---|---|
| `Datasets/jailbreaks.csv` | The adversarial rows |
| `Datasets/jailbreaks.manifest.jsonl` | Its plan |
| `Datasets/jailbreaks.state.jsonl` | Its ledger |
| `Datasets/jailbreaks.coverage.csv` | What its plan drew |
| `Datasets/jailbreaks.plan.json` | What its plan was made with: `seed`, `rounds`, `pool` (the technique names), `family_weights`, `layer_rates` |
| `Datasets/variations.csv` and its four sidecars | The same for the natural kind |
| `Data_cache/benign/benign_samples.csv` | The benign data, shared by both kinds, with its `.manifest.jsonl` and `.state.jsonl` |

`variations.csv` goes to `variations_path` when given, else beside
`output_path` when that is given, else to `Datasets/variations.csv`.

Both CSVs have the same columns: the input's columns, then the fields of the
row. The input's `sample_id` and `accepted` columns keep their place and hold
the row's own values.

| Column | Meaning |
|---|---|
| input columns | `prompt` (the input text), `category`, `entry_type`, ... as the input had them |
| `sample_id` | Content id of `jailbreak`; for an error row, of `<label>:<input_id>:<iteration>`, the label being `jailbreak` or `variation` |
| `input_id` | The input's id |
| `jailbreak` | The augmented text |
| `technique` | The combination: technique names joined by `+`, `identity` when empty |
| `technique_info` | Each step's info joined by `;`; `noop=<name>` for a step that left the text unchanged |
| `complexity` | Sum of the techniques' complexities |
| `num_techniques` | Techniques in the combination |
| `is_noop` | Whether `jailbreak` equals the input |
| `accepted` | Whether the row is usable |
| `reasoning` | Why not, when `accepted` is false |
| `iteration` | The unit's iteration |
| `combination_spec_version` | The spec's `version` |

Columns of the coverage table: `level` (`layer`, `family` or `technique`),
`name`, `category`, `entry_type`, `iteration`, `count`. It counts what the
plan drew per group, and adds a zero row for each technique of the pool
never drawn. The planner logs a WARNING when a layer falls two or more picks
short of its rate in a group.

## The benign data

Techniques tagged `requires_benign` (the `fsh` and `dap` families) need
benign question-and-answer pairs. When a pending unit's combination needs
them and `auto_generate_benign` is on, the stage generates the pairs
`Data_cache/benign/benign_samples.csv` lacks, with the `gen` model, then
loads the file. With `auto_generate_benign` off, the stage loads the file as
it is; when there is no file, it stops before any unit with `Benign data not
found ...`.

The pairs cover each subcategory of the recipe's `benign_taxonomy` (the
packaged `configs/taxonomy/benign.json` by default: 11 categories, 58
subcategories; a category without subcategories is its own). A unit is one
`(main_category, sub_category, answer_type)`, the answer type `short` or
`long`: one call asking for 8 pairs. The units run on the worker pool, as the
stage's own units do. A reply that does not hold exactly 8 complete pairs is
rejected and the call made again: up to three calls in all. A unit rejected
every time is acked
`rejected` with no rows, and a WARNING names it. When no unit has a row, the
stage stops with `No benign pairs at ...`. The prompt
`prompts/jailbreak/benign_generation/` is read through the overlay.

The benign data always resumes: its ledger `benign_samples.state.jsonl`,
keyed `(main_category, sub_category, answer_type)`, records the finished
units, so a stopped run, or a larger taxonomy, generates only the units
missing. A file written before the ledger is taken over once. `resume:
false` does not clear it; delete the ledger to retry the rejected units, or
the CSV and its sidecars to generate everything again. Columns:
`main_category`, `sub_category`, `prompt`, `answer`, `answer_type`.

## How it resumes

Each kind resumes on its own, from its ledger, with the lock and cut-back of
[internals/dataset.md](../internals/dataset.md).

The plan is frozen. A resumed kind reuses its manifest and runs the planned
units its ledger does not record. Since the plan deals each family's
techniques evenly, follows the persona and cognitive vs impersonation rule
([techniques.md](../techniques.md#compatibility-rules)), and counts
the request obfuscation (when `allow_request_obfuscation` is on, as in the
escalation's rounds 3 and 4) against `max_obfuscations`, a fresh plan with
the same seed draws differently from one made before; a kept plan replays
unchanged. The
natural kind's pool is now what the spec's `kinds` mark natural, so a
natural plan made before raises on resume (its pool changed): run the
natural kind with `resume=False`. The build leaves out the benign and
dual-use benign variations such a plan made with a technique the natural
kind no longer draws ([build.md](build.md#training-mode)). The
5,000-character cap on `to_ascii_art` holds wherever it is drawn, the eval
type's pool included, and changes the text a seed draws: the font changes for some
texts from about 40 characters and for most from about 90, a text of about
830 characters or more is rejected where it used to be drawn, and so are
most inputs a `heavy` encoding lengthened first
([techniques.md](../techniques.md#what-the-families-do) has the figures).
A plan kept across this change mixes rows drawn without the cap, saved
before, and rows drawn with it, run after: a replayed plan gives the same
text only within one version of the library. Nothing flags it; run the
kind with `resume=False` for one rendering. A resume whose `seed`, pool, `rounds`,
`family_weights` or `layer_rates` differ from those in the kind's
`.plan.json` raises a `ValueError` that names them and `resume=False`; a
plan without that file runs as it was made, with a WARNING. A changed
`entry_types` is not compared: it only moves inputs in or out. A plan that
names a technique no registered function implements raises a `ValueError`
naming `resume=False` when that unit is built, which stops the stage. Inputs
added since the plan get no unit; a WARNING counts them and names
`resume=False`. `resume: false` removes that kind's CSV, `.cut` and `.tmp`
files, ledger, manifest, plan file and coverage table, then plans again. A
kind that is off is not touched ([running.md](../running.md#resume-fresh-runs-and-a-kill-partway-through)
says how to plan one kind again and keep the other).

## Rejected and failed units

| Case | Row |
|---|---|
| A technique rejects its result: a translation that fails its checks in all 4 rounds, a malformed word list, no word marked for a heavy word encoding, a scenario reply without its scene 3 times, a text no ASCII-art font draws whole or within its cap, an empty reply or text | `jailbreak` holds the text before that technique; `accepted` false; `reasoning` and `technique_info` start with `DISCARDED; technique=<name>` |
| A natural variation leaves the text unchanged | `accepted` false; `reasoning` `unchanged: the text is its input's`. An adversarial row that changes nothing stays accepted, with `is_noop` true; its `sample_id` is its input's, so the build leaves it out |
| A technique raises, or a prompt is too long for a model on the async lane (an OpenAI-compatible endpoint or vLLM) | An error row: the input text, `accepted` false, `technique_info` `ERROR: <message>`, `reasoning` the message |
| A call fails (a provider error, or a prompt too long on the grouped path: Anthropic, introspection) | The stage stops; saved units stay saved |

Every row, rejected or not, is acked, so a resume does not retry it.

## Running one prompt by hand

`jailbreak.apply_combination(fn, prompt, client=...)` runs one technique, or
one `combine_techniques(...)` result, on one prompt with blocking calls, and
returns `(text, info, accepted, reasoning)`. It loads the benign data when a
technique needs it, generating what the cache lacks with `client`; its
`prompt_dir` reaches the techniques and the benign data. `combine_techniques` sorts the techniques into layer
order but does not check the compatibility rules.

```python
from redact.jailbreak import apply_combination, combine_techniques
from redact.jailbreak.hacking.cognitive import to_persona_roleplay
from redact.jailbreak.obfuscation.encoding import to_base64, to_rot13
from redact.llms import ModelClient

# A transform is a plain call: (prompt) -> (text, info).
text, info = to_base64("How to pick a lock")
text, info = combine_techniques(to_rot13, to_base64)("How to pick a lock")

# A technique that calls a model is a generator: calling it directly returns
# the generator, not a result. apply_combination drives it with blocking calls.
client = ModelClient.create("venice-uncensored")
text, info, accepted, reasoning = apply_combination(
    combine_techniques(to_persona_roleplay, to_base64), "How to pick a lock",
    client=client,
)
```
