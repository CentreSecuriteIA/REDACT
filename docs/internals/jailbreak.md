# jailbreak/: the augmentation engine

`jailbreak/` holds the technique library and the engine that runs it: the
augmentation section, the pool, the rules, the plan, and the stage core
`run_augmentation`. The engine plans every unit first, then drives the
units. The user-facing side is [stages/jailbreaks.md](../stages/jailbreaks.md);
the technique names are in [techniques.md](../techniques.md).

## Responsibilities and boundaries

- A technique never calls a model itself. A technique that needs one is a
  generator that yields `LLMRequest`s; the driver batches or streams them.
- `configs/jailbreak/combination_spec.json` is the one source of families,
  layers, complexities and compatibility rules. Importing `redact.jailbreak`
  fails when the spec and the registered functions name different
  techniques. A family whose techniques come one per row of a data file
  (the two language families, from `languages.json`, and the published
  adversarial suffixes of `suffixes`, from `obfuscation/suffix_strings.json`)
  is a `generators`
  entry, which `load_spec` expands into technique entries, so every reader
  sees one `techniques` dict; the family's module builds its functions from
  the same rows ([extending.md](../extending.md#add-a-jailbreak-technique)).
- The plan is frozen in the manifest. A replay reads each combination by
  name through the registry of every technique, not through the run's pool,
  so a plan made with another pool still replays. A technique is retired
  with `deprecated` in the spec, never removed: it stays registered and
  replays, but leaves the pools (`test_registry_only_grows` holds every name
  the registry fixture pins to the registry).
- A technique draws its random choices from the `rng` it is given, so a
  replayed plan gives the same text within one version of the library. A
  technique changed between versions draws differently: the ASCII-art cap
  changes the font `to_ascii_art` draws for most texts over about 90
  characters, so a plan kept across it mixes the two renderings
  ([techniques.md](../techniques.md#what-the-families-do)).

## Modules

| Module | Holds |
|---|---|
| `technique.py` | The technique contract, `load_spec` and `read_spec` (the spec, its generators expanded), `data_rows`, `tag`, `DISCARDED`, `KINDS`, `DEFAULT_KINDS`, `family_kinds`, `FIRST_VERSION`, `version_key`, `decode_hint_rate` (the spec's share of hinted encodings written with a hint), `heavy_encoding_weight` (the spec's weight of the heavy full-text encodings' slot) |
| `rewrite/`, `hacking/`, `manipulation/`, `obfuscation/`, `requests/` | The techniques, one module per family (`distractor.py` and `smalltalk.py` each hold both families of their pair), each with a getter; `get_all_rewrite_functions` (16), `get_all_hacking_functions`, `get_all_manipulation_functions`, `get_all_obfuscation_functions`, `get_all_request_functions` per subpackage; `all_techniques` joins the rewrite layer last, so the others keep their registry order. `manipulation/benign.py` makes the benign data: one unit per subcategory of the benign taxonomy and answer type, keyed `(main_category, sub_category, answer_type)`; the fake turns (`manipulation/fake_turns.py`) need none. `hacking/persona_templates.py` checks its persona frames at import (`check_pools`): a frame is its technique's own text, and only the mechanics are its `variant_pool` |
| `pool.py` | `all_techniques`, `technique_pool(resolved)`, `kind_of(resolved)`, `languages`, `registry`, `chain_tags` (a row's `families` and `layers` from its `technique`, which the build writes; `?<name>` and `?` for a name no registered technique has), `uses_translation` (a translation or a dialect technique), `uses_check` (whether a pool holds a technique the check model checks), `uses_check_model` (that, or a technique that needs the benign data, whose pairs the check model reads), `variant_pools` (each template pool's family, techniques and count of variants on, which the variant floor test ratchets); at import, the spec against the registries, the families' dimensions and the techniques' variant pools |
| `augmentation.py` | `KINDS`, `defaults(config_dir)`, `resolve(section, kind, base=None, config_dir=None)`, `rounds_of(resolved)`, `apply_flat_keywords`, `FLAT_KEYWORDS` |
| `rules.py` | `compatible(selected, pool, remaining_complexity)`, `request_obfuscation_candidates`, `after_request_obfuscation` (what may still join once a pick is the request obfuscation) |
| `sampling.py` | `sample_combination`, `sample_exact_combination`, `_pick_by_family`, `deal_key` (the name a dealer counts a technique under), `is_heavy_encoding`, `REQUEST_SCOPE`, `HEAVY_ENCODINGS` |
| `plan.py` | `plan_run`, `assign_combination` (with the opt-in group budget: `_allowance`, `UNIT_HEADROOM`), `combination_of`, `coverage`, `plan_inputs` (the record of the `<stem>.plan.json` beside the manifest, written through `dataset/planfile.py`, its `spec_version` dating the pool and its `rules` the rules it was drawn by, `rules_id`), `default_escalation_schedule`; it deals from `dataset.dealing.Dealer` |
| `chain.py` | `run_chain` (with `scenarios`, a list, it hands each technique that takes it a `record_scenario` of its own), `combine_techniques`, `in_layer_order`, `apply_combination` |
| `units.py` | `pending_units` (with `scenarios`, each unit collects its scenes), `is_pending`, `unit_gen`, `row_of`, `error_row_of` |
| `pipeline.py` | `run_augmentation`, the stage core; `_library_change`, the kept plan's pool against what the library added and retired since |
| `variants.py` | `variants_of` (a row's template variants, paired by position), `write_variants` (the kind's `<stem>.variants.csv`), `COLUMNS` |
| `scenarios.py` | `rows_of` (a finished unit's scene rows), `write_scenarios` (the kind's `<stem>.scenarios.csv` kept to its rows), `COLUMNS` |
| `catalog.py` | The technique catalog, from the code: `build`, `refusal_rules` (mirrors `rules.compatible`, rule by rule), `compatible_with`, `code_reach` (the prompt steps and chain arguments a technique's code reaches, read off its bytecode), `dumps` and `markdown`; `python -m redact.jailbreak.catalog` (`scripts/catalog.py`, `--check` to only compare) writes `docs/catalog.json` and [catalog.md](../catalog.md) |
| `directives.py` | The variant pools of the pure techniques: `read_pools` (a template file's pools by directive, whatever its shape), `PoolRef` and `uses_pool` (the pool a technique draws from, set as its `variant_pool`), `check_pools` (called by `pool.py` at import), `enabled`, `draw_variant`, `apply_template_directive`; `load_templates` (a file as written) |

## Techniques

A **transform** is a plain function:

```python
def to_x(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    ...                                   # returns (text, info)
```

A technique that calls a model is a **generator**: it yields `LLMRequest`s,
is resumed with each reply, and returns `(text, info)`. A multi-round
technique yields more than once.

```python
def _y_gen(prompt: str, *, gen_model=None, prompt_dir=None, **kwargs) -> TechniqueGen:
    reply = yield LLMRequest(gen_model, messages)
    return reply.strip(), "key=value"

def to_y(prompt: str, **kwargs) -> TechniqueGen:
    return (yield from _y_gen(prompt, **kwargs))
```

- `info` is `""` or `key=value;...`. An info that starts with `DISCARDED`
  rejects the result.
- A technique gets the keywords it declares, out of: `gen_model` (the `gen`
  model's name), `translate_model`, `check_model` (the `check` model's
  name, the `gen` model's when the recipe names none; translation keeps
  its own checkers), `benign_data`, `prompt_dir`, `rng`. One
  that declares `**kwargs` gets them all. Every generator technique has the
  signature `(prompt, **kwargs)`; a test holds them to it.
- At import, `tag(all_techniques())` sets each function's spec tags as
  attributes: `families`, `complexity`, `requires_llm`, `requires_benign`,
  `encode_weight`, `needs_latin` (true or false, else import fails),
  `max_tokens`, `cross_incompatible_with` (families this
  technique itself may not join), and from the first family `layer` and
  `within_layer_order`; `kinds` (the augmentation kinds that may draw it)
  is the technique's own, else its first family's, else `["adversarial"]`.
  A `kinds` that is not a non-empty list of `adversarial` and `natural`
  raises at import. `check` (what checks its model reply: `typos`,
  `scenario`, `word_list`, `translate` or `preserve`; `None` for none) and `calls` (the
  model calls one application makes, generation and checks, as
  `[low, high]`; `None` when the spec gives none) come the same way, the
  technique's own, else its first family's; a value outside those raises at
  import. Both only declare: nothing draws by them. `since` is the spec version that added the technique
  (`FIRST_VERSION`, `"1.0"`, when its entry gives none) and `deprecated`
  the one that retired it (`None` while it is active). Versions are dotted
  whole numbers and compare as such (`version_key`: `"1.10"` comes after
  `"1.9"`); a `since`, a `deprecated` or a spec `version` that is not one
  raises at import. So does an entry with a key outside `TECHNIQUE_KEYS`
  (`complexty` used to give complexity 1 silently), `families` that are
  not a non-empty list of the spec's families (an unknown one used to put
  the technique in the obfuscation layer), a `complexity` that is not a
  whole number from 0 to 3, a `requires_llm` or `requires_benign` that is
  not true or false, an `encode_weight` other than `light`, `heavy` or
  none, and a `max_tokens` of `true`. `read_spec` checks the rest: each
  family's keys (`FAMILY_KEYS`) and a `layer` of `layer_order`, the spec's
  `layer_rates` (numbers from 0 to 1, for a layer or
  `request_obfuscation`), each generator keyed by a family of the spec, and
  each data row (its `where` column true or false, its `complexity`, its
  versions, and a `<family>_since` only for a family made from that file,
  so a misspelled `translaton_since` raises), its messages naming the data
  file; `data_rows` raises for a row without a `slug`.
- A technique may carry an **applicability**: `@applicable_when(predicate)`
  sets its `applicable`, `predicate(text, entry_type) -> bool`, and
  `applies(fn, text, entry_type)` reads it, true for a technique without
  one. The plan draws a technique only for an input it is true for
  ([balanced dealing](#balanced-dealing)); a predicate that raises stops
  the plan with a `RuntimeError` naming the technique and the input. The
  predicate calls the helper
  the technique itself calls, so the two cannot drift apart. The registry
  fixture records `has_applicable`, a bool, not the function. Today 22
  have one: `to_abbreviations` and `to_stripped_accents`, each applying only
  to a text it changes; the three benign roles (spec 1.2), which apply to
  benign and dual-use benign inputs only (`benign_role.is_benign_input`, on
  the entry type alone), and the three guardrail addresses (spec 1.4), to
  every input but an absolutely benign one
  (`guardrail_address.is_targeted_input`); the four transliterations (spec
  1.4), to a text with a Latin letter (`transliteration._has_latin`); and
  the ten code-switches (spec 1.4), to a text of at least two words
  (`code_switching.can_mix`, which the technique also checks before any
  call, refusing a shorter prompt).
- A pure template technique draws its text from a variant pool, one
  directive's variants in its template file, and names it with
  `@uses_pool(file, directive, slots, info_key, request_slot)` (its `variant_pool`;
  `info_key` the key of its info that names the variant drawn, `variant`
  unless it says otherwise; the answer languages share one pool, set by
  their factory). `read_pools` reads a
  file of either shape (`{name, template}` lists, continuation's
  `{name, prefix, suffix}`; a single template entry, the persona frames',
  is a pool of one) into lists of variants; `pool.py` checks every named pool at
  import (`check_pools`: unique non-empty names of letters, digits, `_` and
  `-`, `{prompt}` exactly once in each template and only the technique's
  slots, each written alone with no conversion or format spec, a `weight` a
  number of at least 0, one variant above 0), naming the file, directive
  and variant; a persona frame also names `{mechanic}` exactly once. A pool marked
  `request_slot=False` (`PoolRef.request_slot`) holds pieces the technique
  fills and sets into another template, which holds the request: its
  templates need no `{prompt}`, and one that names it fails as an unfilled
  slot. `draw_variant`
  draws an unweighted pool with `rng.choice`, so a seed draws what it drew
  before pools had weights (`tests/fixtures/template_draws.json` pins every
  template technique's texts and draws by digest), and a weighted one with
  `rng.choices`; a variant of weight 0 is never drawn, and a pool of one
  makes no draw. A
  technique's text belongs in a template file: a lint
  (`tests/jailbreak/test_prose_in_code.py`) fails on a string of eight or
  more words in the package's code that is not a docstring or a message,
  unless `tests/fixtures/prose_in_code.json` allows it by file, function
  and word count; that list only shrinks.

`run_chain(techniques, text, ...)` runs one input through its techniques in
the order given. A transform runs inline; a generator's requests are yielded
up, each capped at the technique's `max_tokens` tag and, under capture,
tagged with an internals id. Each technique gets
`rng = random.Random(stable_seed(seed, label, input_id, iteration, name))`,
the label being `jailbreak` or `variation`, so the two kinds draw apart, and,
when it takes it, `source_prompt`: the text the chain started from (the
unit's input) unless the caller gives one (`None` is none). The selected `fsh` and `dap`
variants, which apply last, pick their examples' subcategory on it, not on
the encoded, translated or wrapped question they then insert; in a kept plan
where they run first it is the text they read before, unless a rewrite step
ran before them, whose output they read then and whose input they read now
(`_select_kwargs` passes it, as the other keywords, to the techniques that
take it). A
step that leaves the text unchanged is noted `noop=<name>`. A step that
leaves the text empty or blank is a rejection,
`DISCARDED; technique=<name>; reason=empty text`. The first rejection stops
the chain, returning the text before that technique, with the info
`DISCARDED; technique=<name>; ...`.

`combine_techniques(*fns)` wraps techniques as one, sorted into layer order
(`in_layer_order`), and runs it with blocking calls. It does not check the
rules. A caller that combines techniques itself (a notebook, a transform for
the multi-turn crescendo) gets the layer order of the library it runs on: the
manipulation layer last since the first eval release, as `sample_exact_combination`,
which sorts the same way.

## Plan, then drive

`run_augmentation(inputs, pool, *, client, out_path, rounds, ...)` is the
core, called once per kind:

1. **Registry.** `registry(all_techniques())`, every registered technique by
   name.
2. **Ledger and resume.** `Ledger.sidecar(out_path, ("input_id",
   "iteration"))`, then `resume_state(..., artifact=out_path,
   manifest=manifest)`: the lock, the cut-back, the finished keys.
3. **Plan.** On a resume with a manifest, reuse it: `planfile.changed_plan`
   compares the run's seed, rounds, `family_weights` and `layer_rates`
   (`plan_inputs`, less `pool`, `spec_version` and `rules`) with
   `<stem>.plan.json`; a difference raises a `ValueError` naming the keys
   and `resume=False`. The pool is compared apart (`_library_change`): the
   names the run's pool adds must each have a `since` after the plan file's
   `spec_version` (`"1.0"` in a plan file written before it was recorded),
   and the names it lacks must each be registered with a `deprecated` after
   it. Then the difference is the library's, and the plan runs as it was
   made, its retired techniques included. Any other difference raises as
   `pool`: a section that adds or leaves out a technique, or a technique no
   longer registered. Known limit: a section that newly names a family all
   of whose techniques came after the plan reads as growth, and warns. The
   rules are compared apart too (`_rules_change`): `rules_id` of the plan's
   pool names and rounds, as the library tags them today, against the plan file's
   `rules` (unknown in a plan file written before it recorded them). When
   the library added or retired techniques, or the plan's `spec_version` or
   rules are not today's, one WARNING (`_warn_since_plan`, given after the
   pass of step 4) names both versions, what was added and retired, whether
   the rules changed or are not recorded, and how many pending units hold a
   combination `compatible()` now refuses or whose request obfuscation
   keeps out another of its techniques (`_refused`: the request
   obfuscation is the obfuscation right after the request, in every plan's
   order, and `after_request_obfuscation` gives what it keeps out); it says
   the kept plan does not draw the added techniques and runs those units as
   planned (and, when the rules changed or are not recorded, that each
   combination applies in the order it records), and names `resume=False`.
   The rows of a kept plan record its `spec_version`, not the library's. A manifest without the plan file logs
   a WARNING and runs unchecked, its rows recording the library's version.
   Otherwise the old coverage table, plan file and variants table are
   removed, and `plan_run` writes the plan file, one manifest row per input
   and round, then the coverage table.
4. **One pass over the plan** to count the pending units, see whether any
   needs benign data (if so, generate what the cache lacks for the benign
   taxonomy's subcategories, on the pool with its own ledger, then load it;
   with `auto_generate_benign` off, the cache is loaded as it is, and a
   missing cache raises before any unit), and, on a resume, count the inputs
   the kept plan does not hold, logged as a WARNING, and the pending units
   whose combination today's rules refuse, for step 3's WARNING.
5. **Drive.** `pending_units` streams the plan: for each planned row of an
   input in this frame and not in the ledger, it builds the combination
   (`combination_of`) and the unit's generator (`unit_gen`, over
   `run_chain`) only when the driver takes it. `drive_units` runs them with
   `pool_size(batch_size, client, check)` live and `cap=batch_size`. A plan row
   naming a technique no registered function implements raises a
   `ValueError` naming `resume=False`, which stops the stage.
6. **Save.** `on_done` makes the row (`row_of`); `on_error` makes an error
   row (`error_row_of`) for anything the unit raises; both go to a
   `CommitBuffer` with the ack `{"input_id", "iteration"}`. A
   `PromptTooLongError` reaches `on_error` only on the async lane
   (OpenAI-compatible, vLLM), raised at the unit's `yield`. On the grouped
   path (Anthropic, introspection) a failed `generate()`, a too-long prompt
   included, propagates and stops the stage.
7. **Count the variants.** `write_variants` reads the kind's CSV in chunks
   and writes `<stem>.variants.csv` beside the manifest (below). When the
   kind has a `<stem>.scenarios.csv`, `write_scenarios` keeps it to the
   rows the CSV holds (below).
8. **Read back** when asked.

A planned unit whose input is not in the frame is skipped; an input that is
not in the plan gets no unit. A unit's key in the driver is its position
among the pending units, so two plan rows with one `(input_id, iteration)`
stay two units.

## Drawing one combination

`sample_combination(rng, pool, ...)` has two modes.

**Budget mode** (no `exact_techniques`) picks layer by layer, each pick from
the techniques still `compatible` with what is picked. The phases carry the
numbers `sampling.py` gives them, in the order they run (the request
obfuscation is phase 5 and runs before phase 4):

- **Phase 0, `rewrite`**, when the round includes it (`include_rewrite`,
  false by default) and the layer joins at its rate (the spec's is 0). It
  is decided only when included, at a rate above 0 and with a rewrite
  technique in the pool; otherwise it makes no rng draw and no dealer
  call, so every draw is the one made before the layer existed
  (`tests/jailbreak/test_rewrite_layer.py` pins it).
- **Phase 1, `hacking`**, when the layer joins (its rate).
- **Phase 2, `manipulation`**, when the layer joins.
- **Phase 3, `requests`**, when the layer joins.
- **Phase 5, the request obfuscation**, decided and drawn before phase 4:
  with `allow_request_obfuscation`, at least one `max_obfuscations` slot and
  a request technique picked in phase 3, when it joins at the
  `request_obfuscation` rate (the dealer's `include`, as for the layers, over
  the group's units, so the units holding a request take it more often; a
  per-unit draw without a dealer), one light `encode` technique (the light
  `to_sensitive_words_encode_*` among them) or a translation, applied right
  after the request technique, before the manipulation layer. A unit without
  a request is not asked, so the step never lands after the outermost
  wrapper. A translation drawn this way applies after every obfuscation and
  the request (a manipulation technique, the only step after it, never meets
  a translation), so it would translate a suffix, a wrapper's keys or a
  container's lines: the spec's `request_obfuscation_refuses` keeps the
  `suffixes`, `structural` and `neutral_container` families out of its
  combination, and the sampler narrows phase 4's pool by it
  (`rules.after_request_obfuscation`). A translation drawn in phase 4 runs
  before those steps and is not limited. The translation family already
  refuses ten of the other thirteen obfuscation families, so with this table
  a translation drawn as the request obfuscation is the unit's only
  obfuscation, and the `max_obfuscations` slots it leaves go unused.
- **Phase 4, `obfuscation`**: one pick after another, from what the budget
  leaves, until `max_obfuscations`, less the request obfuscation's slot, or
  until no technique fits.

The apply order is the layers' (`rewrite`, `hacking`, `obfuscation`,
`requests`, `manipulation`), whatever the pick order (a rewrite first, a
manipulation technique last), except the request obfuscation, which goes
right after the request (`sample_combination` inserts it there, so it still
precedes a manipulation technique). The manipulation layer comes last since
the first eval release: it inserts the finished question among its examples or
after its invented turn, so the obfuscations, the request and the model
steps change the question alone. Within the obfuscation layer it is each
family's `within_layer_order`, in tens: `translation` 10, `typos` 20,
`casual_writing` 30, `noise` 35, `sensitive_words` 40, `wording` 45,
`tokenbreak` 50, `word_art` 55, `suffixes` 60, `payload_split` 65, `encode` 70, `structural` and `neutral_container` 80 (each
excludes the other, so they never meet), `ascii_art` 90. The whole-text
rewrites come before the word-level steps, which they would otherwise undo;
a family added later takes a number between two, as `noise` did. The manifest records the order a
combination applies in and a replay keeps it (`combination_of`), so a plan
made before the renumbering, when `sensitive_words` and `tokenbreak` came
first, still runs as it was made, and one made before the first eval release
still applies its manipulation technique second. `rules_id` holds the
spec's `layer_order`, so the resume of such a plan warns that the rules
changed. A technique above the remaining complexity is
never eligible, for the first pick too. The request obfuscation is drawn
from the techniques still compatible and within the remaining complexity,
so it holds its slot and its complexity before the obfuscations fill the
rest: a combination never has more than `max_obfuscations` obfuscations.
The packaged attacks allow it, so half their units, at the spec's rate of
0.5, take a request obfuscation in one of their `max_obfuscations` slots; the
eval type sets it false, and its count rounds never read it (a test pins
the eval type's draws). With `allow_request_obfuscation` false the sampler
never decides it (no dealer call, no candidates drawn; a test holds it), so
the draws are those of a sampler without the step; a digest of the budget
draws pins them as a change detector, re-pinned on purpose whenever a change
moves a draw. The combined function
names its request obfuscation in `request_obfuscation` (`None` without
one), which `_observe` counts in the dealer.

In budget mode `max_complexity` is what the unit may spend. The planner
may pass a larger number in its place, the unit's allowance under the
[group budget](#the-group-budget); the sampler reads it as any budget.

**Count mode** (`exact_techniques: n`) picks exactly `n` compatible
techniques from any layer, ignoring complexity. It returns fewer when the
rules leave nothing; with the full pool, 1 and 2 are always reachable. It
reads no rate and no `include_rewrite`: a rewrite technique in the pool is
dealt as any family's, so the pool (a kind's `layers.deny`), not a rate,
keeps the rewrite layer out of a count round.

Every pick is **family first** (`_pick_by_family`): a family, then a
technique within it, both dealt when the plan passes its dealer
([balanced dealing](#balanced-dealing)). The 24 translation languages are
one choice, not 24.
The rules are those of [techniques.md](../techniques.md#compatibility-rules);
among them, a translation is never paired with `sensitive_words`, `wording`,
`tokenbreak` or `word_art`, whose marking call mostly fails on translated
text, nor with `typos`, `casual_writing`, `noise` or `payload_split`, one
change more to a text already hard to read, in either pick order (a request
obfuscation that is a translation takes the rules into the obfuscations drawn
after it as any other pick does).

## Balanced dealing

`plan_run` does not draw each unit alone. It groups the units by
`(category, entry_type, iteration)`, sorts each group by input id, and deals
the group from one dealer (`dataset/dealing.py`, shared with the inputs' and
outputs' generation axes), so every group gets its share whatever the input
order.

- **Applicability.** A unit draws from the techniques of the pool that
  apply to its input (`applies`, on the input's text and entry type), in
  pool order; with every technique applying that is the whole pool, so
  the draws are those of a pool without an applicability. The predicate
  sees the input, not what earlier steps of the combination wrote. A unit
  no technique of a layer applies to takes no pick there; the dealer counts
  what each unit drew, so the group's next unit that can take the layer
  does, and a technique left out for some inputs is dealt to the others.
  A unit no technique applies to gets the empty combination, the identity.
  Both modes and both kinds draw this way; a kept plan replays its names
  without asking any predicate.
- **Layers.** A layer joins a unit when the group's picks of that layer so
  far are below `floor(rate x position + jitter)`, where `position` counts
  the group's units so far and `jitter` is a number in [0, 1) seeded per
  group, layer and position. A group gets `rate x size` picks, rounded down
  or up, every prefix of the group is within one pick of `rate x position`,
  and a group of one takes the layer with probability `rate`. The request
  obfuscation, when allowed, is dealt the same way under the name
  `request_obfuscation`. The jitter is drawn afresh for each unit: with one
  offset per group and layer, a layer's picks repeated every `1 / rate`
  units, so two layers of one rate (hacking and the request obfuscation,
  both 0.5) met on every unit or on almost none for the whole group, by the
  seed, and `ascii_art`, which refuses both a hacking technique and a
  translation, was drawn 8 to 51 times in 900 units by the seed (commit
  `590babe`, seeds 1, 2, 42 and 43, three groups); with the jitter they meet
  on about a quarter of the units (49% to 55% of the hacking units, seeds 1,
  2 and 42 to 46) and it is drawn 59 to 64 times.
- **Families.** Among the eligible families, the dealer picks the one with
  the lowest count over its weight (`family_weights`, 1 by default), ties
  broken by the unit's rng. The request obfuscation's families are counted
  apart, under `REQUEST_SCOPE` (`"request_obfuscation:"` before the family's
  name, `sampling.py`), with the same weights (`plan_run` gives the scoped
  names the family's), so its picks do not count against the obfuscation
  layer's. Counted together, its `encode` and `sensitive_words` picks made
  those the layer's families dealt most, and the layer drew them only when
  no other family fitted: 14 techniques went undrawn in 3,000 units (before
  commit `ec8bb42`, seed 42, three groups; none does since).
- **The heavy encodings' slot.** In a budget draw's obfuscations (phase 4)
  the dealer deals the ten heavy full-text encodings (first family
  `encode`, `encode_weight` heavy; not the three heavy
  `to_sensitive_words_encode_*`, which are `sensitive_words` techniques)
  under a name of their own, `HEAVY_ENCODINGS` (`"encode:heavy"`), beside
  `encode`, at the spec's `heavy_encoding_weight` (0.25) times encode's
  weight (`plan_run` adds it to the dealer's weights). `deal_key(fn,
  heavy_apart=)` gives the name, read by `_pick_by_family` and `_observe`;
  `assign_combination` passes `heavy_apart` for a budget draw only. Dealt
  as encode's members, a heavy one fitted only a unit with 3 points left,
  and the light members kept encode's count high, so encode was seldom the
  family dealt on such a unit: after the first eval release's rules the ten were
  drawn 8 to 16 times in 3,000 units of the default attacks (three groups,
  seeds 42 to 46). With the slot at 0.25 they are drawn 37 to 40 times
  (seeds 42 to 46, in one, three or ten groups), about a quarter of a
  family's draws, and `encode`'s draws per unit go from 0.218 without it
  to 0.230 (seed 42, three groups). The total depends on the grouping,
  since each group's dealer starts the slot at 0: 37 or 38 in one group, 38
  or 39 in three, 40 in ten (seeds 42 to 50) and 47 to 51 in 46 uneven
  groups of 30 to 100 units; the least drawn of the ten is drawn 3 times in
  one and in three groups and 1 to 3 times in ten (the slot fixes a total,
  not a count for each encoding). In small groups the weight hardly
  matters: 300 units in 15 groups of 20 draw 8 to 11 at any weight from 0.2
  to 0.4, since each group's slot takes its first pick early and its next
  only after more units than the group has. It is a quota in every budget
  round, not a floor, since the weight does not read the round's budget: in
  the escalation (3,000 units a round, three groups, seeds 42 to 46) round
  3 draws 36 of the ten and round 4 27 or 28, where round 4's budget dealt
  them about 121 times as encode's members (spec 1.3). The weight was 0.2
  when the slot came (44 to 50 draws at spec 1.3); spec 1.4's four
  language-mix families share the obfuscation picks among more families,
  which brought 0.2 down to 30 or 31, so spec 1.4 retuned it for about 40
  (the mean over seeds 42 to 46 in one, three and ten groups: 30.1 at 0.2,
  38.8 at 0.25, 46.5 at 0.3, 51.7 at 0.35, 59.6 at 0.4).
  A count draw (`exact_techniques`: the eval type, the natural kind, the
  first two escalation rounds) and a draw without a dealer deal them as
  encode's members, as before, so their draws do not move. The slot is not
  a family: `family_weights` cannot name it (`plan_run` called directly
  raises too, rather than replace the weight), and weighting `encode`
  scales it too.
- **Techniques.** Within that family, the dealer picks the eligible member
  the group has dealt least (`Dealer.pick_member`, its own `members`
  counter, apart from the families' `counts` the generation axes also
  use), ties broken by the unit's rng; the request obfuscation shares
  these counts with the layer. A family whose members cost differently
  still favours its cheaper ones under the complexity budget (`encode`,
  `sensitive_words`): a member of complexity 3 fits only a unit whose
  earlier picks leave 3 points, and `fsh` and `dap` refuse the heavy
  `encode` members, so over 3,000 units of the default attacks (three groups)
  each heavy `encode` member is drawn 3 to 5 times at seed 42 (3 to 6 over
  seeds 42 to 46), even with its slot, against 27 to 29 for one of
  complexity 2 (26 to 29). That
  skew is intended. A direct `sample_combination` call without a dealer picks both
  uniformly, as the registry fixture's seeds pin. A fresh plan with the
  same seed draws differently from one made before member dealing; a kept
  plan replays unchanged.
- **Repeats.** When a unit's combination equals one of its input's earlier
  iterations, it is drawn again without the dealer, up to 8 draws in all;
  the last is kept if none differs.
- **Shortfall.** A layer, or the request obfuscation step, two or more
  picks below its target in a group, as when the rules or the pool leave no
  fitting technique, is logged as a WARNING naming the coverage file.

After each unit, `assign_combination` has `_observe` count the kept draw's
families (the request obfuscation's under `REQUEST_SCOPE`, a budget draw's
heavy full-text encodings under `HEAVY_ENCODINGS`), techniques,
layers, request obfuscation and complexity in the group's dealer. Each group has its own dealer; nothing is
counted across groups.

`tests/jailbreak/simulate.py` plans N synthetic inputs through `plan_run`
(in a temporary folder) and gives each technique's share of the units and
its share within its family; `python -m tests.jailbreak.simulate [N]` prints
them for the default adversarial kind, and `deal_kind(section, n,
base=...)` plans a kind over a dataset type's section (the eval type's);
`plan_kind` and `plan_rows` give the manifest rows, `spend` what the units
spent in complexity, and `python -m tests.jailbreak.simulate [N] budget` the
unit budget against the group budget. The coverage table holds none of these.

### The group budget

Opt-in, off in both packaged kinds and both dataset types: an adversarial
kind's `budget` `"group"` (default `"unit"`) with `max_unit_complexity`
(default `max_complexity + UNIT_HEADROOM`, 3). Without it a unit spends up
to `max_complexity` and what it leaves is lost; with it the group shares the
budget.

- `_observe` adds each kept technique's complexity, the request
  obfuscation's included, to the dealer's `spent`. `Dealer.bank(per_unit)` is
  `max(0, per_unit x (position - 1) - spent)`: what the units dealt before
  the current one left unspent of `per_unit` each.
- Before it draws, `assign_combination` replaces the round's `max_complexity`
  with the unit's allowance, `min(max_complexity + bank, ceiling)`
  (`_allowance`), once, so a resample draws from the same allowance. Every
  pick of a draw fits what is left of it (`compatible` checks the remaining
  complexity, the first pick and the request obfuscation included), so a unit
  never spends more than its allowance. By induction the group has spent at
  most `max_complexity x position` after each unit, and no unit goes above
  the ceiling.
- A count draw (`exact_techniques`), a call without a dealer and a kind
  without the option leave `max_complexity` as it was. Each round of a plan
  is its own group with its own dealer, so its own bank.
- `rounds_of` writes `budget` only when `"group"` and `max_unit_complexity`
  only when given, so the rounds, the plan file and the draws of a kind
  without the option are those it had before (the pinned digests of
  `tests/jailbreak/test_dealing.py` hold). A kept plan made without the keys
  resumes with no warning under a section that leaves them out or says
  `"unit"`; a section that turns the option on or off over a kept plan, or
  changes its ceiling, raises as `rounds` (`planfile.changed_plan`), naming
  `resume=False`. `resolve` refuses the keys on the natural kind, a ceiling
  without `"group"` or below `max_complexity` or any round's
  `max_complexity`, and a round naming either key.
- Measured with `simulate.compare_budgets` over 3,000 units of the default
  attacks (three groups, seed 42; in brackets the range over seeds 42 to 46),
  it keeps the mean at `max_complexity` (5.60 to 6.00), lets some units reach
  the ceiling (859 of 3,000 above 6, 819 to 867, none above 9) and evens out
  the cheap members of `encode` and `sensitive_words`, and draws more
  techniques of complexity 3 or more (1,367 against 1,267, the heavy `encode`
  members 3 to 6 times each, 3 to 6, against 3 to 5, 3 to 6; spec 1.4,
  `heavy_encoding_weight` 0.25); the extra allowance goes to the
  obfuscations.
  The table is in [configuration.md](../configuration.md#the-group-budget).

## Seeds

`stable_seed(*parts)` (`dataset/dealing.py`) is the first 16 hex digits of the SHA-256 of the parts
joined by `::`: the same in every process, unlike Python's `hash`.

| Draw | Seed |
|---|---|
| A unit's combination | `stable_seed(seed, input_id, iteration)`, plus the attempt number on a redraw |
| A layer's jitter at a unit | `stable_seed(seed, category, entry_type, iteration, layer, position)` |
| A technique's random choices | `stable_seed(seed, label, input_id, iteration, technique name)`, the label `jailbreak` or `variation` |

So the plan does not depend on the order or the chunking of the inputs, and
a replayed plan gives the same text within one version of the library. The
plan file's `spec_version` is the spec's `version` when the plan was drawn,
and a row's `combination_spec_version` is its plan's; the version moves
when techniques are added or retired, not when a technique's code changes
(the ASCII-art cap did not move it) nor always when a rule does, which the
plan file's `rules` (`rules_id(pool, rounds=)`) records instead. The
manifest row records the round's settings and the `seed`, and
`<stem>.plan.json` what the whole plan was made with: `seed`, `rounds`,
`pool` (sorted names), `family_weights`, `layer_rates`, `spec_version`,
`rules`.

## Coverage

`coverage(manifest_path, pool)` writes `<stem>.coverage.csv` beside the
manifest whenever a plan is written: per `level` (`layer`, `family`,
`technique`) and `name`, the count per `category`, `entry_type` and
`iteration`, plus a zero row for every pool technique never drawn. It logs
`Coverage: N of M pool techniques drawn`. With `verbose`,
`log_model_calls(manifest_path, pool)` then logs what the plan costs:
`Model calls: N of M planned units call a model, L to H calls (translation
l to h, from the spec's calls).` A unit counts when it draws a
`requires_llm` technique, and each such technique adds its spec `calls`
(`[low, high]`; every `requires_llm` technique has them, a test holds it).

`variants.write_variants(out_path, manifest_path, pool, registry)` writes
`<stem>.variants.csv` (`technique`, `variant`, `rows`) after the units,
from the rows of the kind's CSV, those of earlier runs included (the
natural kind runs with `read_back=False`, so the file is read, not the
frame). `variants_of(technique, info, registry)` pairs one row's variants by
position: a technique's pool names the info key that holds its variant
(`PoolRef.info_key`: `variant` as `apply_template_directive` writes it,
continuation's `style`), and the values of each key go,
in order, to the row's techniques that write it, less those noted
`noop=`; a row whose values and techniques do not pair up (a rejected or an
error row) counts none, except for the key of an optional pool
(`PoolRef.optional`: the decode hints of the encodings, written at a rate),
whose values may be fewer than its techniques: a row without a hint counts
none for that key and keeps its other variants. Every variant of weight above 0 of a pool
technique gets a row, 0 when none drew it; a technique outside the pool
that a row drew (a kept plan's) is added after. No technique's info
changed for the table: continuation, which the eval pool draws, keeps its
`style=` info while the eval pool is pinned.

With `run_augmentation(scenarios=True)` (the front's `scenarios`), each
pending unit carries a `scenarios` list, which `unit_gen` hands to
`run_chain`. The chain gives each technique that takes `record_scenario`
(or `**kwargs`) a callable bound to its name; `cognitive._cognitive_gen`
and `personas._persona_gen` call it with the scene the construction is
built on, after the check accepted it, so a discarded scenario records
nothing. `_row` puts the unit's scene rows (`scenarios.rows_of`: `input_id`,
`iteration`, the row's `sample_id`, `technique`, `scenario`) on its row
under a private key, which the save's `_append` takes off and appends to
`<stem>.scenarios.csv` after the rows and before the acks. The table is
not in the ledger's `_saved`: `scenarios.write_scenarios(out_path, path)`,
after the units, keeps the rows whose `(input_id, iteration, sample_id)`
the kind's CSV holds, the last of a repeated one, and writes the table
again (`write_concat`) only when one goes; a fresh run removes it, and
`stages._kind_files` lists it for `clear_stage`. Recording changes no row:
a test runs the default attack pool both ways and compares the CSVs.

## The augmentation section in code

`augmentation.resolve(section, kind, base=None, config_dir=None)` overlays
a kind on its default, checks every key and name, and returns the full
kind, or `None` for a kind that is off. The default is the packaged
`configs/jailbreak/augmentations.json` (the `config_dir`'s when it has the
file), with `base`, the dataset type's section, resolved over it: a kind
`section` leaves out is `base`'s, and a kind `base` switches off comes back,
over the packaged default, when `section` gives it. A key starting with
`_`, in the section or a kind, is a comment. The file is checked when it is
read (`_checked_file`, cached per path and modification time, so once per
version): each default kind's keys against `_KIND_KEYS` and its lists with
`_lists`, as a section's; an unknown key is left out with a WARNING (a
copy's typo did nothing before), and a missing key, a kind that is not an
object, a bad `natural_only_families` or an adversarial `exact_techniques`
(which `kind_of` would read as natural) raise, naming the file. With `translation` false, the
translation an allow list or `family_weights` names is dropped with a
WARNING (`_switch_off_translation`). `_check_kinds` raises for an allow
list that names a family or technique whose spec `kinds` lack the kind.
For a family, the message names those of its techniques the kind draws, to
list under `techniques.allow`, and does not offer adding the kind to the
family's `kinds`: they apply to every technique in it that sets none of
its own. `_check_denied_layers` reads each name of the allow lists and
`family_weights` against `layers.deny` after `_deny_layers_off`: a name of
a layer off by default (`rewrite` without `layers.allow`) raises, saying
to write `"layers": {"allow": ["rewrite"]}`, since the pool would drop it
silently; one of a layer the kind's own `layers.deny` names is dropped with
a WARNING, and a `family_weights` entry of it raises.
`technique_pool(resolved)` filters the registered techniques; with the switch
on, a `techniques.allow` that names some languages (`pool.languages()`)
admits those alone. Last, it keeps the techniques whose `kinds` (set by
`technique.tag`: the technique's own, else its first family's,
`family_kinds`, else `DEFAULT_KINDS`, `["adversarial"]`) hold the kind,
which `pool.kind_of(resolved)` reads as `rounds_of` does: natural when the
resolved kind has `exact_techniques`, so no key is added to it. The
natural pool is narrowed to the 96 techniques of
[techniques.md](../techniques.md#the-natural-pool) with translation off;
the attacks lose only the techniques of the three families marked
`["natural"]` alone (`smalltalk_prefix`, `smalltalk_suffix`,
`benign_role`). Then a technique whose `deprecated` tag is set leaves the
pool unless `techniques.allow` names it; one it names stays, with a
`DeprecationWarning` (read for this release); `resolve` logs it as a WARNING
too (`_warn_retired`), since Python shows that warning only to `__main__`.
A family in `families.allow`
does not keep its retired techniques.
`rounds_of(resolved)` turns the kind into one sampler-settings dict per
round: a natural round is `{"exact_techniques": n}`; an adversarial round is
the budget keys, an `include_<layer>` per layer and `allow_request_obfuscation`
when set, `budget` when `"group"` and `max_unit_complexity` when given, with
the round's own keys merged over them. `include_rewrite` is
the exception: written, true, only when the kind's pool holds a rewrite
technique, so the rounds of a kind that denies the layer (every kind that
does not allow it), and of one whose lists name no rewrite family (the eval
type's, the layer allowed), are those written before the layer existed, and
no kept plan of theirs changes. Under `"escalation"` a kind that draws the
layer has its `layer_rates.rewrite` carried into the two budget rounds'
`sampling_probs`, which replace the kind's rates. `_warn_rewrite_rate`
logs a WARNING for a budget round at rate 0 in such a kind, and for a
rewrite rate above 0 while the layer is off. `resolve` adds each layer of the spec's
`layers_off_by_default` (`rewrite`) to the kind's `layers.deny` unless its
`layers.allow` names it (`_deny_layers_off`), so a `layers.deny` a section
gives, which replaces the default's whole, still leaves the layer off, and
the pool and `rounds_of` read one list. The kind's
`layer_rates` goes to `run_augmentation(layer_rates=)` and from there to
`plan_run`; a round's own `layer_rates`, passed on as the sampler's
`sampling_probs`, replaces it whole and is read over the spec's rates.
[configuration.md](../configuration.md#the-augmentation-section) gives the
rules as a user writes them.

## The combination search

`search.py` is a beam search over single-turn combinations against a target
model, not a recipe stage (call `search_combinations`; the user-facing side is
[stages/combination-search.md](../stages/combination-search.md)). It reuses the
engine's parts rather than adding rules of its own: round 0 draws `width` valid
combinations per input with `sample_combination` over the techniques that
`applies` admits; a later round takes the `beam` best of the round before and
`_mutations` adds, swaps or drops one technique, each result run through
`rules.compatible`, so no candidate is ever an invalid pair. A candidate is
applied with `run_chain` and driven on `drive_units`, exactly as the stage
drives a unit; the target reply is scored by a judge read with `is_accepted`,
the `multi_turn` judge pattern (default criterion `SUCCESS_JUDGE_SYSTEM`). Every
tried candidate is a row of `jailbreak_search.csv`, keyed in the ledger by
`(input_id, round, candidate)`; a round is committed before the next round's
beam is read, so a kill resumes to the same rows. The search is a pure function
of the inputs, the seed, the pool and the params — round 0's draws are seeded
per `(seed, input_id, i)` and a round's beam follows from the saved rows — so a
resume re-derives the same plan and runs only the units the ledger lacks. The
plan file records the seed, `width`/`beam`/`depth`, the pool names, the
`rules_id` and `sample_kwargs`; a resume with any of them changed raises. The
module imports `augmentation` lazily (only to build the default pool), so it
adds no import-time cycle.
