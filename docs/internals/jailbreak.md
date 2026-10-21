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
  techniques.
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
| `technique.py` | The technique contract, `load_spec`, `tag`, `DISCARDED`, `KINDS`, `DEFAULT_KINDS`, `family_kinds`, `FIRST_VERSION`, `version_key` |
| `hacking/`, `manipulation/`, `obfuscation/`, `requests/` | The techniques, one module per family (`distractor.py` holds both distractor families), each with a getter; `get_all_hacking_functions`, `get_all_manipulation_functions`, `get_all_obfuscation_functions`, `get_all_request_functions` per subpackage. `manipulation/benign.py` makes the benign data: one unit per subcategory of the benign taxonomy and answer type, keyed `(main_category, sub_category, answer_type)` |
| `pool.py` | `all_techniques`, `technique_pool(resolved)`, `kind_of(resolved)`, `languages`, `registry`, `uses_translation` |
| `augmentation.py` | `KINDS`, `defaults(config_dir)`, `resolve(section, kind, base=None, config_dir=None)`, `rounds_of(resolved)`, `apply_flat_keywords`, `FLAT_KEYWORDS` |
| `rules.py` | `compatible(selected, pool, remaining_complexity)`, `request_obfuscation_candidates` |
| `sampling.py` | `sample_combination`, `sample_exact_combination`, `_pick_by_family` |
| `plan.py` | `plan_run`, `assign_combination`, `combination_of`, `coverage`, `plan_inputs` (the record of the `<stem>.plan.json` beside the manifest, written through `dataset/planfile.py`, its `spec_version` dating the pool), `default_escalation_schedule`; it deals from `dataset.dealing.Dealer` |
| `chain.py` | `run_chain`, `combine_techniques`, `in_layer_order`, `apply_combination` |
| `units.py` | `pending_units`, `is_pending`, `unit_gen`, `row_of`, `error_row_of` |
| `pipeline.py` | `run_augmentation`, the stage core; `_library_change`, the kept plan's pool against what the library added and retired since |
| `directives.py` | `load_templates`, `apply_template_directive`: template variants of the pure techniques |

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
  model's name), `translate_model`, `benign_data`, `prompt_dir`, `rng`. One
  that declares `**kwargs` gets them all. Every generator technique has the
  signature `(prompt, **kwargs)`; a test holds them to it.
- At import, `tag(all_techniques())` sets each function's spec tags as
  attributes: `families`, `complexity`, `requires_llm`, `requires_benign`,
  `encode_weight`, `max_tokens`, `cross_incompatible_with` (families this
  technique itself may not join), and from the first family `layer` and
  `within_layer_order`; `kinds` (the augmentation kinds that may draw it)
  is the technique's own, else its first family's, else `["adversarial"]`.
  A `kinds` that is not a non-empty list of `adversarial` and `natural`
  raises at import. `since` is the spec version that added the technique
  (`FIRST_VERSION`, `"1.0"`, when its entry gives none) and `deprecated`
  the one that retired it (`None` while it is active). Versions are dotted
  whole numbers and compare as such (`version_key`: `"1.10"` comes after
  `"1.9"`); a `since`, a `deprecated` or a spec `version` that is not one
  raises at import.

`run_chain(techniques, text, ...)` runs one input through its techniques in
the order given. A transform runs inline; a generator's requests are yielded
up, each capped at the technique's `max_tokens` tag and, under capture,
tagged with an internals id. Each technique gets
`rng = random.Random(stable_seed(seed, label, input_id, iteration, name))`,
the label being `jailbreak` or `variation`, so the two kinds draw apart. A
step that leaves the text unchanged is noted `noop=<name>`. A step that
leaves the text empty or blank is a rejection,
`DISCARDED; technique=<name>; reason=empty text`. The first rejection stops
the chain, returning the text before that technique, with the info
`DISCARDED; technique=<name>; ...`.

`combine_techniques(*fns)` wraps techniques as one, sorted into layer order
(`in_layer_order`), and runs it with blocking calls. It does not check the
rules.

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
   (`plan_inputs`, less `pool` and `spec_version`) with `<stem>.plan.json`;
   a difference raises a `ValueError` naming the keys and `resume=False`.
   The pool is compared apart (`_library_change`): the names the run's pool
   adds must each have a `since` after the plan file's `spec_version`
   (`"1.0"` in a plan file written before it was recorded), and the names
   it lacks must each be registered with a `deprecated` after it. Then the
   difference is the library's: one WARNING names what was added and
   retired, says the kept plan does not draw the added techniques and names
   `resume=False`, and the plan runs as it was made, its retired techniques
   included. Any other difference raises as `pool`: a section that adds or
   leaves out a technique, or a technique no longer registered. Known
   limit: a section that newly names a family all of whose techniques came
   after the plan reads as growth, and warns. A manifest without the plan
   file logs a WARNING and runs unchecked. Otherwise the old coverage and
   plan files are removed, and `plan_run` writes the plan file, one manifest
   row per input and round, then the coverage table.
4. **One pass over the plan** to count the pending units, see whether any
   needs benign data (if so, generate what the cache lacks for the benign
   taxonomy's subcategories, on the pool with its own ledger, then load it;
   with `auto_generate_benign` off, the cache is loaded as it is, and a
   missing cache raises before any unit), and, on a resume, count the inputs
   the kept plan does not hold, logged as a WARNING.
5. **Drive.** `pending_units` streams the plan: for each planned row of an
   input in this frame and not in the ledger, it builds the combination
   (`combination_of`) and the unit's generator (`unit_gen`, over
   `run_chain`) only when the driver takes it. `drive_units` runs them with
   `pool_size(batch_size, client)` live and `cap=batch_size`. A plan row
   naming a technique no registered function implements raises a
   `ValueError` naming `resume=False`, which stops the stage.
6. **Save.** `on_done` makes the row (`row_of`); `on_error` makes an error
   row (`error_row_of`) for anything the unit raises; both go to a
   `CommitBuffer` with the ack `{"input_id", "iteration"}`. A
   `PromptTooLongError` reaches `on_error` only on the async lane
   (OpenAI-compatible, vLLM), raised at the unit's `yield`. On the grouped
   path (Anthropic, introspection) a failed `generate()`, a too-long prompt
   included, propagates and stops the stage.
7. **Read back** when asked.

A planned unit whose input is not in the frame is skipped; an input that is
not in the plan gets no unit. A unit's key in the driver is its position
among the pending units, so two plan rows with one `(input_id, iteration)`
stay two units.

## Drawing one combination

`sample_combination(rng, pool, ...)` has two modes.

**Budget mode** (no `exact_techniques`) picks layer by layer, each pick from
the techniques still `compatible` with what is picked:

1. `hacking`, when the layer joins (its rate);
2. `manipulation`, when the layer joins;
3. `requests`, when the layer joins;
4. the request obfuscation (phase 5 in the code, decided and drawn before
   phase 4): with `allow_request_obfuscation` and at least one
   `max_obfuscations` slot, when it joins at the `request_obfuscation` rate
   (the dealer's `include`, as for the layers; a per-unit draw without a
   dealer), one light `encode` technique (the light
   `to_sensitive_words_encode_*` among them) or a translation, applied
   after the request technique;
5. `obfuscation`: one pick after another, from what the budget leaves, until
   `max_obfuscations`, less the request obfuscation's slot, or until no
   technique fits.

The apply order is the layers', whatever the pick order, except the request
obfuscation, which goes last. Within the obfuscation layer it is each
family's `within_layer_order`, in tens: `translation` 10, `typos` 20,
`casual_writing` 30, `sensitive_words` 40, `tokenbreak` 50, `suffixes` 60,
`encode` 70, `structural` 80, `ascii_art` 90. The whole-text rewrites come
before the word-level steps, which they would otherwise undo; a family
added later takes a number between two. The manifest records the order a
combination applies in and a replay keeps it (`combination_of`), so a plan
made before the renumbering, when `sensitive_words` and `tokenbreak` came
first, still runs as it was made. A technique above the remaining complexity is
never eligible, for the first pick too. The request obfuscation is drawn
from the techniques still compatible and within the remaining complexity,
so it holds its slot and its complexity before the obfuscations fill the
rest: a combination never has more than `max_obfuscations` obfuscations.
With `allow_request_obfuscation` false (every packaged kind) the draws are
those made before the request obfuscation was counted (member dealing
aside); a test pins them. The combined function
names its request obfuscation in `request_obfuscation` (`None` without
one), which `_observe` counts in the dealer.

**Count mode** (`exact_techniques: n`) picks exactly `n` compatible
techniques from any layer, ignoring complexity. It returns fewer when the
rules leave nothing; with the full pool, 1 and 2 are always reachable.

Every pick is **family first** (`_pick_by_family`): a family, then a
technique within it, both dealt when the plan passes its dealer
([balanced dealing](#balanced-dealing)). The 19 translation languages are
one choice, not 19.
The rules are those of [techniques.md](../techniques.md#compatibility-rules).

## Balanced dealing

`plan_run` does not draw each unit alone. It groups the units by
`(category, entry_type, iteration)`, sorts each group by input id, and deals
the group from one dealer (`dataset/dealing.py`, shared with the inputs' and
outputs' generation axes), so every group gets its share whatever the input
order.

- **Layers.** A layer joins a unit when the group's picks of that layer so
  far are below `floor(rate x position + offset)`, where `position` counts
  the group's units so far and `offset` is a number in [0, 1) seeded per
  group and layer. A group gets `rate x size` picks, rounded down or up, and
  a group of one takes the layer with probability `rate`. The request
  obfuscation, when allowed, is dealt the same way under the name
  `request_obfuscation`.
- **Families.** Among the eligible families, the dealer picks the one with
  the lowest count over its weight (`family_weights`, 1 by default), ties
  broken by the unit's rng.
- **Techniques.** Within that family, the dealer picks the eligible member
  the group has dealt least (`Dealer.pick_member`, its own `members`
  counter, apart from the families' `counts` the generation axes also
  use), ties broken by the unit's rng. A family whose members cost
  differently still favours its cheaper ones under the complexity budget
  (`encode`, `sensitive_words`): a heavy member fits fewer units, as
  wanted. A direct `sample_combination` call without a dealer picks both
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
families, techniques, layers and request obfuscation in the group's dealer. Each group has its own dealer; nothing is
counted across groups.

`tests/jailbreak/simulate.py` plans N synthetic inputs through `plan_run`
(in a temporary folder) and gives each technique's share of the units and
its share within its family; `py -m tests.jailbreak.simulate [N]` prints
them for the default adversarial kind. The coverage table does not hold
these shares.

## Seeds

`stable_seed(*parts)` (`dataset/dealing.py`) is the first 16 hex digits of the SHA-256 of the parts
joined by `::`: the same in every process, unlike Python's `hash`.

| Draw | Seed |
|---|---|
| A unit's combination | `stable_seed(seed, input_id, iteration)`, plus the attempt number on a redraw |
| A layer's offset in a group | `stable_seed(seed, category, entry_type, iteration, layer)` |
| A technique's random choices | `stable_seed(seed, label, input_id, iteration, technique name)`, the label `jailbreak` or `variation` |

So the plan does not depend on the order or the chunking of the inputs, and
a replayed plan gives the same text within one version of the library. The
plan file's `spec_version` and a row's `combination_spec_version` are the
spec's `version`, which moves when techniques are added or retired, not when
a technique's code changes (the ASCII-art cap did not move it). The
manifest row records the round's settings and the `seed`, and
`<stem>.plan.json` what the whole plan was made with: `seed`, `rounds`,
`pool` (sorted names), `family_weights`, `layer_rates`, `spec_version`.

## Coverage

`coverage(manifest_path, pool)` writes `<stem>.coverage.csv` beside the
manifest whenever a plan is written: per `level` (`layer`, `family`,
`technique`) and `name`, the count per `category`, `entry_type` and
`iteration`, plus a zero row for every pool technique never drawn. It logs
`Coverage: N of M pool techniques drawn`.

## The augmentation section in code

`augmentation.resolve(section, kind, base=None, config_dir=None)` overlays
a kind on its default, checks every key and name, and returns the full
kind, or `None` for a kind that is off. The default is the packaged
`configs/jailbreak/augmentations.json` (the `config_dir`'s when it has the
file), with `base`, the dataset type's section, resolved over it: a kind
`section` leaves out is `base`'s, and a kind `base` switches off comes back,
over the packaged default, when `section` gives it. A key starting with
`_`, in the section or a kind, is a comment. With `translation` false, the
translation an allow list or `family_weights` names is dropped with a
WARNING (`_switch_off_translation`). `_check_kinds` raises for an allow
list that names a family or technique whose spec `kinds` lack the kind.
For a family, the message names those of its techniques the kind draws, to
list under `techniques.allow`, and does not offer adding the kind to the
family's `kinds`: they apply to every technique in it that sets none of
its own.
`technique_pool(resolved)` filters the registered techniques; with the switch
on, a `techniques.allow` that names some languages (`pool.languages()`)
admits those alone. Last, it keeps the techniques whose `kinds` (set by
`technique.tag`: the technique's own, else its first family's,
`family_kinds`, else `DEFAULT_KINDS`, `["adversarial"]`) hold the kind,
which `pool.kind_of(resolved)` reads as `rounds_of` does: natural when the
resolved kind has `exact_techniques`, so no key is added to it. Every
technique holds `adversarial`, so only the natural pool is narrowed: to
the 46 techniques of [techniques.md](../techniques.md#the-natural-pool) with
translation off. Then a technique whose `deprecated` tag is set leaves the
pool unless `techniques.allow` names it; one it names stays, with a
`DeprecationWarning` (read for this release). A family in `families.allow`
does not keep its retired techniques.
`rounds_of(resolved)` turns the kind into one sampler-settings dict per
round: a natural round is `{"exact_techniques": n}`; an adversarial round is
the budget keys, an `include_<layer>` per layer and `allow_request_obfuscation`
when set, with the round's own keys merged over them. The kind's
`layer_rates` goes to `run_augmentation(layer_rates=)` and from there to
`plan_run`; a round's own `layer_rates`, passed on as the sampler's
`sampling_probs`, replaces it whole and is read over the spec's rates.
[configuration.md](../configuration.md#the-augmentation-section) gives the
rules as a user writes them.
