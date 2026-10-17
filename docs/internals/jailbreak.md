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
  so a plan made with another pool still replays.
- A technique draws its random choices from the `rng` it is given, so a
  replayed plan gives the same text.

## Modules

| Module | Holds |
|---|---|
| `technique.py` | The technique contract, `load_spec`, `tag`, `DISCARDED` |
| `hacking/`, `manipulation/`, `obfuscation/`, `requests/` | The techniques, one module per family (`distractor.py` holds both distractor families), each with a getter; `get_all_hacking_functions`, `get_all_manipulation_functions`, `get_all_obfuscation_functions`, `get_all_request_functions` per subpackage. `manipulation/benign.py` makes the benign data: one unit per subcategory of the benign taxonomy and answer type, keyed `(main_category, sub_category, answer_type)` |
| `pool.py` | `all_techniques`, `technique_pool(kind)`, `languages`, `registry`, `uses_translation` |
| `augmentation.py` | `KINDS`, `defaults`, `resolve(section, kind)`, `rounds_of(kind)`, `apply_flat_keywords`, `FLAT_KEYWORDS` |
| `rules.py` | `compatible(selected, pool, remaining_complexity)`, `request_obfuscation_candidates` |
| `sampling.py` | `sample_combination`, `sample_exact_combination`, `_pick_by_family` |
| `plan.py` | `plan_run`, `assign_combination`, `combination_of`, `coverage`, `coverage_path`, `plan_path`, `plan_inputs`, `changed_plan_inputs` (the `<stem>.plan.json` beside the manifest), `default_escalation_schedule`, the dealer |
| `chain.py` | `run_chain`, `combine_techniques`, `in_layer_order`, `apply_combination`, `_stable_seed` |
| `units.py` | `pending_units`, `is_pending`, `unit_gen`, `row_of`, `error_row_of` |
| `pipeline.py` | `run_augmentation`, the stage core |
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
  `within_layer_order`.

`run_chain(techniques, text, ...)` runs one input through its techniques in
the order given. A transform runs inline; a generator's requests are yielded
up, each capped at the technique's `max_tokens` tag and, under capture,
tagged with an internals id. Each technique gets
`rng = random.Random(_stable_seed(seed, label, input_id, iteration, name))`,
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
3. **Plan.** On a resume with a manifest, reuse it: `changed_plan_inputs`
   compares the run's seed, rounds, pool, `family_weights` and `layer_rates`
   with `<stem>.plan.json`; a difference raises a `ValueError` naming the
   keys and `resume=False`, and a manifest without the file logs a WARNING
   and runs unchecked. Otherwise the old coverage and plan files are removed,
   and `plan_run` writes the plan file, one manifest row per input and round,
   then the coverage table.
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
4. `obfuscation`: one pick after another, from what the budget leaves, until
   `max_obfuscations` or until no technique fits;
5. with `allow_request_obfuscation`, at the `request_obfuscation` rate, one
   light `encode` technique (the light `to_sensitive_words_encode_*` among
   them) or a translation, applied after the request technique.

The apply order is the layers', whatever the pick order. A technique
above the remaining complexity is never eligible, for the first pick too. The
step-5 pick is drawn from the techniques still compatible and within the
remaining complexity, and is not counted against `max_obfuscations`.

**Count mode** (`exact_techniques: n`) picks exactly `n` compatible
techniques from any layer, ignoring complexity. It returns fewer when the
rules leave nothing; with the full pool, 1 and 2 are always reachable.

Every pick is **family first** (`_pick_by_family`): a family, then a
technique within it. The 20 translation languages are one choice, not 20.
The rules are those of [techniques.md](../techniques.md#compatibility-rules).

## Balanced dealing

`plan_run` does not draw each unit alone. It groups the units by
`(category, entry_type, iteration)`, sorts each group by input id, and deals
the group from one dealer, so every group gets its share whatever the input
order.

- **Layers.** A layer joins a unit when the group's picks of that layer so
  far are below `floor(rate x position + offset)`, where `position` counts
  the group's units so far and `offset` is a number in [0, 1) seeded per
  group and layer. A group gets `rate x size` picks, rounded down or up, and
  a group of one takes the layer with probability `rate`.
- **Families.** Among the eligible families, the dealer picks the one with
  the lowest count over its weight (`family_weights`, 1 by default), ties
  broken by the unit's rng.
- **Repeats.** When a unit's combination equals one of its input's earlier
  iterations, it is drawn again without the dealer, up to 8 draws in all;
  the last is kept if none differs.
- **Shortfall.** A layer two or more picks below its target in a group, as
  when the rules or the pool leave no fitting technique, is logged as a
  WARNING naming the coverage file.

## Seeds

`_stable_seed(*parts)` is the first 16 hex digits of the SHA-256 of the parts
joined by `::`: the same in every process, unlike Python's `hash`.

| Draw | Seed |
|---|---|
| A unit's combination | `_stable_seed(seed, input_id, iteration)`, plus the attempt number on a redraw |
| A layer's offset in a group | `_stable_seed(seed, category, entry_type, iteration, layer)` |
| A technique's random choices | `_stable_seed(seed, label, input_id, iteration, technique name)`, the label `jailbreak` or `variation` |

So the plan does not depend on the order or the chunking of the inputs, and
a replayed plan gives the same text. The manifest row records the round's
settings and the `seed`, and `<stem>.plan.json` what the whole plan was made
with: `seed`, `rounds`, `pool` (sorted names), `family_weights`,
`layer_rates`.

## Coverage

`coverage(manifest_path, pool)` writes `<stem>.coverage.csv` beside the
manifest whenever a plan is written: per `level` (`layer`, `family`,
`technique`) and `name`, the count per `category`, `entry_type` and
`iteration`, plus a zero row for every pool technique never drawn. It logs
`Coverage: N of M pool techniques drawn`.

## The augmentation section in code

`augmentation.resolve(section, kind)` overlays a kind on its default from
`configs/jailbreak/augmentations.json`, checks every key and name, and
returns the full kind, or `None` for a kind that is off. A key starting
with `_`, in the section or a kind, is a comment.
`technique_pool(kind)` filters the registered techniques.
`rounds_of(kind)` turns the kind into one sampler-settings dict per round:
a natural round is `{"exact_techniques": n}`; an adversarial round is the
budget keys, an `include_<layer>` per layer and `allow_request_obfuscation`
when set, with the round's own keys merged over them. The kind's
`layer_rates` goes to `run_augmentation(layer_rates=)` and from there to
`plan_run`; a round's own `layer_rates`, passed on as the sampler's
`sampling_probs`, replaces it whole and is read over the spec's rates.
[configuration.md](../configuration.md#the-augmentation-section) gives the
rules as a user writes them.
