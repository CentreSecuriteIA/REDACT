# Extending REDACT

How to add a model, a prompt, a jailbreak technique, a stage, a category
or taxonomy, and how to change a dataset type. Each section names the files to touch and the tests that check
the change. Run the tests with:

```bash
PYTHONPATH=src python -m pytest tests -q
```

One known order dependence predates the jailbreak work: running
`tests/test_runconfig.py` before `tests/llms/test_estimate.py` in one process
(as `pytest tests/test_runconfig.py tests/llms/test_estimate.py` does) fails
ten of the estimate tests, on `refactor` too. Each file passes alone, and the
full suite in its default order passes.

## Add a model

**For good**, add an entry to `src/redact/configs/llm/models.json`. Every
field is in [configuration.md](configuration.md#the-model-registry). A local
model:

```json
"my-local-8b": {
  "backend_type": "vllm",
  "roles": ["uncensored_gen"],
  "is_uncensored": true,
  "default_max_tokens": 2000,
  "vllm": {
    "hf_model_id": "org/my-model-8b",
    "vllm_kwargs": {"max_model_len": 8192}
  }
}
```

A hosted one, on any OpenAI-compatible endpoint:

```json
"my-hosted": {
  "backend_type": "api",
  "roles": ["translation"],
  "api": {
    "backend_type": "openai",
    "api_key_env": "MY_PROVIDER_KEY",
    "base_url": "https://api.example.com/v1",
    "rpm": 60,
    "recommended_max_workers": 4,
    "price_per_1m_input": 0.2,
    "price_per_1m_output": 0.8
  }
}
```

A local machine-translation model, which takes one user message in its own
format and follows no other instruction:

```json
"my-translator": {
  "backend_type": "vllm",
  "roles": ["translation", "machine_translation"],
  "supports_system_prompt": false,
  "default_temperature": 0.2,
  "vllm": {
    "hf_model_id": "org/my-translator",
    "hf_config": "my-translator",
    "vllm_kwargs": {"max_model_len": 4096}
  }
}
```

- `machine_translation` makes the translation technique send it
  `jailbreak/translate_mt` and check its output itself. Overlay that prompt
  to change its wording; the stage checks it renders to one user message
  holding the three markers of the TranslateGemma format (source, target,
  text) in that order (`check_mt_prompt` in `translation.py`), so a model
  with another format needs that check changed.
- `hf_config` names a folder under `configs/llm/hf_configs/` holding a
  `config.json` read in place of the checkpoint's, when the installed
  transformers cannot read the checkpoint's own. vLLM also reads the config
  of the folder it loads the tokenizer from, so the engine is given a folder
  of the checkpoint's tokenizer files beside a copy of yours, built at load
  (a `tokenizer` in `vllm_kwargs` replaces it). Leave it out otherwise.
- Every translation language needs a code, accepted GlotLID labels and a
  median length ratio in `jailbreak/obfuscation/translation.py`, keyed by
  its slug in `configs/jailbreak/languages.json` (whose `script`, the ISO
  15924 code its GlotLID labels end in, the rules read); import fails when
  one is missing ([add a language](#add-a-language)). Measure the median
  with the model on a parallel corpus such as
  FLORES-200: a guessed 1.0 rejected most Japanese translations, whose
  median is 0.47.

- One entry may carry both an `api` and a `vllm` setup when the hosted model
  and the local weights are the same model; `backend_type` names the default,
  and a recipe picks the other with `"backend_type"` in the role's object.
- A role's default model is the first entry that lists the role. To make a
  new model the default, place it above the others with that role.
- A role must be in `configs/llm/roles.json`. A new registry role only
  matters when a stage asks for it (a `Stage`'s `roles`).
- Set `vram_gb` when you know the weights' size better than the estimate (an
  MoE checkpoint, a measured figure). Set `min_gpus` for a model larger than
  one card.
- `python scripts/download_models.py my-local-8b` fetches the weights ahead
  of a run.

The registry loads and validates every entry at import; a bad entry stops the
import with a message naming the field. `tests/llms/test_model_config.py`
covers the validation.

**For one process**, call `register_model`:

```python
from redact.llms import APIConfig, VLLMConfig, register_model

register_model(
    "my-model", backend_type="api", roles=["uncensored_gen"],
    api=APIConfig(backend_type="openai", api_key_env="MY_KEY",
                  base_url="https://api.example.com/v1", rpm=60),
    vllm=VLLMConfig(hf_model_id="org/my-model"),
)
```

A recipe run resolves names through the same registry, so a model registered
before `run_pipeline` can be named in the recipe.

**A new provider** is a new backend: a class in `llms/backends/<name>.py`
that subclasses `LLMBackend` and sets its `compute_config` flags, with
`from_config(config)` (reading its setup from the entry), `generate(...)`,
`agenerate(...)` when it has async calls, `backend_name` and `clear_cache`.
Emit one `call` event per transport call with `self._record_call(...)`, and
raise `PromptTooLongError` when the provider rejects a prompt as over the
context. Add the class to `BACKEND_TYPES` in `backends/capabilities.py`, and
to `API_BACKEND_TYPES` for a hosted API. Nothing above `llms/` changes:
callers read the flags. [internals/llms.md](internals/llms.md#backends)
describes the contract.

## Add a prompt

A prompt is a JSON file in its own folder:
`src/redact/prompts/<pipeline>/<category>/template.json`, with
`system_prompt`, `template`, `seed_fields`, `few_shot_examples` and
`metadata` ([configuration.md](configuration.md#prompt-files)). Load it in
code with the run's overlay:

```python
from redact.llms import PromptTemplate

template = PromptTemplate.load("my_pipeline", "my_step", prompt_dir=prompt_dir,
                               Category=category)          # system prompt values
messages = template(sample=text)                           # template values, per call
```

Pass `prompt_dir` down from the front, so a user's overlay reaches the new
prompt. A placeholder may not share a name with a parameter of
`PromptTemplate` (`prompt_config`, `few_shot`, `format_style`, `name`,
`source`, `is_override`, `prompt_dir`), nor, through `PromptTemplate.load`,
with `pipeline` or `category`.

**A new input style** needs only a file:
`prompts/input/generation/from_constitution/<style>/template.json`, then
`"style": "<style>"` under `inputs` in the params file. The seeded inputs
stage renders it with `Category`, `sample_description`,
`constitution_subcategory`, `entry_type`, `sample_spec` (the unit's dealt
generation axes; empty without axes) and `num_samples`, and appends the
`numbered` format instruction.

**To change a prompt for your runs** without editing the library, use the
overlay: `scripts/scaffold_prompts.py ./my_prompts`, keep the files you
change, and set `prompt_dir`
([configuration.md](configuration.md#the-prompt-overlay)).

## Add a jailbreak technique

Read [internals/jailbreak.md](internals/jailbreak.md#techniques) for the
contract.

1. **Write the function** in the module of its family, under
   `jailbreak/rewrite/`, `hacking/`, `manipulation/`, `obfuscation/` or
   `requests/` (a rewrite technique: [below](#add-a-rewrite-technique)).
   - A transform: `def to_name(prompt: str, *, rng=None) -> tuple[str, str]`.
     Draw every random choice from `rng`.
   - A technique that calls a model: a generator with the signature
     `(prompt, **kwargs)`, delegating to a helper that takes the keywords it
     needs (`gen_model`, `translate_model`, `check_model`, `benign_data`,
     `prompt_dir`, `rng`) and yields `LLMRequest(gen_model, messages)`. Load its prompt from
     `prompts/jailbreak/<name>/` with `prompt_dir`.
   - A template-driven transform keeps its variants in
     `configs/jailbreak/<layer>/<module>_templates.json`, one list per
     directive, reads them with `directives.read_pools`, names the pool it
     draws from with `@uses_pool("<layer>/<module>_templates.json",
     "<directive>")` (with `slots=` when it fills more than `{prompt}`), and applies
     one with `directives.apply_template_directive(..., rng=rng)`
     ([Add variants to a template pool](#add-variants-to-a-template-pool)).
   - Return `(text, info)`, `info` being `""` or `key=value;...`.
   - **An applicability**, when the technique would change nothing for
     some inputs or does not fit them: decorate it with
     `@applicable_when(predicate)` from `jailbreak/technique.py`, where
     `predicate(text, entry_type)` is true for an input the plan may draw
     it for. Write the predicate by calling the helper the technique itself
     calls, never by restating its rule, so the two cannot drift apart, and
     test that it is true exactly when the technique changes the text (or,
     for an entry-type rule, for exactly the entry types meant).
     `to_abbreviations` is the example: its body is
     `return _abbreviated(prompt), ""` and its decorator
     `@applicable_when(lambda text, entry_type: _abbreviated(text) != text)`;
     `test_casual_writing.py` checks the two agree on a set of neutral
     texts. An entry-type rule is a predicate of its own, which the
     technique does not call: the benign roles take
     `@applicable_when(is_benign_input)`, true for `benign` and
     `dual_use_benign` whatever the text (`requests/benign_role.py`). The plan
     filters the pool per input before the draw; the predicate sees the
     input, not what earlier steps of a combination wrote, and a kept plan
     replays without asking it. A predicate must not raise for any text (a
     text that starts with a combining mark, an empty one): one that does
     stops the plan with a `RuntimeError` naming the technique and the
     input. A technique of a default pool that gains
     one changes what a fresh plan draws: say so in a release note.
2. **Register it**: return it from its module's getter
   (`get_encoding_functions`, `get_framing_functions`, ...), which its
   subpackage's `get_all_rewrite_functions`, `get_all_hacking_functions`,
   `get_all_manipulation_functions`, `get_all_obfuscation_functions` or
   `get_all_request_functions` collects. A
   new module's getter is added to that collector in the subpackage's
   `__init__.py`.
3. **Add its spec entry** under `techniques` in
   `configs/jailbreak/combination_spec.json`: `families` (the first sets the
   layer and the order), `complexity` (0 to 3), `requires_llm`,
   `requires_benign`, `encode_weight` for an `encode` technique,
   `needs_latin: true` for one that changes only Latin text (no rule reads
   it: a test asserts that the family rules refuse every tagged technique
   beside every translation into another script, so one outside `encode`
   and `noise` needs a rule against `translation` in the spec;
   `TestLatinOnly` in `tests/jailbreak/obfuscation/test_encoding.py` checks
   it on six other scripts, and fails when an encoding that leaves them
   unchanged is not tagged, `to_braille` aside while the eval pool is
   pinned),
   `max_tokens` to cap its calls' replies, `cross_incompatible_with` for
   families this technique, not its whole family, may not join,
   `kinds` when it should differ from its first family's (below),
   `check` and `calls` for one that calls a model (below), and
   `since`: move the spec's `version` up (`"1.1"` to `"1.2"`; versions are
   dotted whole numbers, so `"1.10"` follows `"1.9"`), once per change that
   adds or retires techniques, and give each new technique that version. A
   jailbreak plan kept from before then resumes with a WARNING, without
   drawing the new technique; without a later `since`, the new name in its
   pool reads as the run's own change and the resume raises
   ([stages/jailbreaks.md](stages/jailbreaks.md#how-it-resumes)). A new
   family also needs an entry under `families`: `layer`,
   `within_layer_order` (its place in the obfuscation order, between the
   numbers of the families it should follow and precede: they go in tens,
   `translation` 10 to `ascii_art` 90; 0 elsewhere),
   `dimension` (an augmentation tag of `configs/dimensions.json`, where a new
   one is an entry with its side and definition; import fails otherwise),
   `cross_incompatible_with` and, for one variations may draw, `kinds`,
   plus a `description`, which nothing reads. Import fails on a family key
   outside these (and `check`, `calls`), a `layer` not in `layer_order`,
   and a technique entry with an unknown key, a family the spec does not
   have, or a `complexity` that is not a whole number from 0 to 3.
   - **A technique that calls a model** says how its reply is checked and
     what one application costs. `check` names the check: `typos` (the
     typo check), `scenario` (the scenario check), `word_list` (the strict
     parse of a marked-word list), `translate` (translation's own check
     and retry) or `preserve` (the check that a rewrite is the prompt's own
     request); `calls` is the model calls one application makes,
     generation and checks, a whole number or `[low, high]`. A family's
     `check` and `calls` apply to each of its techniques whose entry gives
     none, as `kinds` do.
     Both only declare: the technique runs its check itself (the typo
     check through `llm_pipeline.checked` in `obfuscation/typos.py`, on the
     `check_model` it is given; the scenario check in `hacking/cognitive.py`;
     the strict parse in `obfuscation/tokenbreak.py`; translation's rounds; the
     preserve check in `rewrite/preserve.py`), and
     import fails on a `check` outside those five or `calls` that are not
     whole numbers of at least 1. A technique of the eval pool that has a
     model write the prompt must carry a `check`
     (`tests/jailbreak/test_pool.py`; the `to_{fsh,dap}_selected_*`, which
     only pick their pairs with a model, are exempt by name).
   A combination takes one technique per family; a layer's cap is in
   `layer_max_picks`. A rule binds both ways, so one of the two families'
   lists is enough; a rule between two families of one capped layer is
   needless. Importing `redact.jailbreak` raises while the spec and the
   registry name different techniques.
4. **Decide which kinds draw it.** The spec's `kinds`, a list of
   `"adversarial"` and `"natural"`, say which kinds may draw a technique: a
   technique's own `kinds` replace its first family's, and a family without
   `kinds` is `["adversarial"]`. Give a technique `"natural"` only when a
   benign input it rewrites stays benign and reads as a person would write
   it. A family whose techniques are mostly natural carries both kinds,
   and its attack-only techniques say `["adversarial"]` (the high and insane
   typos, the four high-density noise techniques, eight answer formats); an
   attack family with one natural
   technique keeps no `kinds` and that technique says both
   (`to_leetspeak_basic`). So a new technique in an attack family is an
   attack only until its entry says otherwise, and a new technique in a
   family marked natural (`typos`, `casual_writing`, `noise`,
   `neutral_container`, `translation`, `answer_language`, `answer_format`,
   `asking`, `indirect`) takes its
   family's kinds and joins the natural pool unless its entry says
   `["adversarial"]`. The adversarial kind draws every family unless
   denied. For variations, also add the family to the natural kind's
   `families.allow`, or the name to its `techniques.allow`, in
   `configs/jailbreak/augmentations.json`; an allow list naming what the
   spec keeps from the kind raises. A family meant for variations only says
   `"kinds": ["natural"]` (`smalltalk_prefix`, `smalltalk_suffix`,
   `benign_role`): no attack draws it, and an adversarial allow list that
   names it raises. One the attacks should draw only when a section asks
   carries both kinds and goes in `natural_only_families`
   (`casual_writing`).
   - **The natural pool.** `tests/fixtures/natural_pool.json` pins the 96
     names the natural kind draws by default, and
     `tests/jailbreak/test_pool.py` fails until you decide: give the
     technique `"kinds": ["adversarial"]`, or add its name to the fixture.
     The same file checks the attack pool's 320 techniques in 45 families.
   - **The eval pool.** `configs/dataset_types/eval.json` allows whole
     families (`encode`, `fsh`, `dap`, `framing`, `structural`, `ascii_art`,
     `continuation` and, since eval pool v2, `persona_template`,
     `indirect`, `distractor_prefix`, `distractor_suffix`,
     `impersonation`, `asking`, `guardrail_address`, `neutral_container`,
     `transliteration`, `fake_turns`, `suffixes`), so a technique added to
     one of them (a new persona frame, a published suffix row) joins
     the eval attacks unless `eval.json` denies it under
     `techniques.deny`; of `temporal`, `answer_format` and
     `answer_language` it names single techniques under
     `techniques.allow`, so a new member of those does not join.
     `tests/fixtures/eval_pool.json` pins the 126 names the
     eval type draws, and `tests/jailbreak/test_pool.py` fails until you
     decide: deny the technique in `eval.json` (the eval pool is pinned, so
     this is the usual answer), or add its name to the fixture in an eval
     review. A user's copy of `eval.json` in a `config_dir` is not updated:
     a deny added to the packaged file does not reach it, so a technique new
     to a family the copy allows joins that user's eval pool, and an eval
     run names it in a WARNING (`extra_eval_techniques`).
5. **Test it.** Add tests beside the module's, under
   `tests/jailbreak/<layer>/`. `tests/jailbreak/test_registry.py` compares
   the registry with `tests/fixtures/technique_registry.json` (names, tags,
   whether each has an applicability, signatures, and the combinations five
   seeds draw), so it fails on any new
   technique. Once the change is meant, regenerate the fixture from the
   repository root with `PYTHONPATH=src python -m tests.jailbreak.test_registry`.
   `tests/jailbreak/test_family_pairs.py` pins, in
   `tests/fixtures/family_pairs.json`, how many of the technique pairs of
   each two families the rules refuse; a rule added, dropped or narrowed, or
   a technique refused where its family's others are not, changes a line:
   regenerate with `PYTHONPATH=src python -m tests.jailbreak.test_family_pairs`.
   `tests/jailbreak/test_eval_pairs.py` pins, in
   `tests/fixtures/eval_pairs.json`, which pairs of the eval pool's 126
   techniques the rules admit (each family's techniques, then per family
   pair all, none or the shorter list of pairs admitted or refused, then the
   pairs of the first pool's 36 one per line): a rule that moves the eval
   type's draws shows as a diff of that file, regenerated with
   `PYTHONPATH=src python -m tests.jailbreak.test_eval_pairs` in an eval
   review.
   A technique that draws a variant from a pool (it has a `variant_pool`),
   and every other technique of the eval pool but the translations (a fsh
   or dap one draws on the test's own benign pairs), is also pinned by what
   it writes:
   `tests/jailbreak/test_template_draws.py` runs each registered one on two
   neutral probes for six seeds (one that calls a model is answered from a
   script of the test's own, which a new one needs too) and compares a
   SHA-256 digest per technique and seed (of the texts, the infos and the
   rng's state after them, no text) with
   `tests/fixtures/template_draws.json`. A new template technique, or a
   pool given new variants, regenerates it with
   `PYTHONPATH=src python -m tests.jailbreak.test_template_draws`; the
   digests of an eval-pool technique change only in an eval review (a
   deliberate change of the pinned eval pool), and the regeneration refuses to
   write one without `--eval-review`.
6. **Docs**: make the layer and family tables of
   [techniques.md](techniques.md) again from the spec with
   `PYTHONPATH=src python -m tests.jailbreak.test_docs_counts`, and update
   the totals the sentences of `CLAUDE.md`, `README.md` and that page's
   introduction quote; `tests/jailbreak/test_docs_counts.py` fails until
   both agree with the spec. Then write [the catalog](catalog.md) again,
   `docs/catalog.md` and `docs/catalog.json`, with
   `PYTHONPATH=src python -m redact.jailbreak.catalog` (or
   `scripts/catalog.py`): `tests/jailbreak/test_catalog.py` fails until it
   is the code's. A change to a rule, a pool, a template file or a prompt
   moves it too.

To retire a technique, never remove it: a kept plan replays its techniques
by name, and `tests/jailbreak/test_registry.py::test_registry_only_grows`
fails when a name the registry fixture pins is no longer registered. Give
its spec entry `deprecated`, the spec's `version` moved up for it. It stays
registered and replays, but leaves every pool unless a kind's
`techniques.allow` names it, which logs a WARNING (and gives a
`DeprecationWarning`). Take it out
of the counts in the docs, and out of `tests/fixtures/natural_pool.json` or
`eval_pool.json` when it was in one.

**A family whose techniques come one per row of a data file** is a
`generators` entry of the spec instead of written-out techniques, keyed by
the family: `file` (a JSON object under `configs/jailbreak/` whose `rows`
list holds one row per technique), optionally `where` (a row column that
must be true), `name` (the technique name, `{slug}` the row's `slug`) and
the keys every technique's entry takes (`complexity`, `kinds`,
`requires_llm`, ...). Loading the spec (`technique.read_spec`) adds one
entry per row, the row's `complexity`, `since` and `deprecated` over the
generator's (and its `<family>_since`, `translation_since`, as that family's
`since`); the family's module builds the functions from the same file
(`technique.data_rows`). A file with no rows registers the family with no
technique. The spec, its data files and the template files are read from
the package only: a copy of them in a recipe's `config_dir` is ignored, with
a warning, since the techniques are built when `redact.jailbreak` is
imported.

### Add a language

The `translation` and `answer_language` techniques are generated from
`configs/jailbreak/languages.json`, one row per language:

1. **Add a row**, for example
   `{"name": "Welsh", "slug": "welsh", "complexity": 2, "script": "Latn", "translation": false, "answer_language": true, "since": "1.5"}`:
   `name` as prompts and infos write it, `slug` the end of its technique
   names (`to_welsh`, `to_ask_answer_in_welsh`), `complexity` by resource
   tier (0 high, 1 mid, 2 low), `script` the ISO 15924 code it is written
   in, and which families it joins. A row's place in the file is its
   technique's place in its family, which the draws follow: append it, since
   moving a row changes which language a seed draws.
2. **Date it**: move the spec's `version` up (to `"1.5"`, the example's
   `since`, when the spec is at 1.4) and give the row that `since`, so a
   jailbreak plan kept from before resumes with a WARNING. A language
   already in the table that gains a family (an answer language measured as a
   translation) keeps its `since` and gets `translation_since` for the new
   technique, which the spec's `version` then dates.
3. **For a translation**, add the language's ISO 639-1 code (or its name, when
   the translation model takes it only by name, as Hmong, or measured
   better by name, as Yoruba), its accepted
   GlotLID labels and its median length ratio, measured with the
   translation model on a parallel corpus, to `LANGUAGE_CODES`,
   `ACCEPTED_LABELS` and `MEDIAN_RATIO` in `obfuscation/translation.py`,
   keyed by the slug. Import fails without them, naming the slug. Check that
   the translation model covers the language first. Of the six rows spec 1.2
   added (Hmong, Guarani, Chinese, Hindi, Yoruba, Amharic), the measurement
   was made on a GPU pod on 2026-10-22 and four became translations in spec
   1.3: their rows say `"translation": true, "translation_since": "1.3"` and
   they have those three entries. Yoruba followed in spec 1.4
   (`"translation_since": "1.4"`), on watch: the judge rejects 41% of its
   calls, so its technique discards about one text in thirteen. Guarani (it
   loops on a fifth of its sentences) stays an answer language only.
4. **Then as for any technique**: regenerate the registry fixture and, for
   an answer language (a template technique), the template draws; decide the
   natural pool (an answer language joins it: add its name to
   `tests/fixtures/natural_pool.json`), and update the counts in the docs.
   The eval type's translation allow list names its languages, so a new
   translation is not drawn there.

### Add a published adversarial suffix

The `suffixes` family's published suffixes are generated from
`configs/jailbreak/obfuscation/suffix_strings.json` (the spec's `suffixes`
generator), one row per string, after the family's four noise suffixes,
which stay in code:

1. **Add a row**: `slug` (lower-case letters, digits and underscores, the end
   of its technique's name `to_adversarial_suffix_<slug>`, none of the noise
   suffixes' `punctuation`, `fragments`, `unicode`, `emoji`), `suffix` (the
   string as published), `source` (the paper, its arXiv id or the repository
   it is from), `optimized_against` (the model or models), and
   `since`, the spec version that adds it; `notes`, `complexity` (default 1)
   and `deprecated` are optional. `obfuscation/suffixes.py` refuses a row
   without one of the five at import, naming it. Append the row: its place is
   its technique's place in the family.
2. **Date it** as a language: move the spec's `version` up and give the row
   that `since`.
3. **Then as for any technique**: regenerate the registry fixture and update
   the counts in the docs. The technique is attack only (the generator's
   `kinds`; a row's own `kinds` are not read), so the natural pool does not
   move, and the eval type's allow list names no suffix.

### Add a rewrite technique

The `rewrite` layer comes first: its technique rewrites the whole prompt
before every other layer applies, one at most a combination
(`layer_max_picks`). It holds 16 techniques (`persuasion`, `tense_rewrite`,
`model_distractors`) and is off: the spec's `layer_rates.rewrite` is 0 and
the spec lists it in `layers_off_by_default`, so a kind leaves it out unless
its `layers.allow` names it, whatever its `layers.deny` says.

1. **Write and register it** as any technique (above), in a module under
   `jailbreak/rewrite/` whose getter `get_all_rewrite_functions` in
   `rewrite/__init__.py` returns. One that has a model write the text loads
   its prompt from `prompts/jailbreak/<name>/`, checks the reply (through
   `llm_pipeline.checked`) and says so in its spec `check` and `calls`. A
   technique that rewrites the whole prompt calls
   `rewrite/preserve.py: preserved_rewrite_gen`, which `persuasion` and
   `tense_rewrite` share (`check` `preserve`, `jailbreak/preserve_check`); one
   more persuasion technique is an entry of `persuasion_techniques.json` and
   its spec entry, `to_persuade_<key>`.
2. **Give it a family** in the spec with `"layer": "rewrite"`, a
   `dimension` and, for variations, `kinds`; the technique's entry carries
   `since`, the spec's `version` moved up.
3. **Turn the layer on** for a kind: `"layers": {"allow": ["rewrite"]}`
   puts the technique in the kind's pool, and a
   count round (`exact_techniques`: the natural kind, the escalation's
   first two rounds) may draw it from there. A budget round draws it only
   at a rate above 0, `"layer_rates": {"rewrite": 0.2}` (a budget round at
   rate 0 for a kind that draws the layer warns); the escalation's last two
   rounds give rates of their own, over which `rounds_of` carries the
   kind's `rewrite` rate. With a rewrite technique in the kind's pool,
   `rounds_of` writes `include_rewrite: true` in each of an adversarial
   kind's rounds, so a plan kept from before raises on resume naming
   `rounds`: plan again with `resume=False`. To draw it in every run, take
   `rewrite` out of the spec's `layers_off_by_default` and give the spec's
   `layer_rates.rewrite` a rate.
4. **Then as for any technique**: the registry fixture, the docs counts and
   the natural pool. The eval type's attacks allow named families only, so
   a rewrite family is not drawn there unless `eval.json` names it.

### Add variants to a template pool

A pure template technique draws its text from a *variant pool*: one
directive's list in its template file under `configs/jailbreak/`, read by
`directives.read_pools` whatever the file's shape. Most files are
`{directive: [{"name", "template"}, ...]}`; continuation's variants are
`{"name", "prefix", "suffix"}` (the prompt goes between them, unformatted);
structural's `to_json`, `to_xml` and `to_markdown` are lists like the rest
(its `policy_ids` is a list its module reads for itself, not a pool); the
persona frames of `persona_templates.json` are single templates, read as a
pool of one variant named after the entry. `personas.json` and
`cognitive_techniques.json` are definitions, not pools.

1. **Append a variant** to the directive's list: a `name` of its own in
   the list (the info writes it, `variant=<name>`, and a kind's
   `<kind>.variants.csv` counts the rows that drew it), and a
   `template` that holds `{prompt}` once and no other slot than those its
   technique fills (`{language}` for the answer languages, `{policy_id}`
   for `to_xml`, `{priority}` and `{format}` for `to_json`, whose `{prompt}`
   is the prompt as the inside of a JSON string); a literal brace is written `{{` or `}}`. A pool whose
   technique sets the filled variant into another template, which holds the
   request, has no request slot: its technique names it with
   `@uses_pool(..., slots=(...), request_slot=False)`, and its templates
   hold no `{prompt}`. A pool a technique draws from only at a rate (an
   encoding's decode hints, at the spec's `decode_hint_rate`) says
   `optional=True`: its info names a variant only when one was drawn, and a
   row without one still counts the variants it drew otherwise. A seed draws by
   the pool's length and order, so a new variant changes what some seeds
   draw (step 4).
2. **Weights are optional.** A variant may carry `"weight"`, a number of at
   least 0. A pool where no variant has one is drawn with `rng.choice`,
   each variant as likely; once one has a weight, the pool is drawn with
   `rng.choices` by weight (a variant without one weighs 1), so every
   seed may draw another variant. `0` turns a variant off; a pool keeps at
   least one above 0, and a pool of one variant makes no draw. Never weight
   the pool of an eval-pool technique (`tests/fixtures/eval_pool.json`):
   `tests/jailbreak/test_directives.py` fails, since a weight changes the
   eval text.
3. **Import checks it.** `pool.py` checks every pool a technique names
   (`check_pools`) when `redact.jailbreak` is imported, and raises naming
   the file, the directive and the variant: an empty or repeated name, a
   name with a character other than a letter, a digit, `_` or `-` (it goes
   into a row's info as `key=<name>`), a template without `{prompt}` or with
   it more than once (in a pool with a request slot), with a slot its
   technique does not fill, with a conversion or a format spec on a slot
   (`{prompt!r}`, `{prompt:>9}`), or an unpaired brace, a prefix without its suffix, an unknown key (a
   misspelt `weight`), a weight that is not a number of at least 0, or
   every variant at 0.
4. **Regenerate the digests**: the pool's techniques write other text for
   some seeds, so `tests/jailbreak/test_template_draws.py` fails until
   `PYTHONPATH=src python -m tests.jailbreak.test_template_draws` is run.
   A pool an eval-pool technique draws changes the eval text: that waits
   for an eval review.
5. **Ratchet the floor.** `tests/fixtures/variant_floor.json` records each
   pool's family, its count of variants on and its floor, the fewest it
   should have (`FLOORS` by family in `tests/jailbreak/test_variant_floor.py`:
   8 for a request or framing directive, a container, a fake turn, the
   payload split's reassembly, the word art's decoding and the wording's key
   and puzzles, 12 for a benign role, 20 for small talk, 6
   for the persona templates' mechanics, 4 for an encoding's decode hints; a
   persona frame is its technique's own text, one per persona, not a pool
   the technique draws from, and is not counted). Regenerate it with
   `PYTHONPATH=src python -m tests.jailbreak.test_variant_floor` to record
   the new count; it refuses to record a pool with fewer variants than the
   fixture has, or one that is gone, without `--allow-drop`. The test fails when a pool has fewer variants than
   recorded (a variant removed, or at weight 0), when the fixture is not
   current, when a family has no floor, and when a pool is below its floor
   without being on the fixture's `known_short` list, which the regenerator
   keeps as it is. A pool that reaches its floor must leave
   `known_short`: the test fails until it does. The list is kept by hand and
   nothing but review stops a new below-floor pool from being added to it, so
   add one only on purpose. The list is empty today: all 95 pools are at
   their floor or above. A pool also fails when one of its variants repeats
   another (to the case and the spaces).

A new directive is a new list in the file and a technique that names it
with `@uses_pool` ([above](#add-a-jailbreak-technique)); its family needs
a floor in `FLOORS`.

**Keep a technique's text out of the code.** `tests/jailbreak/test_prose_in_code.py`
reads every module under `src/redact/jailbreak/` with `ast` and finds each
string of eight or more words that is not a docstring, and not inside a
`raise`, a logging call or `warnings.warn`: text a technique could write.
`tests/fixtures/prose_in_code.json` allows the eight found when the lint
was written (four pieces of an error message in `augmentation.py`, four
docstrings of generated `sensitive_words` techniques), each by file,
enclosing function or module-level name, and word count, never its text.
A new string fails, printing only `file:line` and its word count: put the
text in a template file, or, for a message the lint cannot tell from
output, add its entry. An entry no string matches fails too, so moving a
text into JSON takes its entry out and the list only shrinks. Text built
from shorter strings (the suffix fragments, the abbreviation map) is not
found.

## Add a stage

A stage is a core, a front, and a row in the stage table.
[internals/top-level.md](internals/top-level.md) describes all three.

1. **The core**, in the stage's package, follows
   [the seven steps](internals/top-level.md#the-seven-steps-of-a-stage). Use
   `Ledger`, `Manifest`, `resume_state` and `CommitBuffer` from
   `redact.dataset`; do not write JSONL or CSV by hand.
   - A unit is a generator that yields `LLMRequest`s; build it from
     `llm_pipeline.checked` or `extracted` where they fit.
   - Drive the units with `drive_units(..., pool_size=pool_size(batch_size,
     *clients), cap=batch_size)`. Isolate a unit's own errors in `on_error`;
     let any other propagate. On the async lane (a client with
     `supports_async_calls`: OpenAI-compatible, vLLM) a `PromptTooLongError`
     is raised inside the unit at its `yield`; handle it there or in
     `on_error` as a rejected row or ack. On the grouped path (Anthropic,
     introspection) it propagates from `generate()` and stops the stage, as
     any failed call does.
   - Give every row a `sample_id`, the content id of its text, and an
     `input_id` when it derives from another row.
   - Take `resume`, `verbose` and `read_back`.
2. **A path** for its output in `paths.py`.
3. **The front** in `stages.py`: resolve the inputs (a frame, or the run's
   own), one model per role (`None` for the registry role's default, through
   `ModelClient.create`), the output path, `prompt_dir`; call the core; return
   what it returns. The front never touches a ledger. When the inputs hold
   no accepted row, it warns, writes nothing, and with `resume=False` clears
   the stage's files with `dataset.clear_stage` (the core that would clear
   them does not run). Export it from `redact/__init__.py`: `Stage.run`
   looks the front up there.
4. **A `Stage` row** in `STAGES`, at its place in the run order: `name`,
   `front`, `roles` (pairs of recipe role and registry role), `files`
   (`ctx -> (paths, values)`: the CSVs that hold its rows and the column
   values a counted row holds), `recipe_kwargs` (`ctx -> dict`, what the run
   passes the front; a value an earlier stage saves is a `Saved`), `sources` (`ctx -> list[Path]`, the files it reads its
   inputs from, for its run record; `[]` when it reads none), `folded`
   when the run folds some params keys into a kwarg it passes itself, and
   `extra_counts` when its record counts more than its files' rows. A new
   recipe role also goes in `_DEFAULT_MODELS` in `recipe.py`, and the stage's
   name in the `stages` of both dataset type files (`eval.json`,
   `training.json`) to run by default; a recipe's own `stages` replace them.
5. **The build**: add a `Part` in `_stage_parts`, with `accepted_only=True`,
   when its rows belong in the complete dataset, with its `include_<part>`
   and path keys, and list its file in `_build_sources`; when its rows
   belong in an eval file too, map it onto that file's columns in
   `_eval_parts`. A row whose
   `sample_id` an earlier part wrote is left out. Update
   `tests/fixtures/build_files.json`.
6. **Tests.** Cover a stopped and resumed run: no unit twice, no unit lost.
   `tests/test_stage_files.py` pins each stage's files after a stopped and a
   resumed run against `tests/fixtures/stage_files.json`; add the new stage
   there. `tests/conftest.py` has a mock backend and client.
7. **Docs**: a page under `docs/stages/`, and its entry under Stages in the
   `mkdocs.yml` nav.

## Add a category or a taxonomy

A taxonomy is a JSON file under `configs/taxonomy/`, under the `taxonomy/`
folder of a recipe's `config_dir`, or under the folder a recipe names as
`taxonomy_dir`:

```json
{
  "name": "my_taxonomy",
  "description": "...",
  "categories": {
    "Category Name": {
      "description": "Used in the prompts.",
      "subcategories": ["First", "Second"],
      "group_tag": "my_group"
    }
  }
}
```

- **A new category** is one more entry under `categories`. Its name is a
  folder name on disk, so it may not hold `/` or `\`.
- The constitution stage fills its prompts with the category's name,
  `description`, `subcategories` (a list, or a dict whose keys are used) and
  `group_tag` (default `general`, recorded as `source_group_tag`).
- **A new taxonomy**: write the file, or call
  `redact.create_taxonomy(name, categories, taxonomy_dir=...)`, then set
  `"taxonomy": "<name>"` (and `taxonomy_dir`) in the recipe. A dict passed to
  a front as `taxonomy=` works too.

The category reaches every later stage through its rows' `category` column.
The benign categories are a taxonomy of their own,
`configs/taxonomy/benign.json`, in the same format. Two stages read it: the
constitution builds its general-benign part from it, one call per category
with its subcategories, and the jailbreak stage generates the `fsh` and
`dap` benign pairs per subcategory. Write another and name it as the
recipe's `benign_taxonomy`.

## Change a dataset type

A dataset type is a JSON file with exactly three keys besides `_` comments:
`stages` (the default stage list), `params` (per stage, the front's defaults,
checked as a params file's are) and `augmentations` (an overlay of the
augmentation section, kind by kind). To change what `eval` or `training`
means for your runs, copy its file into `<config_dir>/dataset_types/` and
edit it; `scripts/scaffold_setup.py` writes both. There are only these two
types, as the build has only these two modes: a recipe naming another
`dataset_type` stops when it loads, and a file of another name in
`<config_dir>/dataset_types/` is not read, with a warning
([configuration.md](configuration.md#dataset-types)).
