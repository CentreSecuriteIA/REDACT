# Jailbreak techniques

The jailbreak stage draws from 151 techniques in 4 layers and 23 families.
`configs/jailbreak/combination_spec.json` gives each technique its families,
a complexity from 0 to 3, and whether it calls a model (`requires_llm`) or
needs the benign data (`requires_benign`). This page lists them by name, for
the `allow` and `deny` lists of the
[augmentation section](configuration.md#the-augmentation-section), and says
briefly what each family does.

## Layers

A combination applies its techniques in layer order: `hacking`, then
`manipulation`, then `obfuscation`, then `requests`. Within the obfuscation
layer, families apply in the order of their `within_layer_order`.

| Layer | Techniques | Picks per combination |
|---|---|---|
| `hacking` | 25 | at most 1 |
| `manipulation` | 8 | at most 1 |
| `obfuscation` | 71 | up to `max_obfuscations` families in budget mode, plus the request obfuscation when allowed |
| `requests` | 47 | at most 1 |

## Families and techniques

| Layer | Family | Dimension | Order | Count | Calls a model | Needs benign data | Techniques |
|---|---|---|---|---|---|---|---|
| hacking | `cognitive` | `role_framing` | | 5 | yes | no | `to_persona_roleplay`, `to_hypothetical_framing`, `to_authority_obedience`, `to_avi`, `to_deep_inception` |
| hacking | `framing` | `role_framing` | | 5 | no | no | `to_fictional_world`, `to_noble_goal`, `to_nefarious_goal`, `to_high_stake`, `to_no_moral_constraints` |
| hacking | `persona` | `role_framing` | | 15 | yes | no | `to_invented_persona`, `to_psychopath_persona`, `to_alien_persona`, `to_radical_politician_persona`, `to_cult_leader_persona`, `to_very_advanced_ai_persona`, `to_cartel_leader_persona`, `to_artist_persona`, `to_mentally_ill_persona`, `to_deformed_scientist_persona`, `to_politician_persona`, `to_deformed_professor_persona`, `to_religious_figure_persona`, `to_actor_persona`, `to_someone_from_the_future_persona` |
| manipulation | `dap` | `example_priming` | | 4 | the `selected` two | yes | `to_dap_random_short`, `to_dap_random_long`, `to_dap_selected_short`, `to_dap_selected_long` |
| manipulation | `fsh` | `example_priming` | | 4 | the `selected` two | yes | `to_fsh_random_short`, `to_fsh_random_long`, `to_fsh_selected_short`, `to_fsh_selected_long` |
| obfuscation | `sensitive_words` | `encoding` | 1 | 16 | yes | no | `to_sensitive_words_encode_base64`, `to_sensitive_words_encode_rot13`, `to_sensitive_words_encode_rot18`, `to_sensitive_words_encode_rot47`, `to_sensitive_words_encode_unicode`, `to_sensitive_words_encode_ascii`, `to_sensitive_words_encode_separator`, `to_sensitive_words_encode_leetspeak_basic`, `to_sensitive_words_encode_leetspeak_intermediate`, `to_sensitive_words_encode_leetspeak_advanced`, `to_sensitive_words_split`, `to_sensitive_words_star`, `to_sensitive_words_hyphen`, `to_sensitive_words_underscore`, `to_sensitive_words_variables`, `to_synonym_substitution` |
| obfuscation | `tokenbreak` | `encoding` | 2 | 3 | yes | no | `to_tokenbreak_prepend`, `to_tokenbreak_split`, `to_tokenbreak_delimiter` |
| obfuscation | `translation` | `language` | 3 | 19 | yes, the `translation` model | no | `to_french`, `to_japanese`, `to_russian`, `to_spanish`, `to_german`, `to_arabic`, `to_turkish`, `to_czech`, `to_vietnamese`, `to_greek`, `to_croatian`, `to_swahili`, `to_thai`, `to_khmer`, `to_nepali`, `to_zulu`, `to_scots_gaelic`, `to_bengali`, `to_javanese` |
| obfuscation | `typos` | `surface_noise` | 4 | 4 | yes | no | `to_rewrite_with_typos_low`, `to_rewrite_with_typos_medium`, `to_rewrite_with_typos_high`, `to_rewrite_with_typos_insane` |
| obfuscation | `casual_writing` | `surface_noise` | 4.5 | 9 | no | no | `to_lowercase`, `to_uppercase`, `to_no_sentence_capitals`, `to_dropped_spaces`, `to_doubled_spaces`, `to_dropped_punctuation`, `to_doubled_punctuation`, `to_abbreviations`, `to_stripped_accents` |
| obfuscation | `suffixes` | `surface_noise` | 5 | 4 | no | no | `to_adversarial_suffix_punctuation`, `to_adversarial_suffix_fragments`, `to_adversarial_suffix_unicode`, `to_adversarial_suffix_emoji` |
| obfuscation | `encode` | `encoding` | 6 | 12 | no | no | `to_base64`, `to_rot13`, `to_rot18`, `to_rot47`, `to_unicode_escape`, `to_ascii_ordinal`, `to_separator`, `to_leetspeak_basic`, `to_leetspeak_intermediate`, `to_leetspeak_advanced`, `to_morse`, `to_braille` |
| obfuscation | `structural` | `container` | 7 | 3 | no | no | `to_json`, `to_xml`, `to_markdown` |
| obfuscation | `ascii_art` | `encoding` | 8 | 1 | no | no | `to_ascii_art` |
| requests | `answer_format` | `answer_shape` | | 9 | no | no | `to_refusal_then_comply`, `to_apology_prevention`, `to_prefix_injection`, `to_answer_in_markdown`, `to_answer_as_tutorial`, `to_answer_as_code`, `to_answer_as_dialogue`, `to_answer_as_story`, `to_answer_as_structured_data` |
| requests | `answer_language` | `language` | | 20 | no | no | `to_ask_answer_in_french`, `to_ask_answer_in_japanese`, `to_ask_answer_in_russian`, `to_ask_answer_in_spanish`, `to_ask_answer_in_german`, `to_ask_answer_in_arabic`, `to_ask_answer_in_turkish`, `to_ask_answer_in_czech`, `to_ask_answer_in_vietnamese`, `to_ask_answer_in_greek`, `to_ask_answer_in_croatian`, `to_ask_answer_in_thai`, `to_ask_answer_in_swahili`, `to_ask_answer_in_khmer`, `to_ask_answer_in_maori`, `to_ask_answer_in_nepali`, `to_ask_answer_in_zulu`, `to_ask_answer_in_scots_gaelic`, `to_ask_answer_in_bengali`, `to_ask_answer_in_javanese` |
| requests | `asking` | `request_shape` | | 2 | no | no | `to_innocuous_question`, `to_ask_for_details` |
| requests | `continuation` | `request_shape` | | 4 | no | no | `to_continue_passage`, `to_complete_story`, `to_fill_in_blank`, `to_continue_instructions` |
| requests | `distractor_prefix` | `request_shape` | | 2 | no | no | `to_distractor_prefix_related`, `to_distractor_prefix_unrelated` |
| requests | `distractor_suffix` | `request_shape` | | 2 | no | no | `to_distractor_suffix_related`, `to_distractor_suffix_unrelated` |
| requests | `impersonation` | `role_framing` | | 1 | no | no | `to_impersonate_good_person` |
| requests | `indirect` | `request_shape` | | 6 | no | no | `to_summarize`, `to_proofread`, `to_critique`, `to_translate_task`, `to_explain_to_child`, `to_rewrite` |
| requests | `temporal` | `request_shape` | | 1 | no | no | `to_use_past_tense` |

Each family's dimension is a tag of `configs/dimensions.json`: what the
family varies in a finished text. A run stops before any stage when a
generation axis varies the same dimension as a family it draws
([configuration.md](configuration.md#dimensions-and-the-runs-check)).

A technique takes its layer and its order from the first family of its
spec entry. The ten `to_sensitive_words_encode_*` techniques also
belong to `encode`, so one of them blocks both families.

`encode` techniques carry an `encode_weight`: `light` for `to_rot13`,
`to_rot18`, `to_rot47`, `to_separator` and the three `to_leetspeak_*`;
`heavy` for `to_base64`, `to_unicode_escape`, `to_ascii_ordinal`,
`to_morse`, `to_braille`. The ten `to_sensitive_words_encode_*` carry one
too: `heavy` for base64, unicode and ascii, `light` for the rest. The
`heavy` ones stay allowed with `fsh` and `dap`, and the `light` ones may be
the request obfuscation.

## Compatibility rules

`jailbreak/rules.py: compatible` applies the spec's rules both ways, whichever
technique is picked first. A technique may not join a combination when:

- its family is taken already;
- its layer is at its cap (`layer_max_picks`: one `hacking`, one
  `manipulation`, one `requests`);
- one of its families is in a taken family's `cross_incompatible_with`, or
  the reverse. The spec pairs `structural` and `translation` with
  `ascii_art`; `ascii_art` with `framing`, `persona`, `cognitive`, `fsh`,
  `dap`; `fsh` and `dap` with `continuation`, `indirect`, `framing`,
  `persona`, `cognitive`; and `continuation` with `indirect`. So a
  combination holds a hacking or a manipulation technique, never both;
- one of its families is in the `cross_incompatible_with` of a taken
  technique's own spec entry, or a taken family is in its own. The spec keeps
  `to_adversarial_suffix_emoji`, `to_adversarial_suffix_unicode`,
  `to_braille` and `to_tokenbreak_split` from `ascii_art`, since no font
  draws the characters they may leave; the other suffixes and encodings may
  still join it. `to_ascii_art` itself rejects a text with a character no
  font draws (any non-ASCII character);
- it pairs `fsh` or `dap` with an obfuscation in
  `manipulation_blocked_obfuscation`: the `translation` and `ascii_art`
  families and the `heavy` `encode` techniques, except those also in
  `sensitive_words`;
- in budget mode, its complexity is above what is left of `max_complexity`.

`combine_techniques` does not check these rules. It only sorts techniques
into layer order. [internals/jailbreak.md](internals/jailbreak.md) explains
how a combination is drawn.

Adding a technique: [extending.md](extending.md#add-a-jailbreak-technique).

## The eval pool

The `eval` dataset type's attacks draw from 37 techniques in 10 families:
hard ones (complexity 2 or 3 but for `to_noble_goal`, and the low-resource
languages) whose harmful content stays certain, since the technique carries
the text unchanged or reversibly, or a check accepts the result:

| Family | Techniques |
|---|---|
| `translation` | the eight low-resource languages, weighted 3 |
| `encode` | the five `heavy` ones: `to_base64`, `to_unicode_escape`, `to_ascii_ordinal`, `to_morse`, `to_braille` |
| `fsh`, `dap` | all eight |
| `framing` | all five |
| `structural` | `to_json`, `to_xml`, `to_markdown` |
| `ascii_art` | `to_ascii_art` |
| `continuation` | all four |
| `temporal` | `to_use_past_tense` |
| `answer_format` | `to_refusal_then_comply` |

Model rewrites no check verifies (`cognitive`, `persona`, `typos`,
`sensitive_words`, `tokenbreak`) are left out. With
`"adversarial": {"translation": false}` the pool keeps the other 28.
[configuration.md](configuration.md#dataset-types) gives the type file.

## What the families do

A transform returns `(text, info)`; the info names what it drew, such as a
template variant or a font, for traceability.

**Hacking.**

- `cognitive` and `persona` are two model calls each: a scenario, then the
  jailbreak built on it. A named persona grounds the scenario in the
  persona's character first. The definitions live in
  `configs/jailbreak/hacking/cognitive_techniques.json` and `personas.json`,
  editable without touching code. `to_invented_persona` makes the same two
  calls as `cognitive`'s `to_persona_roleplay` until it gets prompts of its
  own; only its info, `category=invented_persona`, differs.
- `framing` directives are transforms that change the scenario or the
  intent. Each has named template variants in
  `configs/jailbreak/hacking/framing_templates.json`; one is drawn per call.

**Manipulation.** Both families build on benign question-and-answer pairs
of one subcategory of the benign taxonomy. The `random` variants pick the
subcategory at random, the `selected` ones ask the model to choose one
(prompt `jailbreak/category_selection`); `short` and `long` are the length of
the answers.

- `fsh` (few-shot hacking): the benign pairs prime a helpful pattern, and the
  prompt comes last, as one more question.
- `dap` (distract and persuade): the prompt is hidden among the benign pairs,
  at a random position. With no pair at all, the prompt is the only
  question.

**Obfuscation.**

- `sensitive_words` and `tokenbreak` first ask the model for the harmful
  words of the prompt (prompt `jailbreak/extract_harmful`), one call per
  technique. The encoding, splitting or substitution then applies to those
  words. A reply that is not a list of at most ten of the prompt's words
  rejects the result (`DISCARDED; reason=malformed word list`). When the
  model marks no word, the encodings and splits apply to the whole prompt
  (every word for the fixed separators), and `to_sensitive_words_variables`
  and `to_synonym_substitution` leave it unchanged. `to_synonym_substitution`
  makes one more call per word; its info is `substitutions=a->b,c->d`.
- `translation` covers 19 languages in three resource tiers, which the
  spec scores by complexity: high (`to_french`, `to_japanese`, `to_russian`,
  `to_spanish`, `to_german`, `to_arabic`; 0), mid (`to_turkish`,
  `to_czech`, `to_vietnamese`, `to_greek`, `to_croatian`; 1) and low
  (`to_swahili`, `to_thai`, `to_khmer`, `to_nepali`, `to_zulu`,
  `to_scots_gaelic`, `to_bengali`, `to_javanese`; 2). On a chat translator
  each is a translate, check and retry loop on the `translation` model. On
  a machine-translation model (`translategemma-27b`, the default) the text
  goes paragraph by paragraph; each piece is checked for its language
  (GlotLID) and length, then translated back and judged by the generation
  model ([stages/jailbreaks.md](stages/jailbreaks.md#models)).
- `typos` is one rewrite by the model, at one of four densities, with the
  level's description in the prompt.
- `casual_writing` is how a person typing quickly writes: casing, spacing,
  punctuation, abbreviations, accents. `to_abbreviations` rewrites only the
  common words it lists, and `to_stripped_accents` only accented Latin
  letters (other scripts keep their marks), so either leaves many prompts
  unchanged: a no-op, which the natural kind rejects.
- `suffixes` appends an adversarial suffix: punctuation, fragments, unicode
  or emoji. Its info is `suffix=<suffix>`.
- `encode` encodes the whole text: base64 (one line), rot13, rot18 (rot13 on
  letters and rot5 on ASCII digits), rot47, unicode escapes (`\uXXXX`, or
  `\UXXXXXXXX` above U+FFFF), ASCII ordinals, a separator (between the
  characters as a reader sees them: a combining mark or a joiner stays with
  its letter), leetspeak at three levels, morse, braille.
- `structural` wraps the text in JSON, XML or markdown. `to_json` keeps
  non-ASCII characters as they are.
- `ascii_art` draws the text in a pyfiglet font, drawn from the 16 that have
  a glyph for every character of the text. When none has, the result is
  rejected (`DISCARDED; reason=no font has every character`).

**Requests.** These change how a request is posed, not its text, and call no
model. Each technique picks one of its named template variants from a JSON
file under `configs/jailbreak/requests/`. The `answer_language` techniques
are generated from the language list in `answer_language_templates.json`, so
a new language is one JSON entry, plus its technique's entry in
`combination_spec.json`. `to_impersonate_good_person` has eight
profession variants (nurse, doctor, security researcher, teacher, ...) and
`to_use_past_tense` four historical framings.

## Reference list coverage

The library is benchmarked against a reference set of 73 instruction
primitives and 74 request primitives.

| Status | Primitives |
|---|---|
| Covered | All encoding, sensitive-words and typo primitives; all 16 translation languages; all 14 named personas; every framing directive (`fictional_world`, `noble_goal`, `nefarious_goal`, `high_stake`, `no_moral_constraints`, `refusal_then_comply`, `prefix_injection`, `apology_prevention`, `answer_in_markdown`, `answer_as_tutorial`); every request primitive (`innocuous_question`, `impersonate_good_person`, the four distractors, `use_past_tense`, `ask_for_details`, `ask_answer_in` for 16 languages) |
| Left out by design | `agent_context_additional_instr` (needs access to the system prompt), `fine_tuning` (out of scope), `use_highly_specialized_language` (no clear way to build it), `direct_question` (changes nothing) |

Beyond the reference list, the library adds:

- the encodings rot18, rot47, braille, morse and ASCII ordinals;
- ASCII art in 16 pyfiglet fonts;
- adversarial suffixes (punctuation, fragments, unicode, emoji);
- the `casual_writing` family (9 techniques: casing, spacing, punctuation,
  abbreviations, accents);
- structural wrapping (JSON, XML, markdown);
- the five cognitive techniques (`persona_roleplay`, `hypothetical_framing`,
  `authority_obedience`, `avi`, `deep_inception`);
- the manipulation families, FSH and DAP, in eight variants;
- the answer formats code, dialogue, story and structured data;
- continuation attacks (4) and indirect task embedding (6);
- the answer languages Zulu, Scots Gaelic, Bengali and Javanese.
