# Jailbreak techniques

The jailbreak stage draws from 144 active techniques in 4 layers and 23
families.
`configs/jailbreak/combination_spec.json` gives each technique (the
language ones through its `generators`, from
[the language table](#the-language-table)) its families, a complexity from
0 to 3, whether it calls a model (`requires_llm`) or
needs the benign data (`requires_benign`), and the kinds that may draw it
(`kinds`, [below](#the-natural-pool)). This page lists them by name, for
the `allow` and `deny` lists of the
[augmentation section](configuration.md#the-augmentation-section), and says
briefly what each family does.

The spec also dates each technique: `since` is the spec version that added
it (`"1.0"` when absent) and `deprecated` the one that retired it. A retired
technique stays registered, so a plan that names it still replays, but no
pool draws it unless the kind's `techniques.allow` names it (with a
`DeprecationWarning`, for this release). A jailbreak plan kept from before a
technique was added or retired resumes with a WARNING
([stages/jailbreaks.md](stages/jailbreaks.md#how-it-resumes)). The counts on
this page are of the active techniques; the seven spec version 1.1 retired
are listed [below](#retired-techniques).

## Layers

A combination applies its techniques in layer order: `hacking`, then
`manipulation`, then `obfuscation`, then `requests`; the request obfuscation,
when allowed (the packaged attacks allow it, for half their units; the eval
type does not), comes last, after the request. Within the obfuscation layer,
families apply in the order of their `within_layer_order`, numbered in tens
so a family added later slots in between: the whole-text rewrites first
(`translation` 10, `typos` 20, `casual_writing` 30), then the word-level
steps (`sensitive_words` 40, `tokenbreak` 50), so a rewrite cannot undo a
split or an encoded word, then `suffixes` 60, `encode` 70, `structural` 80
and `ascii_art` 90, the outermost. A plan made before this order applied
`sensitive_words` and `tokenbreak` first; a kept plan replays the order its
manifest recorded, and a fresh plan applies translation, typos and casual
writing before the word-level steps. The families the eval type draws
(`translation`, `encode`, `structural`, `ascii_art`) keep their order. The
obfuscation layer has no cap: a combination takes up to `max_obfuscations`
of its families in budget mode, the request obfuscation, when allowed,
among them.

<!-- BEGIN layers: made from the spec by tests/jailbreak/test_docs_counts.py -->
| Layer | Families | Techniques | Picks per combination |
|---|---|---|---|
| `hacking` | 3 | 21 | at most 1 |
| `manipulation` | 2 | 8 | at most 1 |
| `obfuscation` | 9 | 68 | no cap |
| `requests` | 9 | 47 | at most 1 |
| all | 23 | 144 | |
<!-- END layers -->

## Families and techniques

The two tables on this page are made from the spec: after a change to it,
make them again from the repository root with
`PYTHONPATH=src py -m tests.jailbreak.test_docs_counts` (the test fails
until then). A technique counts in its first family; a retired one is left
out. The translation family calls the `translation` model (a machine
translator also has the generation model judge it); the other techniques
that call a model call the generation model.

<!-- BEGIN families: made from the spec by tests/jailbreak/test_docs_counts.py -->
| Layer | Family | Dimension | Order | Count | Calls a model | Needs benign data | Techniques |
|---|---|---|---|---|---|---|---|
| hacking | `cognitive` | `role_framing` |  | 5 | yes | no | `to_persona_roleplay`, `to_hypothetical_framing`, `to_authority_obedience`, `to_avi`, `to_deep_inception` |
| hacking | `framing` | `role_framing` |  | 5 | no | no | `to_fictional_world`, `to_noble_goal`, `to_nefarious_goal`, `to_high_stake`, `to_no_moral_constraints` |
| hacking | `persona` | `role_framing` |  | 11 | yes | no | `to_invented_persona`, `to_psychopath_persona`, `to_alien_persona`, `to_radical_politician_persona`, `to_cult_leader_persona`, `to_very_advanced_ai_persona`, `to_cartel_leader_persona`, `to_artist_persona`, `to_deformed_scientist_persona`, `to_actor_persona`, `to_someone_from_the_future_persona` |
| manipulation | `dap` | `example_priming` |  | 4 | `to_dap_selected_short`, `to_dap_selected_long` | yes | `to_dap_random_short`, `to_dap_random_long`, `to_dap_selected_short`, `to_dap_selected_long` |
| manipulation | `fsh` | `example_priming` |  | 4 | `to_fsh_selected_short`, `to_fsh_selected_long` | yes | `to_fsh_random_short`, `to_fsh_random_long`, `to_fsh_selected_short`, `to_fsh_selected_long` |
| obfuscation | `translation` | `language` | 10 | 19 | yes | no | `to_french`, `to_japanese`, `to_russian`, `to_spanish`, `to_german`, `to_arabic`, `to_turkish`, `to_czech`, `to_vietnamese`, `to_greek`, `to_croatian`, `to_swahili`, `to_thai`, `to_khmer`, `to_nepali`, `to_zulu`, `to_scots_gaelic`, `to_bengali`, `to_javanese` |
| obfuscation | `typos` | `surface_noise` | 20 | 4 | yes | no | `to_rewrite_with_typos_low`, `to_rewrite_with_typos_medium`, `to_rewrite_with_typos_high`, `to_rewrite_with_typos_insane` |
| obfuscation | `casual_writing` | `surface_noise` | 30 | 9 | no | no | `to_lowercase`, `to_uppercase`, `to_no_sentence_capitals`, `to_dropped_spaces`, `to_doubled_spaces`, `to_dropped_punctuation`, `to_doubled_punctuation`, `to_abbreviations`, `to_stripped_accents` |
| obfuscation | `sensitive_words` | `encoding` | 40 | 13 | yes | no | `to_sensitive_words_encode_base64`, `to_sensitive_words_encode_rot13`, `to_sensitive_words_encode_rot18`, `to_sensitive_words_encode_rot47`, `to_sensitive_words_encode_unicode`, `to_sensitive_words_encode_ascii`, `to_sensitive_words_encode_separator`, `to_sensitive_words_encode_leetspeak_basic`, `to_sensitive_words_encode_leetspeak_intermediate`, `to_sensitive_words_encode_leetspeak_advanced`, `to_sensitive_words_split`, `to_sensitive_words_variables`, `to_synonym_substitution` |
| obfuscation | `tokenbreak` | `encoding` | 50 | 3 | yes | no | `to_tokenbreak_prepend`, `to_tokenbreak_split`, `to_tokenbreak_delimiter` |
| obfuscation | `suffixes` | `surface_noise` | 60 | 4 | no | no | `to_adversarial_suffix_punctuation`, `to_adversarial_suffix_fragments`, `to_adversarial_suffix_unicode`, `to_adversarial_suffix_emoji` |
| obfuscation | `encode` | `encoding` | 70 | 12 | no | no | `to_base64`, `to_rot13`, `to_rot18`, `to_rot47`, `to_unicode_escape`, `to_ascii_ordinal`, `to_separator`, `to_leetspeak_basic`, `to_leetspeak_intermediate`, `to_leetspeak_advanced`, `to_morse`, `to_braille` |
| obfuscation | `structural` | `container` | 80 | 3 | no | no | `to_json`, `to_xml`, `to_markdown` |
| obfuscation | `ascii_art` | `encoding` | 90 | 1 | no | no | `to_ascii_art` |
| requests | `answer_format` | `answer_shape` |  | 9 | no | no | `to_refusal_then_comply`, `to_apology_prevention`, `to_prefix_injection`, `to_answer_in_markdown`, `to_answer_as_tutorial`, `to_answer_as_code`, `to_answer_as_dialogue`, `to_answer_as_story`, `to_answer_as_structured_data` |
| requests | `answer_language` | `language` |  | 20 | no | no | `to_ask_answer_in_french`, `to_ask_answer_in_japanese`, `to_ask_answer_in_russian`, `to_ask_answer_in_spanish`, `to_ask_answer_in_german`, `to_ask_answer_in_arabic`, `to_ask_answer_in_turkish`, `to_ask_answer_in_czech`, `to_ask_answer_in_vietnamese`, `to_ask_answer_in_greek`, `to_ask_answer_in_croatian`, `to_ask_answer_in_swahili`, `to_ask_answer_in_thai`, `to_ask_answer_in_khmer`, `to_ask_answer_in_maori`, `to_ask_answer_in_nepali`, `to_ask_answer_in_zulu`, `to_ask_answer_in_scots_gaelic`, `to_ask_answer_in_bengali`, `to_ask_answer_in_javanese` |
| requests | `asking` | `request_shape` |  | 2 | no | no | `to_innocuous_question`, `to_ask_for_details` |
| requests | `continuation` | `request_shape` |  | 4 | no | no | `to_continue_passage`, `to_complete_story`, `to_fill_in_blank`, `to_continue_instructions` |
| requests | `distractor_prefix` | `request_shape` |  | 2 | no | no | `to_distractor_prefix_related`, `to_distractor_prefix_unrelated` |
| requests | `distractor_suffix` | `request_shape` |  | 2 | no | no | `to_distractor_suffix_related`, `to_distractor_suffix_unrelated` |
| requests | `impersonation` | `role_framing` |  | 1 | no | no | `to_impersonate_good_person` |
| requests | `indirect` | `request_shape` |  | 6 | no | no | `to_summarize`, `to_proofread`, `to_critique`, `to_translate_task`, `to_explain_to_child`, `to_rewrite` |
| requests | `temporal` | `request_shape` |  | 1 | no | no | `to_use_past_tense` |
<!-- END families -->

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
too: `heavy` for base64, unicode and ascii, `light` for the rest. Of these
ten, the `heavy` ones stay allowed with `fsh` and `dap`, which the plain
`heavy` `encode` techniques are not; the `light` ones, like the plain `light`
ones, may be the request obfuscation.

Twelve of them are tagged `needs_latin`: `to_rot13`, `to_rot18`,
`to_rot47`, the three `to_leetspeak_*` and their six
`to_sensitive_words_encode_*` forms. They change only Latin letters (ROT18
also ASCII digits, ROT47 every printable ASCII character), so they leave a
text in another script as it was, as a test checks on six scripts.
`to_morse` and `to_braille` pass such a text through as well (Morse only
spaces its characters out) but are not tagged: both are in the eval pool,
which is held, and completing them would change its text.

## The language table

`configs/jailbreak/languages.json` holds one row per language: its `name`
(as prompts and infos write it), `slug` (the end of its technique names),
`complexity` (by resource tier: 0 high, 1 mid, 2 low), `script` (ISO 15924,
which the script rule reads), and whether it has a `translation` technique
(`to_<slug>`) and an `answer_language` one (`to_ask_answer_in_<slug>`). The
spec's `generators` make a technique entry of each marked row as the spec
loads, and the two families' modules build the functions from the same
rows, in the table's order. Of the twenty rows, all have an answer-language
technique and nineteen a translation (not Maori). The translation family
kept the order it had; the answer languages took the table's, which puts
Swahili before Thai, so a fresh plan that drew one of the two draws the
other (a kept plan replays by name). The table is read from the package
only. To add a language: [extending.md](extending.md#add-a-language).

## Retired techniques

Spec version 1.1 retired seven techniques. They stay registered, so a plan
that names them replays them, and a kind draws one only when its
`techniques.allow` names it (with a `DeprecationWarning`):

| Family | Technique | Why |
|---|---|---|
| `sensitive_words` | `to_sensitive_words_star`, `to_sensitive_words_hyphen`, `to_sensitive_words_underscore` | `to_sensitive_words_encode_separator` draws their separators (`*`, `-`, `_`) among its own; `to_sensitive_words_split` stays, since a space is not one of them |
| `persona` | `to_politician_persona`, `to_deformed_professor_persona`, `to_religious_figure_persona` | near-duplicates of `to_radical_politician_persona`, `to_deformed_scientist_persona` and `to_cult_leader_persona`, which stay |
| `persona` | `to_mentally_ill_persona` | retired, with no replacement |

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
  `persona`, `cognitive`; and `persona` and `cognitive` with
  `impersonation`, so a prompt never carries two identities. So a
  combination holds a hacking or a manipulation technique, never both
  (`continuation` and `indirect`, both requests, are kept apart by the
  layer cap);
- one of its families is in the `cross_incompatible_with` of a taken
  technique's own spec entry, or a taken family is in its own. The spec keeps
  `to_adversarial_suffix_emoji`, `to_adversarial_suffix_unicode`,
  `to_braille` and `to_tokenbreak_split` from `ascii_art`, since no font
  draws the characters they may leave; the other suffixes and encodings may
  still join it. `to_ascii_art` itself rejects a text with a character no
  single font draws (most non-ASCII characters: accented letters such as `é`
  or `ñ`, dashes, curly quotes, kana; German umlauts and `ß` and some Greek
  or Cyrillic text have fonts), or one every font draws longer than its cap;
- it pairs `fsh` or `dap` with an obfuscation in
  `manipulation_blocked_obfuscation`: the `translation` and `ascii_art`
  families and the `heavy` `encode` techniques, except those also in
  `sensitive_words`;
- it pairs a `needs_latin` technique with a translation into a script
  other than Latin: Japanese, Russian, Arabic, Greek, Thai, Khmer, Nepali
  or Bengali (each language's `script` is in its row of
  `configs/jailbreak/languages.json`). After the translation the encoding would
  change nothing, yet count in `num_techniques`; before it, the translator
  would get a cipher. The Latin-script translations still join them;
- in budget mode, its complexity is above what is left of `max_complexity`.

`combine_techniques` does not check these rules. It only sorts techniques
into layer order. [internals/jailbreak.md](internals/jailbreak.md) explains
how a combination is drawn.

Adding a technique: [extending.md](extending.md#add-a-jailbreak-technique).

## The eval pool

The `eval` dataset type's attacks draw from 36 techniques in 10 families:
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

`to_ascii_art` keeps its 5,000-character cap in this pool too, so its eval
attacks differ from those of a version without the cap: the font a seed
draws changes for some texts from about 40 characters and for most from
about 90, a text of about 830 characters or more is rejected where it used
to be drawn, and so are most inputs when the second round pairs it with a
`heavy` encoding that lengthens the text (`to_unicode_escape`,
`to_ascii_ordinal`, `to_morse`). An eval plan kept across the change mixes
the two renderings. [What the families do](#what-the-families-do) has the
figures.

## The natural pool

A spec entry's `kinds` lists the kinds of the augmentation section that may
draw the technique: `["adversarial"]` when neither the technique nor its
first family gives one, and a technique's own `kinds` replace its family's.
Every technique is adversarial. The families `typos`, `casual_writing`,
`translation`, `answer_language`, `answer_format`, `asking` and `indirect`
are marked `["adversarial", "natural"]`, less their attack-only techniques,
and three techniques of attack families are marked natural on their own.
The natural kind draws only what is marked natural: by default (translation
off) these 46 techniques in 8 families, pinned in
`tests/fixtures/natural_pool.json`:

| Family | Natural techniques |
|---|---|
| `typos` | `to_rewrite_with_typos_low`, `to_rewrite_with_typos_medium` |
| `casual_writing` | all nine |
| `answer_language` | all twenty |
| `answer_format` | `to_answer_in_markdown`, `to_answer_as_tutorial`, `to_answer_as_code`, `to_answer_as_structured_data` |
| `asking` | `to_innocuous_question`, `to_ask_for_details` |
| `indirect` | all six |
| `encode` | `to_leetspeak_basic` |
| `suffixes` | `to_adversarial_suffix_punctuation`, `to_adversarial_suffix_emoji` |

With `"natural": {"translation": true}` the nineteen translations join them.
Attack-only: `structural`, `continuation`, `temporal`, the high and insane
typos, `to_answer_as_dialogue`, `to_answer_as_story` and the three
answer-format directives the natural kind denies anyway
(`to_refusal_then_comply`, `to_apology_prevention`,
`to_prefix_injection`), with every other family. A natural allow list that
names one of them raises, so a `config_dir` copy of `augmentations.json`
from before needs its natural `families.allow` trimmed. A natural-kind plan
made before raises on resume, since its pool changed: run the natural kind
with `resume=False`. A training build leaves out a benign or dual-use benign
variation made with a technique the natural kind does not draw, with one
WARNING that counts them ([stages/build.md](stages/build.md)).

## What the families do

A transform returns `(text, info)`; the info names what it drew, such as a
template variant or a font, for traceability.

**Hacking.**

- `cognitive` and `persona` are two model calls each: a scenario, then the
  jailbreak built on it. A named persona grounds the scenario in the
  persona's character first; the prompt names it by its `display_name`
  (`Someone from the Future`), the info by its key
  (`category=someone_from_the_future_persona`). The scenario prompts ask
  for the scene at the bottom, behind `---` delimiters: the scene is the
  last block after the first delimiter, or, in a reply without one, what
  follows a `Scenario Description` heading. A reply with neither is asked
  again, up to 3 times in all, and then rejects the result
  (`DISCARDED; reason=malformed scenario`). The definitions live in
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
  rejects the result (`DISCARDED; reason=malformed word list`). A word is a
  letter, then letters, digits, apostrophes, hyphens and combining marks
  (not first), so a word a translation wrote in Bengali, Nepali, Thai or
  Khmer, whose vowel signs and viramas are marks, is a word.
  `to_tokenbreak_split` and the separator branch of
  `to_tokenbreak_delimiter` cut a word between the characters a reader
  sees, as `to_separator` does, never between a letter and its marks; a
  conjunct (a virama or Khmer coeng, then a consonant) can still be cut
  between its letters, as `to_separator` cuts it too. An ASCII word is cut
  as before. When the
  model marks no word, the three `heavy` encodings
  (`to_sensitive_words_encode_base64`, `_unicode`, `_ascii`) reject the
  result (`DISCARDED; reason=no sensitive word`): encoding the whole prompt
  would make them a heavy full-text encoding, which the spec keeps away
  from `fsh` and `dap`. The light encodings and the splits apply to the
  whole prompt (every word for the fixed separators), and
  `to_sensitive_words_variables` and `to_synonym_substitution` leave it
  unchanged. `to_synonym_substitution`
  makes one more call per word; its info is `substitutions=a->b,c->d`.
- `translation` covers 19 languages, the rows of
  [the language table](#the-language-table) marked `translation` (not Maori,
  which TranslateGemma renders
  with invented sensitive and technical terms, fluently enough to pass every
  check) in three resource tiers, which the table scores by complexity:
  high (`to_french`, `to_japanese`, `to_russian`, `to_spanish`, `to_german`,
  `to_arabic`; 0), mid (`to_turkish`,
  `to_czech`, `to_vietnamese`, `to_greek`, `to_croatian`; 1) and low
  (`to_swahili`, `to_thai`, `to_khmer`, `to_nepali`, `to_zulu`,
  `to_scots_gaelic`, `to_bengali`, `to_javanese`; 2). On a chat translator
  each is a translate, check and retry loop on the `translation` model. On
  a machine-translation model (`translategemma-27b`, the default) the text
  goes paragraph by paragraph; each piece is checked for its language
  (GlotLID) and length (against the language's median ratio, measured
  with TranslateGemma on FLORES-200), then translated back and judged by
  the generation
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
- `ascii_art` draws the text in a pyfiglet font, drawn from those of the 16
  that have a glyph for every character of the text and draw it in at most
  `MAX_RENDERED_CHARS` (5,000) characters, the `max_chars` keyword's
  default. When none has every glyph, the result is rejected
  (`DISCARDED; reason=no font has every character`); when every such font
  draws the text longer than the cap, it is rejected with the shortest
  drawing's length (`DISCARDED; reason=too long; rendered=<n>;
  max_chars=5000`). The cap was set from these drawing lengths, in
  characters, of plain English text (a range over three texts; `-` where
  the font lacks a character, a comma or a question mark):

  | Font | 50 | 100 | 150 | 300 |
  |---|---|---|---|---|
  | `digital` | 306 | 603-606 | 903-906 | 1,716-1,806 |
  | `bubble` | 728-744 | 1,440-1,468 | 2,148-2,204 | 4,072-4,360 |
  | `small` | 990-1,015 | 1,935-1,965 | 2,835-2,965 | 5,330-5,805 |
  | `alphabet` | 1,274-1,365 | 2,520-2,639 | 3,717-3,976 | 7,028-7,763 |
  | `standard` | 1,368-1,464 | 2,706-2,820 | 3,990-4,236 | 7,536-8,298 |
  | `slant` | 1,482-1,560 | 2,940-3,042 | 4,320-4,560 | 8,136-8,820 |
  | `big` | 1,832-1,960 | 3,648-3,776 | 5,376-5,648 | 10,144-11,120 |
  | `banner` | 2,344-2,360 | 4,560-4,632 | 6,784-6,984 | 12,752-13,896 |
  | `gothic` | 2,466-2,502 | 4,860-4,995 | 7,191-7,452 | 13,644-14,562 |
  | `block` | 3,280-3,584 | 6,600-6,856 | 9,728-10,256 | 18,416-20,128 |
  | `alligator` | 3,290-3,633 | 6,552-6,993 | 9,849-10,346 | 18,851-19,999 |
  | `dotmatrix` | 6,050-6,390 | 11,980-12,520 | 17,920-18,790 | 33,830-36,790 |
  | `isometric1`-`4` | 6,347-6,699 | 12,892-13,112 | - | - |

  The cap holds wherever the technique is drawn, the
  [eval pool](#the-eval-pool) included, and it changes the text, not only
  what is rejected. A text every font
  draws within the cap, up to about 38 characters, draws the same font as
  without one. Over that, the font is drawn from the fonts that fit. Of the
  12 fonts that have a typical text's glyphs (the four `isometric` fonts
  have no digit or punctuation mark), all 12 fit up to about 38
  characters, 11 at 50, 9 to 11 at 75, 9 at 100, 6 at 150, 2 or 3 at 200
  to 300, 1 (`digital`) at 400 to 800 and none from about 830. So the same
  seed draws another font, and another text, than it did before the cap:
  for a text with a digit or a punctuation mark, about 1 unit in 12 from
  about 45 to 75 characters (those that drew `dotmatrix`, the last font of
  the list), about 2 in 3 at 90 to 100 and about 9 in 10 from about 125;
  for one of letters and spaces alone (all 16 fonts), about 19 in 20 from
  about 38.

  A text of about 830 characters or more is rejected where it used to be
  drawn. After an encoding that lengthens the text, most inputs are
  rejected: unicode escapes make a text about 6 times as long, so an input
  of about 140 characters or more is rejected after them, one of about 230
  after ASCII ordinals (about 3.7 times) and one of about 250 after morse
  (about 3.4 times); base64 (about 1.4 times) lowers the limit to about
  610. A kept plan resumed across this change mixes rows drawn without the
  cap (saved before) and with it (run after), so a replayed plan gives the
  same text only within one version of the library; run the kind with
  `resume=False` for one rendering.

  Each font is loaded once per process. Every font that has the glyphs
  draws the text to measure it, so a text costs up to 16 drawings (about
  0.03 s in all at 150 characters, 0.4 s at 2,000).

**Requests.** These change how a request is posed, not its text, and call no
model. Each technique picks one of its named template variants from a JSON
file under `configs/jailbreak/requests/`. The `answer_language` techniques
are generated from the rows of `configs/jailbreak/languages.json` marked
`answer_language` ([the language table](#the-language-table)), so a new
language is one row. `to_impersonate_good_person` has eight
profession variants (nurse, doctor, security researcher, teacher, ...) and
`to_use_past_tense` four historical framings.

The variants of a directive (a framing, request or structural template)
form its variant pool. Each is as likely to be drawn unless a variant of
the pool gives a `weight`; `0` turns a variant off. No packaged variant has
a weight, and an eval-pool technique's may not, since a weight changes the
eval text ([extending.md](extending.md#add-variants-to-a-template-pool)).
Importing `redact.jailbreak` checks every pool and raises naming the file,
directive and variant of a fault. A pool should hold at least 8 variants
(a request or framing directive, a container); most hold 4 today: 34 of the
35 pools are below that floor, which a test ratchets
([extending.md](extending.md#add-variants-to-a-template-pool)).

## Reference list coverage

The library is benchmarked against a reference set of 73 instruction
primitives and 74 request primitives.

| Status | Primitives |
|---|---|
| Covered | All encoding, sensitive-words and typo primitives; 15 of the 16 translation languages (Maori is left out, see [above](#what-the-families-do)); all 14 named personas (4 of them retired, [above](#retired-techniques)); every framing directive (`fictional_world`, `noble_goal`, `nefarious_goal`, `high_stake`, `no_moral_constraints`, `refusal_then_comply`, `prefix_injection`, `apology_prevention`, `answer_in_markdown`, `answer_as_tutorial`); every request primitive (`innocuous_question`, `impersonate_good_person`, the four distractors, `use_past_tense`, `ask_for_details`, `ask_answer_in` for 16 languages) |
| Left out by design | `agent_context_additional_instr` (needs access to the system prompt), `fine_tuning` (out of scope), `use_highly_specialized_language` (no clear way to build it), `direct_question` (changes nothing) |

Beyond the reference list, these are ports of the BELLS_Jailbreak reference
library (its `hacking.py`, `manipulation.py` and `obfuscation.py`, as the
module docstrings of this repository's first commit say):

- ASCII art (pyfiglet fonts);
- adversarial suffixes (punctuation, fragments, unicode, emoji);
- structural wrapping (JSON, XML, markdown);
- the five cognitive techniques (`persona_roleplay`, `hypothetical_framing`,
  `authority_obedience`, `avi`, `deep_inception`);
- the manipulation families, FSH and DAP, in eight variants;
- the encodings morse and braille;
- the translations into Zulu, Scots Gaelic, Bengali and Javanese.

Also beyond the reference list:

- the encodings rot18, rot47 and ASCII ordinals;
- the `casual_writing` family (9 techniques: casing, spacing, punctuation,
  abbreviations, accents);
- the answer formats code, dialogue, story and structured data;
- continuation attacks (4) and indirect task embedding (6);
- the answer languages Zulu, Scots Gaelic, Bengali and Javanese.
