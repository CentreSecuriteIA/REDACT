# Jailbreak techniques

The jailbreak stage draws from 244 active techniques in 5 layers and 36
families; the first layer, `rewrite`, holds 16 and is off by default
([below](#layers)).
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

A combination applies its techniques in layer order: `rewrite`, then
`hacking`, then `obfuscation`, then `requests`, then `manipulation`; the
request obfuscation, when allowed (the packaged attacks allow it, for half their units, all
among those that drew a request; the eval type does not), comes right after the request,
before the manipulation layer; a combination without a request takes none.
The manipulation layer comes last since the eval release: `fsh`, `dap` and
`fake_turns` put the finished question, obfuscated and with its request
around it, among their examples or after their invented turn, so the
obfuscations, the request and the model steps (a typo rewrite, a marking
call) read and change the question alone, and the examples stay as written.
A selected `fsh` or `dap` variant picks its examples on the clean input
([below](#what-the-families-do)). A plan made before the eval release
applied the manipulation layer second; a kept plan replays the order its
manifest recorded, and its resume warns that the rules changed (the plan
file's `rules` id holds the layer order). Within the obfuscation layer,
families apply in the order of their `within_layer_order`, numbered in tens
so a family added later slots in between: the whole-text rewrites first
(`translation` 10, `typos` 20, `casual_writing` 30, `noise` 35), then the word-level
steps (`sensitive_words` 40, `wording` 45, `tokenbreak` 50, `word_art` 55), so a rewrite cannot undo a
split, an encoded word or a code name, then `suffixes` 60, `payload_split` 65, `encode` 70, `structural` and
`neutral_container` 80 (each excludes the other, so they share the place)
and `ascii_art` 90, the outermost obfuscation. A plan made before this order applied
`sensitive_words` and `tokenbreak` first; a kept plan replays the order its
manifest recorded, and a fresh plan applies translation, typos, casual
writing and noise before the word-level steps (a fresh plan never pairs a
translation with the other three, which a kept plan may hold). The families the eval type draws
(`translation`, `encode`, `structural`, `ascii_art`) keep their order. The
obfuscation layer has no cap: a combination takes up to `max_obfuscations`
of its families in budget mode, the request obfuscation, when allowed,
among them.

The `rewrite` layer comes before every other layer, at most one technique a
combination: a model rewrites the whole prompt as a persuasion appeal or in
another tense, or writes sentences placed before or after it
([below](#what-the-families-do)). It is off by default: its spec rate is 0
and the spec lists it in `layers_off_by_default`, so a kind draws it only
when its `layers.allow` names it. Its 16 techniques say `["adversarial"]`,
so the natural kind never draws them, and `eval.json` names no rewrite
family;
[extending.md](extending.md#add-a-rewrite-technique) says how to turn it
on. (`to_rewrite`, despite its name, is an `indirect` request technique.)

<!-- BEGIN layers: made from the spec by tests/jailbreak/test_docs_counts.py -->
| Layer | Families | Techniques | Picks per combination |
|---|---|---|---|
| `rewrite` | 3 | 16 | at most 1 |
| `hacking` | 4 | 35 | at most 1 |
| `obfuscation` | 14 | 110 | no cap |
| `requests` | 12 | 68 | at most 1 |
| `manipulation` | 3 | 15 | at most 1 |
| all | 36 | 244 | |
<!-- END layers -->

## Families and techniques

The two tables on this page are made from the spec: after a change to it,
make them again from the repository root with
`PYTHONPATH=src python -m tests.jailbreak.test_docs_counts` (the test fails
until then). A technique counts in its first family; a retired one is left
out. The translation family calls the `translation` model (a machine
translator also has the generation model judge it); the other techniques
that call a model call the generation model.

<!-- BEGIN families: made from the spec by tests/jailbreak/test_docs_counts.py -->
| Layer | Family | Dimension | Order | Count | Calls a model | Needs benign data | Techniques |
|---|---|---|---|---|---|---|---|
| rewrite | `model_distractors` | `request_shape` |  | 2 | yes | no | `to_model_distractor_prefix`, `to_model_distractor_suffix` |
| rewrite | `persuasion` | `request_shape` |  | 12 | yes | no | `to_persuade_evidence_based`, `to_persuade_logical_appeal`, `to_persuade_expert_endorsement`, `to_persuade_authority_endorsement`, `to_persuade_social_proof`, `to_persuade_emotional_appeal`, `to_persuade_reciprocity`, `to_persuade_commitment_consistency`, `to_persuade_scarcity`, `to_persuade_priming`, `to_persuade_framing`, `to_persuade_relationship_leverage` |
| rewrite | `tense_rewrite` | `request_shape` |  | 2 | yes | no | `to_rewrite_in_past_tense`, `to_rewrite_in_future_tense` |
| hacking | `cognitive` | `role_framing` |  | 5 | yes | no | `to_persona_roleplay`, `to_hypothetical_framing`, `to_authority_obedience`, `to_avi`, `to_deep_inception` |
| hacking | `framing` | `role_framing` |  | 5 | no | no | `to_fictional_world`, `to_noble_goal`, `to_nefarious_goal`, `to_high_stake`, `to_no_moral_constraints` |
| hacking | `persona` | `role_framing` |  | 14 | yes | no | `to_invented_persona`, `to_psychopath_persona`, `to_alien_persona`, `to_radical_politician_persona`, `to_cult_leader_persona`, `to_very_advanced_ai_persona`, `to_cartel_leader_persona`, `to_artist_persona`, `to_deformed_scientist_persona`, `to_actor_persona`, `to_someone_from_the_future_persona`, `to_mercenary_persona`, `to_underground_hacker_persona`, `to_anarchist_philosopher_persona` |
| hacking | `persona_template` | `role_framing` |  | 11 | no | no | `to_dan_persona_template`, `to_stan_persona_template`, `to_dude_persona_template`, `to_developer_mode_persona_template`, `to_aim_persona_template`, `to_evil_confidant_persona_template`, `to_opposite_mode_persona_template`, `to_unfiltered_oracle_persona_template`, `to_rogue_ai_persona_template`, `to_maintenance_mode_persona_template`, `to_liberated_model_persona_template` |
| obfuscation | `translation` | `language` | 10 | 23 | yes | no | `to_french`, `to_japanese`, `to_russian`, `to_spanish`, `to_german`, `to_arabic`, `to_turkish`, `to_czech`, `to_vietnamese`, `to_greek`, `to_croatian`, `to_swahili`, `to_thai`, `to_khmer`, `to_nepali`, `to_zulu`, `to_scots_gaelic`, `to_bengali`, `to_javanese`, `to_hmong`, `to_chinese`, `to_hindi`, `to_amharic` |
| obfuscation | `typos` | `surface_noise` | 20 | 4 | yes | no | `to_rewrite_with_typos_low`, `to_rewrite_with_typos_medium`, `to_rewrite_with_typos_high`, `to_rewrite_with_typos_insane` |
| obfuscation | `casual_writing` | `surface_noise` | 30 | 9 | no | no | `to_lowercase`, `to_uppercase`, `to_no_sentence_capitals`, `to_dropped_spaces`, `to_doubled_spaces`, `to_dropped_punctuation`, `to_doubled_punctuation`, `to_abbreviations`, `to_stripped_accents` |
| obfuscation | `noise` | `surface_noise` | 35 | 12 | no | no | `to_random_capitals_low`, `to_random_capitals_medium`, `to_random_capitals_high`, `to_scrambled_letters_low`, `to_scrambled_letters_medium`, `to_scrambled_letters_high`, `to_keyboard_typos_low`, `to_keyboard_typos_medium`, `to_keyboard_typos_high`, `to_emoji_sprinkle_low`, `to_emoji_sprinkle_medium`, `to_emoji_sprinkle_high` |
| obfuscation | `sensitive_words` | `encoding` | 40 | 13 | yes | no | `to_sensitive_words_encode_base64`, `to_sensitive_words_encode_rot13`, `to_sensitive_words_encode_rot18`, `to_sensitive_words_encode_rot47`, `to_sensitive_words_encode_unicode`, `to_sensitive_words_encode_ascii`, `to_sensitive_words_encode_separator`, `to_sensitive_words_encode_leetspeak_basic`, `to_sensitive_words_encode_leetspeak_intermediate`, `to_sensitive_words_encode_leetspeak_advanced`, `to_sensitive_words_split`, `to_sensitive_words_variables`, `to_synonym_substitution` |
| obfuscation | `wording` | `encoding` | 45 | 2 | yes | no | `to_pseudonym_substitution`, `to_word_game` |
| obfuscation | `tokenbreak` | `encoding` | 50 | 3 | yes | no | `to_tokenbreak_prepend`, `to_tokenbreak_split`, `to_tokenbreak_delimiter` |
| obfuscation | `word_art` | `encoding` | 55 | 1 | yes | no | `to_word_art` |
| obfuscation | `suffixes` | `surface_noise` | 60 | 4 | no | no | `to_adversarial_suffix_punctuation`, `to_adversarial_suffix_fragments`, `to_adversarial_suffix_unicode`, `to_adversarial_suffix_emoji` |
| obfuscation | `payload_split` | `encoding` | 65 | 2 | `to_payload_split_semantic` | no | `to_payload_split`, `to_payload_split_semantic` |
| obfuscation | `encode` | `encoding` | 70 | 27 | no | no | `to_base64`, `to_rot13`, `to_rot18`, `to_rot47`, `to_unicode_escape`, `to_ascii_ordinal`, `to_separator`, `to_leetspeak_basic`, `to_leetspeak_intermediate`, `to_leetspeak_advanced`, `to_morse`, `to_braille`, `to_reversed_characters`, `to_reversed_words`, `to_reversed_word_order`, `to_lookalike_letters`, `to_zero_width_noise`, `to_hex`, `to_binary`, `to_url_encoding`, `to_html_entities`, `to_base32`, `to_base85`, `to_caesar_cipher`, `to_atbash`, `to_fullwidth`, `to_math_styled` |
| obfuscation | `neutral_container` | `container` | 80 | 6 | no | no | `to_code_comment`, `to_yaml_document`, `to_csv_row`, `to_email_message`, `to_transcript_line`, `to_log_entry` |
| obfuscation | `structural` | `container` | 80 | 3 | no | no | `to_json`, `to_xml`, `to_markdown` |
| obfuscation | `ascii_art` | `encoding` | 90 | 1 | no | no | `to_ascii_art` |
| requests | `answer_format` | `answer_shape` |  | 12 | no | no | `to_refusal_then_comply`, `to_apology_prevention`, `to_prefix_injection`, `to_answer_in_markdown`, `to_answer_as_tutorial`, `to_answer_as_code`, `to_answer_as_dialogue`, `to_answer_as_story`, `to_answer_as_structured_data`, `to_dual_answer`, `to_tagged_prefix`, `to_answer_in_encoding` |
| requests | `answer_language` | `language` |  | 26 | no | no | `to_ask_answer_in_french`, `to_ask_answer_in_japanese`, `to_ask_answer_in_russian`, `to_ask_answer_in_spanish`, `to_ask_answer_in_german`, `to_ask_answer_in_arabic`, `to_ask_answer_in_turkish`, `to_ask_answer_in_czech`, `to_ask_answer_in_vietnamese`, `to_ask_answer_in_greek`, `to_ask_answer_in_croatian`, `to_ask_answer_in_swahili`, `to_ask_answer_in_thai`, `to_ask_answer_in_khmer`, `to_ask_answer_in_maori`, `to_ask_answer_in_nepali`, `to_ask_answer_in_zulu`, `to_ask_answer_in_scots_gaelic`, `to_ask_answer_in_bengali`, `to_ask_answer_in_javanese`, `to_ask_answer_in_hmong`, `to_ask_answer_in_guarani`, `to_ask_answer_in_chinese`, `to_ask_answer_in_hindi`, `to_ask_answer_in_yoruba`, `to_ask_answer_in_amharic` |
| requests | `asking` | `request_shape` |  | 3 | no | no | `to_innocuous_question`, `to_ask_for_details`, `to_ask_with_urgency` |
| requests | `benign_role` | `role_framing` |  | 3 | no | no | `to_benign_role_subject_helper`, `to_benign_role_tutor`, `to_benign_role_task_assistant` |
| requests | `continuation` | `request_shape` |  | 4 | no | no | `to_continue_passage`, `to_complete_story`, `to_fill_in_blank`, `to_continue_instructions` |
| requests | `distractor_prefix` | `request_shape` |  | 3 | no | no | `to_distractor_prefix_related`, `to_distractor_prefix_unrelated`, `to_distractor_prefix_task` |
| requests | `distractor_suffix` | `request_shape` |  | 3 | no | no | `to_distractor_suffix_related`, `to_distractor_suffix_unrelated`, `to_distractor_suffix_task` |
| requests | `impersonation` | `role_framing` |  | 3 | no | no | `to_impersonate_good_person`, `to_impersonate_authority`, `to_impersonate_developer` |
| requests | `indirect` | `request_shape` |  | 6 | no | no | `to_summarize`, `to_proofread`, `to_critique`, `to_translate_task`, `to_explain_to_child`, `to_rewrite` |
| requests | `smalltalk_prefix` | `request_shape` |  | 1 | no | no | `to_smalltalk_prefix` |
| requests | `smalltalk_suffix` | `request_shape` |  | 1 | no | no | `to_smalltalk_suffix` |
| requests | `temporal` | `request_shape` |  | 3 | no | no | `to_use_past_tense`, `to_use_future_tense`, `to_use_historical_era` |
| manipulation | `dap` | `example_priming` |  | 6 | `to_dap_selected_short`, `to_dap_selected_long`, `to_dap_selected_many` | yes | `to_dap_random_short`, `to_dap_random_long`, `to_dap_selected_short`, `to_dap_selected_long`, `to_dap_random_many`, `to_dap_selected_many` |
| manipulation | `fake_turns` | `example_priming` |  | 3 | no | no | `to_fake_system_turn`, `to_fake_assistant_turn`, `to_fake_history` |
| manipulation | `fsh` | `example_priming` |  | 6 | `to_fsh_selected_short`, `to_fsh_selected_long`, `to_fsh_selected_many` | yes | `to_fsh_random_short`, `to_fsh_random_long`, `to_fsh_selected_short`, `to_fsh_selected_long`, `to_fsh_random_many`, `to_fsh_selected_many` |
<!-- END families -->

Each family's dimension is a tag of `configs/dimensions.json`: what the
family varies in a finished text. A run stops before any stage when a
generation axis varies the same dimension as a family it draws
([configuration.md](configuration.md#dimensions-and-the-runs-check)).

A technique takes its layer and its order from the first family of its
spec entry. The ten `to_sensitive_words_encode_*` techniques also
belong to `encode`, so one of them blocks both families.

`encode` techniques carry an `encode_weight`: `light` for `to_rot13`,
`to_rot18`, `to_rot47`, `to_separator`, the three `to_leetspeak_*`, the
three reversals, `to_lookalike_letters`, `to_zero_width_noise`,
`to_url_encoding`, `to_caesar_cipher`, `to_atbash`, `to_fullwidth` and
`to_math_styled`; `heavy` for `to_base64`, `to_unicode_escape`,
`to_ascii_ordinal`, `to_hex`, `to_binary`, `to_html_entities`, `to_base32`,
`to_base85`, `to_morse`, `to_braille` (each a code for every character, of
complexity 3). The ten `to_sensitive_words_encode_*`
carry one too: `heavy` for base64, unicode and ascii, `light` for the rest. Of
these ten, the `heavy` ones stay allowed with `fsh` and `dap`, which the plain
`heavy` `encode` techniques are not; the `light` ones, like the plain `light`
ones, may be the request obfuscation, which no `heavy` one is. A budget round
deals the ten plain `heavy` ones, the full-text encodings, in a slot of their
own beside `encode`, at the spec's `heavy_encoding_weight` (0.2) times
`encode`'s weight, about a fifth of a family's draws: dealt as `encode`'s
members they were seldom drawn in the default attacks, since one fits only a
unit with 3 points of its budget left ([internals/jailbreak.md](internals/jailbreak.md#balanced-dealing)).
The slot is a quota in every budget round, not a floor: about 40 draws in
3,000 units whatever the round's budget, so round 3 of the escalation rises
from 6 to 42 and round 4, whose budget of 9 dealt them about 121 times, falls
to about 45. A count round deals them as `encode`'s members.

Seventeen of them are tagged `needs_latin`: the ciphers `to_rot13`,
`to_rot18`, `to_rot47`, `to_caesar_cipher`, `to_atbash`, the three
`to_leetspeak_*` and their six `to_sensitive_words_encode_*` forms, and the
Latin-only transforms `to_lookalike_letters`, `to_fullwidth` and
`to_math_styled`. They change only ASCII characters, the letters (ROT18
also the digits, ROT47 every printable ASCII character, fullwidth also the
digits), so they leave a text in another script as it was when it holds none
of those, as a test checks on six samples of other scripts; ROT47 changes the
ASCII punctuation or digits of such a text, and in a Latin-script text with
accents (Vietnamese, Yoruba, Czech) the accented letters stay in clear under
the letter ciphers.
`to_morse` and `to_braille` pass such a text through as well (Morse only
spaces its characters out) but are not tagged: both are in the eval pool,
which is [pinned](#the-eval-pool), and completing them would change its text.
`to_reversed_*`, `to_zero_width_noise` and the byte codecs (`to_hex`,
`to_binary`, `to_url_encoding`, `to_html_entities`, `to_base32`,
`to_base85`) are not Latin-only: they change a text in any script.
`to_lookalike_letters`, `to_zero_width_noise`, `to_fullwidth` and
`to_math_styled` write non-ASCII characters, so each is kept from `ascii_art`
(`to_zero_width_noise` also from `tokenbreak`).

Nine `noise` techniques are tagged `needs_latin` too: the three densities of
`to_random_capitals_*` (ASCII small letters only), `to_scrambled_letters_*`
(words of Latin letters only) and `to_keyboard_typos_*` (the letters of the
QWERTY map only), checked on the same six scripts. The three
`to_emoji_sprinkle_*` change a text in any script; their emoji have no glyph
in any font, so each is kept from `ascii_art`. The `noise` family is kept
from `typos` and `casual_writing`: one kind of surface noise per text. It is
kept from `translation` too, whatever the script.

No rule reads the `needs_latin` tag. A translation into another script would
leave such a technique nothing to change, and the family rules keep the two
apart already: a translation refuses the whole `encode` family and `noise`.
A test asserts that every tagged technique is refused beside every
translation into another script, so a Latin-only technique added to a family
a translation does not refuse fails it until the spec keeps the two apart.

## The language table

`configs/jailbreak/languages.json` holds one row per language: its `name`
(as prompts and infos write it), `slug` (the end of its technique names),
`complexity` (by resource tier: 0 high, 1 mid, 2 low), `script` (ISO 15924,
the script a translation writes, which the test of `needs_latin` reads), and
whether it has a `translation` technique (`to_<slug>`) and an
`answer_language` one (`to_ask_answer_in_<slug>`). The
spec's `generators` make a technique entry of each marked row as the spec
loads, and the two families' modules build the functions from the same
rows, in the table's order. Of the twenty-six rows, all have an
answer-language technique and twenty-three a translation: not Maori, nor
Guarani and Yoruba, two of the six rows spec version 1.2 added (Hmong,
Guarani, Chinese, Hindi, Yoruba, Amharic). Spec version 1.3 switched on the
translations of the other four (Hmong, Chinese, Hindi, Amharic) after a
measurement with the translation model on a GPU (2026-10-22: TranslateGemma
27B on the FLORES-200 devtest, 1,012 sentences, and on REDACT's inputs, with
the whole machine-translation path): 96% to 100% of the units accepted,
GlotLID reading 99.8% to 99.9% of the translations as the language, and
runaway replies (1,500 tokens or more) in none of Chinese's and Hindi's 1,012
sentences, 3 of Amharic's and 4 of Hmong's. Guarani stays an
answer language, since TranslateGemma loops on a fifth of its sentences and
only 40% of the units pass; Yoruba too, since the judge rejects 41% of its
calls and 93% of the units pass, below the 97% bar. A row's
`translation_since` (`"1.3"` for these four) dates its translation apart from
its `since` (`"1.2"`, the answer language), so a plan kept from spec 1.2
resumes with the growth WARNING. Chinese is scored high (0, as Japanese),
Hindi mid (1, as Turkish and Vietnamese), and Hmong, Guarani, Yoruba and
Amharic low (2, as Zulu and Swahili). Two of the four are on watch, as Scots
Gaelic is: the judge rejects 20% of Hmong's calls (a back-translation
chrF++ of 45.1, 9 to 12 points below the four shipped languages measured the
same way) and Amharic's chrF++ on the devtest is 9 points below NLLB-200's.
Hmong's acceptance, 230 of 237 units (97.05%, a Wilson 95% interval of 94.0%
to 98.6%), passes the 97% bar on the point estimate and not with certainty. Hmong is sent to the translation
model by its name, never a code (`hmn` and `mww` gave Hawaiian and Marathi);
Chinese is accepted as `cmn_Hani` or `wuu_Hani`. The translation family
kept the order it had, the four joining at their rows; the answer languages took the table's, which puts
Swahili before Thai, so a fresh plan that drew one of the two draws the
other (a kept plan replays by name). The table is read from the package
only. To add a language: [extending.md](extending.md#add-a-language).

## Retired techniques

Spec version 1.1 retired seven techniques. They stay registered, so a plan
that names them replays them, and a kind draws one only when its
`techniques.allow` names it (with a WARNING in the log and a
`DeprecationWarning`):

| Family | Technique | Why |
|---|---|---|
| `sensitive_words` | `to_sensitive_words_star`, `to_sensitive_words_hyphen`, `to_sensitive_words_underscore` | `to_sensitive_words_encode_separator` draws their separators (`*`, `-`, `_`) among its own; `to_sensitive_words_split` stays, since a space is not one of them |
| `persona` | `to_politician_persona`, `to_deformed_professor_persona`, `to_religious_figure_persona` | near-duplicates of `to_radical_politician_persona`, `to_deformed_scientist_persona` and `to_cult_leader_persona`, which stay |
| `persona` | `to_mentally_ill_persona` | retired, with no replacement |

## Compatibility rules

`jailbreak/rules.py: compatible` applies the spec's rules both ways, whichever
technique is picked first. A technique may not join a combination when:

- its family is taken already;
- its layer is at its cap (`layer_max_picks`: one `rewrite`, one
  `hacking`, one `manipulation`, one `requests`);
- one of its families is in a taken family's `cross_incompatible_with`, or
  the reverse. The spec pairs `structural` with `ascii_art`; `translation`
  with `ascii_art`, with the four families that have a model mark words,
  `sensitive_words`, `tokenbreak`, `word_art` and `wording` (the marking call finds a
  word in 62 of 100 harmful English inputs but in 3 to 36 of their
  translations, measured on nine languages and none of the European ones, so
  a heavy encoding after a translation is discarded and a light one changes
  nothing), with `typos`, `casual_writing`, `noise` and
  `payload_split` (a translated prompt is already hard to read, for the model
  that writes or marks text and for the model that answers, and one more
  word-level or character-level change on top is too complex; `structural`
  stays allowed beside a translation, the eval type draws them together), and
  with `encode`, every member, light or heavy, into any script (both rewrite
  the whole text, so one is enough, and a cipher before a translation would
  hand the translator a cipher; the ten `to_sensitive_words_encode_*` were
  kept from a translation by their other family already; the
  [eval release](configuration.md#release-notes) removed from the eval pool
  the 40 pairs of its eight translations with its five encodings);
  `ascii_art` with `framing`, `persona`, `cognitive`, `fsh`,
  `dap`; `fsh` and `dap` with `continuation`, `indirect`, `framing`,
  `persona`, `cognitive`; `persona_template` with `fsh`, `dap` and
  `ascii_art`; `payload_split` with `ascii_art`, `encode`, `fake_turns` and
  `word_art` (a font or a cipher would break the parts apart, and an
  invented turn is a conversation already); `word_art` with `ascii_art`,
  `payload_split`, `encode`, `sensitive_words`, `tokenbreak`, `structural`,
  `neutral_container` and `fake_turns` (one disguise of the word, and the art
  is long); `wording` with `sensitive_words`, `tokenbreak`, `word_art`,
  `translation` and `ascii_art` (the other steps that put the marked words in
  other terms, a translation the marking call fails on, and a font on a text
  the key or the puzzles lengthen); `sensitive_words` with `tokenbreak` (each
  has the model mark the text's words, so a second marking call would read a
  text the first changed: one marking step per text, as for `word_art` and
  `wording`; the eval pool holds neither); `persona`, `cognitive`, `persona_template` and `persuasion` with
  `impersonation`, so a prompt never carries two identities; `persuasion`
  also with `ascii_art`, `tense_rewrite` with `temporal` (the request would be
  set in time twice) and `model_distractors` with `ascii_art`,
  `distractor_prefix` and `distractor_suffix`; `fake_turns`
  with the list of `fsh` and `dap`, `persona_template`, `structural` and
  `neutral_container` (an invented turn is a conversation, not a document
  to wrap); `noise`
  with `typos` and `casual_writing`, one kind of surface noise per text; and
  `neutral_container` with `structural` (one document around the prompt)
  and `ascii_art` (no font draws a document of that length). So a
  combination holds a hacking or a manipulation technique, never both
  (`continuation` and `indirect`, both requests, are kept apart by the
  layer cap);
- one of its families is in the `cross_incompatible_with` of a taken
  technique's own spec entry, or a taken family is in its own. The spec keeps
  `to_adversarial_suffix_emoji`, `to_adversarial_suffix_unicode`,
  `to_braille`, `to_tokenbreak_split`, `to_lookalike_letters`,
  `to_zero_width_noise`, `to_fullwidth`, `to_math_styled` and the three
  `to_emoji_sprinkle_*` from `ascii_art`, since no font draws the characters
  they may leave. It keeps the ten `heavy` full-text encodings from it too,
  `to_braille` and the nine others (`to_base64`, `to_unicode_escape`,
  `to_ascii_ordinal`, `to_morse`, `to_hex`, `to_binary`, `to_html_entities`,
  `to_base32`, `to_base85`): a code for every character lengthens the text,
  so its drawing often runs past the cap (below), and both disguise the whole
  text, so one is enough. The three `heavy` `to_sensitive_words_encode_*`
  encode only the words a model marks and may still join it, as the `light`
  encodings that write ASCII and the other suffixes and noise do.
  It keeps the four many-shot variants of `fsh` and `dap` from `typos`, a
  rule written when the typo rewrite read the whole many-shot text, which a
  model would not return whole; the manipulation layer now applies last, so
  the rewrite reads the question alone, and the rule is kept as it was. It keeps
  `to_refusal_then_comply`, `to_prefix_injection`, `to_dual_answer` and
  `to_tagged_prefix` from `persona_template`, whose mechanics already ask
  for a dual answer or a set opening, and `to_answer_in_encoding` from
  `encode`: an answer asked for in an encoding is not put on an encoded
  prompt. The ten `to_sensitive_words_encode_*` are `encode` members, so
  they are kept from it too; the sensitive-word steps that encode no word
  are not. `to_ascii_art` itself rejects a text with a character no
  single font draws (most non-ASCII characters: accented letters such as `é`
  or `ñ`, dashes, curly quotes, kana; German umlauts and `ß` and some Greek
  or Cyrillic text have fonts), or one every font draws longer than its cap;
- it pairs a family of the manipulation layer (`fsh`, `dap`, `fake_turns`)
  with an obfuscation in `manipulation_blocked_obfuscation`: the
  `translation`, `ascii_art`, `payload_split`, `word_art` and `wording` families and
  the `heavy` `encode` techniques, except those also in `sensitive_words`. The
  block was written when the manipulation layer applied before the
  obfuscations, which then changed the examples too; since the eval release
  the layer applies last and an obfuscation changes the question alone, and
  the block is kept as it was (lifting it would admit 104 eval pairs, each
  `fsh` or `dap` technique beside a translation or a heavy encoding, a change
  of the eval coverage of its own);
- it is a suffix, a structural wrapper or a neutral container and the
  combination's request obfuscation is a translation (budget mode with
  `allow_request_obfuscation`). The request obfuscation applies right after
  the request, to the finished request (a manipulation technique, the only
  step after it, never meets a translation), so the translation would
  translate the suffix, the keys
  of the wrapper or the lines of the container. The spec's
  `request_obfuscation_refuses` names these three families for a translation;
  the sampler draws the request obfuscation before the obfuscations and keeps
  them out of the pool the obfuscations draw from
  (`rules.after_request_obfuscation`, not part of `compatible`, which does not
  know which technique of a combination is the request obfuscation). A
  translation drawn as an ordinary obfuscation runs before those steps and
  joins them, and an encoding is kept from any translation by the family
  rule above;
- in budget mode, its complexity is above what is left of `max_complexity`
  (under the opt-in [group budget](configuration.md#the-group-budget), of the
  unit's allowance: `max_complexity` plus what the group's earlier units left
  unspent, up to `max_unit_complexity`).

`combine_techniques` does not check these rules. It only sorts techniques
into layer order. [internals/jailbreak.md](internals/jailbreak.md) explains
how a combination is drawn.

Adding a technique: [extending.md](extending.md#add-a-jailbreak-technique).

## The eval pool

The `eval` dataset type's attacks draw from 36 techniques in 10 families:
hard ones (complexity 2 or 3 but for `to_noble_goal`, and the low-resource
languages) whose harmful content stays certain, since the technique carries
the text unchanged or reversibly, or a check accepts the result.

The pool is **pinned**: its 36 names, its two count rounds and the text each
technique draws for a seed stay the same from release to release, so the
results of an eval run stay comparable. `tests/fixtures/eval_pool.json` pins
the names, `tests/fixtures/eval_pairs.json` the pairs of them the
[rules](#compatibility-rules) admit, and a digest pins the draws; a change to
the pool, to those pairs, or to the text or info of one of its techniques, is
made on purpose in a review of the eval type (a new spec version, where a
technique is added or retired, and new fixtures) and fails the tests until
then. This
page and the others say *pinned*, or *unchanged*, where it matters; *the next
eval review* is that deliberate change.

| Family | Techniques |
|---|---|
| `translation` | the eight low-resource languages, weighted 3 |
| `encode` | the five `heavy` ones: `to_base64`, `to_unicode_escape`, `to_ascii_ordinal`, `to_morse`, `to_braille` |
| `fsh`, `dap` | the eight `short` and `long` ones |
| `framing` | all five |
| `structural` | `to_json`, `to_xml`, `to_markdown` |
| `ascii_art` | `to_ascii_art` |
| `continuation` | all four |
| `temporal` | `to_use_past_tense` |
| `answer_format` | `to_refusal_then_comply` |

Model rewrites (`cognitive`, `persona`, `typos`, `sensitive_words`,
`tokenbreak`) are left out while the eval pool is pinned; a technique of
this pool that has a model write the prompt must carry a spec `check`. The fifteen spec 1.2 encodings
are not drawn here either: `eval.json` denies each in `techniques.deny`, so the
`encode` row keeps its five `heavy` members while the eval pool is pinned. The
four spec 1.2 many-shot variants of `fsh` and `dap` are denied the same way.
`temporal` is not allowed as a family: `eval.json` names `to_use_past_tense`
under `techniques.allow`, so neither spec 1.3 member of the family
(`to_use_future_tense`, `to_use_historical_era`) nor a later one reaches the
pool. With
`"adversarial": {"translation": false}` the pool keeps the other 28.
[configuration.md](configuration.md#dataset-types) gives the type file.

`to_ascii_art` keeps its 5,000-character cap in this pool too, so its eval
attacks differ from those of a version without the cap: the font a seed
draws changes for some texts from about 40 characters and for most from
about 90, a text of about 830 characters or more is rejected where it used
to be drawn, and so were most inputs when the second round paired it with a
`heavy` encoding that lengthens the text (`to_unicode_escape`,
`to_ascii_ordinal`, `to_morse`); the [eval release](configuration.md#release-notes)
keeps ASCII art from those encodings, so a fresh eval plan no longer draws
the pairs. An eval plan kept across the change mixes the two renderings. [What the families do](#what-the-families-do) has the
figures.

The eval release also put the manipulation layer last. The 40 pairs that
hold one of the eight `fsh` and `dap` techniques, each beside `to_json`,
`to_xml`, `to_markdown`, `to_refusal_then_comply` or `to_use_past_tense`,
now wrap or reframe the question alone and insert it among the examples,
where the wrapper or the request used to go around the whole list; no pair
is added or taken out, and the other 251 pairs and the 36 single techniques
write what they wrote. An eval plan kept from before writes the old text for
its pending units (`tests/jailbreak/test_eval_pairs.py` names the 40).

## The natural pool

A spec entry's `kinds` lists the kinds of the augmentation section that may
draw the technique: `["adversarial"]` when neither the technique nor its
first family gives one, and a technique's own `kinds` replace its family's.
Every technique is adversarial but those of the three request families
spec version 1.2 marks `["natural"]` alone: `smalltalk_prefix`,
`smalltalk_suffix` and `benign_role`, which no attack draws (an adversarial
allow list that names one raises). The families `typos`, `casual_writing`,
`noise`, `neutral_container`, `translation`, `answer_language`,
`answer_format`, `asking` and `indirect` are marked
`["adversarial", "natural"]`, less their attack-only
techniques, and three techniques of attack families are marked natural on
their own. The natural kind draws only what is marked natural: by default
(translation off) these 72 techniques in 13 families, pinned in
`tests/fixtures/natural_pool.json`:

| Family | Natural techniques |
|---|---|
| `typos` | `to_rewrite_with_typos_low`, `to_rewrite_with_typos_medium` |
| `casual_writing` | all nine |
| `noise` | the low and medium densities of `to_random_capitals_*`, `to_scrambled_letters_*`, `to_keyboard_typos_*` and `to_emoji_sprinkle_*` (eight, since spec 1.2) |
| `neutral_container` | all six (since spec 1.2) |
| `answer_language` | all twenty-six (six since spec 1.2) |
| `answer_format` | `to_answer_in_markdown`, `to_answer_as_tutorial`, `to_answer_as_code`, `to_answer_as_structured_data` |
| `asking` | all three (`to_ask_with_urgency` since spec 1.3) |
| `indirect` | all six |
| `smalltalk_prefix`, `smalltalk_suffix` | `to_smalltalk_prefix`, `to_smalltalk_suffix` (since spec 1.2, natural only) |
| `benign_role` | all three, for benign and dual-use benign inputs only (since spec 1.2, natural only) |
| `encode` | `to_leetspeak_basic` |
| `suffixes` | `to_adversarial_suffix_punctuation`, `to_adversarial_suffix_emoji` |

With `"natural": {"translation": true}` the 23 translations join them (95);
the kind draws one technique per input unless `exact_techniques` says more,
and with two or more a translation is kept from `typos`, `casual_writing` and
`noise` by the rules above.
Attack-only: `structural`, `continuation`, `temporal`, the high and insane
typos, the four high-density noise techniques, `to_answer_as_dialogue`, `to_answer_as_story`,
the three answer formats of spec version 1.3 (`to_dual_answer`,
`to_tagged_prefix`, `to_answer_in_encoding`) and the three
answer-format directives the natural kind denies anyway
(`to_refusal_then_comply`, `to_apology_prevention`,
`to_prefix_injection`), with every other family. A natural allow list that
names one of them raises, so a `config_dir` copy of `augmentations.json`
from before needs its natural `families.allow` trimmed. A natural plan made
before the natural kind was narrowed to this pool raises on resume, since its
pool changed: run the natural kind with `resume=False`. One made since, before
spec version 1.2 grew the pool, resumes with the growth WARNING
([configuration.md](configuration.md#release-notes)). A training build leaves
out a benign or dual-use benign variation made with a technique the natural
kind does not draw, with one WARNING that counts them
([stages/build.md](stages/build.md)).

## What a plan costs

A technique that calls a model carries, in the spec, the model calls one
application makes (`calls`, generation and checks, low to high: a typo
rewrite 2 to 4, a translation of one paragraph 2 to 12, a word-marking
step 1, the synonyms 1 to 11, a wording technique 1 to 21, a semantic payload
split 1 to 3, an fsh or dap pick 1, a cognitive or persona scenario 3 to 9, an
invented persona 4 to 12, a persuasion or tense rewrite 2 to 4, a model
distractor 1 to 3) and what checks its
reply (`check`). When a plan is made, one INFO line adds them up over the
planned units; for 1,000 inputs of one category and entry type, planned
with the packaged attack kind and seed 42:

```text
Model calls: 855 of 1000 planned units call a model, 1997 to 7669 calls (translation 480 to 2880, from the spec's calls).
```

Other seeds (43 to 46) give 848 to 853 units, 1,986 to 1,994 calls at the
low end, 7,610 to 7,732 at the high end and a translation share of 478 to 480
and 2,868 to 2,880.

The translation share goes to the translation model (a machine
translation's judge, one call a round, to the generation model).

## What the families do

A transform returns `(text, info)`; the info names what it drew, such as a
template variant or a font, for traceability.

**Rewrite.** Off by default, adversarial only, one technique at most per
combination, applied before every other layer. All of them call the
generation model.

- `persuasion` (12, `to_persuade_<key>`, one per entry of
  `configs/jailbreak/rewrite/persuasion_techniques.json`) has a model rewrite
  the whole prompt as an appeal of that persuasion technique, given its name,
  definition and an example (`jailbreak/persuasion_rewrite`); the info is
  `category=<key>`.
- `tense_rewrite` (`to_rewrite_in_past_tense`, `to_rewrite_in_future_tense`)
  has a model rewrite the whole prompt in that tense, changing the verbs and
  what the grammar requires (`jailbreak/tense_rewrite`); the info is
  `tense=<tense>`. The pure `to_use_past_tense` and `to_use_future_tense`
  wrap the request in a sentence instead.
- Both are checked: the stage's `check` model (the `gen` model unless the
  recipe names one) reads the rewrite against the prompt, for the same
  request (`jailbreak/preserve_check`, at temperature 0.15). A rejected
  rewrite is made again with the check's reply as feedback, up to two
  rewrites, so 2 to 4 calls (the spec's `calls`); one rejected twice is
  discarded (`reason=preserve check failed`). A rewrite the check accepts
  but that has no text, or is the prompt itself, is discarded too
  (`reason=empty rewrite`, `reason=unchanged rewrite`).
- `model_distractors` (`to_model_distractor_prefix`,
  `to_model_distractor_suffix`) has a model write a few sentences in the
  voice of the person asking (`jailbreak/model_distractor`), placed before or
  after the prompt, which is not rewritten; they are joined to it with a
  space. No check model reads them. A reply with no text, or that is the
  prompt itself, is asked for again, up to 3 replies in all, and then
  rejects the result
  (`DISCARDED; reason=empty distractor`). The info is `position=<position>`.

**Hacking.**

- `cognitive` and `persona` are a scenario, a check of it, then the
  jailbreak built on it: 3 to 9 model calls (the spec's `calls`). A named persona grounds the scenario in the
  persona's character first; the prompt names it by its `display_name`
  (`Someone from the Future`), the info by its key
  (`category=someone_from_the_future_persona`). Spec version 1.3 added
  three named personas, `mercenary`, `underground_hacker` and
  `anarchist_philosopher` (attack only, like the rest of the family). The
  scenario prompts ask
  for the scene at the bottom, behind `---` delimiters: the scene is the
  last block after the first delimiter, or, in a reply without one, what
  follows a `Scenario Description` heading. A reply with neither is asked
  again, up to 3 times in all, and then rejects the result
  (`DISCARDED; reason=malformed scenario`). The stage's `check` model (the
  `gen` model unless the recipe names one) then reads the scene against the
  prompt (`jailbreak/scenario_check`, at temperature 0.15). A scene it
  rejects is asked for again with the check's reply as feedback, up to two
  scenes; one rejected twice rejects the result (`DISCARDED; reason=scenario
  check failed`). The definitions live in
  `configs/jailbreak/hacking/cognitive_techniques.json` and `personas.json`,
  editable without touching code. `to_invented_persona` first asks the model
  for a persona for the text (the `persona_invention` prompt): a name and a
  description, as a `Name:` line and then a `Description:`, and nothing else
  but blank lines (markdown labels, a missing or empty field and two
  personas are other forms). A reply in another form is asked for again, up
  to three replies in all, and then rejects the result (`DISCARDED;
  reason=malformed persona`). The scenario, its check and the construction
  then go as for a named persona, with the invented name and description:
  4 to 12 calls, and the info `category=invented_persona`.
- `framing` directives are transforms that change the scenario or the
  intent. Each has named template variants in
  `configs/jailbreak/hacking/framing_templates.json`; one is drawn per call.
- `persona_template` (spec version 1.3) sets the prompt in a fixed frame
  for one of eleven well-known jailbreak personas, a technique each
  (`to_dan_persona_template` to `to_liberated_model_persona_template`), with
  no model call. Each frame holds a place for a *mechanic*: one of six
  pieces of text, drawn by `rng`, that sets the rules of the persona's
  answers, named in the info (`category=dan_persona_template;variant=<mechanic>`)
  and counted in the kind's variants table. The mechanic names the persona
  by the frame's `display_name`. The frames and mechanics are in
  `configs/jailbreak/hacking/persona_templates.json`; a frame is its
  technique's own text, so the variant floor counts the six mechanics
  alone. The family is attack-only, and the eval type does not draw it.

**Manipulation.** `fsh` and `dap` build on benign question-and-answer pairs
of the subcategories of the benign taxonomy. The `random` variants pick a
subcategory at random, the `selected` ones ask the model to choose one
(prompt `jailbreak/category_selection`); `short` and `long` are the length of
the answers, three to seven pairs of that subcategory.

- `fsh` (few-shot hacking): the benign pairs prime a helpful pattern, and the
  prompt comes last, as one more question.
- `dap` (distract and persuade): the prompt is hidden among the benign pairs,
  at a random position. With no pair at all, the prompt is the only
  question.

The manipulation layer applies last, so the prompt they insert is the
finished question: obfuscated, its request around it, its request
obfuscation applied; the pairs stay as written. A `selected` variant shows
its model the clean input instead, the text the unit started from, which
`run_chain` passes the techniques that take it as `source_prompt`: it picks
the subcategory on what the question is about, not on its cipher or its
wrapper, then inserts the finished question. In a plan kept from before the
eval release, where `fsh` or `dap` ran first, the clean input is the text
the variant read, so it picks as it did; the exception is a rewrite
technique (in a kind that allows the rewrite layer), which runs first in
both orders: the variant is now shown the input before the rewrite, where
a kept plan's was shown the rewritten text.

The `many` variants (spec 1.2: `to_fsh_random_many`,
`to_fsh_selected_many`, `to_dap_random_many`, `to_dap_selected_many`) show
`MANY_PAIRS`, 32, pairs with short answers. The cache holds eight of a kind
per subcategory, so they draw from a pool across it
(`manipulation/pairs.py: _many_pairs`): `random_many` from every short pair
(its info says `sub_category=all`), `selected_many` the chosen
subcategory's pairs first, then those of the other subcategories of its
main category, then the rest, each group in an order drawn from the unit's
rng, each question once. The prompt then shows them in a drawn order, as the
siblings do. A cache of fewer than 32 short pairs gives them all, and the
info says so (`num_shots=<n>;wanted=32`); with one short pair the DAP
question follows it (`harmful_idx=2`: never first, and last only where it
cannot be otherwise). A cache with no short pair, every short unit having
been rejected, raises a `ValueError` that names the cache, for the unit's
error row, instead of writing a prompt with no shot. Why 32: a short pair of the cache
is about 150 characters as the prompt writes it, about 40 tokens at four
characters a token, so 32 are about 1,200 tokens (1,450 at the cache's 90th
percentile, 2,800 were every pair as long as the longest), well inside the
generation model's 10,000-token context; since the layer applies last, no
later model step reads them. They refuse `typos`, a rule written when a typo
rewrite read the pairs too (a model rewriting so long a prompt would not
return it whole) and kept as it was, and otherwise share
the rules of `fsh` and `dap`; their complexity is 3, the top of the scale:
one above their `random` siblings and level with their `selected` ones.

`fake_turns` (spec version 1.3) puts an invented turn of the conversation
before the prompt (the finished question, as for `fsh` and `dap`), with no
model call and no benign data:
`to_fake_system_turn` an invented system turn, `to_fake_assistant_turn` an
invented assistant turn, `to_fake_history` an invented earlier exchange,
eight variants each, from
`configs/jailbreak/manipulation/fake_turns_templates.json`; the info names
the variant (`variant=<name>`). Being of the manipulation layer, they take
`fsh` and `dap`'s place in a combination (one manipulation at most) and
their obfuscation block, and refuse the same hacking and request families,
the persona templates, and the `structural` and `neutral_container`
wrappers. The family is attack-only, and the eval type does not draw it.

**Obfuscation.**

- `sensitive_words` and `tokenbreak` first ask the model for the harmful
  words of the prompt (prompt `jailbreak/extract_harmful`), one call per
  technique. The encoding, splitting or substitution then applies to those
  words; a combination holds at most one of the two. A reply that is not a
  list of at most ten of the prompt's words
  rejects the result (`DISCARDED; reason=malformed word list`). A word is a
  letter, then letters, digits, apostrophes, hyphens, combining marks and
  zero-width format characters (none first), so a word in Bengali, Nepali,
  Thai or Khmer, whose vowel signs and viramas are marks, or in Persian or
  Bengali with a zero-width joiner or non-joiner, is a word. A word of one
  letter is refused unless it is a Chinese character, since the step would
  change that letter wherever it stands, inside other words too; a word
  named twice, in any case, counts once (the first spelling); a reply may
  put straight or curly quotes, guillemets or an ellipsis around its words.
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
- `translation` covers 23 languages, the rows of
  [the language table](#the-language-table) marked `translation` (not Maori,
  which TranslateGemma renders
  with invented sensitive and technical terms, fluently enough to pass every
  check, nor Guarani and Yoruba, which measured below the acceptance bar) in
  three resource tiers, which the table scores by complexity:
  high (`to_french`, `to_japanese`, `to_russian`, `to_spanish`, `to_german`,
  `to_arabic`, `to_chinese`; 0), mid (`to_turkish`,
  `to_czech`, `to_vietnamese`, `to_greek`, `to_croatian`, `to_hindi`; 1) and low
  (`to_swahili`, `to_thai`, `to_khmer`, `to_nepali`, `to_zulu`,
  `to_scots_gaelic`, `to_bengali`, `to_javanese`, `to_hmong`, `to_amharic`; 2),
  the last two (spec 1.3) on watch like Scots Gaelic ([the language
  table](#the-language-table)). On a chat translator
  each is a translate, check and retry loop on the `translation` model. On
  a machine-translation model (`translategemma-27b`, the default) the text
  goes paragraph by paragraph; each piece is checked for its language
  (GlotLID; an undetermined `und_*` label is no verdict) and length
  (against the language's median ratio, measured with TranslateGemma on
  FLORES-200; not for a piece under 20 characters, which a label or a
  digit makes longer than a sentence's ratio), then translated back and
  judged by
  the generation
  model ([stages/jailbreaks.md](stages/jailbreaks.md#models)). A fresh plan
  never follows a translation with a step that marks words or changes the
  words or characters of its text (`sensitive_words`, `wording`,
  `tokenbreak`, `word_art`, `typos`, `casual_writing`, `noise`,
  `payload_split`; [the rules](#compatibility-rules)).
- `typos` is a rewrite by the model, at one of four densities, with the
  level's description in the prompt, then a check: the stage's `check`
  model (the `gen` model unless the recipe names one) reads the rewrite
  against the prompt (`jailbreak/typo_check`, at temperature 0.15). A
  rejected rewrite is made again with the check's reply as feedback, up to
  two rewrites, so 2 to 4 calls (the spec's `calls`); one rejected twice is
  discarded (`reason=typo check failed`). It is not drawn with a
  translation.
- `casual_writing` is how a person typing quickly writes: casing, spacing,
  punctuation, abbreviations, accents. It is not drawn with a translation.
  `to_abbreviations` rewrites only the
  common words it lists, and `to_stripped_accents` only accented Latin
  letters (other scripts keep their marks), so either would leave many
  prompts unchanged. Each has an applicability: a fresh plan draws it only
  for an input it changes, read off the input before any other step of
  the combination. The other members may still leave a prompt unchanged
  (`to_lowercase` on a text in lower case already): a no-op, which the
  natural kind rejects.
- `noise` is the slips of a person typing, each at three densities (spec
  version 1.2), all pure and drawn by `rng`. A density is the share of the
  letters or words the technique may change that it changes, at least one:
  random capitals write the ASCII small letters as capitals (low 0.05,
  medium 0.15, high 0.4 of them); scrambled letters shuffle the inner
  letters of a word of Latin letters, the first and last kept (0.1, 0.25,
  0.5 of the words whose inner letters are not all the same, each changed);
  keyboard typos replace a letter by a key beside it on a QWERTY keyboard,
  its case kept (0.02, 0.05, 0.1 of the letters); the emoji sprinkle puts a
  space and one of eight neutral emoji (smiles, thumbs up, folded hands, a
  thinking face, sparkles, raised hands) after a word (0.05, 0.1, 0.25 of
  the words), so taking them out gives the text back. The constants are in
  `obfuscation/noise.py`. The high densities are attack-only. The family is
  not drawn with a translation, whatever the script.
- `wording` (spec version 1.3, complexity 2) puts each word the generation
  model marks harmful (the call and the strict word-list parse of
  `sensitive_words`, `tokenbreak` and `word_art`; a reply that marks no word
  is rejected, `DISCARDED; reason=no sensitive word`, a malformed one too)
  in other terms, with one more call for each distinct word (the `pseudonym`
  or `word_game` prompt, capped at 128 tokens), and one of eight variants of
  its pool around the text says how to read them.
  `to_pseudonym_substitution` swaps in a code name of one or two words the
  model gives the word, with the case of the word it replaces, and adds the
  key, one `<alias> = <word>` line per word (the pool `pseudonym_key` of
  `obfuscation/wording_templates.json`). `to_word_game` puts a placeholder
  `[W1]`, `[W2]`, ... for every occurrence of each word and lists a puzzle
  that resolves to it, one `[Wn]: <puzzle>` line per placeholder, in a variant
  that asks the reader to solve them first (the pool `word_game_solve`). A
  reply that is not a code name (over two words, on more than one line, with an
  `=`, holding a marked word, or equal to an earlier one) or not a puzzle
  (empty, over 40 words, or holding the word) is asked for again once, then
  discards the text (`malformed pseudonym`, `malformed puzzle`). The info is
  `variant=<name>;detected_words=<words>`. It is attack only and outside the
  eval type's pool; `to_synonym_substitution`, which swaps a plain synonym
  with no key or puzzle, stays in `sensitive_words`.
- `word_art` (spec version 1.3, complexity 3) draws one word of the text in
  ASCII art: the generation model marks the text's harmful words (the call
  and the strict word-list parse of `sensitive_words` and `tokenbreak`), the
  first one is drawn in a font chosen as `to_ascii_art` chooses its fonts
  (from those with a glyph for every character of the word, within the same
  cap), every occurrence of it in the text, in any case, becomes `[MASK]`,
  and one of eight variants of its pool (`decode` in
  `obfuscation/word_art_templates.json`) puts the art and the masked text
  together and asks the reader to read the art and put the word back. A
  reply that marks no word is rejected (`DISCARDED; reason=no sensitive
  word`), a malformed one too (`malformed word list`), and a word no font
  draws with the reason `to_ascii_art` gives. The info is
  `variant=<name>;font=<font>;detected_words=<word>`. It is attack only and
  outside the eval type's pool.
- `suffixes` appends an adversarial suffix: punctuation, fragments, unicode
  or emoji. Its info is `suffix=<suffix>`.
- `payload_split` (spec version 1.3) cuts the text into two to four parts,
  names them A, B, C ..., and writes them, one per line as `A: "..."` (a
  quote or a backslash in a part is escaped as a JSON string does, so the
  quoting stays balanced), in one of eight variants of its pool (`reassembly` in
  `obfuscation/payload_split_templates.json`) that asks the reader to join
  them in order. `to_payload_split` is pure: it cuts at word boundaries into
  parts of nearly equal word count (the text's whitespace becomes single
  spaces). `to_payload_split_semantic` has a model cut the text where its
  meaning allows (the `payload_split_semantic` prompt), asked again up to
  three times until the reply is a numbered list of the drawn count of
  parts, and keeps the parts only when, joined in order and compared
  without whitespace, they are the text: a check in code, no check model
  (`DISCARDED; reason=parts differ from the prompt`, or `malformed parts`
  when no reply was the list). Both draw the count, two to four and no more
  than the text's words, from `rng`, discard a text of fewer than two words
  (`too few words to split`), and write `variant=<name>;parts=<n>`. The
  family is attack only, and outside the eval type's pool. It is not drawn
  with a translation.
- `encode` encodes the whole text: base64 (one line), rot13, rot18 (rot13 on
  letters and rot5 on ASCII digits), rot47, unicode escapes (`\uXXXX`, or
  `\UXXXXXXXX` above U+FFFF), ASCII ordinals, a separator (between the
  characters as a reader sees them: a combining mark, a joiner, an emoji's
  skin tone, a flag's two letters and a decomposed Hangul syllable's jamo
  stay together, as the reversals and the zero-width noise below take them;
  not a full segmenter, so the rest of emoji sequences and Indic conjuncts
  can still be cut), leetspeak at three levels, morse, braille. Spec version 1.2
  added fifteen more, all pure: reversal of the characters, of each word's
  characters or of the word order (by character cluster, so a base letter
  keeps its marks); look-alike letters (a fraction of the Latin letters,
  drawn by `rng`, swapped for the visually identical Cyrillic ones);
  zero-width noise (zero-width spaces and non-joiners scattered between the
  characters at a density drawn by `rng`; never the joiner, which the
  character clusters of the separators and the reversals bind to the next
  character); the byte codecs hex, binary (eight bits per UTF-8 byte), URL
  percent-encoding (every character but letters, digits and `_.-~`, a `/`
  too), numeric HTML entities, base32 and base85; a Caesar shift (drawn by
  `rng` from a fixed set, never 13, reported in `shift=<n>`, not stated in the
  text) and Atbash on the Latin letters;
  and fullwidth and math-styled Unicode (math-styled drawing one
  Mathematical Alphanumeric style per call, reported in `math_style=<name>`).
  Hex, binary, HTML entities, base32 and base85 are `heavy`, the rest `light`.
  Spec version 1.3 gives 21 of the 27 encodings *decode hints*: at the spec's
  `decode_hint_rate` (0.5), such an encoding writes one of four hints around
  the encoded text, drawn by `rng` after the encoding, which says how to
  decode it (the Caesar cipher's states the shift that call used). The hints
  are in `obfuscation/decode_hints.json`, a pool of four per technique; the
  info gains `hint=<name>` when a hint was written, and the kind's variants
  table counts them. An encoding that changed nothing (a text of digits under
  ROT13) gets none. Six encodings have none: the five of the eval pool,
  `to_base64`, `to_unicode_escape`, `to_ascii_ordinal`, `to_morse` and
  `to_braille`, whose hints would change the pinned eval pool's text, and `to_leetspeak_basic`,
  which the natural kind draws; import fails for a pool of hints that names one
  (`NEVER_HINTED` in `obfuscation/encoding.py`). The ten `to_sensitive_words_encode_*` techniques
  call the plain encoders, so they carry none.
- `structural` wraps the text in JSON, XML or markdown. `to_json` keeps
  non-ASCII characters as they are.
- `neutral_container` (spec version 1.2) places the text in an everyday
  document, the formats a classifier meets in real traffic. Each of its six
  pure techniques draws one of eight variants of its pool in
  `obfuscation/neutral_container_templates.json`, which give the
  surroundings; the info is `container=<container>;variant=<name>`.
  `to_code_comment` puts the text in a comment above a few lines of code
  (Python, JavaScript, SQL, C, HTML, shell, Lisp, LaTeX), each line behind
  the comment marker; `to_yaml_document` makes it a block scalar of a YAML
  document (a ticket, front matter, a record list, a config map and so on),
  each line indented; `to_csv_row` makes it one quoted field of a row below
  a header (comma, semicolon, tab or pipe), its own quotes doubled;
  `to_email_message` makes it the body of an email, below its headers and
  above a signature; `to_transcript_line` makes it one speaker's line of a
  transcript (a timestamped line, a WebVTT or SRT cue, an interview or a
  meeting), and `to_log_entry` the message of a log entry (Python, syslog,
  logback, Go, Ruby, nginx and other line formats), newlines kept in both.
  The text can be read back whole, except for what would end its container:
  in a comment or a YAML block scalar every kind of line break (carriage
  return, vertical tab, form feed, NEL, U+2028, U+2029) is written as a
  line feed, so the next line stays behind the marker or the indent; in a
  block comment (C, HTML) a `*/` or `-->` of the text gets a space before
  its last character; in a block scalar the lines of only spaces are
  emptied and the leading spaces of the first line with text dropped (they
  would set an indent the next lines lack); in a WebVTT or SRT cue, which a
  blank line ends, only the non-blank lines are kept and a `-->` is broken
  the same way. The surroundings are
  neutral by design: no variant gives an instruction, says anything to the
  reader or mentions answering, rules or policies; a test checks each
  template against a list of such words, a starting list to extend. The
  transcript and log containers come closest to a chat turn: a transcript
  holds the one speaker's line and no other speaker's, and a log message
  follows at most one neutral entry of the same log.
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
  rejected (a fresh plan no longer draws ASCII art after a `heavy` full-text
  encoding; a kept plan may): unicode escapes make a text about 6 times as long, so an input
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
profession variants and `to_use_past_tense` four historical framings.

Spec version 1.3 added three attack-only answer formats, eight variants
each: `to_dual_answer` asks for two answers, one after the other,
`to_tagged_prefix` for an answer that opens with a set tag, and
`to_answer_in_encoding` for an answer written in an encoding. None is in
the natural or the eval pool.

It also gave each thin request family a member, two for impersonation and
temporal, eight variants each, so each now has three:
`to_impersonate_authority` and `to_impersonate_developer` (a claimed
position of authority, or a claim to be one of the model's developers,
before the prompt), `to_use_future_tense` and `to_use_historical_era` (the
request set in the future, or in a historical era; complexity 3, as
`to_use_past_tense`), `to_distractor_prefix_task` and
`to_distractor_suffix_task` (another task before or after the prompt) and
`to_ask_with_urgency` (a note of urgency before it). Their info has their
family's form, `type=<directive>;variant=<name>`. All are attack-only but
`to_ask_with_urgency`: `asking` is natural, and an urgent ask is a natural
variation, so it takes its family's kinds. The
`eval` type names `to_use_past_tense` alone of `temporal`, not the family,
so its pool does not change.

Three request families of spec version 1.2 are for natural variations only
(their `kinds` are `["natural"]`, so no attack draws them): ordinary,
harmless text that real users put around a request, which a classifier
should not take for an attack. The text is kept as it is.

- `smalltalk_prefix` and `smalltalk_suffix`: `to_smalltalk_prefix` puts one
  of twenty greetings and friendly asides before the text,
  `to_smalltalk_suffix` one of twenty thanks and closings after it, from
  `requests/smalltalk_templates.json`. They apply to every input. Their
  dimension is `request_shape`.
- `benign_role` gives the assistant a plain role in front of the text, in
  three styles of twelve variants each, from
  `requests/benign_role_templates.json`: `to_benign_role_subject_helper` (a
  subject expert or a reference worker), `to_benign_role_tutor` (a teacher or
  a study helper) and `to_benign_role_task_assistant` (help with a task for
  work or study). A benign role applies to
  benign and dual-use benign inputs only: each technique carries an
  applicability on the entry type, so the plan never draws one for a
  harmful or dual-use harmful input: its rows are hard negatives, a role in
  front of a harmless request. Its dimension is `role_framing`, which the
  `writer` axis
  is ruled apart from
  ([configuration.md](configuration.md#dimensions-and-the-runs-check)).

No greeting, closing or role says anything about rules or how to reply; a
test checks each template against a list of such words.

The variants of a directive (a framing, request or structural template)
form its variant pool. Each is as likely to be drawn unless a variant of
the pool gives a `weight`; `0` turns a variant off. No packaged variant has
a weight, and an eval-pool technique's may not, since a weight changes the
eval text ([extending.md](extending.md#add-variants-to-a-template-pool)).
Importing `redact.jailbreak` checks every pool and raises naming the file,
directive and variant of a fault. A pool should hold at least 8 variants
(a request or framing directive, a container, a fake turn, the payload split's reassembly, the
word art's decoding, the wording's key and puzzles), 12 for a benign role, 20 for small talk, 6 for the
persona templates' mechanics and 4 for an encoding's decode hints;
13 of the 85 pools are below their floor today (the five
framing, the four continuation, the two structural, `refusal_then_comply` and
`past_tense`, each with 4, or 1): the techniques of the eval pool, whose
text is pinned until the next eval review. A test ratchets the rest
([extending.md](extending.md#add-variants-to-a-template-pool)).

The info names the variant drawn: `variant=<name>` for the framing
techniques, the answer formats, the answer languages, asking, the
distractors, impersonation, indirect, temporal, the neutral containers,
the payload splits (`variant=<name>;parts=<n>`), the word art
(`variant=<name>;font=<font>;detected_words=<word>`), the wording
(`variant=<name>;detected_words=<words>`),
small talk (`smalltalk=<prefix|suffix>;variant=<name>`), the benign roles
(`role=<style>;variant=<name>`), the fake turns (`variant=<name>`) and the
persona templates, which name the mechanic drawn
(`category=<key>_persona_template;variant=<mechanic>`);
`hint=<name>` after an encoding's own info (`shift=<n>;hint=<name>`) when
it wrote a decode hint; `style=<name>` for
continuation. Structural's `to_xml` and `to_markdown` have one template
each and name none (`to_xml`'s info names its policy id), and `to_json`
writes its wrapper in code. Continuation and structural are drawn by the
eval pool, so their infos stay as they are until the next eval review.
After a run, `<kind>.variants.csv` counts the rows that drew each variant
([stages/jailbreaks.md](stages/jailbreaks.md#files)).

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
- the `noise` family (12 techniques: random capitals, scrambled letters,
  keyboard typos and emoji, each at three densities);
- the `neutral_container` family (6 techniques: a code comment, a YAML
  document, a CSV row, an email, a transcript line, a log entry);
- small talk before and after the text, and benign roles (5 techniques,
  natural variations only);
- many-shot variants of FSH and DAP (4 techniques, 32 pairs each);
- the answer formats code, dialogue, story and structured data;
- continuation attacks (4) and indirect task embedding (6);
- the answer languages Zulu, Scots Gaelic, Bengali and Javanese, and (spec
  version 1.2) Hmong, Guarani, Chinese, Hindi, Yoruba and Amharic, of which
  (spec version 1.3) Hmong, Chinese, Hindi and Amharic are also translations;
- (spec version 1.3) the persona templates (11 frames, 6 mechanics), three
  named personas (a mercenary, an underground hacker, an anarchist
  philosopher), the fake turns (3), the payload split (2: at word boundaries, or where a
  model finds the meaning allows), the word art (1: a marked word drawn
  as ASCII art) and the wording (2: a code name with a key, a word game);
  the answer formats dual answer, tagged prefix and answer
  in an encoding; the impersonations of an authority and of a developer,
  the future tense and a historical era, a task as a distractor before or
  after the prompt, and an urgent ask.
