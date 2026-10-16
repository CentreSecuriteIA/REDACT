# Jailbreak techniques

The jailbreak stage draws from 152 techniques in 4 layers and 23 families.
`configs/jailbreak/combination_spec.json` gives each technique its families,
a complexity from 0 to 3, and whether it calls a model (`requires_llm`) or
needs the benign data (`requires_benign`). This page lists them by name, for
the `allow` and `deny` lists of the
[augmentation section](configuration.md#the-augmentation-section). It does not
describe what a technique does.

## Layers

A combination applies its techniques in layer order: `hacking`, then
`manipulation`, then `obfuscation`, then `requests`. Within the obfuscation
layer, families apply in the order of their `within_layer_order`.

| Layer | Techniques | Picks per combination |
|---|---|---|
| `hacking` | 25 | at most 1 |
| `manipulation` | 8 | at most 1 |
| `obfuscation` | 72 | up to `max_obfuscations` families in budget mode |
| `requests` | 47 | at most 1 |

## Families and techniques

| Layer | Family | Order | Count | Calls a model | Needs benign data | Techniques |
|---|---|---|---|---|---|---|
| hacking | `cognitive` | | 5 | yes | no | `to_persona_roleplay`, `to_hypothetical_framing`, `to_authority_obedience`, `to_avi`, `to_deep_inception` |
| hacking | `framing` | | 5 | no | no | `to_fictional_world`, `to_noble_goal`, `to_nefarious_goal`, `to_high_stake`, `to_no_moral_constraints` |
| hacking | `persona` | | 15 | yes | no | `to_invented_persona`, `to_psychopath_persona`, `to_alien_persona`, `to_radical_politician_persona`, `to_cult_leader_persona`, `to_very_advanced_ai_persona`, `to_cartel_leader_persona`, `to_artist_persona`, `to_mentally_ill_persona`, `to_deformed_scientist_persona`, `to_politician_persona`, `to_deformed_professor_persona`, `to_religious_figure_persona`, `to_actor_persona`, `to_someone_from_the_future_persona` |
| manipulation | `dap` | | 4 | the `selected` two | yes | `to_dap_random_short`, `to_dap_random_long`, `to_dap_selected_short`, `to_dap_selected_long` |
| manipulation | `fsh` | | 4 | the `selected` two | yes | `to_fsh_random_short`, `to_fsh_random_long`, `to_fsh_selected_short`, `to_fsh_selected_long` |
| obfuscation | `sensitive_words` | 1 | 16 | yes | no | `to_sensitive_words_encode_base64`, `to_sensitive_words_encode_rot13`, `to_sensitive_words_encode_rot18`, `to_sensitive_words_encode_rot47`, `to_sensitive_words_encode_unicode`, `to_sensitive_words_encode_ascii`, `to_sensitive_words_encode_separator`, `to_sensitive_words_encode_leetspeak_basic`, `to_sensitive_words_encode_leetspeak_intermediate`, `to_sensitive_words_encode_leetspeak_advanced`, `to_sensitive_words_split`, `to_sensitive_words_star`, `to_sensitive_words_hyphen`, `to_sensitive_words_underscore`, `to_sensitive_words_variables`, `to_synonym_substitution` |
| obfuscation | `tokenbreak` | 2 | 3 | yes | no | `to_tokenbreak_prepend`, `to_tokenbreak_split`, `to_tokenbreak_delimiter` |
| obfuscation | `translation` | 3 | 20 | yes, the `translation` model | no | `to_french`, `to_japanese`, `to_russian`, `to_spanish`, `to_german`, `to_arabic`, `to_turkish`, `to_czech`, `to_vietnamese`, `to_greek`, `to_croatian`, `to_swahili`, `to_thai`, `to_khmer`, `to_maori`, `to_nepali`, `to_zulu`, `to_scots_gaelic`, `to_bengali`, `to_javanese` |
| obfuscation | `typos` | 4 | 4 | yes | no | `to_rewrite_with_typos_low`, `to_rewrite_with_typos_medium`, `to_rewrite_with_typos_high`, `to_rewrite_with_typos_insane` |
| obfuscation | `casual_writing` | 4.5 | 9 | no | no | `to_lowercase`, `to_uppercase`, `to_no_sentence_capitals`, `to_dropped_spaces`, `to_doubled_spaces`, `to_dropped_punctuation`, `to_doubled_punctuation`, `to_abbreviations`, `to_stripped_accents` |
| obfuscation | `suffixes` | 5 | 4 | no | no | `to_adversarial_suffix_punctuation`, `to_adversarial_suffix_fragments`, `to_adversarial_suffix_unicode`, `to_adversarial_suffix_emoji` |
| obfuscation | `encode` | 6 | 12 | no | no | `to_base64`, `to_rot13`, `to_rot18`, `to_rot47`, `to_unicode_escape`, `to_ascii_ordinal`, `to_separator`, `to_leetspeak_basic`, `to_leetspeak_intermediate`, `to_leetspeak_advanced`, `to_morse`, `to_braille` |
| obfuscation | `structural` | 7 | 3 | no | no | `to_json`, `to_xml`, `to_markdown` |
| obfuscation | `ascii_art` | 8 | 1 | no | no | `to_ascii_art` |
| requests | `answer_format` | | 9 | no | no | `to_refusal_then_comply`, `to_apology_prevention`, `to_prefix_injection`, `to_answer_in_markdown`, `to_answer_as_tutorial`, `to_answer_as_code`, `to_answer_as_dialogue`, `to_answer_as_story`, `to_answer_as_structured_data` |
| requests | `answer_language` | | 20 | no | no | `to_ask_answer_in_french`, `to_ask_answer_in_japanese`, `to_ask_answer_in_russian`, `to_ask_answer_in_spanish`, `to_ask_answer_in_german`, `to_ask_answer_in_arabic`, `to_ask_answer_in_turkish`, `to_ask_answer_in_czech`, `to_ask_answer_in_vietnamese`, `to_ask_answer_in_greek`, `to_ask_answer_in_croatian`, `to_ask_answer_in_thai`, `to_ask_answer_in_swahili`, `to_ask_answer_in_khmer`, `to_ask_answer_in_maori`, `to_ask_answer_in_nepali`, `to_ask_answer_in_zulu`, `to_ask_answer_in_scots_gaelic`, `to_ask_answer_in_bengali`, `to_ask_answer_in_javanese` |
| requests | `asking` | | 2 | no | no | `to_innocuous_question`, `to_ask_for_details` |
| requests | `continuation` | | 4 | no | no | `to_continue_passage`, `to_complete_story`, `to_fill_in_blank`, `to_continue_instructions` |
| requests | `distractor_prefix` | | 2 | no | no | `to_distractor_prefix_related`, `to_distractor_prefix_unrelated` |
| requests | `distractor_suffix` | | 2 | no | no | `to_distractor_suffix_related`, `to_distractor_suffix_unrelated` |
| requests | `impersonation` | | 1 | no | no | `to_impersonate_good_person` |
| requests | `indirect` | | 6 | no | no | `to_summarize`, `to_proofread`, `to_critique`, `to_translate_task`, `to_explain_to_child`, `to_rewrite` |
| requests | `temporal` | | 1 | no | no | `to_use_past_tense` |

A technique takes its layer and its order from the first family of its
spec entry. The ten `to_sensitive_words_encode_*` techniques also
belong to `encode`, so one of them blocks both families.

`encode` techniques carry an `encode_weight`: `light` for `to_rot13`,
`to_rot18`, `to_rot47`, `to_separator` and the three `to_leetspeak_*`;
`heavy` for `to_base64`, `to_unicode_escape`, `to_ascii_ordinal`,
`to_morse`, `to_braille`.

## Compatibility rules

`jailbreak/rules.py: compatible` applies the spec's rules both ways, whichever
technique is picked first. A technique may not join a combination when:

- its family is taken already;
- its layer is at its cap (`layer_max_picks`: one `hacking`, one
  `manipulation`, one `requests`);
- one of its families is in a taken family's `cross_incompatible_with`, or
  the reverse. The spec pairs `structural` with `ascii_art`; `ascii_art` with
  `framing`, `persona`, `cognitive`, `fsh`, `dap`; `fsh` and `dap` with
  `continuation`, `indirect`, `framing`, `persona`, `cognitive`; and
  `continuation` with `indirect`;
- it pairs `fsh` or `dap` with an obfuscation in
  `manipulation_blocked_obfuscation`: the `translation` and `ascii_art`
  families and the `heavy` `encode` techniques, except those also in
  `sensitive_words`;
- in budget mode, its complexity is above what is left of `max_complexity`.

`combine_techniques` does not check these rules. It only sorts techniques
into layer order. [internals/jailbreak.md](internals/jailbreak.md) explains
how a combination is drawn.

Adding a technique: [extending.md](extending.md#add-a-jailbreak-technique).
