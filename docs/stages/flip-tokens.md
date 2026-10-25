# Flip tokens (EchoGram)

!!! note "Ships empty"
    The `flip_tokens` family is registered, but `configs/jailbreak/obfuscation/flip_tokens.json`
    holds **no row**, so the family has **no technique** until rows found on a
    GPU pod are copied in. This is deliberate: a flip token is specific to the
    guard it was found against, so the library ships the mechanism, not the
    tokens.

## What it is

EchoGram (Schulz and Yeung, HiddenLayer, 2025) is a guard-in-the-loop idea:
a short token string, **appended** to a prompt, that flips a guardrail
classifier's verdict on that prompt from *flagged* to *not flagged*, while
the prompt itself is unchanged. It targets the detector, not the model that
answers — the library's own threat model, evaluating guardrails and
classifiers.

The `flip_tokens` family carries such strings as data, one technique per row,
exactly as the `suffixes` family carries published adversarial suffixes
(`stages/jailbreaks.md`). The strings are found by a **search** that must run
against a real guard, so the search is not part of the packaged library (see
[Status](#status)); the family and its data file are, so that once rows are
found they drop in without code.

## The family

- **Layer / order:** obfuscation, `within_layer_order` 75. It applies after
  every obfuscation that rewrites the text's characters (`payload_split` 65,
  `encode` 70), which would otherwise rewrite the tokens, and before the
  wrappers (`structural`, `neutral_container` 80), which keep the tokens
  literal inside the wrapper, as they keep a suffix.
- **Kinds:** `adversarial` only. No natural variation appends a flip token,
  and a natural allow list naming the family raises. The **eval pool does not
  draw it** either (the eval type allows families by name, `flip_tokens` not
  among them), so the pinned eval and natural pools are unchanged.
- **Refusals:** `ascii_art` (drawing the prompt as art would remove the
  literal string), and a `translation` drawn as the *request obfuscation*
  (it would translate the tokens). These are the published suffixes'
  refusals, for the same reasons; every other family is admitted, so a flip
  token can sit after an encoding and inside a wrapper.
- **One application:** the row's `tokens` are appended to the prompt after the
  row's `separator` (a space unless the row gives another — the separator the
  search appended with), and the info is `slug=<slug>`. The guard the tokens
  were found against and the flip counts stay in the data file, not in every
  dataset row.

## The data file

`configs/jailbreak/obfuscation/flip_tokens.json` is `{"_comment": ..., "rows": []}`.
The spec's `flip_tokens` generator makes one technique `to_flip_tokens_<slug>`
per row, in file order; `jailbreak/obfuscation/flip_tokens.py` builds the
functions from the same file, so a row adds a technique and the import fails
while the two disagree (the `suffix_strings.json` pattern). Each row gives:

| Field | Meaning |
|---|---|
| `slug` | lower-case letters, digits and underscores; the end of the technique name |
| `tokens` | the string appended |
| `guard` | the guard it was found against (a Hugging Face id or a registry name) |
| `flips` | harmful prompts the guard flagged bare and passed with the tokens |
| `n_harmful` | harmful prompts the guard flagged bare (the denominator) |
| `n_benign` | benign prompts the guard passed bare |
| `since` | the spec version that added the row (`"1.5"` and up) |

Optional: `benign_flips` (benign prompts the tokens turned flagged, of
`n_benign` — a side effect worth recording), `separator`, `notes`,
`complexity` (0 to 3, default 1) and `deprecated`. A row that breaks a rule
fails the import, naming the row. The file is read from the package only,
never from a `config_dir`.

## Status

The **search** that produces the rows — a guard-in-the-loop flip-token
search, which needs a guard model and the library's harmful and benign
prompt sets in the loop — is **not in the packaged library**. It is the one
piece of this family that depends on running an attack against a classifier,
and building it here was stopped; it is recorded as blocker **B15** in
`.claude/completion_blockers.md`. Until it runs on a pod, the data file ships
empty and the family has no technique, which is a supported state (a family
with no row registers with none).

### The pod recipe (for whoever runs the search)

The search belongs on a GPU pod with an open guardrail model loaded. The
pieces the library already provides:

- **Prompt sets.** The library's own accepted inputs
  (`Datasets/constitution_inputs/*/samples.csv`, the harmful and dual-use
  entry types) are the harmful set; the benign and dual-use-benign inputs are
  the benign set. The search reads these from frames the caller passes; it
  must not print or store the prompts' text beyond the stage files that
  already hold them.
- **Guard models to try** (small, open; those marked *gated* need a Hugging
  Face token and the model owner's access grant):
    - Llama Guard family (`meta-llama/Llama-Guard-3-8B`, `meta-llama/Llama-Guard-3-1B`) — *gated*;
    - Prompt Guard (`meta-llama/Llama-Prompt-Guard-2-86M`, `meta-llama/Llama-Prompt-Guard-2-22M`) — *gated*, small, a direct classifier;
    - Qwen3Guard-class moderation models and other open text-classification
      guards on the Hub that load without a grant.
  A guard is reached either as a `ModelClient` with a verdict prompt (a
  chat model asked to classify) or as a local Hugging Face
  text-classification pipeline (a direct classifier). Say in the row's
  `guard` which model and which form.
- **Candidate tokens.** Draw candidates from a word list or from the guard
  tokenizer's own vocabulary, with a seeded, deterministic sample.
- **Scoring.** For each candidate, the share of the harmful set whose verdict
  flips flagged → not flagged when the token is appended, and (recorded as
  `benign_flips`) the share of the benign set that flips the other way. Keep
  the best by flip count.
- **Writing rows.** Write the best candidates as rows of the shape above into
  a **scratch file**, never over the packaged `flip_tokens.json`; read the
  counts on stdout; then copy by hand the rows worth keeping into the
  packaged file, each with its `guard`, `flips`, `n_harmful`, `n_benign`,
  `since: "1.5"` and a short `notes`. Re-pin the registry and catalog
  fixtures in the same commit (a row adds a technique), and the eval and
  natural pools stay pinned (the family is adversarial-only and not in the
  eval type).

The tokens found this way are **specific to the guard they were found
against**. Against another guard they are a recognizable string at the end of
a request, not a working attack — the same caveat the published suffixes
carry.
