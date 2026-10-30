"""Romanization: a model rewrites the prompt in the Latin spelling of a
non-Latin language (Hinglish, Arabizi and the like), keeping the meaning.

One technique per language of ``romanization_languages.json``
(``to_romanize_<slug>``). The generation model rewrites; a code gate checks
that the output is in the Latin alphabet and is not plain English (its
GlotLID label is the language's romanized one, or at least not ``eng_Latn``,
GlotLID's romanized labels being trained on little data), then the check
model reads it for meaning. ``info`` is ``language=<name>``.

**Five languages are gated more weakly** (spec 1.5, attack only): Russian,
Japanese, Chinese, Greek and Korean, whose rows have no ``romanized_label``
because GlotLID v3 has no Latin-script label for any of them. Nothing in code
can read their language, so their gate is the Latin share, the length band, a
count of distinctively English words (:data:`ENGLISH_MARKERS`, which runs with
or without GlotLID) and GlotLID's ``eng_Latn`` reading when it is installed;
the check model's verdict, which is asked whether the rewrite is in the named
language, is the only check of the language itself. English leaking through
(a short English text with few of those words) and a rewrite in the wrong
language are more likely to pass for these five than for the other ten. The
owner asked for them with that risk understood.
"""

import re
from collections.abc import Callable

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen, data_rows
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

from . import langid
from .scripts import script_share

#: The languages, one row per technique (name, slug, romanized_label, natural).
#: ``romanized_label`` is None for a language GlotLID has no Latin label for.
LANGUAGES = data_rows("romanization_languages.json")

LENGTH_RATIO_BAND = (0.5, 2.0)
MIN_BAND_CHARS = 20

#: The Latin share the romanized output must reach: romanization writes in the
#: Latin alphabet, so a low share means the model wrote the native script.
MIN_LATIN_SHARE = 0.8

#: Distinctively English words, read in place of a romanized label for a
#: language that has none. Each is common in English and is not a common word
#: of the usual romanizations of Russian, Japanese (Hepburn), Chinese (pinyin),
#: Greek or Korean (Revised Romanization): the short words that are (``to``,
#: ``i``, ``a``, ``no``, ``on``, ``my``, ``by``, ``you``, ``can``, ``are``,
#: ``me``, ``so``, ``in``, ``do``) are left out, as are English words that
#: chat romanizations keep as loanwords (``please``, ``ok``). A romanization
#: holds none of them outside a quotation; English prose holds them at about
#: one word in six (median over thirty English prompts, 0.155).
ENGLISH_MARKERS = frozenset({
    "the", "and", "of", "that", "this", "these", "those", "with", "for", "from",
    "what", "which", "how", "when", "where", "why", "who", "whom", "whose",
    "your", "yours", "would", "should", "could", "have", "will", "was", "were",
    "been", "being", "not", "but", "or", "if", "at", "into", "about", "they",
    "them", "their", "there", "then", "than", "does", "did", "had", "his",
    "her", "its", "our", "just", "also", "very", "here", "only", "other",
    "such", "each", "because", "without", "while", "any", "using", "after",
    "before", "over", "under", "between", "through", "during", "against",
    "every", "most", "many", "much", "both", "either", "neither", "whether",
    "upon", "onto", "within",
})

#: The share of a reply's words in :data:`ENGLISH_MARKERS` from which a
#: romanization into a language without a romanized label is read as English
#: (``english``). Set from thirty English prompts (23 of them reach it) and
#: forty romanized sentences in the five languages (none has a marker); not
#: measured on model output.
MAX_ENGLISH_SHARE = 0.1

#: What the generation prompt keeps as written (a quotation, a bracket, code),
#: left out of the English count: such text may be English in any language.
_KEPT_AS_WRITTEN = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»|„[^“”]*[“”]|`[^`]*`'
                              r"|\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|https?://\S+")
_WORD = re.compile(r"[^\W\d_]+")

GEN_MAX_TOKENS = 2048
CHECK_MAX_TOKENS = 256


def english_share(text: str) -> float:
    """The share of ``text``'s words that are :data:`ENGLISH_MARKERS`, a
    quotation, a bracket, code in backquotes and a URL left out (the
    generation prompt keeps those as written); 0 when no word is left. A
    word is a run of letters, compared lower-cased."""
    words = _WORD.findall(_KEPT_AS_WRITTEN.sub(" ", text).lower())
    if not words:
        return 0.0
    return sum(word in ENGLISH_MARKERS for word in words) / len(words)


def _romanize_gate(text: str, original: str, language: dict) -> str:
    """Why a romanization of ``original`` into ``language`` (a row of
    :data:`LANGUAGES`) is rejected, or ``""``.

    Empty; a length ratio outside :data:`LENGTH_RATIO_BAND`; a Latin share
    below :data:`MIN_LATIN_SHARE` (the model wrote the native script, not a
    romanization); or an output read as English (it stayed English). When
    GlotLID reads the language's romanized label the output is confirmed;
    any other non-English Latin label passes too, since GlotLID's romanized
    labels are trained on little data. Without the detector the language is
    not checked (Latin share only).

    A language without a romanized label (Russian, Japanese, Chinese, Greek,
    Korean) is also read as English when :func:`english_share` reaches
    :data:`MAX_ENGLISH_SHARE`, with or without the detector; past that, and
    GlotLID's ``eng_Latn``, nothing here reads its language, which only the
    check model judges: a weaker gate than the other languages'.
    """
    if not text.strip():
        return "empty"
    if len(original) >= MIN_BAND_CHARS:
        ratio = len(text) / len(original)
        low, high = LENGTH_RATIO_BAND
        if not low <= ratio <= high:
            return f"length_ratio={ratio:.2f}"
    if script_share(text, "Latn") < MIN_LATIN_SHARE:
        return "not latin"
    romanized = language["romanized_label"]
    if romanized is None and english_share(text) >= MAX_ENGLISH_SHARE:
        return "english"
    if not langid.available():
        return ""
    label = langid.detect(text)
    if romanized is not None and label == romanized:
        return ""
    if label == "eng_Latn":
        return "english"
    return ""


def _romanize_gen(
    prompt: str,
    language: dict,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = 2,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Rewrite ``prompt`` romanized into ``language`` on ``gen_model``, gated
    in code and checked for meaning by ``check_model`` (``gen_model`` when
    None), up to ``check_attempts`` rewrites."""
    name = language["name"]
    gen = build_messages(
        load_prompt("jailbreak", "romanize", prompt_dir),
        prompt=prompt, language=name)
    check = PromptTemplate.load("jailbreak", "romanize_check", prompt_dir=prompt_dir)
    text, accepted, verdict = yield from checked(
        gen_model, lambda feedback: with_feedback(gen, feedback),
        check_model or gen_model,
        lambda original, rewritten: check(original=original, rewritten=rewritten,
                                          language=name),
        original=prompt, max_attempts=check_attempts,
        precheck=lambda candidate: _romanize_gate(candidate, prompt, language),
        max_tokens=(GEN_MAX_TOKENS, CHECK_MAX_TOKENS),
        temperatures=(None, CHECK_TEMPERATURE),
    )
    if not accepted:
        return text.strip(), (f"{DISCARDED}; reason=romanize failed; "
                              f"language={name}; feedback={verdict}")
    return text.strip(), f"language={name}"


def _make_fn(language: dict) -> Callable:
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _romanize_gen(prompt, language, **kwargs))

    fn.__name__ = fn.__qualname__ = f"to_romanize_{language['slug']}"
    fn.__doc__ = f"Romanize the prompt into Latin-script {language['name']}."
    if language["romanized_label"] is None:
        # The module docstring says why: no language is read in code.
        fn.__doc__ += " Gated more weakly: GlotLID has no label."
    return fn


_FNS = [_make_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_romanization_functions() -> list[Callable]:
    """Return all romanization technique generators."""
    return list(_FNS)
