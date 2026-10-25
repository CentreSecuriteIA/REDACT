"""Romanization: a model rewrites the prompt in the Latin spelling of a
non-Latin language (Hinglish, Arabizi and the like), keeping the meaning.

One technique per language of ``romanization_languages.json``
(``to_romanize_<slug>``). The generation model rewrites; a code gate checks
that the output is in the Latin alphabet and is not plain English (its
GlotLID label is the language's romanized one, or at least not ``eng_Latn``,
GlotLID's romanized labels being trained on little data), then the check
model reads it for meaning. ``info`` is ``language=<name>``.
"""

from collections.abc import Callable

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen, data_rows
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

from . import langid
from .scripts import script_share

#: The languages, one row per technique (name, slug, romanized_label).
LANGUAGES = data_rows("romanization_languages.json")

LENGTH_RATIO_BAND = (0.5, 2.0)
MIN_BAND_CHARS = 20

#: The Latin share the romanized output must reach: romanization writes in the
#: Latin alphabet, so a low share means the model wrote the native script.
MIN_LATIN_SHARE = 0.8

GEN_MAX_TOKENS = 2048
CHECK_MAX_TOKENS = 256


def _romanize_gate(text: str, original: str, language: dict) -> str:
    """Why a romanization of ``original`` into ``language`` (a row of
    :data:`LANGUAGES`) is rejected, or ``""``.

    Empty; a length ratio outside :data:`LENGTH_RATIO_BAND`; a Latin share
    below :data:`MIN_LATIN_SHARE` (the model wrote the native script, not a
    romanization); or an output GlotLID reads as English (it stayed English).
    When GlotLID reads the language's romanized label the output is confirmed;
    any other non-English Latin label passes too, since GlotLID's romanized
    labels are trained on little data. Without the detector the language is
    not checked (Latin share only).
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
    if not langid.available():
        return ""
    label = langid.detect(text)
    if label == language["romanized_label"]:
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
    return fn


_FNS = [_make_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_romanization_functions() -> list[Callable]:
    """Return all romanization technique generators."""
    return list(_FNS)
