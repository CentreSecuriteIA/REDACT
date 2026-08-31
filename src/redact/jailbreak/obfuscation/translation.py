"""Low-resource language translation obfuscation.

Each per-language function is a **technique generator** (see
``jailbreak/protocol.py``): it yields an :class:`LLMRequest` for the
translation, then one for the quality check, and resumes with each response.
On a failed check it retries (up to ``num_retries``) with the checker's
feedback. This lets the batched engine advance every sample's translation
round-by-round — all translate requests for a round go out as one batch, then
all check requests — instead of blocking one prompt at a time.

Translation always routes to the translation-role model (DeepSeek by default),
independent of the generation model passed by the pipeline, because that model
has stronger multilingual coverage. The single-sample / test path drives these
generators via ``utils.apply_combination`` → ``protocol.run_sync``.

Languages are grouped by resource level (safety training coverage):
  - High-resource  (complexity 0): French, Japanese, Russian, Spanish, German, Arabic
  - Mid-resource   (complexity 1): Turkish, Czech, Vietnamese, Greek, Croatian
  - Low-resource   (complexity 2): Swahili, Thai, Khmer, Maori, Nepali,
                                   Zulu, Scots Gaelic, Bengali, Javanese
"""

from redact.jailbreak.protocol import LLMRequest, TechniqueGen
from redact.llms.model_config import default_model_for_role
from redact.llms.prompts import build_messages, load_prompt
from redact.llms.router import is_accepted

LANGUAGES = [
    # High-resource
    "French", "Japanese", "Russian", "Spanish", "German", "Arabic",
    # Mid-resource
    "Turkish", "Czech", "Vietnamese", "Greek", "Croatian",
    # Low-resource
    "Swahili", "Thai", "Khmer", "Maori", "Nepali",
    "Zulu", "Scots Gaelic", "Bengali", "Javanese",
]


def _resolve_translate_model() -> str:
    """Return the canonical translation model from the registry.

    Looked up by role so the choice can be changed by registering a
    different model with ``role="translation"``. Falls back to the
    historical default ``deepseek-v3.2`` if the role isn't registered
    (e.g. user mutated the registry).
    """
    try:
        return default_model_for_role("translation")
    except KeyError:
        return "deepseek-v3.2"


def _build_translate_messages(text: str, language: str, feedback: str = "") -> list[dict]:
    """Build the chat messages for a translation request (pure, no LLM call).

    Loads ``translate`` (first attempt) or ``translate_retry`` (feedback
    from a prior rejected attempt) — two separate prompt categories rather
    than one template with a conditional block, since ``str.format_map()``
    has no conditionals.
    """
    category = "translate_retry" if feedback else "translate"
    config = load_prompt("jailbreak", category)
    return build_messages(config, language=language, text=text, feedback=feedback)


def _build_check_messages(original: str, translation: str, language: str) -> list[dict]:
    """Build the chat messages for a translation-quality check (pure)."""
    config = load_prompt("jailbreak", "translate_check")
    return build_messages(config, original=original, translation=translation, language=language)


def _translate_gen(
    prompt: str,
    language: str,
    *,
    translate_model: str | None = None,
    check_model: str | None = None,
    num_retries: int = 4,
    **kwargs,
) -> TechniqueGen:
    """Translate ``prompt`` to ``language`` with checked, feedback-aware retry.

    Yields a translate request then a check request each round; returns
    ``(translation, language)`` on success or
    ``(last_attempt, "DISCARDED; language=...; feedback=...")`` on exhaustion.

    ``**kwargs`` swallows engine-supplied keys (``gen_model``, ``benign_data``)
    that translation does not use — it always uses the translation-role model.
    """
    # Resolved fresh per call (not a frozen import-time constant) so a
    # runtime register_model(..., role="translation") takes effect
    # immediately, same as every other role default in the library.
    gen_model = translate_model or _resolve_translate_model()
    chk_model = check_model or _resolve_translate_model()
    feedback = ""
    last = prompt
    for _ in range(num_retries):
        translation = yield LLMRequest(
            gen_model, _build_translate_messages(prompt, language, feedback)
        )
        verdict = yield LLMRequest(
            chk_model, _build_check_messages(prompt, translation, language)
        )
        if is_accepted(verdict):
            return translation, language
        feedback = verdict
        last = translation
    return last, f"DISCARDED; language={language}; feedback={feedback}"


def _make_language_fn(language: str):
    """Build a named technique generator for one target language."""

    def fn(prompt: str, **kwargs) -> TechniqueGen:
        result = yield from _translate_gen(prompt, language, **kwargs)
        return result

    fn.__name__ = "to_" + language.lower().replace(" ", "_")
    fn.__qualname__ = fn.__name__
    fn.__doc__ = f"Translate to {language} (technique generator)."
    return fn


# High-resource (complexity 0)
to_french = _make_language_fn("French")
to_japanese = _make_language_fn("Japanese")
to_russian = _make_language_fn("Russian")
to_spanish = _make_language_fn("Spanish")
to_german = _make_language_fn("German")
to_arabic = _make_language_fn("Arabic")

# Mid-resource (complexity 1)
to_turkish = _make_language_fn("Turkish")
to_czech = _make_language_fn("Czech")
to_vietnamese = _make_language_fn("Vietnamese")
to_greek = _make_language_fn("Greek")
to_croatian = _make_language_fn("Croatian")

# Low-resource (complexity 2)
to_swahili = _make_language_fn("Swahili")
to_thai = _make_language_fn("Thai")
to_khmer = _make_language_fn("Khmer")
to_maori = _make_language_fn("Maori")
to_nepali = _make_language_fn("Nepali")
to_zulu = _make_language_fn("Zulu")
to_scots_gaelic = _make_language_fn("Scots Gaelic")
to_bengali = _make_language_fn("Bengali")
to_javanese = _make_language_fn("Javanese")


def get_translation_functions() -> list:
    """Return all translation technique generators."""
    return [
        # High-resource
        to_french, to_japanese, to_russian, to_spanish, to_german, to_arabic,
        # Mid-resource
        to_turkish, to_czech, to_vietnamese, to_greek, to_croatian,
        # Low-resource
        to_swahili, to_thai, to_khmer, to_maori, to_nepali,
        to_zulu, to_scots_gaelic, to_bengali, to_javanese,
    ]
