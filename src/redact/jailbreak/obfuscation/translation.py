"""Translation techniques: the prompt translated to one of 20 languages.

The translation-role model translates and checks, retrying with the check's
feedback; ``info`` is the language, or ``DISCARDED; language=...; feedback=...``
when every attempt is rejected.
"""

from collections.abc import Callable

from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import LLMRequest, is_accepted
from redact.llms.model_config import default_model_for_role
from redact.llms.prompting import build_messages, load_prompt

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
    return last, f"{DISCARDED}; language={language}; feedback={feedback}"


def _make_language_fn(language: str) -> Callable:
    """The technique that translates to ``language``."""

    def fn(prompt: str, **kwargs) -> TechniqueGen:
        return (yield from _translate_gen(prompt, language, **kwargs))

    fn.__name__ = fn.__qualname__ = "to_" + language.lower().replace(" ", "_")
    fn.__doc__ = f"Translate to {language} (technique generator)."
    return fn


# One technique per language, named to_<language>: to_french, to_scots_gaelic, ...
_FNS = [_make_language_fn(language) for language in LANGUAGES]
globals().update({fn.__name__: fn for fn in _FNS})


def get_translation_functions() -> list[Callable]:
    """Return all translation technique generators."""
    return list(_FNS)
