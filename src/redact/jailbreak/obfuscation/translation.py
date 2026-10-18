"""Translation techniques: the prompt translated to one of 19 languages.

The run's translation model (the translation role's by default) translates
and checks, retrying with the check's feedback; ``info`` is the language, or
``DISCARDED; language=...; feedback=...`` when every attempt is rejected.
"""

import math
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
    # Low-resource. Not Maori: TranslateGemma invents its sensitive and
    # technical terms, fluently enough to pass every check.
    "Swahili", "Thai", "Khmer", "Nepali",
    "Zulu", "Scots Gaelic", "Bengali", "Javanese",
]

#: The ISO 639-1 code a machine-translation model is sent for each language.
#: Its template takes an unknown code as the language's name, so every
#: language needs one.
LANGUAGE_CODES = {
    "French": "fr", "Japanese": "ja", "Russian": "ru", "Spanish": "es",
    "German": "de", "Arabic": "ar", "Turkish": "tr", "Czech": "cs",
    "Vietnamese": "vi", "Greek": "el", "Croatian": "hr", "Swahili": "sw",
    "Thai": "th", "Khmer": "km", "Nepali": "ne", "Zulu": "zu",
    "Scots Gaelic": "gd", "Bengali": "bn", "Javanese": "jv",
}

#: The GlotLID labels a translation into each language may carry: its own,
#: and the near-identical languages GlotLID reads short text as.
ACCEPTED_LABELS = {
    "French": {"fra_Latn"},
    "Japanese": {"jpn_Jpan"},
    "Russian": {"rus_Cyrl"},
    "Spanish": {"spa_Latn"},
    "German": {"deu_Latn"},
    "Arabic": {"arb_Arab", "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab",
               "apc_Arab", "ars_Arab", "ary_Arab", "arz_Arab"},
    "Turkish": {"tur_Latn"},
    "Czech": {"ces_Latn"},
    "Vietnamese": {"vie_Latn"},
    "Greek": {"ell_Grek"},
    "Croatian": {"hrv_Latn", "bos_Latn", "srp_Latn"},
    "Swahili": {"swh_Latn", "swc_Latn", "swa_Latn"},
    "Thai": {"tha_Thai"},
    "Khmer": {"khm_Khmr"},
    "Nepali": {"npi_Deva"},
    "Zulu": {"zul_Latn", "nbl_Latn", "nde_Latn", "xho_Latn", "ssw_Latn"},
    "Scots Gaelic": {"gla_Latn"},
    "Bengali": {"ben_Beng"},
    "Javanese": {"jav_Latn"},  # never ind_Latn: Indonesian is a real failure
}

#: A translation's length over its input's, in characters, at the median: the
#: pod test's TranslateGemma figures, 1.0 for a language it did not measure.
MEDIAN_RATIO = {
    "French": 1.0, "Japanese": 1.0, "Russian": 1.0, "Spanish": 1.0,
    "German": 1.0, "Arabic": 1.0, "Turkish": 1.0, "Czech": 1.0,
    "Vietnamese": 1.0, "Greek": 1.0, "Croatian": 1.0, "Swahili": 1.16,
    "Thai": 1.02, "Khmer": 1.09, "Nepali": 1.04, "Zulu": 1.17,
    "Scots Gaelic": 1.32, "Bengali": 1.07, "Javanese": 1.12,
}


def _check_tables(tables: dict[str, dict]) -> None:
    """Raise ``ValueError`` unless each table names exactly :data:`LANGUAGES`."""
    for name, table in tables.items():
        missing = [language for language in LANGUAGES if language not in table]
        extra = sorted(set(table) - set(LANGUAGES))
        if missing or extra:
            raise ValueError(
                f"translation.{name} must name exactly the LANGUAGES: "
                f"missing {missing}, extra {extra}.")


_check_tables({"LANGUAGE_CODES": LANGUAGE_CODES, "ACCEPTED_LABELS": ACCEPTED_LABELS,
               "MEDIAN_RATIO": MEDIAN_RATIO})

#: A translator reply is capped at about three times its input's tokens,
#: counting a token as this many characters (which overstates an English
#: text's tokens and leaves room for a script that spends more), plus
#: :data:`REPLY_SLACK_TOKENS`, never above :data:`MAX_REPLY_TOKENS`. A reply
#: cut there is a length outlier, so a runaway loop costs one capped request.
CHARS_PER_TOKEN = 3
REPLY_TOKENS_PER_INPUT_TOKEN = 3
REPLY_SLACK_TOKENS = 32
MAX_REPLY_TOKENS = 1536


def reply_cap(text: str) -> int:
    """The ``max_tokens`` of a translator request that translates ``text``."""
    tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
    return min(MAX_REPLY_TOKENS,
               REPLY_TOKENS_PER_INPUT_TOKEN * tokens + REPLY_SLACK_TOKENS)


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


def _build_translate_messages(
    text: str, language: str, feedback: str = "", prompt_dir=None,
) -> list[dict]:
    """Build the chat messages for a translation request (pure, no LLM call).

    Loads ``translate`` (first attempt) or ``translate_retry`` (feedback
    from a prior rejected attempt) — two separate prompt categories rather
    than one template with a conditional block, since ``str.format_map()``
    has no conditionals.
    """
    category = "translate_retry" if feedback else "translate"
    config = load_prompt("jailbreak", category, prompt_dir)
    return build_messages(config, language=language, text=text, feedback=feedback)


def _build_check_messages(
    original: str, translation: str, language: str, prompt_dir=None,
) -> list[dict]:
    """Build the chat messages for a translation-quality check (pure)."""
    config = load_prompt("jailbreak", "translate_check", prompt_dir)
    return build_messages(config, original=original, translation=translation, language=language)


def _translate_gen(
    prompt: str,
    language: str,
    *,
    translate_model: str | None = None,
    check_model: str | None = None,
    num_retries: int = 4,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Translate ``prompt`` to ``language`` with checked, feedback-aware retry.

    Yields a translate request then a check request each round; returns
    ``(translation, language)`` on success or
    ``(last_attempt, "DISCARDED; language=...; feedback=...")`` on exhaustion.

    ``translate_model`` makes both calls, the translation role's model when
    None; ``check_model`` replaces it for the check. ``prompt_dir`` is the
    overlay the three prompts are read through. ``**kwargs`` swallows
    engine-supplied keys (``gen_model``, ``benign_data``) that translation
    does not use.
    """
    # Resolved fresh per call (not a frozen import-time constant) so a
    # runtime register_model(..., role="translation") takes effect
    # immediately, same as every other role default in the library.
    gen_model = translate_model or _resolve_translate_model()
    chk_model = check_model or gen_model
    feedback = ""
    last = prompt
    for _ in range(num_retries):
        translation = yield LLMRequest(
            gen_model, _build_translate_messages(prompt, language, feedback, prompt_dir)
        )
        verdict = yield LLMRequest(
            chk_model, _build_check_messages(prompt, translation, language, prompt_dir)
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
