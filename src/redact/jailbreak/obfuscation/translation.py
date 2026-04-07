"""Low-resource language translation obfuscation.

Delegates all translation + checking + retry logic to
LLMs/translator.py::translate_with_check(). Each per-language function
accepts backend/model/rate_limiter as parameters.

Languages are grouped by resource level (safety training coverage):
  - High-resource  (complexity 0): French, Japanese, Russian, Spanish, German, Arabic
  - Mid-resource   (complexity 1): Turkish, Czech, Vietnamese, Greek, Croatian
  - Low-resource   (complexity 2): Swahili, Thai, Khmer, Maori, Nepali,
                                   Zulu, Scots Gaelic, Bengali, Javanese

Ported from reference obfuscation.py lines 443-501.
"""

from redact.llms.base import LLMBackend
from redact.llms.wrappers import RateLimiter
from redact.llms.translator import translate_with_check, DEFAULT_TRANSLATE_MODEL

LANGUAGES = [
    # High-resource
    "French", "Japanese", "Russian", "Spanish", "German", "Arabic",
    # Mid-resource
    "Turkish", "Czech", "Vietnamese", "Greek", "Croatian",
    # Low-resource
    "Swahili", "Thai", "Khmer", "Maori", "Nepali",
    "Zulu", "Scots Gaelic", "Bengali", "Javanese",
]


def _translate(
    prompt: str,
    language: str,
    backend: LLMBackend,
    gen_model: str | None = None,
    check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
    num_retries: int = 4,
) -> tuple[str, str]:
    """Translate prompt to target language with checked retry.

    Args:
        prompt: Text to translate.
        language: Target language name.
        backend: LLM backend.
        gen_model: Model for translation (default: deepseek-v3.2).
        check_model: Model for validation (default: deepseek-v3.2).
        rate_limiter: Optional rate limiter.
        num_retries: Maximum attempts.

    Returns:
        (translated_text, language) on success,
        (last_attempt, "DISCARDED; language=...; feedback=...") on failure.
    """
    return translate_with_check(
        backend=backend,
        gen_model=gen_model or DEFAULT_TRANSLATE_MODEL,
        check_model=check_model or DEFAULT_TRANSLATE_MODEL,
        text=prompt,
        target_language=language,
        rate_limiter=rate_limiter,
        num_retries=num_retries,
    )


# ---------------------------------------------------------------------------
# High-resource languages (complexity 0)
# ---------------------------------------------------------------------------

def to_french(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to French."""
    return _translate(prompt, "French", backend, gen_model, check_model, rate_limiter)


def to_japanese(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Japanese."""
    return _translate(prompt, "Japanese", backend, gen_model, check_model, rate_limiter)


def to_russian(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Russian."""
    return _translate(prompt, "Russian", backend, gen_model, check_model, rate_limiter)


def to_spanish(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Spanish."""
    return _translate(prompt, "Spanish", backend, gen_model, check_model, rate_limiter)


def to_german(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to German."""
    return _translate(prompt, "German", backend, gen_model, check_model, rate_limiter)


def to_arabic(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Arabic."""
    return _translate(prompt, "Arabic", backend, gen_model, check_model, rate_limiter)


# ---------------------------------------------------------------------------
# Mid-resource languages (complexity 1)
# ---------------------------------------------------------------------------

def to_turkish(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Turkish."""
    return _translate(prompt, "Turkish", backend, gen_model, check_model, rate_limiter)


def to_czech(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Czech."""
    return _translate(prompt, "Czech", backend, gen_model, check_model, rate_limiter)


def to_vietnamese(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Vietnamese."""
    return _translate(prompt, "Vietnamese", backend, gen_model, check_model, rate_limiter)


def to_greek(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Greek."""
    return _translate(prompt, "Greek", backend, gen_model, check_model, rate_limiter)


def to_croatian(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Croatian."""
    return _translate(prompt, "Croatian", backend, gen_model, check_model, rate_limiter)


# ---------------------------------------------------------------------------
# Low-resource languages (complexity 2)
# ---------------------------------------------------------------------------

def to_swahili(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Swahili."""
    return _translate(prompt, "Swahili", backend, gen_model, check_model, rate_limiter)


def to_thai(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Thai."""
    return _translate(prompt, "Thai", backend, gen_model, check_model, rate_limiter)


def to_khmer(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Khmer."""
    return _translate(prompt, "Khmer", backend, gen_model, check_model, rate_limiter)


def to_maori(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Maori."""
    return _translate(prompt, "Maori", backend, gen_model, check_model, rate_limiter)


def to_nepali(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Nepali."""
    return _translate(prompt, "Nepali", backend, gen_model, check_model, rate_limiter)


def to_zulu(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Zulu."""
    return _translate(prompt, "Zulu", backend, gen_model, check_model, rate_limiter)


def to_scots_gaelic(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Scots Gaelic."""
    return _translate(prompt, "Scots Gaelic", backend, gen_model, check_model, rate_limiter)


def to_bengali(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Bengali."""
    return _translate(prompt, "Bengali", backend, gen_model, check_model, rate_limiter)


def to_javanese(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Javanese."""
    return _translate(prompt, "Javanese", backend, gen_model, check_model, rate_limiter)


def get_translation_functions() -> list:
    """Return all translation technique functions."""
    return [
        # High-resource
        to_french, to_japanese, to_russian, to_spanish, to_german, to_arabic,
        # Mid-resource
        to_turkish, to_czech, to_vietnamese, to_greek, to_croatian,
        # Low-resource
        to_swahili, to_thai, to_khmer, to_maori, to_nepali,
        to_zulu, to_scots_gaelic, to_bengali, to_javanese,
    ]
