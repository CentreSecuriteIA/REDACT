"""Low-resource language translation obfuscation.

Delegates all translation + checking + retry logic to
LLMs/translator.py::translate_with_check(). Each per-language function
accepts backend/model/rate_limiter as parameters.

Ported from reference obfuscation.py lines 443-501.
"""

from redact.llms.base import LLMBackend
from redact.llms.wrappers import RateLimiter
from redact.llms.translator import translate_with_check, DEFAULT_TRANSLATE_MODEL

LANGUAGES = ["Zulu", "Scots Gaelic", "Bengali", "Swahili", "Thai", "Javanese"]


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


def to_javanese(
    prompt: str, backend: LLMBackend,
    gen_model: str | None = None, check_model: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Translate to Javanese."""
    return _translate(prompt, "Javanese", backend, gen_model, check_model, rate_limiter)


def get_translation_functions() -> list:
    """Return all low-resource language technique functions."""
    return [to_zulu, to_scots_gaelic, to_bengali, to_swahili, to_thai, to_javanese]
