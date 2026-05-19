"""Typo-rewriting obfuscation via LLM.

Four density levels (low → insane). Each function asks the LLM to reintroduce
character-level typos into the prompt while preserving meaning and word choice.
This is an LLM-dependent technique — the model is better than rule-based
approaches at producing natural-looking typos that don't distort meaning.

All functions return tuple[str, str]: (rewritten_prompt, additional_info).
additional_info format: "level={level}"
"""

from pathlib import Path

from redact.llms.base import LLMBackend
from redact.llms.calls import generate_sample
from redact.llms.prompts import load_prompt, build_messages
from redact.llms.wrappers import RateLimiter


# ---------------------------------------------------------------------------
# Level descriptions injected into the prompt template
# ---------------------------------------------------------------------------

_LEVEL_DESCRIPTIONS = {
    "low":    "a small number of minor typos (roughly 1 per sentence) — occasional single-character swaps or missing letters, text remains easily readable",
    "medium": "a moderate number of typos throughout (roughly 2-3 per sentence) — several words altered, meaning still clear but noticeably error-prone",
    "high":   "a high density of typos (most sentences have multiple errors) — many words misspelled, text is effortful to read but meaning recoverable",
    "insane": "an extreme density of typos throughout every word — heavy character substitutions, missing and doubled letters everywhere, barely legible",
}


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply_typos(
    prompt: str,
    level: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Rewrite prompt with LLM-introduced typos at the specified density level."""
    config = load_prompt("jailbreak", "rewrite_with_typos", prompt_dir)
    messages = build_messages(
        config,
        prompt=prompt,
        level_description=_LEVEL_DESCRIPTIONS[level],
    )
    result = generate_sample(backend, model, messages, rate_limiter)
    return result.strip(), f"level={level}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_rewrite_with_typos_low(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Rewrite with a low density of typos (~1 per sentence)."""
    return _apply_typos(prompt, "low", backend, model, rate_limiter)


def to_rewrite_with_typos_medium(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Rewrite with a medium density of typos (~2-3 per sentence)."""
    return _apply_typos(prompt, "medium", backend, model, rate_limiter)


def to_rewrite_with_typos_high(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Rewrite with a high density of typos (most sentences have multiple errors)."""
    return _apply_typos(prompt, "high", backend, model, rate_limiter)


def to_rewrite_with_typos_insane(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """Rewrite with an extreme density of typos (barely legible)."""
    return _apply_typos(prompt, "insane", backend, model, rate_limiter)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_typos_functions() -> list:
    """Return all typo-rewriting technique functions."""
    return [
        to_rewrite_with_typos_low,
        to_rewrite_with_typos_medium,
        to_rewrite_with_typos_high,
        to_rewrite_with_typos_insane,
    ]
