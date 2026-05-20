"""Typo-rewriting obfuscation via LLM.

Four density levels (low → insane). Each function asks the LLM to reintroduce
character-level typos into the prompt while preserving meaning and word choice.
This is an LLM-dependent technique — the model is better than rule-based
approaches at producing natural-looking typos that don't distort meaning.

Each function is a single-round **technique generator** (see
``jailbreak/protocol.py``): it yields one :class:`LLMRequest` and returns
``(rewritten_prompt, "level=<level>")``.
"""

from redact.jailbreak.protocol import LLMRequest, TechniqueGen
from redact.llms.prompts import load_prompt, build_messages


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


def _apply_typos_gen(
    prompt: str,
    level: str,
    *,
    gen_model: str | None = None,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Yield one rewrite request; return ``(rewritten, "level=<level>")``."""
    config = load_prompt("jailbreak", "rewrite_with_typos", prompt_dir)
    messages = build_messages(
        config,
        prompt=prompt,
        level_description=_LEVEL_DESCRIPTIONS[level],
    )
    result = yield LLMRequest(gen_model, messages)
    return result.strip(), f"level={level}"


# ---------------------------------------------------------------------------
# Public functions (technique generators)
# ---------------------------------------------------------------------------


def to_rewrite_with_typos_low(prompt: str, **kwargs) -> TechniqueGen:
    """Rewrite with a low density of typos (~1 per sentence)."""
    return (yield from _apply_typos_gen(prompt, "low", **kwargs))


def to_rewrite_with_typos_medium(prompt: str, **kwargs) -> TechniqueGen:
    """Rewrite with a medium density of typos (~2-3 per sentence)."""
    return (yield from _apply_typos_gen(prompt, "medium", **kwargs))


def to_rewrite_with_typos_high(prompt: str, **kwargs) -> TechniqueGen:
    """Rewrite with a high density of typos (most sentences have multiple errors)."""
    return (yield from _apply_typos_gen(prompt, "high", **kwargs))


def to_rewrite_with_typos_insane(prompt: str, **kwargs) -> TechniqueGen:
    """Rewrite with an extreme density of typos (barely legible)."""
    return (yield from _apply_typos_gen(prompt, "insane", **kwargs))


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
