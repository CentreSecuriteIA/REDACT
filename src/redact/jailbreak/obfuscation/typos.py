"""Typo techniques: a model rewrites the prompt with typos at one of four densities.

``info`` is ``level=<level>``.
"""

from collections.abc import Callable

from redact.jailbreak.technique import TechniqueGen
from redact.llm_pipeline import LLMRequest
from redact.llms.prompting import build_messages, load_prompt


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
        level_description=config["level_descriptions"][level],
    )
    result = yield LLMRequest(gen_model, messages)
    return result.strip(), f"level={level}"


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


def get_typos_functions() -> list[Callable]:
    """Return all typo-rewriting technique functions."""
    return [
        to_rewrite_with_typos_low,
        to_rewrite_with_typos_medium,
        to_rewrite_with_typos_high,
        to_rewrite_with_typos_insane,
    ]
