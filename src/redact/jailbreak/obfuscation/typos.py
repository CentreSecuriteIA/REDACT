"""Typo techniques: a model rewrites the prompt with typos at one of four
densities, and a check model reads the rewrite against the prompt.

``info`` is ``level=<level>``; a rewrite the check rejects every time is
``DISCARDED; reason=typo check failed; level=<level>; feedback=<verdict>``.
"""

from collections.abc import Callable

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import CHECK_TEMPERATURE, checked
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt


def _apply_typos_gen(
    prompt: str,
    level: str,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = 2,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Have ``gen_model`` rewrite ``prompt`` with typos at ``level``, and
    ``check_model`` (``gen_model`` when None) check the rewrite against it
    (``jailbreak/typo_check``, at :data:`CHECK_TEMPERATURE`).

    A rejected rewrite is made again with the verdict as feedback, up to
    ``check_attempts`` rewrites, so one application makes 2 to
    ``2 * check_attempts`` calls. Returns ``(rewrite, "level=<level>")``,
    or the last rewrite with a ``DISCARDED`` info when every one is rejected.
    """
    config = load_prompt("jailbreak", "rewrite_with_typos", prompt_dir)
    rewrite = build_messages(
        config,
        prompt=prompt,
        level_description=config["level_descriptions"][level],
    )
    check = PromptTemplate.load("jailbreak", "typo_check", prompt_dir=prompt_dir)
    text, accepted, verdict = yield from checked(
        gen_model, lambda feedback: with_feedback(rewrite, feedback),
        check_model or gen_model,
        lambda original, typos: check(original=original, typo_version=typos),
        original=prompt, max_attempts=check_attempts,
        temperatures=(None, CHECK_TEMPERATURE),
    )
    if not accepted:
        return text.strip(), (f"{DISCARDED}; reason=typo check failed; "
                              f"level={level}; feedback={verdict}")
    return text.strip(), f"level={level}"


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
