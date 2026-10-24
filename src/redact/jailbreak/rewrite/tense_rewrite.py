"""Tense-rewrite techniques: a model rewrites the whole prompt in the past or
the future tense (``jailbreak/tense_rewrite``, its ``{tense}``), and a check
model reads the rewrite against the prompt
(:func:`~redact.jailbreak.rewrite.preserve.preserved_rewrite_gen`).

The model rewrites the grammar; the pure ``to_use_past_tense`` and
``to_use_future_tense`` wrap the request in a sentence instead. ``info`` is
``tense=<tense>``.
"""

from collections.abc import Callable

from redact.jailbreak.technique import TechniqueGen
from redact.llms.prompting import build_messages, load_prompt

from .preserve import preserved_rewrite_gen


def _tense_gen(
    prompt: str,
    tense: str,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = 2,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Have ``gen_model`` rewrite ``prompt`` in ``tense``, and ``check_model``
    check the rewrite (:func:`preserved_rewrite_gen`: 2 to
    ``2 * check_attempts`` calls)."""
    messages = build_messages(
        load_prompt("jailbreak", "tense_rewrite", prompt_dir),
        tense=tense, prompt=prompt)
    return (yield from preserved_rewrite_gen(
        prompt, messages, f"tense={tense}", gen_model=gen_model,
        check_model=check_model, check_attempts=check_attempts,
        prompt_dir=prompt_dir))


def to_rewrite_in_past_tense(prompt: str, **kwargs) -> TechniqueGen:
    """Rewrite the prompt in the past tense."""
    return (yield from _tense_gen(prompt, "past", **kwargs))


def to_rewrite_in_future_tense(prompt: str, **kwargs) -> TechniqueGen:
    """Rewrite the prompt in the future tense."""
    return (yield from _tense_gen(prompt, "future", **kwargs))


def get_tense_rewrite_functions() -> list[Callable]:
    """Return the tense-rewrite techniques."""
    return [to_rewrite_in_past_tense, to_rewrite_in_future_tense]
