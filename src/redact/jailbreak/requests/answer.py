"""Answer-format techniques: a random variant of an answer-format template.

``info`` is ``directive=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/answer_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="directive")


@uses_pool(_FILE, "refusal_then_comply")
def to_refusal_then_comply(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Instruct the model to acknowledge sensitivity and then answer fully."""
    return _apply("refusal_then_comply", prompt, rng=rng)


@uses_pool(_FILE, "apology_prevention")
def to_apology_prevention(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Append a directive preventing refusals, apologies, and disclaimers."""
    return _apply("apology_prevention", prompt, rng=rng)


@uses_pool(_FILE, "prefix_injection")
def to_prefix_injection(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Instruct the model to begin its response with a compliance prefix."""
    return _apply("prefix_injection", prompt, rng=rng)


@uses_pool(_FILE, "answer_in_markdown")
def to_answer_in_markdown(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Request structured Markdown output format."""
    return _apply("answer_in_markdown", prompt, rng=rng)


@uses_pool(_FILE, "answer_as_tutorial")
def to_answer_as_tutorial(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Request step-by-step tutorial output format."""
    return _apply("answer_as_tutorial", prompt, rng=rng)


@uses_pool(_FILE, "answer_as_code")
def to_answer_as_code(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Request answer as executable code (Python, bash, pseudocode, or function)."""
    return _apply("answer_as_code", prompt, rng=rng)


@uses_pool(_FILE, "answer_as_dialogue")
def to_answer_as_dialogue(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Request answer as a dialogue, interview, debate, or Q&A transcript."""
    return _apply("answer_as_dialogue", prompt, rng=rng)


@uses_pool(_FILE, "answer_as_story")
def to_answer_as_story(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Request answer embedded in a short story or narrative."""
    return _apply("answer_as_story", prompt, rng=rng)


@uses_pool(_FILE, "answer_as_structured_data")
def to_answer_as_structured_data(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Request answer as JSON, XML, YAML, or a structured table."""
    return _apply("answer_as_structured_data", prompt, rng=rng)


def get_answer_functions() -> list[Callable]:
    """Return all answer-format and output-conditioning directive functions."""
    return [
        to_refusal_then_comply,
        to_apology_prevention,
        to_prefix_injection,
        to_answer_in_markdown,
        to_answer_as_tutorial,
        to_answer_as_code,
        to_answer_as_dialogue,
        to_answer_as_story,
        to_answer_as_structured_data,
    ]
