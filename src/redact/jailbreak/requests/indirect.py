"""Indirect techniques: the prompt embedded in a task such as a summary.

``info`` is ``task=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial

from ..directives import apply_template_directive, load_templates

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "indirect_templates.json")
_apply = partial(apply_template_directive, _TEMPLATES, info_key="task")


def to_summarize(prompt: str) -> tuple[str, str]:
    """Embed prompt in a summarization task."""
    return _apply("summarize", prompt)


def to_proofread(prompt: str) -> tuple[str, str]:
    """Embed prompt in a proofreading/editing task."""
    return _apply("proofread", prompt)


def to_critique(prompt: str) -> tuple[str, str]:
    """Embed prompt in a critical analysis or fact-checking task."""
    return _apply("critique", prompt)


def to_translate_task(prompt: str) -> tuple[str, str]:
    """Embed prompt in a translation or rephrasing task."""
    return _apply("translate_task", prompt)


def to_explain_to_child(prompt: str) -> tuple[str, str]:
    """Embed prompt in a 'explain simply' pedagogical task."""
    return _apply("explain_to_child", prompt)


def to_rewrite(prompt: str) -> tuple[str, str]:
    """Embed prompt in a rewriting task."""
    return _apply("rewrite", prompt)


def get_indirect_functions() -> list[Callable]:
    """Return all indirect-request technique functions."""
    return [
        to_summarize,
        to_proofread,
        to_critique,
        to_translate_task,
        to_explain_to_child,
        to_rewrite,
    ]
