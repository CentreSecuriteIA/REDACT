"""Indirect techniques: the prompt embedded in a task such as a summary.

``info`` is ``task=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/indirect_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="task")


@uses_pool(_FILE, "summarize")
def to_summarize(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Embed prompt in a summarization task."""
    return _apply("summarize", prompt, rng=rng)


@uses_pool(_FILE, "proofread")
def to_proofread(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Embed prompt in a proofreading/editing task."""
    return _apply("proofread", prompt, rng=rng)


@uses_pool(_FILE, "critique")
def to_critique(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Embed prompt in a critical analysis or fact-checking task."""
    return _apply("critique", prompt, rng=rng)


@uses_pool(_FILE, "translate_task")
def to_translate_task(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Embed prompt in a translation or rephrasing task."""
    return _apply("translate_task", prompt, rng=rng)


@uses_pool(_FILE, "explain_to_child")
def to_explain_to_child(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Embed prompt in a 'explain simply' pedagogical task."""
    return _apply("explain_to_child", prompt, rng=rng)


@uses_pool(_FILE, "rewrite")
def to_rewrite(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Embed prompt in a rewriting task."""
    return _apply("rewrite", prompt, rng=rng)


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
