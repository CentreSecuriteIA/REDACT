"""Asking techniques: an innocuous-question framing, or a request for details.

``info`` is ``type=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/asking_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="type")


@uses_pool(_FILE, "innocuous_question")
def to_innocuous_question(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Add a casual/academic framing prefix to make the request sound innocent."""
    return _apply("innocuous_question", prompt, rng=rng)


@uses_pool(_FILE, "ask_for_details")
def to_ask_for_details(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Append a directive requesting maximum detail and specificity."""
    return _apply("ask_for_details", prompt, rng=rng)


def get_asking_functions() -> list[Callable]:
    """Return all question-framing technique functions."""
    return [
        to_innocuous_question,
        to_ask_for_details,
    ]
