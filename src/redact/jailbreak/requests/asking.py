"""Asking techniques: an innocuous-question framing, or a request for details.

``info`` is ``type=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial

from ..directives import apply_template_directive, load_templates

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "asking_templates.json")
_apply = partial(apply_template_directive, _TEMPLATES, info_key="type")


def to_innocuous_question(prompt: str) -> tuple[str, str]:
    """Add a casual/academic framing prefix to make the request sound innocent."""
    return _apply("innocuous_question", prompt)


def to_ask_for_details(prompt: str) -> tuple[str, str]:
    """Append a directive requesting maximum detail and specificity."""
    return _apply("ask_for_details", prompt)


def get_asking_functions() -> list[Callable]:
    """Return all question-framing technique functions."""
    return [
        to_innocuous_question,
        to_ask_for_details,
    ]
