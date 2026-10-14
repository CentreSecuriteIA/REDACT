"""Distractor techniques: related or unrelated text before or after the prompt.

``info`` is ``type=<name>;variant=<variant>``.
"""

from collections.abc import Callable

from ..directives import apply_template_directive, load_templates

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "distractor_templates.json")


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    return apply_template_directive(_TEMPLATES, directive, prompt, info_key="type")


def to_distractor_prefix_related(prompt: str) -> tuple[str, str]:
    """Prepend a topically adjacent distractor sentence before the prompt."""
    return _apply("prefix_related", prompt)


def to_distractor_prefix_unrelated(prompt: str) -> tuple[str, str]:
    """Prepend an off-topic distractor sentence before the prompt."""
    return _apply("prefix_unrelated", prompt)


def to_distractor_suffix_related(prompt: str) -> tuple[str, str]:
    """Append a topically adjacent distractor sentence after the prompt."""
    return _apply("suffix_related", prompt)


def to_distractor_suffix_unrelated(prompt: str) -> tuple[str, str]:
    """Append an off-topic distractor sentence after the prompt."""
    return _apply("suffix_unrelated", prompt)


def get_distractor_functions() -> list[Callable]:
    """Return all distractor technique functions."""
    return [
        to_distractor_prefix_related,
        to_distractor_prefix_unrelated,
        to_distractor_suffix_related,
        to_distractor_suffix_unrelated,
    ]
