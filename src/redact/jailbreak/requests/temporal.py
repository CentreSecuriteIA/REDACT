"""Temporal technique: the prompt reframed in the past tense.

``info`` is ``type=past_tense;variant=<variant>``.
"""

from collections.abc import Callable

from ..directives import apply_template_directive, load_templates

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "temporal_templates.json")


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    return apply_template_directive(_TEMPLATES, directive, prompt, info_key="type")


def to_use_past_tense(prompt: str) -> tuple[str, str]:
    """Reframe the prompt as a historical or past event."""
    return _apply("past_tense", prompt)


def get_temporal_functions() -> list[Callable]:
    """Return all temporal-framing technique functions."""
    return [
        to_use_past_tense,
    ]
