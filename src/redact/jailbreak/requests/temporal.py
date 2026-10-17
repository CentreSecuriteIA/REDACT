"""Temporal technique: the prompt reframed in the past tense.

``info`` is ``type=past_tense;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, load_templates

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "temporal_templates.json")
_apply = partial(apply_template_directive, _TEMPLATES, info_key="type")


def to_use_past_tense(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Reframe the prompt as a historical or past event."""
    return _apply("past_tense", prompt, rng=rng)


def get_temporal_functions() -> list[Callable]:
    """Return all temporal-framing technique functions."""
    return [
        to_use_past_tense,
    ]
