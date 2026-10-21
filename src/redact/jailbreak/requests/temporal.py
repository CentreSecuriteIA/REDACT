"""Temporal technique: the prompt reframed in the past tense.

``info`` is ``type=past_tense;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/temporal_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="type")


@uses_pool(_FILE, "past_tense")
def to_use_past_tense(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Reframe the prompt as a historical or past event."""
    return _apply("past_tense", prompt, rng=rng)


def get_temporal_functions() -> list[Callable]:
    """Return all temporal-framing technique functions."""
    return [
        to_use_past_tense,
    ]
