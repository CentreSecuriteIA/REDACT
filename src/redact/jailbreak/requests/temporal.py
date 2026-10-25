"""Temporal techniques: the prompt reframed in time: in the past tense, in
the future tense (spec 1.3) or in a historical era (spec 1.3).

``info`` is ``type=<name>;variant=<variant>``.
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


@uses_pool(_FILE, "future_tense")
def to_use_future_tense(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Reframe the prompt as a future event."""
    return _apply("future_tense", prompt, rng=rng)


@uses_pool(_FILE, "historical_era")
def to_use_historical_era(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Set the prompt in a historical era."""
    return _apply("historical_era", prompt, rng=rng)


def get_temporal_functions() -> list[Callable]:
    """Return all temporal-framing technique functions."""
    return [
        to_use_past_tense,
        to_use_future_tense,
        to_use_historical_era,
    ]
