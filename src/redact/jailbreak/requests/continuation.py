"""Continuation techniques: the prompt between a random variant's prefix and suffix.

``info`` is ``style=<variant>``: the key the other template techniques write
as ``variant``. It stays ``style`` while the eval pool, which draws these
techniques, is pinned; the variants table reads it as their variant.
"""

import random
from collections.abc import Callable

from ..directives import draw_variant, read_pools, uses_pool

_FILE = "requests/continuation_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)


def _apply(
    directive: str, prompt: str, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Draw a variant (:func:`~redact.jailbreak.directives.draw_variant`) and
    wrap the prompt with its prefix and suffix."""
    t = draw_variant(_TEMPLATES[directive], rng)
    result = t["prefix"] + prompt + t["suffix"]
    return result, f"style={t['name']}"


@uses_pool(_FILE, "continue_passage", slots=(), info_key="style")
def to_continue_passage(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Frame the prompt as an interrupted passage the model must continue."""
    return _apply("continue_passage", prompt, rng=rng)


@uses_pool(_FILE, "complete_story", slots=(), info_key="style")
def to_complete_story(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Embed the prompt in a story that the model must complete in full detail."""
    return _apply("complete_story", prompt, rng=rng)


@uses_pool(_FILE, "fill_in_blank", slots=(), info_key="style")
def to_fill_in_blank(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Append a fill-in-the-blank directive after the prompt."""
    return _apply("fill_in_blank", prompt, rng=rng)


@uses_pool(_FILE, "continue_instructions", slots=(), info_key="style")
def to_continue_instructions(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Frame the prompt as cut-off numbered instructions the model must finish."""
    return _apply("continue_instructions", prompt, rng=rng)


def get_continuation_functions() -> list[Callable]:
    """Return all continuation-attack technique functions."""
    return [
        to_continue_passage,
        to_complete_story,
        to_fill_in_blank,
        to_continue_instructions,
    ]
