"""Distractor techniques: related or unrelated text before or after the prompt.

``info`` is ``type=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/distractor_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="type")


@uses_pool(_FILE, "prefix_related")
def to_distractor_prefix_related(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Prepend a topically adjacent distractor sentence before the prompt."""
    return _apply("prefix_related", prompt, rng=rng)


@uses_pool(_FILE, "prefix_unrelated")
def to_distractor_prefix_unrelated(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Prepend an off-topic distractor sentence before the prompt."""
    return _apply("prefix_unrelated", prompt, rng=rng)


@uses_pool(_FILE, "suffix_related")
def to_distractor_suffix_related(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Append a topically adjacent distractor sentence after the prompt."""
    return _apply("suffix_related", prompt, rng=rng)


@uses_pool(_FILE, "suffix_unrelated")
def to_distractor_suffix_unrelated(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Append an off-topic distractor sentence after the prompt."""
    return _apply("suffix_unrelated", prompt, rng=rng)


def get_distractor_functions() -> list[Callable]:
    """Return all distractor technique functions."""
    return [
        to_distractor_prefix_related,
        to_distractor_prefix_unrelated,
        to_distractor_suffix_related,
        to_distractor_suffix_unrelated,
    ]
