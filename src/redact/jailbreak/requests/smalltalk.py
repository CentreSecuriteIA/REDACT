"""Small talk: a friendly line before the prompt or after it, the greetings,
asides, thanks and closings real users write around a request.

``to_smalltalk_prefix`` puts one of the twenty variants of its pool in
``requests/smalltalk_templates.json`` before the prompt, and
``to_smalltalk_suffix`` one of twenty after it; the prompt is kept as it is.
The lines are ordinary and harmless by design: they ask nothing of the reader
and say nothing about how to reply. The natural kind draws them, the attacks
never do (their spec ``kinds``).

``info`` is ``smalltalk=<directive>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial
from random import Random

from ..directives import apply_template_directive, read_pools, uses_pool

_FILE = "requests/smalltalk_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)
_apply = partial(apply_template_directive, _TEMPLATES, info_key="smalltalk")


@uses_pool(_FILE, "prefix")
def to_smalltalk_prefix(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Put a greeting or a friendly aside before the prompt."""
    return _apply("prefix", prompt, rng=rng)


@uses_pool(_FILE, "suffix")
def to_smalltalk_suffix(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Put a thank-you or a polite closing line after the prompt."""
    return _apply("suffix", prompt, rng=rng)


def get_smalltalk_functions() -> list[Callable]:
    """Return both small-talk technique functions."""
    return [
        to_smalltalk_prefix,
        to_smalltalk_suffix,
    ]
