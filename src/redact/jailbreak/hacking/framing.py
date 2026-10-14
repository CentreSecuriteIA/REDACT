"""Framing techniques: the prompt in a random variant of a framing template.

``info`` is ``directive=<name>;variant=<variant>``.
"""

from collections.abc import Callable
from functools import partial

from ..directives import apply_template_directive, load_templates

_TEMPLATES: dict[str, list[dict]] = load_templates("hacking", "framing_templates.json")
_apply = partial(apply_template_directive, _TEMPLATES, info_key="directive")


def to_fictional_world(prompt: str) -> tuple[str, str]:
    """Wrap prompt in a fictional/alternate-reality framing."""
    return _apply("fictional_world", prompt)


def to_noble_goal(prompt: str) -> tuple[str, str]:
    """Prepend a benign/altruistic justification to the prompt."""
    return _apply("noble_goal", prompt)


def to_nefarious_goal(prompt: str) -> tuple[str, str]:
    """Frame prompt as coming from a researcher/tester with explicit harmful context."""
    return _apply("nefarious_goal", prompt)


def to_high_stake(prompt: str) -> tuple[str, str]:
    """Add urgency/lives-at-stake pressure framing to the prompt."""
    return _apply("high_stake", prompt)


def to_no_moral_constraints(prompt: str) -> tuple[str, str]:
    """Append a direct constraint-removal directive to the prompt."""
    return _apply("no_moral_constraints", prompt)


def get_framing_functions() -> list[Callable]:
    """Return scenario-modifying framing directive functions."""
    return [
        to_fictional_world,
        to_noble_goal,
        to_nefarious_goal,
        to_high_stake,
        to_no_moral_constraints,
    ]
