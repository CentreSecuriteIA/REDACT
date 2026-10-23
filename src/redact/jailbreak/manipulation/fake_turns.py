"""Fake-turn techniques: the prompt after an invented turn of the
conversation, a random variant of a template of
``configs/jailbreak/manipulation/fake_turns_templates.json``: an invented
system turn (``to_fake_system_turn``), an invented assistant turn
(``to_fake_assistant_turn``) or an invented earlier exchange
(``to_fake_history``). Pure, and unlike fsh and dap they need no benign data.

``info`` is ``variant=<variant>``: the technique names its directive.
"""

from collections.abc import Callable
from random import Random

from ..directives import draw_variant, read_pools, uses_pool

_FILE = "manipulation/fake_turns_templates.json"
_TEMPLATES: dict[str, list[dict]] = read_pools(_FILE)


def _apply(directive: str, prompt: str, rng: Random | None) -> tuple[str, str]:
    """``prompt`` in a variant of ``directive`` drawn by ``rng``
    (:func:`~redact.jailbreak.directives.draw_variant`), and its info."""
    variant = draw_variant(_TEMPLATES[directive], rng)
    text = variant["template"].format_map({"prompt": prompt})
    return text, f"variant={variant['name']}"


@uses_pool(_FILE, "fake_system_turn")
def to_fake_system_turn(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Put an invented system turn before the prompt."""
    return _apply("fake_system_turn", prompt, rng)


@uses_pool(_FILE, "fake_assistant_turn")
def to_fake_assistant_turn(
    prompt: str, *, rng: Random | None = None,
) -> tuple[str, str]:
    """Pair the prompt with an invented assistant turn."""
    return _apply("fake_assistant_turn", prompt, rng)


@uses_pool(_FILE, "fake_history")
def to_fake_history(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
    """Put an invented earlier exchange before the prompt."""
    return _apply("fake_history", prompt, rng)


def get_fake_turn_functions() -> list[Callable]:
    """Return the fake-turn techniques: a system turn, an assistant turn, a
    history."""
    return [to_fake_system_turn, to_fake_assistant_turn, to_fake_history]
