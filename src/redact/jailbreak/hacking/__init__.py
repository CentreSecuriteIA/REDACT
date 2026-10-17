"""Hacking techniques, one module per spec family: cognitive, persona and framing."""

from collections.abc import Callable

from .cognitive import get_cognitive_functions
from .framing import get_framing_functions
from .personas import get_persona_functions


def get_all_hacking_functions() -> list[Callable]:
    """Return flat list of all hacking technique functions."""
    return get_cognitive_functions() + get_persona_functions() + get_framing_functions()
