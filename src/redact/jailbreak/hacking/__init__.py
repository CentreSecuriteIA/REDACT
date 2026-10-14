"""Cognitive/psychological hacking jailbreak techniques.

Five cognitive techniques plus 15 named persona variants,
all using a two-step scenario -> meta-prompt pipeline.
"""

from collections.abc import Callable

from .cognitive import get_hacking_functions
from .framing import get_framing_functions
from .personas import get_persona_functions


def get_all_hacking_functions() -> list[Callable]:
    """Return flat list of all hacking technique functions."""
    return get_hacking_functions() + get_persona_functions() + get_framing_functions()
