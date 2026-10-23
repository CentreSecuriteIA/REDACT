"""Manipulation techniques: fsh and dap (sharing ``pairs.py``) and the benign
Q&A data they draw from (``benign.py``), and the fake turns, which need
none."""

from collections.abc import Callable

from .benign import get_or_generate_benign_data
from .dap import get_dap_functions
from .fake_turns import get_fake_turn_functions
from .fsh import get_fsh_functions


def get_all_manipulation_functions() -> list[Callable]:
    """Return flat list of all manipulation technique functions; the fake
    turns, the newest family, come last."""
    return get_fsh_functions() + get_dap_functions() + get_fake_turn_functions()
