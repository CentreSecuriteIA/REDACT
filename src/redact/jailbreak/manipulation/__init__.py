"""Manipulation-based jailbreak techniques: FSH and DAP.

- FSH (Few-Shot Hacking): benign Q&A pairs before harmful question
- DAP (Distract and Persuade): harmful question hidden among benign pairs
"""

from collections.abc import Callable

from .benign import get_or_generate_benign_data
from .dap import get_dap_functions
from .fsh import get_fsh_functions


def get_all_manipulation_functions() -> list[Callable]:
    """Return flat list of all manipulation technique functions."""
    return get_fsh_functions() + get_dap_functions()
