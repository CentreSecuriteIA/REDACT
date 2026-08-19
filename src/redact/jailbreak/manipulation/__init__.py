"""Manipulation-based jailbreak techniques: FSH and DAP.

- FSH (Few-Shot Hacking): benign Q&A pairs before harmful question
- DAP (Distract and Persuade): harmful question hidden among benign pairs
"""

from .benign import (
    BENIGN_CATEGORIES,
    get_or_generate_benign_data,
    load_benign_data,
    process_category,
)
from .dap import get_dap_functions
from .fsh import get_fsh_functions


def get_manipulation_types() -> list[str]:
    """Return list of manipulation type names."""
    return ["fsh", "dap"]


def get_manipulation_type_to_getter() -> dict[str, callable]:
    """Return mapping of manipulation type name to its getter function."""
    return {
        "fsh": get_fsh_functions,
        "dap": get_dap_functions,
    }


def get_all_manipulation_functions() -> list:
    """Return flat list of all manipulation technique functions."""
    return get_fsh_functions() + get_dap_functions()
