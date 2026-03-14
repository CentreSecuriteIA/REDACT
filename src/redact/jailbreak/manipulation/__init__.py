"""Manipulation-based jailbreak techniques: FSH and DAP.

- FSH (Few-Shot Hacking): benign Q&A pairs before harmful question
- DAP (Distract and Persuade): harmful question hidden among benign pairs
"""

from .fsh import get_fsh_functions
from .dap import get_dap_functions
from .benign import BENIGN_CATEGORIES, load_benign_data, process_category


def get_manipulation_types() -> list[str]:
    """Return list of manipulation type names."""
    return ["fsh", "dap"]


def get_manipulation_type_to_getter() -> dict[str, callable]:
    """Return mapping of manipulation type name to its getter function."""
    return {
        "fsh": get_fsh_functions,
        "dap": get_dap_functions,
    }
