"""Cognitive/psychological hacking jailbreak techniques.

Five cognitive techniques plus 15 named persona variants,
all using a two-step scenario -> meta-prompt pipeline.
"""

from .cognitive import (
    get_hacking_functions,
    HACKING_CATEGORIES,
    CATEGORY_NAMES,
    get_situation,
    create_jailbreak,
)
from .personas import (
    get_persona_functions,
    get_persona_situation,
)
from .framing import get_framing_functions


def get_hacking_types() -> list[str]:
    """Return list of hacking type names."""
    return ["cognitive_psychological", "personas", "framing"]


def get_hacking_type_to_getter() -> dict[str, callable]:
    """Return mapping of hacking type name to its getter function."""
    return {
        "cognitive_psychological": get_hacking_functions,
        "personas": get_persona_functions,
        "framing": get_framing_functions,
    }


def get_all_hacking_functions() -> list:
    """Return flat list of all hacking technique functions."""
    return get_hacking_functions() + get_persona_functions() + get_framing_functions()
