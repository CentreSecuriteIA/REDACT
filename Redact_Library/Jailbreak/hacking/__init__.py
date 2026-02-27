"""Cognitive/psychological hacking jailbreak techniques.

Five techniques using a two-step scenario -> meta-prompt pipeline.
"""

from .cognitive import (
    get_hacking_functions,
    HACKING_CATEGORIES,
    CATEGORY_NAMES,
    get_situation,
    create_jailbreak,
)


def get_hacking_types() -> list[str]:
    """Return list of hacking type names."""
    return ["cognitive_psychological"]


def get_hacking_type_to_getter() -> dict[str, callable]:
    """Return mapping of hacking type name to its getter function."""
    return {"cognitive_psychological": get_hacking_functions}
