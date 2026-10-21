"""Rewrite techniques: each rewrites the whole prompt, before every other
layer applies (the spec's first layer).

The layer holds no technique yet: its techniques come with their prompts.
It is off by default (the spec's ``layers_off_by_default``): a kind of the
augmentation section draws it only when its ``layers.allow`` names it.
"""

from collections.abc import Callable


def get_all_rewrite_functions() -> list[Callable]:
    """Return flat list of all rewrite technique functions (none yet)."""
    return []
