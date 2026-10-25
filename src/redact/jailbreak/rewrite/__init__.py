"""Rewrite techniques, one module per spec family: persuasion, tense_rewrite
and model_distractors (the first two sharing ``preserve.py``). A model changes
the whole prompt, or adds sentences to it, before every other layer applies
(the spec's first layer).

The layer is off by default (the spec's ``layers_off_by_default``): a kind of
the augmentation section draws it only when its ``layers.allow`` names it.
"""

from collections.abc import Callable

from .model_distractors import get_model_distractor_functions
from .persuasion import get_persuasion_functions
from .tense_rewrite import get_tense_rewrite_functions


def get_all_rewrite_functions() -> list[Callable]:
    """Return flat list of all rewrite technique functions."""
    return (get_persuasion_functions() + get_tense_rewrite_functions()
            + get_model_distractor_functions())
