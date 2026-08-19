"""Balanced dataset splitting — re-exports from redact.dataset.split.

The generalized splitting logic now lives in redact/dataset/split.py.
This module re-exports the functions under their original names so that
existing jailbreak code continues to work.
"""

from redact.dataset.split import (
    balanced_counts,
    deterministic_balanced_assign,
)
from redact.dataset.split import (
    normalize_origin as merge_groups,
)
from redact.dataset.split import (
    split_by_functions as split_by_technique_functions,
)

__all__ = [
    "merge_groups",
    "balanced_counts",
    "deterministic_balanced_assign",
    "split_by_technique_functions",
]
