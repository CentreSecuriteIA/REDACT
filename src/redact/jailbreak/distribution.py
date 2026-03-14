"""Balanced dataset splitting — re-exports from Dataset_Functions.

The generalized splitting logic now lives in Dataset_Functions/split.py.
This module re-exports the functions under their original names so that
existing jailbreak code continues to work.
"""

from Redact_Library.Dataset_Functions.split import (
    normalize_origin as merge_groups,
    balanced_counts,
    deterministic_balanced_assign,
    split_by_functions as split_by_technique_functions,
)

__all__ = [
    "merge_groups",
    "balanced_counts",
    "deterministic_balanced_assign",
    "split_by_technique_functions",
]
