"""Shared type definitions for REDACT.

This module hosts types that are referenced by multiple subpackages.
Keep it dependency-free — nothing in this file should import from any
other ``redact.*`` module. Otherwise we reintroduce the circular-import
risk that motivated splitting these out in the first place.
"""

from enum import Enum


class EntryType(str, Enum):
    """Severity level for a generated dataset sample.

    Spectrum:
        HARMFUL <-> DUAL_USE_HARMFUL <-> DUAL_USE_BENIGN <-> BENIGN

    Originally introduced by the constitution pipeline; now shared with
    content moderation so the same checker can evaluate samples across
    the full severity spectrum.
    """

    HARMFUL = "harmful"
    DUAL_USE_HARMFUL = "dual_use_harmful"
    DUAL_USE_BENIGN = "dual_use_benign"
    BENIGN = "benign"


ALL_ENTRY_TYPES: list[EntryType] = list(EntryType)
