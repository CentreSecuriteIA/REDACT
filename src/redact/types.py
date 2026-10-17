"""The severity of a sample; imports nothing from the package."""

from enum import Enum


class EntryType(str, Enum):  # noqa: UP042
    """From ``HARMFUL`` through the two dual-use levels to ``BENIGN``."""

    HARMFUL = "harmful"
    DUAL_USE_HARMFUL = "dual_use_harmful"
    DUAL_USE_BENIGN = "dual_use_benign"
    BENIGN = "benign"


ALL_ENTRY_TYPES: list[EntryType] = list(EntryType)
