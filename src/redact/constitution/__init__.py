"""Constitution generation for constitutional classifiers.

Generates structured category hierarchies across 4 severity levels
(harmful, dual-use harmful, dual-use benign, benign).

Usage::

    from redact.constitution import ConstitutionPipeline
    from redact.llms import ModelClient

    pipeline = ConstitutionPipeline(ModelClient.create("claude-opus-4-6"))
    result = pipeline.run(taxonomy, num_categories=10)
"""

from ..types import ALL_ENTRY_TYPES, EntryType
from .generation import (
    ConstitutionEntry,
    ConstitutionPipeline,
    ConstitutionResult,
)

__all__ = [
    "EntryType",
    "ALL_ENTRY_TYPES",
    "ConstitutionEntry",
    "ConstitutionResult",
    "ConstitutionPipeline",
]
