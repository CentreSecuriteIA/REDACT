"""Constitution generation for constitutional classifiers.

Generates structured category hierarchies across 4 severity levels
(harmful, dual-use harmful, dual-use benign, benign).

Usage::

    from redact.constitution import ConstitutionPipeline
    from redact.llms import ModelClient

    pipeline = ConstitutionPipeline(ModelClient.create("claude-opus-4-6"))
    constitution = pipeline.run(taxonomy, num_categories=10)
"""

from ..types import ALL_ENTRY_TYPES, EntryType
from .generation import ConstitutionPipeline, read_constitution

__all__ = [
    "ALL_ENTRY_TYPES",
    "ConstitutionPipeline",
    "EntryType",
    "read_constitution",
]
