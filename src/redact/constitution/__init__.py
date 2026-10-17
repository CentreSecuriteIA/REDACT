"""The constitution stage: per taxonomy category, a hierarchy of constitution
categories and sample descriptions at four severities, which seeds the inputs.

Usage::

    from redact.constitution import ConstitutionPipeline
    from redact.dataset import load_taxonomy
    from redact.llms import ModelClient

    pipeline = ConstitutionPipeline(ModelClient.create("claude-opus-4-6"))
    constitution = pipeline.run(load_taxonomy("harmful"))
"""

from ..types import ALL_ENTRY_TYPES, EntryType
from .generation import ConstitutionPipeline, read_constitution

__all__ = [
    "ALL_ENTRY_TYPES",
    "ConstitutionPipeline",
    "EntryType",
    "read_constitution",
]
