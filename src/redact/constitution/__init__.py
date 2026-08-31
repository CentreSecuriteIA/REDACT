"""Constitution generation and input expansion for constitutional classifiers.

Two pipelines:
    1. **generation** — Generates structured category hierarchies across 4
       severity levels (harmful, dual-use harmful, dual-use benign, benign).
    2. **input_generation** — Expands constitution entries into full realistic
       prompts using the content moderation InputPipeline.

Usage::

    from redact.constitution import ConstitutionPipeline, EntryType
    from redact.constitution import ConstitutionInputPipeline
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
from .input_generation import (
    ConstitutionInputPipeline,
    ConstitutionInputResult,
    get_available_styles,
)

__all__ = [
    # Constitution generation
    "EntryType",
    "ALL_ENTRY_TYPES",
    "ConstitutionEntry",
    "ConstitutionResult",
    "ConstitutionPipeline",
    # Constitution-to-input generation
    "ConstitutionInputPipeline",
    "ConstitutionInputResult",
    "get_available_styles",
]
