"""Constitution generation for constitutional classifiers.

Generates structured category hierarchies across 4 severity levels
(harmful, dual-use harmful, dual-use benign, benign) that later seed
input sample generation for classifier training.

Usage::

    from redact.constitution import ConstitutionPipeline, EntryType
    from redact.llms import get_backend, RateLimiter

    backend = get_backend("claude-opus-4-6")
    pipeline = ConstitutionPipeline(backend, "claude-opus-4-6", RateLimiter())
    result = pipeline.run(taxonomy, num_categories=10)
"""

from .pipeline import (
    EntryType,
    ConstitutionEntry,
    ConstitutionResult,
    ConstitutionPipeline,
)

__all__ = [
    "EntryType",
    "ConstitutionEntry",
    "ConstitutionResult",
    "ConstitutionPipeline",
]
