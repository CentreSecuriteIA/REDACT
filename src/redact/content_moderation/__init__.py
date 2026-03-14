"""Content moderation dataset generation pipeline.

Usage:
    from redact.content_moderation import InputPipeline
    from redact.content_moderation.checker import (
        build_quality_checker,
        build_category_checker,
    )
    from redact.content_moderation.metaprompt import (
        generate_instruction_prompt,
    )
"""

from .generation import InputPipeline, SampleResult, TurnResult, CategoryResult
from .checker import build_quality_checker, build_category_checker
from .metaprompt import (
    generate_category_description,
    generate_seeds,
    generate_abstract_seeds,
    generate_instruction_prompt,  # legacy alias
)
