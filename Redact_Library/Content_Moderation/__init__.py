"""Content moderation dataset generation pipeline.

Usage:
    from Redact_Library.Content_Moderation import InputPipeline
    from Redact_Library.Content_Moderation.checker import (
        build_quality_checker,
        build_category_checker,
    )
    from Redact_Library.Content_Moderation.metaprompt import (
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
