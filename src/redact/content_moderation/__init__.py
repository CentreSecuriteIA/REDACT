"""Content moderation stages: inputs, outputs and paraphrases, with their checkers.

The seeded inputs stage is ``InputPipeline.run_from_constitution``; the
deprecated standalone path is :func:`run_standalone_inputs`. The fronts in
:mod:`redact.stages` resolve models and files and call these.

Usage::

    from redact import paths
    from redact.content_moderation import run_output_generation
    from redact.llms import ModelClient

    client = ModelClient.create("venice-uncensored")
    rows = run_output_generation(
        inputs, client, check_outputs=True, check=client, batch_size=32,
        out_path=paths.output_responses_csv("./runs/eval"),
    )
"""

from .checker import (
    build_output_quality_checker,
    build_paraphrase_checker,
    build_quality_checker,
)
from .inputs import InputPipeline
from .metaprompt import (
    generate_category_description,
    generate_seeds,
)
from .outputs import run_output_generation
from .paraphrase import paraphrase_batch
from .results import (
    CategoryResult,
    ConstitutionInputResult,
    SampleResult,
    TurnResult,
)
from .standalone import run_standalone_inputs

__all__ = [
    "CategoryResult",
    "ConstitutionInputResult",
    "InputPipeline",
    "SampleResult",
    "TurnResult",
    "build_output_quality_checker",
    "build_paraphrase_checker",
    "build_quality_checker",
    "generate_category_description",
    "generate_seeds",
    "paraphrase_batch",
    "run_output_generation",
    "run_standalone_inputs",
]
