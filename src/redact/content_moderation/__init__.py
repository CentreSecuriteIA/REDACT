"""Content moderation stages: inputs, outputs and paraphrases, with their checkers.

The inputs stage is ``InputPipeline.run_from_constitution``. The fronts in
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
from .outputs import run_output_generation
from .paraphrase import paraphrase_batch
from .results import ConstitutionInputResult

__all__ = [
    "ConstitutionInputResult",
    "InputPipeline",
    "build_output_quality_checker",
    "build_paraphrase_checker",
    "build_quality_checker",
    "paraphrase_batch",
    "run_output_generation",
]
