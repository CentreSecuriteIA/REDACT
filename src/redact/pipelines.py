"""Re-exports of :mod:`redact.stages`, kept for one release.

Importing this module warns (``DeprecationWarning``): import the fronts from
:mod:`redact`, and ``RunContext`` and ``Stage`` from :mod:`redact.stages`.
"""

import warnings

from redact.stages import (
    STAGES,
    RunContext,
    Stage,
    build_dataset,
    generate_constitution,
    generate_inputs,
    generate_jailbreaks,
    generate_outputs,
    generate_paraphrases,
)

warnings.warn(
    "redact.pipelines is now redact.stages: import the fronts from redact, and "
    "RunContext and Stage from redact.stages. The former module is read for "
    "this release.", DeprecationWarning, stacklevel=2)

__all__ = [
    "STAGES",
    "RunContext",
    "Stage",
    "build_dataset",
    "generate_constitution",
    "generate_inputs",
    "generate_jailbreaks",
    "generate_outputs",
    "generate_paraphrases",
]
