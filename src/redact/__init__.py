"""REDACT: a taxonomy and a task in, a dataset out.

Every module logs under the ``redact`` logger, which gets a ``NullHandler``
and no level: configuring logging is the application's call
(``scripts/run.py`` does it). On import the package loads ``.env`` from
:func:`get_output_dir`, its values winning over the shell's.
"""

import logging
import random

from .paths import get_output_dir

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _load_env() -> None:
    """Load ``.env`` from :func:`get_output_dir`, over the shell's values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidate = get_output_dir() / ".env"
    if candidate.exists():
        load_dotenv(candidate, override=True)


_load_env()

DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed the global ``random`` module and, if installed, ``numpy``."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


# Imported after .env is loaded, so a library reading the environment at
# import time sees its values.
from . import (  # noqa: E402
    constitution,
    content_moderation,
    dataset,
    jailbreak,
    llm_pipeline,
    llms,
    multi_turn,
    multiturn_attacks,
    optimization,
    paths,
    residency,
    telemetry,
)
from .dataset import create_taxonomy  # noqa: E402
from .driver import (  # noqa: E402
    read_manifest,
    read_record,
    run_pipeline,
    write_manifest,
    write_record,
)
from .jailbreak import default_escalation_schedule  # noqa: E402
from .llms.prompting import scaffold_setup  # noqa: E402
from .multi_turn import evaluate_conversations, generate_conversations  # noqa: E402
from .recipe import load_params, load_recipe  # noqa: E402
from .stages import (  # noqa: E402
    STAGES,
    build_dataset,
    generate_constitution,
    generate_inputs,
    generate_jailbreaks,
    generate_outputs,
    generate_paraphrases,
)
from .types import ALL_ENTRY_TYPES, EntryType  # noqa: E402

__all__ = [
    "ALL_ENTRY_TYPES",
    "DEFAULT_SEED",
    "STAGES",
    "EntryType",
    "build_dataset",
    "constitution",
    "content_moderation",
    "create_taxonomy",
    "dataset",
    "default_escalation_schedule",
    "evaluate_conversations",
    "generate_constitution",
    "generate_conversations",
    "generate_inputs",
    "generate_jailbreaks",
    "generate_outputs",
    "generate_paraphrases",
    "get_output_dir",
    "jailbreak",
    "llm_pipeline",
    "llms",
    "load_params",
    "load_recipe",
    "multi_turn",
    "multiturn_attacks",
    "optimization",
    "paths",
    "read_manifest",
    "read_record",
    "residency",
    "run_pipeline",
    "scaffold_setup",
    "set_seed",
    "telemetry",
    "write_manifest",
    "write_record",
]
