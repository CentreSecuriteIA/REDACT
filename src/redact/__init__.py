"""REDACT — Red-team Dataset Automation & Construction Toolkit.

A modular, extensible pipeline for automated red teaming dataset generation.
Designed for content moderation and jailbreak dataset construction.

Subpackages:
    llms               — Model-agnostic LLM abstraction layer
    content_moderation — Content moderation input/output generation
    jailbreak          — Jailbreak technique library
    dataset            — Data handling utilities (I/O, merge, split, taxonomy)

On import, this module:
    1. Attempts to load a .env file (via python-dotenv if available)
    2. Sets the default random seed for reproducibility
    3. Makes all subpackages importable
"""

import logging
import os
import random
from pathlib import Path

__version__ = "0.1.0"

# Every module logs via logging.getLogger(__name__) under this "redact" root —
# never print(). As a library we attach no handler and set no level (that's the
# application's call); NullHandler only silences the "no handlers found"
# warning if the caller never configures logging at all. To see progress,
# configure it yourself, e.g. logging.basicConfig(level=logging.INFO), or use
# scripts/run.py, which already does this (--quiet -> WARNING, --debug -> DEBUG).
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Output directory resolution
# ---------------------------------------------------------------------------

def get_output_dir() -> Path:
    """Return the base directory for generated data (Datasets/, Data_cache/).

    Resolution order:
      1. ``REDACT_OUTPUT_DIR`` env var (if set)
      2. Walk upward from the redact package source directory until a
         ``pyproject.toml`` or ``.git`` marker is found — stable regardless
         of the caller's cwd (same approach used by pytest, ruff, black).
      3. ``Path.cwd()`` — fallback for installed wheels with no source tree.

    Override by passing explicit paths to pipeline functions, or set
    the ``REDACT_OUTPUT_DIR`` environment variable.
    """
    env = os.environ.get("REDACT_OUTPUT_DIR")
    if env:
        return Path(env).resolve()

    markers = {"pyproject.toml", ".git"}
    current = Path(__file__).resolve().parent
    while True:
        if any((current / m).exists() for m in markers):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return Path.cwd()


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def _load_env() -> bool:
    """Load .env file from the project root returned by ``get_output_dir()``.

    Uses ``override=True`` so .env values take precedence over any
    pre-existing (potentially stale) shell environment variables.

    Returns True if a .env file was loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    candidate = get_output_dir() / ".env"
    if candidate.exists():
        load_dotenv(candidate, override=True)
        return True
    return False


_env_loaded = _load_env()


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------

DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set the global random seed for reproducibility.

    Sets seeds for the ``random`` module and, if available, ``numpy``.
    Called automatically on library import with DEFAULT_SEED.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# Apply default seed on import
set_seed(DEFAULT_SEED)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

class Config:
    """Library-wide configuration loaded from environment variables.

    Keys are accessed via ``Config.get()`` or attribute-style constants.
    Values reflect the current environment (including .env values).

    Usage::

        from redact import Config

        Config.validate()  # raises if required keys are missing
        client = ModelClient.create("venice-uncensored")  # resolves its backend
    """

    # Known environment variable names
    VENICE_API_KEY = "VENICE_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    HF_TOKEN = "HF_TOKEN"
    REDACT_OUTPUT_DIR = "REDACT_OUTPUT_DIR"
    # Telemetry sink: "jsonl" (default) | "wandb" | "off". See redact.telemetry.
    REDACT_TRACE = "REDACT_TRACE"
    # Key into configs/llm/gpu_pricing.json, e.g. "runpod". Default "local" (rate 0).
    REDACT_GPU_PROVIDER = "REDACT_GPU_PROVIDER"

    @classmethod
    def get(cls, key: str, default: str | None = None) -> str | None:
        """Get an environment variable value.

        Args:
            key: Environment variable name.
            default: Fallback if not set.

        Returns:
            The value, or *default*.
        """
        return os.environ.get(key, default)

    @classmethod
    def validate(cls, require: list[str] | None = None) -> None:
        """Validate that required environment variables are set.

        Args:
            require: List of env var names to check.
                Defaults to ``["VENICE_API_KEY"]``.

        Raises:
            ValueError: If any required variable is missing.
        """
        if require is None:
            require = ["VENICE_API_KEY"]

        missing = [key for key in require if not os.environ.get(key)]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                f"Set them in your shell or in a .env file."
            )

    @classmethod
    def is_configured(cls, *keys: str) -> bool:
        """Check whether all given env vars are set (non-empty).

        Args:
            *keys: Environment variable names to check.

        Returns:
            True if all are set and non-empty.
        """
        return all(os.environ.get(k) for k in keys)


# ---------------------------------------------------------------------------
# Subpackage availability — lazy, no heavy imports on library load
# ---------------------------------------------------------------------------

# Deliberately one import per line (not merged into one `from . import (...)`
# block): each line's own noqa suppression stays attached to the line ruff
# actually reports E402 on — a merged block only lets the *opening* line
# suppress E402, silently un-suppressing it for every other name inside it.
from . import llms  # noqa: E402, F401, I001
from . import content_moderation  # noqa: E402, F401
from . import jailbreak  # noqa: E402, F401
from . import dataset  # noqa: E402, F401
from . import constitution  # noqa: E402, F401
from . import multi_turn  # noqa: E402, F401
from . import optimization  # noqa: E402, F401
from . import multiturn_attacks  # noqa: E402, F401

# Convenience re-export: the built-in 4-round escalation schedule for
# generate_jailbreaks(settings_per_iteration=...).
from .jailbreak import default_escalation_schedule  # noqa: E402, F401

# Multi-turn conversation datasets.
from .multi_turn import (  # noqa: E402, F401
    evaluate_conversations,
    generate_conversations,
)

# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------
from .pipelines import (  # noqa: E402
    build_dataset,
    create_taxonomy,
    generate_constitution,
    generate_inputs,
    generate_jailbreaks,
    generate_outputs,
    generate_paraphrases,
)

# Config-driven runs (recipe + input-params + per-stage manifests).
from .runconfig import (  # noqa: E402, F401
    load_params,
    load_recipe,
    read_manifest,
    run_pipeline,
    write_manifest,
)
from .types import ALL_ENTRY_TYPES, EntryType  # noqa: E402, F401
