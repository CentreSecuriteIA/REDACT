"""REDACT — Red-team Dataset Automation & Construction Toolkit.

A modular, extensible pipeline for automated red teaming dataset generation.
Designed for content moderation and jailbreak dataset construction.

Subpackages:
    LLMs              — Model-agnostic LLM abstraction layer
    Content_Moderation — Content moderation input/output generation
    Jailbreak          — Jailbreak technique library
    Dataset_Functions  — Data handling utilities (I/O, merge, split, taxonomy)

On import, this module:
    1. Attempts to load a .env file (via python-dotenv if available)
    2. Sets the default random seed for reproducibility
    3. Makes all subpackages importable
"""

import os
import random
from pathlib import Path

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def _load_env() -> bool:
    """Load .env file if python-dotenv is available.

    Searches for .env in:
      1. The Redact_Library package directory
      2. The project root (one level up from the package)

    Returns True if a .env file was loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    pkg_dir = Path(__file__).parent
    for candidate in [pkg_dir / ".env", pkg_dir.parent / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)
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

        from Redact_Library import Config

        Config.validate()  # raises if required keys are missing
        backend = APIBackend(api_key=Config.get("VENICE_API_KEY"), ...)
    """

    # Known environment variable names
    VENICE_API_KEY = "VENICE_API_KEY"
    HF_TOKEN = "HF_TOKEN"

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

from . import LLMs  # noqa: E402, F401
from . import Content_Moderation  # noqa: E402, F401
from . import Jailbreak  # noqa: E402, F401
from . import Dataset_Functions  # noqa: E402, F401

# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------

from .pipelines import (  # noqa: E402
    create_taxonomy,
    generate_inputs,
    generate_outputs,
    generate_jailbreaks,
    build_dataset,
)
