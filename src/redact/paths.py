"""Single source of truth for on-disk layout.

Every default file/directory the pipelines read or write is derived here from a
single working root, so a caller can relocate an entire run by pointing one
``data_dir`` at a new folder (or setting ``REDACT_OUTPUT_DIR``). Previously the
same ``Datasets/`` and ``Data_cache/...`` paths were recomputed independently in
``pipelines.py``, ``dataset/io.py``, ``jailbreak/manipulation/benign.py`` and the
two constitution modules; those now all delegate here.

All functions take an optional ``root``; when ``None`` they fall back to
:func:`redact.get_output_dir` (env ``REDACT_OUTPUT_DIR`` → project marker → cwd).
"""

from pathlib import Path

from redact import get_output_dir

# ---------------------------------------------------------------------------
# Layout constants — the library-defined folder/file names live in one place.
# ---------------------------------------------------------------------------

DATASETS_DIRNAME = "Datasets"
DATA_CACHE_DIRNAME = "Data_cache"
SAMPLES_FILENAME = "samples.csv"

JAILBREAKS_FILENAME = "jailbreaks.csv"
OUTPUT_RESPONSES_FILENAME = "output_responses.csv"
COMPLETE_DATASET_FILENAME = "complete_dataset.csv"


def _root(root: str | Path | None = None) -> Path:
    """Resolve the working root — explicit ``root`` or ``get_output_dir()``."""
    return Path(root) if root is not None else get_output_dir()


# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------

def datasets(root: str | Path | None = None) -> Path:
    """The ``Datasets/`` root (per-category CSVs + merged handoff files)."""
    return _root(root) / DATASETS_DIRNAME


def data_cache(root: str | Path | None = None) -> Path:
    """The ``Data_cache/`` root (benign, constitution, scenarios, …)."""
    return _root(root) / DATA_CACHE_DIRNAME


# ---------------------------------------------------------------------------
# Per-artifact paths
# ---------------------------------------------------------------------------

def category_csv(
    category: str,
    base: str | Path | None = None,
    filename: str = SAMPLES_FILENAME,
) -> Path:
    """Per-category CSV path: ``{base or Datasets}/{category}/{filename}``."""
    base_dir = Path(base) if base is not None else datasets()
    return base_dir / category / filename


def jailbreaks_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / JAILBREAKS_FILENAME


def output_responses_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / OUTPUT_RESPONSES_FILENAME


def complete_dataset_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / COMPLETE_DATASET_FILENAME


def benign_csv(root: str | Path | None = None) -> Path:
    return data_cache(root) / "benign" / "benign_samples.csv"


def constitution_dir(root: str | Path | None = None) -> Path:
    return data_cache(root) / "constitution"


def constitution_inputs_dir(root: str | Path | None = None) -> Path:
    return datasets(root) / "constitution_inputs"


def taxonomy_dir() -> Path:
    """Package-bundled taxonomy config dir (``src/redact/configs/taxonomy``).

    Package data, not run output — resolved relative to this file, independent
    of the working root.
    """
    return Path(__file__).resolve().parent / "configs" / "taxonomy"
