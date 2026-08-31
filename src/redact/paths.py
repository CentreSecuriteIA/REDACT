"""Single source of truth for on-disk layout.

Two independent families of paths live here:

**Run outputs**, derived from a single working root — every default
file/directory the pipelines read or write, so a caller can relocate an
entire run by pointing one ``data_dir`` at a new folder (or setting
``REDACT_OUTPUT_DIR``). These take an optional ``root``; when ``None`` they
fall back to :func:`redact.get_output_dir` (env ``REDACT_OUTPUT_DIR`` →
project marker → cwd).

**Package resources** — :func:`package_dir`, :func:`prompts_dir`,
:func:`configs_dir`, :func:`taxonomy_dir`, :func:`seeds_dir`,
:func:`jailbreak_configs_dir`. These resolve relative to the installed
package, independent of the working root, and take no ``root``.
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
PARAPHRASES_INPUTS_FILENAME = "paraphrases_inputs.csv"
PARAPHRASES_OUTPUTS_FILENAME = "paraphrases_outputs.csv"
PARAPHRASED_FILENAME = "paraphrased.csv"


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


def paraphrases_inputs_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / PARAPHRASES_INPUTS_FILENAME


def paraphrases_outputs_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / PARAPHRASES_OUTPUTS_FILENAME


def paraphrased_csv(root: str | Path | None = None) -> Path:
    """The eval-mode merged paraphrase artifact (paraphrased inputs + outputs)."""
    return datasets(root) / PARAPHRASED_FILENAME


CONVERSATIONS_FILENAME = "conversations.csv"


def conversations_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / CONVERSATIONS_FILENAME


def benign_csv(root: str | Path | None = None) -> Path:
    return data_cache(root) / "benign" / "benign_samples.csv"


def constitution_dir(root: str | Path | None = None) -> Path:
    return data_cache(root) / "constitution"


def constitution_inputs_dir(root: str | Path | None = None) -> Path:
    return datasets(root) / "constitution_inputs"


def package_dir() -> Path:
    """Root of the installed ``redact`` package (``src/redact/``).

    Package data, not run output — resolved relative to this file,
    independent of the working root. Single source of truth for the
    package-root-relative lookup previously recomputed independently (with
    inconsistent ``.resolve()`` usage and per-file-tuned ``.parent`` depth)
    in ``constitution/input_generation.py``, ``dataset/loading.py``,
    ``dataset/taxonomy.py``, ``llms/prompts.py``,
    ``jailbreak/manipulation/benign.py``, ``jailbreak/directives.py``,
    ``jailbreak/spec.py``, and ``jailbreak/requests/continuation.py``.
    """
    return Path(__file__).resolve().parent


def prompts_dir() -> Path:
    """Package-bundled prompt templates (``src/redact/prompts``)."""
    return package_dir() / "prompts"


def configs_dir() -> Path:
    """Package-bundled config root (``src/redact/configs``)."""
    return package_dir() / "configs"


def llm_configs_dir() -> Path:
    """Package-bundled LLM config dir (``src/redact/configs/llm``)."""
    return configs_dir() / "llm"


def models_json() -> Path:
    """The shipped model registry (``configs/llm/models.json``)."""
    return llm_configs_dir() / "models.json"


def roles_json() -> Path:
    """The shipped role catalogue (``configs/llm/roles.json``)."""
    return llm_configs_dir() / "roles.json"


def gpu_pricing_json() -> Path:
    """The shipped GPU rate table (``configs/llm/gpu_pricing.json``).

    Provider -> GPU name -> hourly USD, used by ``telemetry`` to price local
    engine time. Data, meant to be edited.
    """
    return llm_configs_dir() / "gpu_pricing.json"


def vram_cache_json(root: str | Path | None = None) -> Path:
    """Measured VRAM footprints (``Data_cache/vram.json``).

    Written after a real local load and read by the residency planner, so a
    second run plans from what a checkpoint actually took rather than an
    estimate. Under ``Data_cache/`` because it is an intermediate artifact, not
    dataset output.
    """
    return data_cache(root) / "vram.json"


def taxonomy_dir() -> Path:
    """Package-bundled taxonomy config dir (``src/redact/configs/taxonomy``)."""
    return configs_dir() / "taxonomy"


def seeds_dir() -> Path:
    """Package-bundled seed prompts (``src/redact/configs/seeds``)."""
    return configs_dir() / "seeds"


def jailbreak_configs_dir() -> Path:
    """Package-bundled jailbreak technique configs (``src/redact/configs/jailbreak``)."""
    return configs_dir() / "jailbreak"
