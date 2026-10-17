"""The on-disk layout: the working root and every file under it.

Run outputs derive from one working root, so a run moves by pointing one
``data_dir`` (or ``REDACT_OUTPUT_DIR``) at another folder; their functions
take ``root``, and ``None`` means :func:`get_output_dir`. Package resources
(:func:`package_dir`, :func:`prompts_dir`, :func:`configs_dir`, ...) resolve
relative to the installed package and take no root.
"""

import os
from pathlib import Path

DATASETS_DIRNAME = "Datasets"
DATA_CACHE_DIRNAME = "Data_cache"
SAMPLES_FILENAME = "samples.csv"

JAILBREAKS_FILENAME = "jailbreaks.csv"
VARIATIONS_FILENAME = "variations.csv"
OUTPUT_RESPONSES_FILENAME = "output_responses.csv"
COMPLETE_DATASET_FILENAME = "complete_dataset.csv"
PARAPHRASES_INPUTS_FILENAME = "paraphrases_inputs.csv"
PARAPHRASES_OUTPUTS_FILENAME = "paraphrases_outputs.csv"
PARAPHRASED_FILENAME = "paraphrased.csv"
CONVERSATIONS_FILENAME = "conversations.csv"
CONVERSATIONS_SCORED_FILENAME = "conversations_scored.csv"
CONSTITUTION_INPUTS_DIRNAME = "constitution_inputs"
CONSTITUTION_INPUTS_MERGED_FILENAME = "constitution_inputs_merged.csv"

CONSTITUTION_DIRNAME = "constitution"
CONSTITUTION_MERGED_FILENAME = "merged.csv"
BENIGN_DIRNAME = "benign"
BENIGN_FILENAME = "benign_samples.csv"


def get_output_dir() -> Path:
    """The default working root.

    ``REDACT_OUTPUT_DIR`` if set; else the first folder above this package
    that holds a ``pyproject.toml`` or ``.git``; else the current directory
    (an installed wheel has no source tree).
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


def _root(root: str | Path | None = None) -> Path:
    """Resolve the working root — explicit ``root`` or ``get_output_dir()``."""
    return Path(root) if root is not None else get_output_dir()


def datasets(root: str | Path | None = None) -> Path:
    """The ``Datasets/`` root (the stages' CSVs and the inputs' category tree)."""
    return _root(root) / DATASETS_DIRNAME


def data_cache(root: str | Path | None = None) -> Path:
    """The ``Data_cache/`` root (benign, constitution, scenarios, …)."""
    return _root(root) / DATA_CACHE_DIRNAME


def category_csv(
    category: str,
    root: str | Path | None = None,
    filename: str = SAMPLES_FILENAME,
) -> Path:
    """Per-category CSV path: ``{root}/{category}/{filename}``.

    ``root`` is the folder that holds the category folders, by default
    :func:`constitution_inputs_dir`.

    Raises:
        ValueError: The category name holds a path separator. Its files
            would land in a nested folder that nothing looks in.
    """
    if "/" in category or "\\" in category:
        raise ValueError(
            f"Category name {category!r} holds a path separator. A category is "
            "stored under one folder name; rename it without '/' or '\\'."
        )
    base_dir = Path(root) if root is not None else constitution_inputs_dir()
    return base_dir / category / filename


def jailbreaks_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / JAILBREAKS_FILENAME


def variations_csv(root: str | Path | None = None) -> Path:
    """The natural variations of the inputs, which the jailbreak stage writes too."""
    return datasets(root) / VARIATIONS_FILENAME


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


def conversations_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / CONVERSATIONS_FILENAME


def conversations_scored_csv(root: str | Path | None = None) -> Path:
    return datasets(root) / CONVERSATIONS_SCORED_FILENAME


def constitution_inputs_dir(root: str | Path | None = None) -> Path:
    return datasets(root) / CONSTITUTION_INPUTS_DIRNAME


def constitution_inputs_merged_csv(root: str | Path | None = None) -> Path:
    """The accepted seeded inputs of every style, in one file for ``build_dataset``."""
    return datasets(root) / CONSTITUTION_INPUTS_MERGED_FILENAME


def benign_csv(root: str | Path | None = None) -> Path:
    return data_cache(root) / BENIGN_DIRNAME / BENIGN_FILENAME


def constitution_dir(root: str | Path | None = None) -> Path:
    return data_cache(root) / CONSTITUTION_DIRNAME


def constitution_merged_csv(root: str | Path | None = None) -> Path:
    """Every constitution entry, of every entry type, in one file."""
    return constitution_dir(root) / CONSTITUTION_MERGED_FILENAME


def package_dir() -> Path:
    """Root of the installed ``redact`` package (``src/redact/``)."""
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


def taxonomy_dir() -> Path:
    """Package-bundled taxonomy config dir (``src/redact/configs/taxonomy``)."""
    return configs_dir() / "taxonomy"


def dataset_configs_dir() -> Path:
    """Package-bundled HuggingFace dataset configs (``src/redact/configs/datasets``)."""
    return configs_dir() / "datasets"


def jailbreak_configs_dir() -> Path:
    """Package-bundled jailbreak technique configs (``src/redact/configs/jailbreak``)."""
    return configs_dir() / "jailbreak"
