"""CSV read/write per category folder.

All output is saved as CSVs in Datasets/{category}/samples.csv.
Supports incremental appending with MD5-based deduplication for
checkpoint-style writing during generation runs.

Extra metadata columns (reasoning, technique, language, etc.) are
preserved — the standard columns are a minimum, not a fixed schema.
"""

import hashlib
from pathlib import Path

import pandas as pd

# Default dataset directory: Datasets/ relative to the calling script
from redact import get_output_dir


def _default_dataset_dir() -> Path:
    return get_output_dir() / "Datasets"


# Standard input-sample CSV schema. Extra columns (technique, language,
# source_sample_description, etc.) are allowed and preserved on read/write.
# This is the contract the jailbreak pipeline (handled in a separate
# session) consumes from generated input CSVs.
SAMPLE_COLUMNS: list[str] = [
    "id",                # MD5 hash of sample text
    "sample",            # The generated text content
    "category",          # Harm or benign category name
    "subcategory",       # Constitution subcategory, or ""
    "entry_type",        # harmful / dual_use_harmful / dual_use_benign / benign
    "turn",              # Generation turn index
    "accepted",          # Whether the sample passed the checker
    "rejection_reason",  # Checker reasoning on reject; "" on accept
    "source",            # metaprompt / constitution / handcrafted / generated
]


def _hash_text(text: str) -> str:
    """MD5 hash of sample text, truncated to 16 hex chars."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _resolve_path(
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = "samples.csv",
) -> Path:
    """Resolve the CSV path for a category."""
    if dataset_dir is None:
        dataset_dir = _default_dataset_dir()
    return Path(dataset_dir) / category / filename


def read_category_csv(
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = "samples.csv",
) -> pd.DataFrame:
    """Read the CSV for a category. Returns empty DataFrame if not found.

    Args:
        category: Category name (used as subdirectory).
        dataset_dir: Root dataset directory.
        filename: CSV filename within the category directory.

    Returns:
        DataFrame with at least SAMPLE_COLUMNS (may have extra columns).
    """
    path = _resolve_path(category, dataset_dir, filename)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=SAMPLE_COLUMNS)


def write_category_csv(
    df: pd.DataFrame,
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = "samples.csv",
) -> Path:
    """Write a DataFrame as the category CSV (full overwrite).

    Creates the category directory if it doesn't exist.

    Args:
        df: DataFrame to write.
        category: Category name.
        dataset_dir: Root dataset directory.
        filename: CSV filename.

    Returns:
        Path to the written file.
    """
    path = _resolve_path(category, dataset_dir, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def get_existing_samples(
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = "samples.csv",
) -> set[str]:
    """Return the set of sample texts already in the category CSV.

    Used by the generation pipeline to build a "prohibited outputs"
    exclusion list so the LLM doesn't regenerate existing samples.

    Args:
        category: Category name.
        dataset_dir: Root dataset directory.
        filename: CSV filename.

    Returns:
        Set of sample text strings.
    """
    df = read_category_csv(category, dataset_dir, filename)
    if df.empty or "sample" not in df.columns:
        return set()
    return set(df["sample"].dropna().tolist())


def append_samples(
    samples: list[str],
    category: str,
    turn: int,
    accepted: list[bool] | None = None,
    source: str = "generated",
    extra_columns: list[dict] | None = None,
    dataset_dir: str | Path | None = None,
    filename: str = "samples.csv",
) -> pd.DataFrame:
    """Append new samples to a category CSV incrementally.

    Generates IDs via MD5 hash. Deduplicates against existing IDs.
    This is the primary checkpoint-save function during generation.

    Args:
        samples: List of sample text strings.
        category: Category name.
        turn: Generation turn index.
        accepted: Per-sample acceptance flags. If None, all are True.
        source: Source label for the samples.
        extra_columns: Optional list of dicts with per-sample metadata.
            Must have same length as samples. Keys become extra CSV columns
            (e.g. {"reasoning": "...", "technique": "..."}).
        dataset_dir: Root dataset directory.
        filename: CSV filename.

    Returns:
        Updated DataFrame including the new samples.
    """
    if accepted is None:
        accepted = [True] * len(samples)
    if extra_columns is None:
        extra_columns = [{}] * len(samples)

    existing = read_category_csv(category, dataset_dir, filename)
    existing_ids = set(existing["id"].tolist()) if not existing.empty else set()

    new_rows: list[dict] = []
    for i, (text, acc) in enumerate(zip(samples, accepted)):
        sample_id = _hash_text(text)
        if sample_id not in existing_ids:
            row = {
                "id": sample_id,
                "sample": text,
                "category": category,
                "turn": turn,
                "accepted": acc,
                "source": source,
            }
            # Merge extra metadata columns
            if i < len(extra_columns):
                row.update(extra_columns[i])
            new_rows.append(row)
            existing_ids.add(sample_id)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        write_category_csv(combined, category, dataset_dir, filename)
        return combined

    return existing
