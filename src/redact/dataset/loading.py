"""HuggingFace dataset loading with config-driven filtering.

Provides standardized functions for loading datasets from HuggingFace Hub,
applying column selection and row filtering, and loading from saved JSON
config files.
"""

import json
from pathlib import Path

import pandas as pd

from .. import paths

_DEFAULT_CONFIG_DIR = paths.configs_dir()


def load_hf_dataset(
    dataset_id: str,
    split: str = "train",
    columns: list[str] | None = None,
    filters: dict[str, str | list[str]] | None = None,
    exclude: dict[str, str | list[str]] | None = None,
    drop_columns: list[str] | None = None,
    to_pandas: bool = True,
) -> pd.DataFrame:
    """Load a HuggingFace dataset with optional filtering.

    Args:
        dataset_id: HF dataset ID (e.g. "bells-o-project/content-moderation-input").
        split: Dataset split name.
        columns: Optional list of columns to keep (applied after loading).
        filters: Optional inclusion filters {column: value_or_list}.
                 Only rows where column value is in the list are kept.
        exclude: Optional exclusion filters {column: value_or_list}.
                 Rows where column value is in the list are dropped.
        drop_columns: Optional list of columns to drop.
        to_pandas: If True (default), convert to pandas DataFrame.

    Returns:
        Filtered DataFrame (or HF Dataset if to_pandas=False).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace loading. "
            "Install with: pip install datasets"
        )

    dataset = load_dataset(dataset_id, split=split)

    # Apply inclusion filters
    if filters:
        for col, values in filters.items():
            if isinstance(values, str):
                values = [values]
            dataset = dataset.filter(lambda x, c=col, v=values: x[c] in v)

    # Apply exclusion filters
    if exclude:
        for col, values in exclude.items():
            if isinstance(values, str):
                values = [values]
            dataset = dataset.filter(lambda x, c=col, v=values: x[c] not in v)

    if not to_pandas:
        return dataset

    df = dataset.to_pandas()

    # Drop columns
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    # Keep only specified columns
    if columns:
        available = [c for c in columns if c in df.columns]
        df = df[available]

    return df


def load_from_config(
    config_name: str,
    config_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load a dataset using a saved JSON config file.

    Config files define dataset_id, split, filters, exclusions, etc.
    See Dataset_Configs/ for examples.

    Args:
        config_name: Config filename (with or without .json extension).
        config_dir: Directory containing config files.

    Returns:
        Filtered DataFrame.
    """
    if config_dir is None:
        config_dir = _DEFAULT_CONFIG_DIR
    config_path = Path(config_dir) / config_name
    if not config_path.suffix:
        config_path = config_path.with_suffix(".json")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    return load_hf_dataset(
        dataset_id=config["dataset_id"],
        split=config.get("split", "train"),
        columns=config.get("columns"),
        filters=config.get("filters"),
        exclude=config.get("exclude"),
        drop_columns=config.get("drop_columns"),
    )


def filter_dataset(
    df: pd.DataFrame,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    column: str = "category",
) -> pd.DataFrame:
    """Filter rows by column value inclusion/exclusion lists.

    Args:
        df: Source DataFrame.
        include: If provided, keep only rows with column value in this list.
        exclude: If provided, drop rows with column value in this list.
        column: Column to filter on.

    Returns:
        Filtered DataFrame.
    """
    if column not in df.columns:
        return df

    if include is not None:
        df = df[df[column].isin(include)]
    if exclude is not None:
        df = df[~df[column].isin(exclude)]

    return df.reset_index(drop=True)
