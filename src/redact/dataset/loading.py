"""Load a published dataset from the HuggingFace Hub, directly or from a JSON config."""

import json
from pathlib import Path

import pandas as pd

from .. import paths

_DEFAULT_CONFIG_DIR = paths.dataset_configs_dir()


def load_hf_dataset(
    dataset_id: str,
    split: str = "train",
    columns: list[str] | None = None,
    filters: dict[str, str | list[str]] | None = None,
    exclude: dict[str, str | list[str]] | None = None,
    drop_columns: list[str] | None = None,
    to_pandas: bool = True,
) -> pd.DataFrame:
    """Load one split of a HuggingFace dataset, filtered.

    Args:
        dataset_id: The dataset's id on the Hub, such as
            ``"bells-o-project/content-moderation-input"``.
        split: Split name.
        columns: Columns to keep, after ``drop_columns``.
        filters: ``{column: value or values}``. A row is kept when each
            column holds one of its values.
        exclude: ``{column: value or values}``. A row is dropped when a
            column holds one of its values.
        drop_columns: Columns to drop.
        to_pandas: False returns the HuggingFace ``Dataset`` after the row
            filters, with every column.

    Raises:
        ImportError: The ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace loading: "
            "install the library's hf extra (pip install -e \".[hf]\" from the "
            "repository)."
        ) from None

    dataset = load_dataset(dataset_id, split=split)

    if filters:
        for col, values in filters.items():
            if isinstance(values, str):
                values = [values]
            dataset = dataset.filter(lambda x, c=col, v=values: x[c] in v)

    if exclude:
        for col, values in exclude.items():
            if isinstance(values, str):
                values = [values]
            dataset = dataset.filter(lambda x, c=col, v=values: x[c] not in v)

    if not to_pandas:
        return dataset

    df = dataset.to_pandas()

    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    if columns:
        available = [c for c in columns if c in df.columns]
        df = df[available]

    return df


def load_from_config(
    config_name: str,
    config_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load the dataset that the JSON config ``config_name`` describes.

    The config gives ``dataset_id`` and optionally ``split``, ``columns``,
    ``filters``, ``exclude`` and ``drop_columns``, as :func:`load_hf_dataset`
    takes them. ``config_dir`` defaults to the packaged ``configs/datasets``,
    which holds ``content_moderation_input.json``. The name is taken with or
    without its ``.json``. The columns are the published dataset's own: that
    config's id column is ``id``, not the ``sample_id`` the stages read, and
    no stage or recipe reads it.

    Raises:
        FileNotFoundError: The directory holds no such file.
    """
    if config_dir is None:
        config_dir = _DEFAULT_CONFIG_DIR
    config_path = Path(config_dir) / config_name
    if config_path.suffix != ".json":
        config_path = config_path.with_name(config_path.name + ".json")

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
