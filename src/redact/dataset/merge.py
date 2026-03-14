"""Merge per-category CSVs into a single unified dataset.

Each category stores its samples separately in Datasets/{category}/samples.csv.
This module provides functions to combine them on demand.

Also provides general-purpose CSV normalization and directory-based merging,
with presets for jailbreak technique pipelines and content moderation output.
"""

import hashlib
from pathlib import Path

import pandas as pd

from .io import read_category_csv, _default_dataset_dir


def merge_category_csvs(
    categories: list[str],
    dataset_dir: str | Path | None = None,
    filename: str = "samples.csv",
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Merge multiple category CSVs into a single DataFrame.

    Args:
        categories: List of category names to merge.
        dataset_dir: Root dataset directory.
        filename: CSV filename within each category directory.
        accepted_only: If True, only include accepted samples.

    Returns:
        Combined DataFrame with all categories.
    """
    frames: list[pd.DataFrame] = []
    for category in categories:
        df = read_category_csv(category, dataset_dir, filename)
        if not df.empty:
            if accepted_only and "accepted" in df.columns:
                df = df[df["accepted"] == True]  # noqa: E712
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def discover_categories(dataset_dir: str | Path | None = None) -> list[str]:
    """Discover all category subdirectories in the dataset directory.

    Args:
        dataset_dir: Root dataset directory.

    Returns:
        Sorted list of category directory names.
    """
    base = Path(dataset_dir) if dataset_dir is not None else _default_dataset_dir()
    if not base.is_dir():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "samples.csv").exists()
    )


def merge_all(
    dataset_dir: str | Path | None = None,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Discover all categories and merge into one DataFrame.

    Args:
        dataset_dir: Root dataset directory.
        accepted_only: If True, only include accepted samples.

    Returns:
        Combined DataFrame.
    """
    categories = discover_categories(dataset_dir)
    return merge_category_csvs(categories, dataset_dir, accepted_only=accepted_only)


# ---------------------------------------------------------------------------
# General-purpose CSV normalization and directory-based merging
# ---------------------------------------------------------------------------


def normalize_csv(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    text_column: str = "prompt",
    discard_values: list[str] | None = None,
    recompute_ids: bool = True,
    id_column: str = "id",
) -> pd.DataFrame | None:
    """Normalize a CSV DataFrame for merging.

    Renames columns, drops discarded rows, and optionally recomputes
    deterministic IDs from the text column.

    Works for any pipeline — jailbreak techniques, content moderation
    output, or any dataset with a primary text column.

    Args:
        df: Raw DataFrame.
        column_map: Mapping of old column names to new names.
            E.g. ``{"obfuscation_type": "technique_type"}``.
        text_column: Primary text column name (used for discarding
            and ID computation). Defaults to ``"prompt"``.
        discard_values: Row values in *text_column* that mark
            discarded rows (case-insensitive, stripped). Defaults
            to ``["DISCARDED"]``.
        recompute_ids: If True, recompute IDs from the text column.
        id_column: Name of the ID column. Defaults to ``"id"``.

    Returns:
        Normalized DataFrame, or None if *text_column* is missing.
    """
    if discard_values is None:
        discard_values = ["DISCARDED"]

    # Rename columns
    if column_map:
        rename = {old: new for old, new in column_map.items() if old in df.columns}
        if rename:
            df = df.rename(columns=rename)

    # Must have the text column
    if text_column not in df.columns:
        return None

    # Drop discarded rows
    if discard_values:
        upper_discard = {v.upper() for v in discard_values}
        df = df[
            ~df[text_column]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(upper_discard)
        ]

    df = df.copy()

    # Recompute deterministic IDs
    if recompute_ids:
        df[id_column] = df[text_column].apply(
            lambda t: hashlib.md5(str(t).encode()).hexdigest()[:12]
        )

    return df.reset_index(drop=True)


def merge_csvs_from_dirs(
    base_dir: str | Path,
    scan_dirs: list[str] | None = None,
    output_path: str | Path | None = None,
    keep_columns: list[str] | None = None,
    column_map: dict[str, str] | None = None,
    text_column: str = "prompt",
    discard_values: list[str] | None = None,
    recompute_ids: bool = True,
    deduplicate: bool = True,
    glob_pattern: str = "*.csv",
) -> pd.DataFrame:
    """Scan directories for CSVs, normalize, and merge.

    General-purpose merge that works for any pipeline. Recursively
    finds CSV files, normalizes each with :func:`normalize_csv`, and
    concatenates into a single DataFrame.

    Args:
        base_dir: Root directory to scan.
        scan_dirs: Optional list of subdirectory names to limit the scan.
            If None, scans the entire *base_dir* recursively.
        output_path: If provided, save the merged CSV here.
        keep_columns: Final column subset. If None, keeps all columns.
        column_map: Passed to :func:`normalize_csv`.
        text_column: Primary text column. Defaults to ``"prompt"``.
        discard_values: Discard markers. Defaults to ``["DISCARDED"]``.
        recompute_ids: Recompute IDs from text column. Defaults to True.
        deduplicate: Drop duplicate IDs. Defaults to True.
        glob_pattern: File pattern to match. Defaults to ``"*.csv"``.

    Returns:
        Merged DataFrame.
    """
    base = Path(base_dir)

    # Collect CSV paths
    if scan_dirs:
        csv_paths = []
        for sub in scan_dirs:
            subdir = base / sub
            if subdir.is_dir():
                csv_paths.extend(subdir.rglob(glob_pattern))
    else:
        csv_paths = list(base.rglob(glob_pattern))

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(csv_paths):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        normalized = normalize_csv(
            df,
            column_map=column_map,
            text_column=text_column,
            discard_values=discard_values,
            recompute_ids=recompute_ids,
        )
        if normalized is not None and not normalized.empty:
            frames.append(normalized)

    if not frames:
        cols = keep_columns or []
        return pd.DataFrame(columns=cols)

    merged = pd.concat(frames, ignore_index=True)

    # Final column selection
    if keep_columns:
        available = [c for c in keep_columns if c in merged.columns]
        merged = merged[available]

    # Deduplicate by ID
    if deduplicate and "id" in merged.columns:
        merged = merged.drop_duplicates(subset=["id"]).reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

    return merged


# ---------------------------------------------------------------------------
# Pipeline-specific presets
# ---------------------------------------------------------------------------

JAILBREAK_KEEP_COLUMNS = [
    "id", "prompt", "category", "origin",
    "input_prompt", "input_id", "technique", "technique_type",
]

JAILBREAK_COLUMN_MAP = {
    "obfuscation_type": "technique_type",
    "hacking_type": "technique_type",
    "manipulation_type": "technique_type",
}

CONTENT_MOD_KEEP_COLUMNS = [
    "id", "sample", "category", "turn", "accepted", "source",
]

CONTENT_MOD_COLUMN_MAP = {
    "text": "sample",
    "output": "sample",
    "label": "category",
}


def normalize_technique_csv(
    df: pd.DataFrame,
    type_column_map: dict[str, str] | None = None,
) -> pd.DataFrame | None:
    """Normalize a jailbreak technique CSV. Preset wrapper around :func:`normalize_csv`."""
    return normalize_csv(
        df,
        column_map=type_column_map or JAILBREAK_COLUMN_MAP,
        text_column="prompt",
    )


def merge_technique_csvs(
    base_dir: str | Path,
    scan_dirs: list[str] | None = None,
    output_path: str | Path | None = None,
    keep_columns: list[str] | None = None,
    type_column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Merge jailbreak technique CSVs. Preset wrapper around :func:`merge_csvs_from_dirs`."""
    return merge_csvs_from_dirs(
        base_dir,
        scan_dirs=scan_dirs,
        output_path=output_path,
        keep_columns=keep_columns or JAILBREAK_KEEP_COLUMNS,
        column_map=type_column_map or JAILBREAK_COLUMN_MAP,
        text_column="prompt",
    )


def merge_content_mod_csvs(
    base_dir: str | Path,
    scan_dirs: list[str] | None = None,
    output_path: str | Path | None = None,
    keep_columns: list[str] | None = None,
    column_map: dict[str, str] | None = None,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Merge content moderation output CSVs. Preset wrapper around :func:`merge_csvs_from_dirs`.

    Normalizes column names (e.g. ``text`` → ``sample``, ``label`` → ``category``),
    and optionally filters to accepted samples only.

    Args:
        base_dir: Root directory to scan.
        scan_dirs: Optional subdirectory names to limit the scan.
        output_path: If provided, save the merged CSV here.
        keep_columns: Final column subset. Defaults to CONTENT_MOD_KEEP_COLUMNS.
        column_map: Column renaming. Defaults to CONTENT_MOD_COLUMN_MAP.
        accepted_only: If True, filter to rows where ``accepted == True``.

    Returns:
        Merged DataFrame.
    """
    merged = merge_csvs_from_dirs(
        base_dir,
        scan_dirs=scan_dirs,
        output_path=None,  # save after filtering
        keep_columns=keep_columns or CONTENT_MOD_KEEP_COLUMNS,
        column_map=column_map or CONTENT_MOD_COLUMN_MAP,
        text_column="sample",
    )

    if accepted_only and "accepted" in merged.columns:
        merged = merged[merged["accepted"] == True].reset_index(drop=True)  # noqa: E712

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

    return merged


# Keep TYPE_COLUMN_MAP as alias for backward compatibility
TYPE_COLUMN_MAP = JAILBREAK_COLUMN_MAP
