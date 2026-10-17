"""The per-category tree ``Datasets/{category}/samples.csv``: append, read back,
merge, cap per group.

A category CSV may hold columns beyond ``SAMPLE_COLUMNS`` (reasoning,
technique, language, ...); they are kept on read and write.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from redact import paths

from .identity import content_id

# The columns of an input sample row.
SAMPLE_COLUMNS: list[str] = [
    "sample_id",         # MD5 hash of sample text — this row's own identity
    "sample",            # The generated text content
    "category",          # Harm or benign category name
    "subcategory",       # Constitution subcategory, or ""
    "entry_type",        # harmful / dual_use_harmful / dual_use_benign / benign
    "turn",              # Generation turn index
    "accepted",          # Whether the sample passed the checker
    "rejection_reason",  # Checker reasoning on reject; "" on accept
    "source",            # metaprompt / constitution / handcrafted / generated
]


def read_category_csv(
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = paths.SAMPLES_FILENAME,
) -> pd.DataFrame:
    """Read a category's CSV; an empty ``SAMPLE_COLUMNS`` frame when there is none."""
    path = paths.category_csv(category, root=dataset_dir, filename=filename)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=SAMPLE_COLUMNS)


def write_category_csv(
    df: pd.DataFrame,
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = paths.SAMPLES_FILENAME,
) -> Path:
    """Write ``df`` as a category's CSV, replacing the file, and return its path."""
    path = paths.category_csv(category, root=dataset_dir, filename=filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def get_existing_samples(
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = paths.SAMPLES_FILENAME,
) -> set[str]:
    """The sample texts already in a category's CSV."""
    df = read_category_csv(category, dataset_dir, filename)
    if df.empty or "sample" not in df.columns:
        return set()
    return set(df["sample"].dropna().tolist())


#TODO(dataset): append_samples reads and rewrites the whole category CSV on
# every save, so a save slows down as the file grows. Append the new rows only.
# Only the standalone inputs are saved through it.
def append_samples(
    samples: list[str],
    category: str,
    turn: int,
    accepted: list[bool] | None = None,
    source: str = "generated",
    extra_columns: list[dict] | None = None,
    dataset_dir: str | Path | None = None,
    filename: str = paths.SAMPLES_FILENAME,
) -> pd.DataFrame:
    """Add samples to a category's CSV, skipping those whose id is already in it.

    Args:
        samples: Sample texts. A row's ``sample_id`` is the content id of its
            text.
        category: Category name.
        turn: Generation turn index.
        accepted: One flag per sample. ``None`` accepts all.
        source: Source label for the rows.
        extra_columns: One dict of further columns per sample, e.g.
            ``{"reasoning": "...", "technique": "..."}``.
        dataset_dir: Root of the tree.
        filename: CSV file name inside the category folder.

    Returns:
        The category's rows after the append.
    """
    if accepted is None:
        accepted = [True] * len(samples)
    if extra_columns is None:
        extra_columns = [{}] * len(samples)

    existing = read_category_csv(category, dataset_dir, filename)
    existing_ids = set(existing["sample_id"].tolist()) if not existing.empty else set()

    new_rows: list[dict] = []
    for i, (text, acc) in enumerate(zip(samples, accepted)):
        sample_id = content_id(text)
        if sample_id not in existing_ids:
            row = {
                "sample_id": sample_id,
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


def sample_row(
    text: str,
    category: str,
    turn: int,
    accepted: bool = True,
    source: str = "generated",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One input sample row: ``SAMPLE_COLUMNS`` first, the columns of ``extra`` after.

    A new category CSV that takes its columns from such rows has the header
    :func:`append_samples` gives one. ``sample_id`` is the content id of
    ``text``; a ``SAMPLE_COLUMNS`` value that nothing gives is ``""``.
    """
    given = {
        "sample_id": content_id(text),
        "sample": text,
        "category": category,
        "turn": turn,
        "accepted": accepted,
        "source": source,
        **(extra or {}),
    }
    row = {column: given.pop(column, "") for column in SAMPLE_COLUMNS}
    return {**row, **given}


def discover_categories(dataset_dir: str | Path | None = None) -> list[str]:
    """The sorted names of the category folders that hold a samples CSV."""
    base = Path(dataset_dir) if dataset_dir is not None else paths.datasets()
    if not base.is_dir():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / paths.SAMPLES_FILENAME).exists()
    )


def merge_category_csvs(
    categories: list[str],
    dataset_dir: str | Path | None = None,
    filename: str = paths.SAMPLES_FILENAME,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Concatenate the CSVs of ``categories`` into one frame.

    ``accepted_only`` keeps the rows whose ``accepted`` is true. A category
    without a CSV is skipped; with none found the frame is empty.
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


def merge_all(
    dataset_dir: str | Path | None = None,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Merge the CSVs of every category found under ``dataset_dir``."""
    categories = discover_categories(dataset_dir)
    return merge_category_csvs(categories, dataset_dir, accepted_only=accepted_only)


def _input_tree(data_dir: str | Path | None) -> Path:
    """The folder of the input samples of the run under ``data_dir``.

    A run with a constitution seeds its inputs from it and keeps them in
    ``Datasets/constitution_inputs/``. They are the run's inputs when that
    tree holds a category CSV; otherwise the categories directly under
    ``Datasets/`` are.
    """
    seeded = paths.constitution_inputs_dir(data_dir)
    return seeded if discover_categories(seeded) else paths.datasets(data_dir)


def input_files(data_dir: str | Path | None = None) -> list[Path]:
    """The category files of the input samples of the run under ``data_dir``."""
    tree = _input_tree(data_dir)
    return [paths.category_csv(c, root=tree) for c in discover_categories(tree)]


def merge_inputs(data_dir: str | Path | None = None) -> pd.DataFrame:
    """The accepted rows of :func:`input_files`, in one frame."""
    return merge_all(_input_tree(data_dir), accepted_only=True)


def take_per_group(
    df: pd.DataFrame,
    n: int | None,
    group_by: tuple[str, ...] | list[str] = ("category", "entry_type"),
) -> pd.DataFrame:
    """Keep the first ``n`` rows of each group, in the frame's own order.

    Caps a merged, category-ordered frame without skewing it to the
    categories that come first, as a plain ``df.head(n)`` would.

    Args:
        df: Source frame.
        n: Rows to keep per group. ``None`` returns ``df`` unchanged.
        group_by: Columns to group on. Those missing from ``df`` are left
            out; with none present the result is ``df.head(n)``.

    Returns:
        A new frame with the index reset.
    """
    if n is None:
        return df
    valid = [c for c in group_by if c in df.columns]
    if not valid:
        return df.head(n).reset_index(drop=True)
    return (
        df.groupby(valid, sort=False, group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )
