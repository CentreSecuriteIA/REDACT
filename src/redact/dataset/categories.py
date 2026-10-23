"""The per-category tree of the input samples,
``Datasets/constitution_inputs/{category}/samples.csv``: its rows, read back,
merged, capped per group.

A category CSV may hold columns beyond ``SAMPLE_COLUMNS`` (template_style,
rejection_reason, ...); they are kept on read.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from redact import paths

from .identity import content_id
from .rows import _ID_COLUMNS, _blank_missing_text, _has_bytes, accepted_mask

# The columns of an input sample row.
SAMPLE_COLUMNS: list[str] = [
    "sample_id",         # Content id of the text: this row's own identity
    "sample",            # The generated text
    "category",          # The entry's source_category, harmful or benign taxonomy
    "subcategory",       # The entry's constitution subcategory, or ""
    "entry_type",        # harmful / dual_use_harmful / dual_use_benign / benign
    "turn",              # 0: every sample is written in one turn
    "accepted",          # Whether the sample passed the checker
    "rejection_reason",  # The checker's reply on a reject; "" on accept
    "source",            # "constitution": every input is seeded from it
]


# What a folder name may not hold on Windows, besides the path separators
# and the control characters, those below the space.
_NOT_IN_A_FOLDER_NAME = '<>:"|?*'
_SPACE = " "
# The device names Windows reserves, with or without an extension.
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL",
                   *(f"COM{n}" for n in range(1, 10)),
                   *(f"LPT{n}" for n in range(1, 10))}


def check_category_name(name: Any) -> str:
    """``name`` when it can name a category's folder, on every platform.

    A category's input samples are stored under a folder of its name
    (``{category}/samples.csv``), and a run folder may be copied to another
    machine, so a name Windows cannot use as a folder is refused on every
    platform: a path separator, one of ``<>:"|?*`` or a control character, a
    trailing dot or space, an empty name, or a reserved device name such as
    ``CON``.

    Raises:
        ValueError: The name, why, and what to do.
    """
    text = name if isinstance(name, str) else ""
    stem = text.split(".")[0].strip().upper()
    if not text.strip():
        why = "it is empty"
    elif "/" in text or "\\" in text:
        why = "it holds a path separator ('/' or '\\')"
    elif any(c in _NOT_IN_A_FOLDER_NAME or c < _SPACE for c in text):
        why = (f"it holds a character a folder name may not hold "
               f"({_NOT_IN_A_FOLDER_NAME} or a control character)")
    elif text.endswith((".", " ")):
        why = "it ends with a dot or a space"
    elif stem in _RESERVED_NAMES:
        why = f"{stem} is a device name Windows reserves"
    else:
        return text
    raise ValueError(
        f"Category name {name!r} cannot name a folder: {why}. A category's "
        "inputs are stored under a folder of its name, on every platform the "
        "run may be copied to; rename the category in the taxonomy (or the "
        "constitution's source_category).")


def read_category_csv(
    category: str,
    dataset_dir: str | Path | None = None,
    filename: str = paths.SAMPLES_FILENAME,
) -> pd.DataFrame:
    """Read a category's CSV; an empty ``SAMPLE_COLUMNS`` frame when there is none.

    ``dataset_dir`` defaults to ``paths.constitution_inputs_dir()``. Ids are
    read as text, and a missing cell of a text column as ``""``.
    """
    path = paths.category_csv(category, root=dataset_dir, filename=filename)
    if _has_bytes(path):
        return _blank_missing_text(pd.read_csv(path, dtype=_ID_COLUMNS))
    return pd.DataFrame(columns=SAMPLE_COLUMNS)


def sample_row(
    text: str,
    category: str,
    turn: int,
    accepted: bool = True,
    source: str = "generated",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One input sample row: ``SAMPLE_COLUMNS`` first, the columns of ``extra`` after.

    ``sample_id`` is the content id of ``text``; a ``SAMPLE_COLUMNS`` value
    that nothing gives is ``""``.
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
    """The sorted names of the category folders that hold a samples CSV.

    A CSV of 0 bytes, what a save killed before its first byte leaves, is
    none. ``dataset_dir`` defaults to ``paths.constitution_inputs_dir()``.
    """
    base = (Path(dataset_dir) if dataset_dir is not None
            else paths.constitution_inputs_dir())
    if not base.is_dir():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and _has_bytes(d / paths.SAMPLES_FILENAME)
    )


def merge_category_csvs(
    categories: list[str],
    dataset_dir: str | Path | None = None,
    filename: str = paths.SAMPLES_FILENAME,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Concatenate the CSVs of ``categories`` into one frame.

    ``accepted_only`` keeps the rows whose ``accepted`` is true
    (:func:`redact.dataset.accepted_mask`). A category
    without a CSV is skipped; with none found the frame is empty.
    """
    frames: list[pd.DataFrame] = []
    for category in categories:
        df = read_category_csv(category, dataset_dir, filename)
        if not df.empty:
            if accepted_only and "accepted" in df.columns:
                df = df[accepted_mask(df["accepted"])]
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


def input_files(data_dir: str | Path | None = None) -> list[Path]:
    """The category files of the input samples of the run under ``data_dir``,
    in ``Datasets/constitution_inputs/``."""
    tree = paths.constitution_inputs_dir(data_dir)
    return [paths.category_csv(c, root=tree) for c in discover_categories(tree)]


def merge_inputs(data_dir: str | Path | None = None) -> pd.DataFrame:
    """The accepted rows of :func:`input_files`, in one frame."""
    return merge_all(paths.constitution_inputs_dir(data_dir), accepted_only=True)


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
        df.groupby(valid, sort=False, group_keys=False, dropna=False)
        .head(n)
        .reset_index(drop=True)
    )
