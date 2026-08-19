"""Generalized dataset splitting across techniques or parameters.

Distributes rows evenly across N splits, stratified by configurable
grouping columns. Generalized from the jailbreak library's
dataset_distribution.py to work with any pipeline.

Key function: deterministic_balanced_assign() ensures every split
receives a representative sample from every group.
"""

import pandas as pd


def normalize_origin(
    df: pd.DataFrame,
    valid_origins: tuple[str, ...] = ("handcrafted", "generated"),
    fallback: str = "dataset",
) -> pd.DataFrame:
    """Normalize origin column values.

    Any origin value not in valid_origins is replaced with fallback.
    Generalizes the jailbreak library's merge_groups().

    Args:
        df: DataFrame with an 'origin' column.
        valid_origins: Values to keep as-is.
        fallback: Replacement for unrecognized values.

    Returns:
        Copy of df with normalized 'origin' values.
    """
    df = df.copy()
    if "origin" not in df.columns:
        return df
    mask = ~df["origin"].isin(valid_origins)
    df.loc[mask, "origin"] = fallback
    return df


def balanced_counts(n: int, dists: list) -> tuple[list[int], list[int]]:
    """Distribute n items across len(dists) bins, keeping counts balanced.

    Max difference between any two bins is 1. Uses round-robin
    starting from the bin with minimum count.

    Args:
        n: Number of items to distribute.
        dists: Current cumulative counts per bin (mutated in place).

    Returns:
        (new_counts, updated_dists)
    """
    k = len(dists)
    base = n // k
    remainder = n % k

    new_counts = [base] * k
    for i in range(k):
        dists[i] += base

    min_val = min(dists)
    start_idx = next(i for i in range(k) if dists[i] == min_val)

    for i in range(start_idx, start_idx + remainder):
        idx = i % k
        new_counts[idx] += 1
        dists[idx] += 1

    return new_counts, dists


def deterministic_balanced_assign(
    df: pd.DataFrame,
    num_splits: int | list,
    group_by: tuple[str, ...] | list[str] = ("category", "origin"),
) -> list[pd.DataFrame]:
    """Split DataFrame into balanced parts, stratified by group columns.

    Rows are grouped by group_by columns and distributed evenly so
    every split receives a representative sample from every group.

    Args:
        df: Source DataFrame.
        num_splits: Number of output splits (int) or list of labels
                    (length determines split count).
        group_by: Column names to group by for stratification.
                  Only existing columns are used.

    Returns:
        List of DataFrames, one per split.
    """
    k = len(num_splits) if isinstance(num_splits, list) else num_splits

    # Filter group_by to only columns that exist
    valid_group_by = [col for col in group_by if col in df.columns]

    if not valid_group_by:
        # No grouping columns — simple round-robin. Sliced directly (one
        # slice per split) rather than one pd.concat per row, which was
        # O(n^2) (each concat copies the whole growing frame).
        return [df.iloc[i::k].reset_index(drop=True) for i in range(k)]

    # Normalize origin if it's in the group columns
    if "origin" in valid_group_by:
        df = normalize_origin(df)

    groups = df.groupby(valid_group_by, sort=True).groups

    splits = [pd.DataFrame(columns=df.columns) for _ in range(k)]
    distributions = [0] * k

    for _group_key, indices in groups.items():
        group_size = len(indices)
        if group_size == 0:
            continue

        new_counts, distributions = balanced_counts(group_size, distributions)

        cumsum = 0
        for split_idx, count in enumerate(new_counts):
            data_indices = indices[cumsum: cumsum + count]
            new_rows = df.loc[data_indices]
            splits[split_idx] = pd.concat(
                [splits[split_idx], new_rows], ignore_index=True
            )
            cumsum += count

    return splits


def split_by_column(
    df: pd.DataFrame,
    column: str,
) -> dict[str, pd.DataFrame]:
    """Split DataFrame into one part per unique value of a column.

    Args:
        df: Source DataFrame.
        column: Column to split on.

    Returns:
        Dict mapping column value to its DataFrame slice.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in DataFrame")
    return {
        str(value): group.reset_index(drop=True)
        for value, group in df.groupby(column, sort=True)
    }


def take_per_group(
    df: pd.DataFrame,
    n: int | None,
    group_by: tuple[str, ...] | list[str] = ("category", "entry_type"),
) -> pd.DataFrame:
    """Take the first ``n`` rows of each (existing) group, order-preserving.

    A deterministic alternative to a global ``df.head(n)`` for capping a merged,
    category-ordered frame: instead of skewing to whichever categories appear
    first, it keeps up to ``n`` rows from every ``group_by`` group so all
    categories (and severity levels) are represented.

    Args:
        df: Source DataFrame.
        n: Max rows to keep per group. ``None`` returns ``df`` unchanged.
        group_by: Columns to group on. Only columns actually present are used
            (same idiom as :func:`deterministic_balanced_assign`); when none are
            present it falls back to a plain ``df.head(n)``.

    Returns:
        A new DataFrame with at most ``n`` rows per group, overall row order
        preserved, index reset.
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


def split_by_functions(
    df: pd.DataFrame,
    type_to_getter: dict[str, callable],
    technique_type: str | None = None,
    group_by: tuple[str, ...] | list[str] = ("category", "origin"),
) -> dict[str, pd.DataFrame]:
    """Split dataset by technique functions from a registry.

    Works with any module's get_type_to_getter() registry dict.

    Args:
        df: Source DataFrame.
        type_to_getter: Registry dict mapping type names to getter functions.
        technique_type: Optional filter to one type.
        group_by: Columns for stratified splitting.

    Returns:
        Dict mapping function name to its DataFrame slice.
    """
    if technique_type is not None:
        if technique_type not in type_to_getter:
            raise ValueError(
                f"Unknown type '{technique_type}'. Valid: {list(type_to_getter)}"
            )
        funcs = type_to_getter[technique_type]()
    else:
        funcs = []
        for getter in type_to_getter.values():
            funcs.extend(getter())

    splits = deterministic_balanced_assign(df, len(funcs), group_by=group_by)
    return {fn.__name__: split for fn, split in zip(funcs, splits)}
