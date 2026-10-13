"""Balanced splitting of a dataset across jailbreak techniques.

Rows are dealt out group by group, so every part gets its share of every
``(category, origin)`` group.
"""

import pandas as pd

__all__ = [
    "merge_groups",
    "balanced_counts",
    "deterministic_balanced_assign",
    "split_by_technique_functions",
]


def merge_groups(
    df: pd.DataFrame,
    valid_origins: tuple[str, ...] = ("handcrafted", "generated"),
    fallback: str = "dataset",
) -> pd.DataFrame:
    """Replace every ``origin`` that is not in ``valid_origins`` by ``fallback``.

    Returns a copy. A frame without an ``origin`` column is copied as it is.
    """
    df = df.copy()
    if "origin" not in df.columns:
        return df
    mask = ~df["origin"].isin(valid_origins)
    df.loc[mask, "origin"] = fallback
    return df


def balanced_counts(n: int, dists: list) -> tuple[list[int], list[int]]:
    """Share ``n`` items among ``len(dists)`` bins, no two shares more than one apart.

    The items left over after an even share go round-robin from the first
    bin with the lowest total.

    Args:
        n: Number of items to share.
        dists: The running total of each bin. Updated in place.

    Returns:
        The share of each bin, and ``dists``.
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
    """Split ``df`` into parts that each hold their share of every group.

    Args:
        df: Source frame.
        num_splits: The number of parts, or a list with one label per part.
        group_by: Columns to group on. Those missing from ``df`` are left
            out; with none present the rows are dealt out round-robin. With
            ``origin`` among them, :func:`merge_groups` is applied first.

    Returns:
        One frame per part.
    """
    k = len(num_splits) if isinstance(num_splits, list) else num_splits

    valid_group_by = [col for col in group_by if col in df.columns]

    if not valid_group_by:
        # One slice per part: a concat per row would copy the growing frame
        # each time.
        return [df.iloc[i::k].reset_index(drop=True) for i in range(k)]

    if "origin" in valid_group_by:
        df = merge_groups(df)

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


def split_by_technique_functions(
    df: pd.DataFrame,
    type_to_getter: dict[str, callable],
    technique_type: str | None = None,
    group_by: tuple[str, ...] | list[str] = ("category", "origin"),
) -> dict[str, pd.DataFrame]:
    """Split ``df`` into one balanced part per technique function, by function name.

    Args:
        df: Source frame.
        type_to_getter: A registry of technique types, each with the getter
            of its functions, such as ``get_type_to_getter()``.
        technique_type: Limits the split to the functions of one type.
        group_by: As for :func:`deterministic_balanced_assign`.

    Raises:
        ValueError: ``technique_type`` is not in the registry.
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
