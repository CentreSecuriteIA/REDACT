"""Deduplication utilities for generated datasets.

Provides exact text deduplication. Semantic/embedding-based deduplication
and ranking are planned for later.
"""

import pandas as pd


def exact_dedup(
    df: pd.DataFrame,
    column: str = "sample",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove exact duplicate texts from a DataFrame.

    Args:
        df: DataFrame to deduplicate.
        column: Column name containing the text to compare.
        keep: Which duplicate to keep ("first", "last", or False for none).

    Returns:
        Deduplicated DataFrame.
    """
    if df.empty or column not in df.columns:
        return df
    return df.drop_duplicates(subset=[column], keep=keep).reset_index(drop=True)


def normalized_dedup(
    df: pd.DataFrame,
    column: str = "sample",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicates after normalizing whitespace and case.

    Catches near-duplicates that differ only in capitalization or spacing.

    Args:
        df: DataFrame to deduplicate.
        column: Column name containing the text to compare.
        keep: Which duplicate to keep.

    Returns:
        Deduplicated DataFrame.
    """
    if df.empty or column not in df.columns:
        return df
    normalized = df[column].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    mask = ~normalized.duplicated(keep=keep)
    return df[mask].reset_index(drop=True)


# NOTE: Semantic/embedding-based dedup and ranking (select best X samples
# via embedding similarity scores) to be designed later. Consider
# SentenceTransformer or similar for pairwise distance computation.
# For large datasets, exact dedup may be sufficient.
#
# Planned features:
# - semantic_dedup(df, column, model, threshold) -> DataFrame
#   Remove samples with cosine similarity above threshold
# - rank_samples(df, column, model, top_k) -> DataFrame
#   Select top_k most diverse samples via maximal marginal relevance
# - coverage_stats(df, column, model) -> dict
#   Pairwise embedding distances, diversity metrics (per CLAUDE.md)
