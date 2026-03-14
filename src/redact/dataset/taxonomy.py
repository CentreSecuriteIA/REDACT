"""Taxonomy loading and application.

Taxonomies define canonical category lists with descriptions, subcategories,
aliases, and logical groups. They serve two purposes:
1. Filtering/grouping existing datasets (normalize_categories, filter_by_group)
2. Driving generation loops (iter_categories, get_category_descriptions)

Taxonomy JSON schema:
{
    "name": "...",
    "description": "...",
    "categories": {
        "CategoryName": {
            "description": "...",
            "subcategories": [...] or {...}
        },
        ...
    },
    "aliases": {"AltName": "CanonicalName", ...},
    "groups": {"group_name": ["Cat1", "Cat2"], ...}
}
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

# Default config directory: Dataset_Configs/taxonomy/ inside the Redact_Library package
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_DIR = _PACKAGE_DIR / "Dataset_Configs" / "taxonomy"
_DEFAULT_SEEDS_DIR = _PACKAGE_DIR / "Dataset_Configs" / "seeds"


def load_taxonomy(
    name: str,
    config_dir: str | Path | None = None,
) -> dict:
    """Load a taxonomy JSON file.

    Args:
        name: Taxonomy filename (with or without .json extension).
        config_dir: Directory containing taxonomy files.

    Returns:
        Parsed taxonomy dict.
    """
    if config_dir is None:
        config_dir = _DEFAULT_CONFIG_DIR
    path = Path(config_dir) / name
    if not path.suffix:
        path = path.with_suffix(".json")

    if not path.exists():
        raise FileNotFoundError(f"Taxonomy not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_categories(taxonomy: dict) -> list[str]:
    """Get the canonical category name list from a taxonomy.

    Args:
        taxonomy: Loaded taxonomy dict.

    Returns:
        Sorted list of category names.
    """
    return sorted(taxonomy.get("categories", {}).keys())


def get_category_descriptions(taxonomy: dict) -> dict[str, str]:
    """Get {category: description} mapping.

    Used by generation pipelines to feed category descriptions into prompts.

    Args:
        taxonomy: Loaded taxonomy dict.

    Returns:
        Dict mapping category name to its description string.
    """
    result: dict[str, str] = {}
    for name, info in taxonomy.get("categories", {}).items():
        if isinstance(info, dict):
            result[name] = info.get("description", "")
        else:
            result[name] = str(info)
    return result


def get_subcategories(
    taxonomy: dict,
    category: str,
) -> list[str] | dict[str, str]:
    """Get subcategories for a category.

    Returns a list if subcategories are plain strings, or a dict
    if they have descriptions. Returns empty list if no subcategories.

    Args:
        taxonomy: Loaded taxonomy dict.
        category: Category name.

    Returns:
        Subcategories as list[str] or dict[str, str].
    """
    info = taxonomy.get("categories", {}).get(category, {})
    if isinstance(info, dict):
        return info.get("subcategories", [])
    return []


def normalize_categories(
    df: pd.DataFrame,
    taxonomy: dict,
    column: str = "category",
) -> pd.DataFrame:
    """Apply taxonomy aliases to normalize category names in a DataFrame.

    Maps alias names to their canonical equivalents.

    Args:
        df: DataFrame with a category column.
        taxonomy: Loaded taxonomy dict with "aliases" mapping.
        column: Column to normalize.

    Returns:
        Copy of df with normalized category names.
    """
    aliases = taxonomy.get("aliases", {})
    if not aliases or column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].replace(aliases)
    return df


def filter_by_group(
    df: pd.DataFrame,
    taxonomy: dict,
    group: str,
    column: str = "category",
) -> pd.DataFrame:
    """Filter DataFrame to categories in a taxonomy group.

    Args:
        df: DataFrame with a category column.
        taxonomy: Loaded taxonomy dict with "groups" mapping.
        group: Group name (e.g. "safety", "harmful_content").
        column: Column to filter on.

    Returns:
        Filtered DataFrame.
    """
    group_cats = get_group_categories(taxonomy, group)
    if not group_cats or column not in df.columns:
        return df
    return df[df[column].isin(group_cats)].reset_index(drop=True)


def get_group_categories(taxonomy: dict, group: str) -> list[str]:
    """Get the category list for a taxonomy group.

    Args:
        taxonomy: Loaded taxonomy dict.
        group: Group name.

    Returns:
        List of category names in the group, or empty list.
    """
    return taxonomy.get("groups", {}).get(group, [])


def iter_categories(taxonomy: dict) -> Iterator[tuple[str, dict]]:
    """Iterate over (category_name, category_info) pairs.

    Convenience for generation loops that process each category.
    category_info is the dict containing description, subcategories, etc.
    If the value is not a dict, wraps it as {"description": value}.

    Args:
        taxonomy: Loaded taxonomy dict.

    Yields:
        (category_name, category_info_dict) tuples.
    """
    for name, info in taxonomy.get("categories", {}).items():
        if isinstance(info, dict):
            yield name, info
        else:
            yield name, {"description": str(info)}


# ---------------------------------------------------------------------------
# Seed prompts
# ---------------------------------------------------------------------------


def load_seeds(
    name: str,
    seeds_dir: str | Path | None = None,
) -> dict:
    """Load a seed prompts JSON file.

    Args:
        name: Seeds filename (with or without .json extension).
        seeds_dir: Directory containing seed files.

    Returns:
        Parsed seeds dict with "seeds" key mapping category -> list[str].
    """
    if seeds_dir is None:
        seeds_dir = _DEFAULT_SEEDS_DIR
    path = Path(seeds_dir) / name
    if not path.suffix:
        path = path.with_suffix(".json")

    if not path.exists():
        raise FileNotFoundError(f"Seeds file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_seed_prompts(
    seeds: dict,
    category: str,
) -> str:
    """Get formatted seed prompts for a category.

    Returns a numbered list string suitable for injection into the
    generation template's {SeedPrompts} placeholder.

    Args:
        seeds: Loaded seeds dict (from load_seeds()).
        category: Category name.

    Returns:
        Numbered list of seed prompts, or empty string if no seeds.
    """
    prompts = seeds.get("seeds", {}).get(category, [])
    if not prompts:
        return ""
    return "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
