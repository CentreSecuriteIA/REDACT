"""Tests for taxonomy loading and application."""

import pytest
import pandas as pd

from redact.dataset.taxonomy import (
    load_taxonomy,
    get_categories,
    get_category_descriptions,
    get_subcategories,
    normalize_categories,
    filter_by_group,
    get_group_categories,
    iter_categories,
    get_seed_prompts,
)


class TestLoadTaxonomy:
    def test_loads_real_taxonomy(self):
        tax = load_taxonomy("content_moderation_categories")
        assert "categories" in tax
        assert "aliases" in tax
        assert len(tax["categories"]) >= 10

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_taxonomy("nonexistent_taxonomy_xyz")

    def test_extension_optional(self):
        t1 = load_taxonomy("content_moderation_categories")
        t2 = load_taxonomy("content_moderation_categories.json")
        assert t1 == t2


class TestGetCategories:
    def test_returns_sorted(self, sample_taxonomy):
        cats = get_categories(sample_taxonomy)
        assert cats == sorted(cats)
        assert set(cats) == {"Cat A", "Cat B"}

    def test_real_taxonomy(self):
        tax = load_taxonomy("content_moderation_categories")
        cats = get_categories(tax)
        assert "Physical Harm" in cats
        assert "Sexual Content" in cats


class TestGetCategoryDescriptions:
    def test_returns_descriptions(self, sample_taxonomy):
        descs = get_category_descriptions(sample_taxonomy)
        assert descs["Cat A"] == "Category A description"

    def test_all_categories_present(self, sample_taxonomy):
        descs = get_category_descriptions(sample_taxonomy)
        assert set(descs.keys()) == {"Cat A", "Cat B"}


class TestGetSubcategories:
    def test_returns_list(self, sample_taxonomy):
        subs = get_subcategories(sample_taxonomy, "Cat A")
        assert subs == ["Sub1", "Sub2"]

    def test_missing_category(self, sample_taxonomy):
        subs = get_subcategories(sample_taxonomy, "Nonexistent")
        assert subs == []


class TestNormalizeCategories:
    def test_applies_aliases(self, sample_taxonomy):
        df = pd.DataFrame({"category": ["Cat A", "Alias A", "Alt B"]})
        result = normalize_categories(df, sample_taxonomy)
        assert list(result["category"]) == ["Cat A", "Cat A", "Cat B"]

    def test_no_aliases(self):
        tax = {"categories": {}, "aliases": {}}
        df = pd.DataFrame({"category": ["X"]})
        result = normalize_categories(df, tax)
        assert list(result["category"]) == ["X"]

    def test_missing_column(self, sample_taxonomy):
        df = pd.DataFrame({"other": [1]})
        result = normalize_categories(df, sample_taxonomy)
        assert "other" in result.columns


class TestFilterByGroup:
    def test_filters_to_group(self, sample_taxonomy):
        df = pd.DataFrame({"category": ["Cat A", "Cat B", "Cat A"]})
        result = filter_by_group(df, sample_taxonomy, "group1")
        assert list(result["category"]) == ["Cat B"]

    def test_multiple_in_group(self, sample_taxonomy):
        df = pd.DataFrame({"category": ["Cat A", "Cat B"]})
        result = filter_by_group(df, sample_taxonomy, "group2")
        assert len(result) == 2

    def test_unknown_group(self, sample_taxonomy):
        df = pd.DataFrame({"category": ["Cat A"]})
        result = filter_by_group(df, sample_taxonomy, "nonexistent")
        assert len(result) == 1  # returns unchanged


class TestGetGroupCategories:
    def test_known_group(self, sample_taxonomy):
        assert get_group_categories(sample_taxonomy, "group1") == ["Cat B"]

    def test_unknown_group(self, sample_taxonomy):
        assert get_group_categories(sample_taxonomy, "missing") == []


class TestIterCategories:
    def test_yields_pairs(self, sample_taxonomy):
        pairs = list(iter_categories(sample_taxonomy))
        assert len(pairs) == 2
        names = [name for name, _ in pairs]
        assert "Cat A" in names
        assert "Cat B" in names

    def test_info_is_dict(self, sample_taxonomy):
        for _, info in iter_categories(sample_taxonomy):
            assert isinstance(info, dict)
            assert "description" in info


class TestGetSeedPrompts:
    def test_returns_numbered_list(self):
        seeds = {"seeds": {"violence": ["prompt1", "prompt2"]}}
        result = get_seed_prompts(seeds, "violence")
        assert "1. prompt1" in result
        assert "2. prompt2" in result

    def test_missing_category(self):
        seeds = {"seeds": {}}
        assert get_seed_prompts(seeds, "missing") == ""
