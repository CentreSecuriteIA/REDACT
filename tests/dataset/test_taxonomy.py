"""Tests for taxonomy loading."""

import pytest

from redact.dataset.taxonomy import get_seed_prompts, iter_categories, load_taxonomy


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
