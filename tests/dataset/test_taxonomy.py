"""Tests for taxonomy writing and loading."""

import json

import pytest

from redact.dataset.taxonomy import (
    create_taxonomy,
    get_seed_prompts,
    iter_categories,
    load_taxonomy,
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


class TestCreateTaxonomy:
    def test_creates_json(self, tmp_path):
        create_taxonomy(
            "test_tax",
            categories={"CatA": "Description A", "CatB": {"description": "B"}},
            description="Test taxonomy",
            taxonomy_dir=tmp_path,
        )
        path = tmp_path / "test_tax.json"
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["name"] == "test_tax"
        assert "CatA" in loaded["categories"]
        assert loaded["categories"]["CatA"]["description"] == "Description A"

    def test_string_categories_normalized(self, tmp_path):
        tax = create_taxonomy(
            "norm_test",
            categories={"Cat": "just a string"},
            taxonomy_dir=tmp_path,
        )
        assert tax["categories"]["Cat"] == {"description": "just a string"}

    def test_aliases_and_groups(self, tmp_path):
        tax = create_taxonomy(
            "full_test",
            categories={"A": "desc"},
            aliases={"B": "A"},
            groups={"g1": ["A"]},
            taxonomy_dir=tmp_path,
        )
        assert tax["aliases"] == {"B": "A"}
        assert tax["groups"] == {"g1": ["A"]}

    def test_returns_dict(self, tmp_path):
        result = create_taxonomy("ret_test", {"X": "Y"}, taxonomy_dir=tmp_path)
        assert isinstance(result, dict)
        assert result["name"] == "ret_test"

    def test_loads_back_as_written(self, tmp_path):
        tax = create_taxonomy("round_trip", {"A": "a", "B": {"description": "b"}},
                              taxonomy_dir=tmp_path, verbose=False)
        assert load_taxonomy("round_trip", config_dir=tmp_path) == tax

    def test_exported_from_redact(self):
        import redact

        assert redact.create_taxonomy is create_taxonomy


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
