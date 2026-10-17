"""Tests for taxonomy writing and loading."""

import json
import warnings

import pytest

from redact.dataset.taxonomy import create_taxonomy, iter_categories, load_taxonomy


class TestLoadTaxonomy:
    def test_loads_real_taxonomy(self):
        tax = load_taxonomy("harmful")
        assert "categories" in tax
        assert "aliases" in tax
        assert len(tax["categories"]) >= 10

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_taxonomy("nonexistent_taxonomy_xyz")

    def test_extension_optional(self):
        t1 = load_taxonomy("harmful")
        t2 = load_taxonomy("harmful.json")
        assert t1 == t2

    def test_a_name_with_a_dot_is_taken_without_its_extension(self, tmp_path):
        create_taxonomy("my.tax", {"A": "a"}, taxonomy_dir=tmp_path, verbose=False)
        assert (tmp_path / "my.tax.json").exists()
        assert list(load_taxonomy("my.tax", config_dir=tmp_path)["categories"]) == ["A"]
        assert load_taxonomy("my.tax.json", config_dir=tmp_path) == load_taxonomy(
            "my.tax", config_dir=tmp_path)


class TestFormerNames:
    """The names before the config layout, read for one release."""

    def test_the_former_taxonomy_name_loads_harmful_with_a_warning(self):
        with pytest.warns(DeprecationWarning, match="'harmful'"):
            former = load_taxonomy("content_moderation_categories")
        assert former == load_taxonomy("harmful")

    def test_a_folder_with_the_former_file_name_still_serves_harmful(self, tmp_path):
        create_taxonomy("content_moderation_categories", {"A": "a"},
                        taxonomy_dir=tmp_path, verbose=False)
        with pytest.warns(DeprecationWarning, match="rename it harmful.json"):
            tax = load_taxonomy("harmful", config_dir=tmp_path)
        assert list(tax["categories"]) == ["A"]

    def test_a_folder_reads_its_own_file_under_the_former_name(self, tmp_path):
        create_taxonomy("content_moderation_categories", {"A": "a"},
                        taxonomy_dir=tmp_path, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tax = load_taxonomy("content_moderation_categories", config_dir=tmp_path)
        assert list(tax["categories"]) == ["A"]

    def test_a_folder_with_neither_name_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="harmful.json"):
            load_taxonomy("harmful", config_dir=tmp_path)


class TestPackagedLayout:
    def test_each_config_file_is_where_its_loader_reads_it(self):
        from redact import paths

        assert (paths.taxonomy_dir() / "harmful.json").is_file()
        assert (paths.taxonomy_dir() / "benign.json").is_file()
        assert (paths.dataset_configs_dir() / "content_moderation_input.json").is_file()
        hacking = paths.jailbreak_configs_dir() / "hacking"
        assert {"cognitive_techniques.json", "personas.json"} <= {
            p.name for p in hacking.iterdir()}
        assert {p.name for p in paths.taxonomy_dir().iterdir()
                if p.suffix == ".json"} == {"harmful.json", "benign.json"}


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


class TestLoadFromConfig:
    def test_a_name_with_a_dot_is_taken_without_its_extension(
            self, tmp_path, monkeypatch):
        import redact.dataset.loading as loading

        (tmp_path / "my.config.json").write_text(
            json.dumps({"dataset_id": "org/set"}), encoding="utf-8")
        monkeypatch.setattr(loading, "load_hf_dataset",
                            lambda dataset_id, **kwargs: dataset_id)
        assert loading.load_from_config("my.config", config_dir=tmp_path) == "org/set"
