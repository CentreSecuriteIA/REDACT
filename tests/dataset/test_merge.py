"""Tests for CSV merge utilities."""

import pandas as pd

from redact.dataset.io import write_category_csv
from redact.dataset.merge import (
    discover_categories,
    merge_all,
    merge_category_csvs,
    normalize_csv,
)


def _write_test_csv(dataset_dir, category, samples, accepted=None):
    """Helper to create a test category CSV."""
    if accepted is None:
        accepted = [True] * len(samples)
    df = pd.DataFrame({
        "sample_id": [f"id{i}" for i in range(len(samples))],
        "sample": samples,
        "category": [category] * len(samples),
        "turn": [0] * len(samples),
        "accepted": accepted,
        "source": ["generated"] * len(samples),
    })
    write_category_csv(df, category, dataset_dir=dataset_dir)


class TestMergeCategoryCsvs:
    def test_merges_multiple(self, tmp_dataset_dir):
        _write_test_csv(tmp_dataset_dir, "cat1", ["a", "b"])
        _write_test_csv(tmp_dataset_dir, "cat2", ["c"])
        result = merge_category_csvs(["cat1", "cat2"], dataset_dir=tmp_dataset_dir)
        assert len(result) == 3

    def test_accepted_only(self, tmp_dataset_dir):
        _write_test_csv(tmp_dataset_dir, "cat1", ["a", "b"], [True, False])
        result = merge_category_csvs(["cat1"], dataset_dir=tmp_dataset_dir, accepted_only=True)
        assert len(result) == 1

    def test_all_accepted(self, tmp_dataset_dir):
        _write_test_csv(tmp_dataset_dir, "cat1", ["a", "b"], [True, False])
        result = merge_category_csvs(["cat1"], dataset_dir=tmp_dataset_dir, accepted_only=False)
        assert len(result) == 2

    def test_missing_category(self, tmp_dataset_dir):
        result = merge_category_csvs(["nonexistent"], dataset_dir=tmp_dataset_dir)
        assert result.empty


class TestDiscoverCategories:
    def test_finds_categories(self, tmp_dataset_dir):
        _write_test_csv(tmp_dataset_dir, "alpha", ["x"])
        _write_test_csv(tmp_dataset_dir, "beta", ["y"])
        cats = discover_categories(dataset_dir=tmp_dataset_dir)
        assert cats == ["alpha", "beta"]

    def test_ignores_dirs_without_csv(self, tmp_dataset_dir):
        (tmp_dataset_dir / "empty_dir").mkdir()
        cats = discover_categories(dataset_dir=tmp_dataset_dir)
        assert "empty_dir" not in cats

    def test_nonexistent_dir(self, tmp_path):
        cats = discover_categories(dataset_dir=tmp_path / "nope")
        assert cats == []


class TestMergeAll:
    def test_discovers_and_merges(self, tmp_dataset_dir):
        _write_test_csv(tmp_dataset_dir, "a", ["x"])
        _write_test_csv(tmp_dataset_dir, "b", ["y"])
        result = merge_all(dataset_dir=tmp_dataset_dir)
        assert len(result) == 2


class TestNormalizeCsv:
    def test_column_rename(self):
        df = pd.DataFrame({"old_name": ["a"], "prompt": ["text"]})
        result = normalize_csv(df, column_map={"old_name": "new_name"})
        assert "new_name" in result.columns

    def test_discard_values(self):
        df = pd.DataFrame({"prompt": ["real", "DISCARDED", "also real"]})
        result = normalize_csv(df)
        assert len(result) == 2

    def test_recompute_ids(self):
        df = pd.DataFrame({"prompt": ["hello"], "sample_id": ["old_id"]})
        result = normalize_csv(df, recompute_ids=True)
        assert result.iloc[0]["sample_id"] != "old_id"

    def test_missing_text_column_returns_none(self):
        df = pd.DataFrame({"other": ["x"]})
        assert normalize_csv(df) is None
