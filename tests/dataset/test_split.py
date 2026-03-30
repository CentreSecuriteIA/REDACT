"""Tests for dataset splitting utilities."""

import pandas as pd

from redact.dataset.split import (
    normalize_origin,
    balanced_counts,
    deterministic_balanced_assign,
    split_by_column,
)


class TestNormalizeOrigin:
    def test_valid_origins_unchanged(self):
        df = pd.DataFrame({"origin": ["handcrafted", "generated"]})
        result = normalize_origin(df)
        assert list(result["origin"]) == ["handcrafted", "generated"]

    def test_unknown_replaced(self):
        df = pd.DataFrame({"origin": ["handcrafted", "some_dataset", "other"]})
        result = normalize_origin(df)
        assert list(result["origin"]) == ["handcrafted", "dataset", "dataset"]

    def test_custom_fallback(self):
        df = pd.DataFrame({"origin": ["unknown"]})
        result = normalize_origin(df, fallback="external")
        assert result.iloc[0]["origin"] == "external"

    def test_no_origin_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = normalize_origin(df)
        assert "other" in result.columns  # returns unchanged

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"origin": ["unknown"]})
        normalize_origin(df)
        assert df.iloc[0]["origin"] == "unknown"


class TestBalancedCounts:
    def test_even_distribution(self):
        dists = [0, 0, 0]
        new_counts, _ = balanced_counts(9, dists)
        assert new_counts == [3, 3, 3]
        assert sum(new_counts) == 9

    def test_uneven_distribution(self):
        dists = [0, 0, 0]
        new_counts, _ = balanced_counts(10, dists)
        assert sum(new_counts) == 10
        assert max(new_counts) - min(new_counts) <= 1

    def test_single_bin(self):
        dists = [0]
        new_counts, _ = balanced_counts(5, dists)
        assert new_counts == [5]

    def test_cumulative(self):
        dists = [5, 3, 4]
        new_counts, updated = balanced_counts(6, dists)
        assert sum(new_counts) == 6
        assert updated == [5 + new_counts[0], 3 + new_counts[1], 4 + new_counts[2]]


class TestDeterministicBalancedAssign:
    def test_correct_split_count(self):
        df = pd.DataFrame({
            "sample": list(range(12)),
            "category": ["A"] * 6 + ["B"] * 6,
            "origin": ["generated"] * 12,
        })
        splits = deterministic_balanced_assign(df, 3)
        assert len(splits) == 3

    def test_all_rows_assigned(self):
        df = pd.DataFrame({
            "sample": list(range(10)),
            "category": ["A"] * 5 + ["B"] * 5,
            "origin": ["generated"] * 10,
        })
        splits = deterministic_balanced_assign(df, 2)
        total = sum(len(s) for s in splits)
        assert total == 10

    def test_deterministic(self):
        df = pd.DataFrame({
            "sample": list(range(10)),
            "category": ["A"] * 5 + ["B"] * 5,
            "origin": ["generated"] * 10,
        })
        s1 = deterministic_balanced_assign(df, 2)
        s2 = deterministic_balanced_assign(df, 2)
        for a, b in zip(s1, s2):
            assert list(a["sample"]) == list(b["sample"])

    def test_no_group_columns(self):
        df = pd.DataFrame({"sample": [1, 2, 3, 4]})
        splits = deterministic_balanced_assign(df, 2, group_by=("missing",))
        assert len(splits) == 2
        assert sum(len(s) for s in splits) == 4


class TestSplitByColumn:
    def test_basic_split(self):
        df = pd.DataFrame({
            "val": [1, 2, 3],
            "group": ["A", "B", "A"],
        })
        result = split_by_column(df, "group")
        assert set(result.keys()) == {"A", "B"}
        assert len(result["A"]) == 2
        assert len(result["B"]) == 1

    def test_missing_column_raises(self):
        df = pd.DataFrame({"val": [1]})
        import pytest
        with pytest.raises(ValueError, match="not in DataFrame"):
            split_by_column(df, "missing")
