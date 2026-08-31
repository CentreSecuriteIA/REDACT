"""Tests for dataset splitting utilities."""

import pandas as pd

from redact.dataset.split import (
    balanced_counts,
    deterministic_balanced_assign,
    normalize_origin,
    split_by_column,
    take_per_group,
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


class TestTakePerGroup:
    def _frame(self):
        # 2 categories x 2 entry_types x 5 rows, in category-then-entry_type order.
        rows = []
        i = 0
        for cat in ("cbrn", "violence"):
            for et in ("harmful", "benign"):
                for _ in range(5):
                    rows.append({"prompt": f"p{i}", "category": cat, "entry_type": et})
                    i += 1
        return pd.DataFrame(rows)

    def test_caps_per_category_entry_type(self):
        df = self._frame()
        result = take_per_group(df, 2)
        assert len(result) == 8  # 4 groups x 2
        counts = result.groupby(["category", "entry_type"]).size()
        assert (counts == 2).all()

    def test_preserves_order(self):
        df = self._frame()
        result = take_per_group(df, 2)
        # First two of the very first group (cbrn/harmful) lead, in order.
        assert list(result["prompt"])[:2] == ["p0", "p1"]

    def test_fallback_category_only(self):
        df = self._frame().drop(columns=["entry_type"])
        result = take_per_group(df, 3)
        assert len(result) == 6  # 2 categories x 3
        assert (result.groupby("category").size() == 3).all()

    def test_fallback_no_group_columns(self):
        df = pd.DataFrame({"prompt": [f"p{i}" for i in range(10)]})
        result = take_per_group(df, 4)
        assert len(result) == 4
        assert list(result["prompt"]) == ["p0", "p1", "p2", "p3"]

    def test_none_returns_unchanged(self):
        df = self._frame()
        result = take_per_group(df, None)
        assert len(result) == len(df)

    def test_n_larger_than_group_returns_full(self):
        df = self._frame()
        result = take_per_group(df, 100)
        assert len(result) == len(df)  # no padding, no error
