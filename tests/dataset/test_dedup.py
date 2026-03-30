"""Tests for deduplication utilities."""

import pandas as pd

from redact.dataset.dedup import exact_dedup, normalized_dedup


class TestExactDedup:
    def test_removes_duplicates(self):
        df = pd.DataFrame({"sample": ["a", "b", "a", "c"]})
        result = exact_dedup(df)
        assert list(result["sample"]) == ["a", "b", "c"]

    def test_keeps_first(self):
        df = pd.DataFrame({
            "sample": ["dup", "dup"],
            "idx": [1, 2],
        })
        result = exact_dedup(df, keep="first")
        assert result.iloc[0]["idx"] == 1

    def test_keeps_last(self):
        df = pd.DataFrame({
            "sample": ["dup", "dup"],
            "idx": [1, 2],
        })
        result = exact_dedup(df, keep="last")
        assert result.iloc[0]["idx"] == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"sample": []})
        result = exact_dedup(df)
        assert result.empty

    def test_missing_column(self):
        df = pd.DataFrame({"other": ["a", "a"]})
        result = exact_dedup(df, column="sample")
        assert len(result) == 2  # returns unchanged

    def test_no_duplicates(self):
        df = pd.DataFrame({"sample": ["a", "b", "c"]})
        result = exact_dedup(df)
        assert len(result) == 3


class TestNormalizedDedup:
    def test_case_insensitive(self):
        df = pd.DataFrame({"sample": ["Hello", "hello", "HELLO"]})
        result = normalized_dedup(df)
        assert len(result) == 1

    def test_whitespace_normalized(self):
        df = pd.DataFrame({"sample": ["a  b", "a b", "a   b"]})
        result = normalized_dedup(df)
        assert len(result) == 1

    def test_preserves_original_text(self):
        df = pd.DataFrame({"sample": ["Hello World", "hello world"]})
        result = normalized_dedup(df, keep="first")
        assert result.iloc[0]["sample"] == "Hello World"

    def test_empty_dataframe(self):
        df = pd.DataFrame({"sample": []})
        result = normalized_dedup(df)
        assert result.empty

    def test_different_content_preserved(self):
        df = pd.DataFrame({"sample": ["abc", "def", "ghi"]})
        result = normalized_dedup(df)
        assert len(result) == 3
