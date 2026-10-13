"""Tests for CSV merge utilities."""

import pandas as pd

from redact.dataset.merge import normalize_csv


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
