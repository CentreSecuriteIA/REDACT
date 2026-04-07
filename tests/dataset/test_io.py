"""Tests for CSV I/O per category folder."""

import pandas as pd

from redact.dataset.io import (
    SAMPLE_COLUMNS,
    read_category_csv,
    write_category_csv,
    append_samples,
    get_existing_samples,
)


class TestReadCategoryCsv:
    def test_missing_returns_empty(self, tmp_dataset_dir):
        df = read_category_csv("nonexistent", dataset_dir=tmp_dataset_dir)
        assert df.empty
        assert list(df.columns) == SAMPLE_COLUMNS

    def test_reads_existing(self, tmp_dataset_dir):
        cat_dir = tmp_dataset_dir / "test_cat"
        cat_dir.mkdir()
        pd.DataFrame({"sample": ["hello"]}).to_csv(cat_dir / "samples.csv", index=False)
        df = read_category_csv("test_cat", dataset_dir=tmp_dataset_dir)
        assert len(df) == 1
        assert df.iloc[0]["sample"] == "hello"


class TestWriteCategoryCsv:
    def test_creates_dir_and_file(self, tmp_dataset_dir):
        df = pd.DataFrame({"sample": ["test"]})
        path = write_category_csv(df, "new_cat", dataset_dir=tmp_dataset_dir)
        assert path.exists()
        assert (tmp_dataset_dir / "new_cat").is_dir()

    def test_roundtrip(self, tmp_dataset_dir):
        df = pd.DataFrame({"sample": ["a", "b"], "category": ["X", "Y"]})
        write_category_csv(df, "roundtrip", dataset_dir=tmp_dataset_dir)
        loaded = read_category_csv("roundtrip", dataset_dir=tmp_dataset_dir)
        assert list(loaded["sample"]) == ["a", "b"]


class TestAppendSamples:
    def test_creates_file_on_first_append(self, tmp_dataset_dir):
        result = append_samples(
            ["sample1", "sample2"],
            category="test",
            turn=0,
            dataset_dir=tmp_dataset_dir,
        )
        assert len(result) == 2
        assert (tmp_dataset_dir / "test" / "samples.csv").exists()

    def test_deduplicates(self, tmp_dataset_dir):
        append_samples(["dup"], category="test", turn=0, dataset_dir=tmp_dataset_dir)
        result = append_samples(["dup"], category="test", turn=1, dataset_dir=tmp_dataset_dir)
        assert len(result) == 1  # not duplicated

    def test_appends_new(self, tmp_dataset_dir):
        append_samples(["first"], category="test", turn=0, dataset_dir=tmp_dataset_dir)
        result = append_samples(["second"], category="test", turn=1, dataset_dir=tmp_dataset_dir)
        assert len(result) == 2

    def test_accepted_flags(self, tmp_dataset_dir):
        result = append_samples(
            ["a", "b"],
            category="test",
            turn=0,
            accepted=[True, False],
            dataset_dir=tmp_dataset_dir,
        )
        assert list(result["accepted"]) == [True, False]

    def test_extra_columns(self, tmp_dataset_dir):
        result = append_samples(
            ["text"],
            category="test",
            turn=0,
            extra_columns=[{"reasoning": "looks good"}],
            dataset_dir=tmp_dataset_dir,
        )
        assert "reasoning" in result.columns
        assert result.iloc[0]["reasoning"] == "looks good"

    def test_standard_columns_present(self, tmp_dataset_dir):
        result = append_samples(
            ["x"], category="cat", turn=0, dataset_dir=tmp_dataset_dir
        )
        for col in SAMPLE_COLUMNS:
            assert col in result.columns


class TestGetExistingSamples:
    def test_empty_when_no_file(self, tmp_dataset_dir):
        result = get_existing_samples("missing", dataset_dir=tmp_dataset_dir)
        assert result == set()

    def test_returns_sample_texts(self, tmp_dataset_dir):
        append_samples(["a", "b"], category="test", turn=0, dataset_dir=tmp_dataset_dir)
        result = get_existing_samples("test", dataset_dir=tmp_dataset_dir)
        assert result == {"a", "b"}
