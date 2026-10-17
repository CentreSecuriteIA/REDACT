"""Tests for the per-category tree."""

import pandas as pd

from redact import paths
from redact.dataset import append_rows, content_id
from redact.dataset.categories import (
    SAMPLE_COLUMNS,
    append_samples,
    discover_categories,
    get_existing_samples,
    merge_all,
    merge_category_csvs,
    merge_inputs,
    read_category_csv,
    sample_row,
    take_per_group,
    write_category_csv,
)


class TestReadCategoryCsv:
    def test_missing_returns_empty(self, tmp_dataset_dir):
        df = read_category_csv("nonexistent", dataset_dir=tmp_dataset_dir)
        assert df.empty
        assert list(df.columns) == SAMPLE_COLUMNS

    def test_an_empty_file_reads_as_none(self, tmp_dataset_dir):
        """What a save killed before its first byte leaves."""
        path = paths.category_csv("empty", root=tmp_dataset_dir)
        path.parent.mkdir()
        path.touch()
        df = read_category_csv("empty", dataset_dir=tmp_dataset_dir)
        assert df.empty and list(df.columns) == SAMPLE_COLUMNS
        assert discover_categories(tmp_dataset_dir) == ["empty"]
        assert merge_all(tmp_dataset_dir).empty

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


class TestSampleRow:
    _EXTRA = {"entry_type": "harmful", "subcategory": "Sub", "reasoning": "fine",
              "template_style": "long", "rejection_reason": ""}

    def test_the_sample_columns_come_first_then_the_extra_ones(self):
        row = sample_row("a text", "Cyber", turn=2, accepted=False,
                         source="constitution", extra=self._EXTRA)
        assert list(row) == [*SAMPLE_COLUMNS, "reasoning", "template_style"]
        assert row == {
            "sample_id": content_id("a text"), "sample": "a text", "category": "Cyber",
            "subcategory": "Sub", "entry_type": "harmful", "turn": 2, "accepted": False,
            "rejection_reason": "", "source": "constitution", "reasoning": "fine",
            "template_style": "long"}

    def test_a_sample_column_nothing_gives_is_empty(self):
        row = sample_row("a text", "Cyber", turn=0)
        assert list(row) == SAMPLE_COLUMNS
        assert (row["subcategory"], row["entry_type"], row["rejection_reason"]) == (
            "", "", "")
        assert (row["accepted"], row["source"]) == (True, "generated")

    def test_a_new_file_of_such_rows_is_the_one_append_samples_writes(
            self, tmp_dataset_dir):
        """Header and cells, so the two ways to write a category file agree."""
        texts = ["first, with a comma", 'second "quoted"']
        append_samples(texts, category="old", turn=1, accepted=[True, False],
                       source="constitution", extra_columns=[self._EXTRA] * 2,
                       dataset_dir=tmp_dataset_dir)
        new = paths.category_csv("new", root=tmp_dataset_dir)
        new.parent.mkdir()
        append_rows(new, [
            sample_row(text, "old", turn=1, accepted=accepted, source="constitution",
                       extra=self._EXTRA)
            for text, accepted in zip(texts, (True, False))])
        old = paths.category_csv("old", root=tmp_dataset_dir)
        assert new.read_bytes() == old.read_bytes()


class TestGetExistingSamples:
    def test_empty_when_no_file(self, tmp_dataset_dir):
        result = get_existing_samples("missing", dataset_dir=tmp_dataset_dir)
        assert result == set()

    def test_returns_sample_texts(self, tmp_dataset_dir):
        append_samples(["a", "b"], category="test", turn=0, dataset_dir=tmp_dataset_dir)
        result = get_existing_samples("test", dataset_dir=tmp_dataset_dir)
        assert result == {"a", "b"}


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


class TestMergeInputs:
    """A run's inputs: the seeded tree when it holds a category, else the
    categories directly under ``Datasets/``."""

    def test_the_seeded_inputs_are_read_when_their_tree_holds_one(self, tmp_path):
        _write_test_csv(paths.constitution_inputs_dir(tmp_path), "Cyber", ["seeded"])
        _write_test_csv(paths.datasets(tmp_path), "Fraud", ["standalone"])
        assert list(merge_inputs(tmp_path)["sample"]) == ["seeded"]

    def test_the_standalone_inputs_are_read_without_a_seeded_tree(self, tmp_path):
        _write_test_csv(paths.datasets(tmp_path), "Cyber", ["a", "b"])
        _write_test_csv(paths.datasets(tmp_path), "Fraud", ["c"])
        result = merge_inputs(tmp_path)
        pd.testing.assert_frame_equal(result, merge_all(paths.datasets(tmp_path)))
        assert list(result["sample"]) == ["a", "b", "c"]

    def test_a_seeded_folder_without_a_category_csv_is_no_seeded_tree(self, tmp_path):
        seeded = paths.constitution_inputs_dir(tmp_path)
        (seeded / "Cyber").mkdir(parents=True)
        (seeded / "constitution_inputs.state.jsonl").write_text("{}")
        _write_test_csv(paths.datasets(tmp_path), "Fraud", ["standalone"])
        assert list(merge_inputs(tmp_path)["sample"]) == ["standalone"]

    def test_only_accepted_rows_are_returned(self, tmp_path):
        _write_test_csv(paths.constitution_inputs_dir(tmp_path), "Cyber",
                        ["kept", "rejected"], [True, False])
        assert list(merge_inputs(tmp_path)["sample"]) == ["kept"]

    def test_a_seeded_tree_of_rejected_rows_is_still_the_run_s_inputs(self, tmp_path):
        _write_test_csv(paths.constitution_inputs_dir(tmp_path), "Cyber",
                        ["rejected"], [False])
        _write_test_csv(paths.datasets(tmp_path), "Fraud", ["standalone"])
        assert merge_inputs(tmp_path).empty

    def test_no_tree_gives_an_empty_frame(self, tmp_path):
        assert merge_inputs(tmp_path).empty

    def test_ids_and_empty_cells_read_back_as_written(self, tmp_path):
        """A one-row file whose id looks like a number; a cell nothing gave."""
        path = paths.category_csv("Cyber", root=paths.datasets(tmp_path))
        path.parent.mkdir(parents=True)
        append_rows(path, [{**sample_row("a text", "Cyber", turn=0),
                            "sample_id": "0123456789012345"}])
        (row,) = merge_inputs(tmp_path).to_dict("records")
        assert row["sample_id"] == "0123456789012345"
        assert (row["subcategory"], row["entry_type"]) == ("", "")


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

    def test_a_missing_group_value_is_a_group_of_its_own(self):
        df = pd.DataFrame({"prompt": ["p0", "p1", "p2"],
                           "category": ["Cyber", "Cyber", None],
                           "entry_type": ["harmful", None, None]})
        assert list(take_per_group(df, 5)["prompt"]) == ["p0", "p1", "p2"]
