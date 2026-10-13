"""Tests for dataset rows in one CSV."""

import pandas as pd
import pytest

from redact.dataset import Ledger, append_rows, keys_in, read_rows

_UNIT = ("input_id", "iteration")
_UNIT_CASTERS = {"input_id": str, "iteration": int}


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


class TestAppendRows:
    def test_the_first_append_writes_the_header(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        assert _lines(path) == ["input_id,sample", "a,x"]

    def test_later_appends_add_rows_under_the_one_header(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        append_rows(path, [{"input_id": "b", "sample": "y"},
                           {"input_id": "c", "sample": "z"}])
        assert _lines(path) == ["input_id,sample", "a,x", "b,y", "c,z"]


class TestReadRows:
    def test_reads_back_the_rows_appended(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        append_rows(path, [{"input_id": "b", "sample": "y"}])
        frame = read_rows(path)
        assert frame.to_dict("records") == [
            {"input_id": "a", "sample": "x"}, {"input_id": "b", "sample": "y"}]

    def test_no_file_gives_an_empty_frame(self, tmp_path):
        frame = read_rows(tmp_path / "nope.csv")
        assert isinstance(frame, pd.DataFrame) and frame.empty


class TestKeysIn:
    def test_no_file_gives_no_keys(self, tmp_path):
        assert keys_in(tmp_path / "nope.csv", ("input_id",)) == set()

    def test_a_file_with_only_a_header_gives_no_keys(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("input_id,sample\n", encoding="utf-8")
        assert keys_in(path, ("input_id",)) == set()

    def test_one_key_field_gives_each_value(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"},
                           {"input_id": "b", "sample": "y"}])
        assert keys_in(path, ("input_id",)) == {"a", "b"}

    def test_several_key_fields_give_tuples(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "iteration": 0},
                           {"input_id": "a", "iteration": 1}])
        assert keys_in(path, _UNIT, _UNIT_CASTERS) == {("a", 0), ("a", 1)}

    def test_casters_convert_the_values_read(self, tmp_path):
        """An id of digits is read from a CSV as a number."""
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "123", "iteration": 0}])
        (key,) = keys_in(path, _UNIT, _UNIT_CASTERS)
        assert key == ("123", 0)
        assert isinstance(key[0], str) and isinstance(key[1], int)

    def test_keys_equal_the_keys_a_ledger_reads_for_the_same_units(self, tmp_path):
        path = tmp_path / "out.csv"
        ledger = Ledger.sidecar(path, key_fields=_UNIT, casters=_UNIT_CASTERS)
        units = [{"input_id": "123", "iteration": 0}, {"input_id": "b", "iteration": 2}]
        append_rows(path, [{**unit, "sample": "x"} for unit in units])
        ledger.record(units)
        assert keys_in(path, ledger.key_fields, ledger.casters) == ledger.completed()

    def test_a_file_without_a_key_column_gives_no_keys(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        assert keys_in(path, _UNIT, _UNIT_CASTERS) == set()

    def test_a_cast_that_fails_raises(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("input_id,iteration\na,0\nb,\n", encoding="utf-8")
        with pytest.raises(ValueError):
            keys_in(path, _UNIT, _UNIT_CASTERS)
