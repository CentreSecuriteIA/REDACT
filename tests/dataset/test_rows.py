"""Tests for dataset rows in one CSV."""

import os
from pathlib import Path

import pandas as pd
import pytest

import redact.dataset.rows as rows_module
from redact.dataset import (
    Ledger,
    Part,
    accepted_mask,
    append_rows,
    check_columns,
    count_rows,
    has_rows,
    iter_rows,
    keys_in,
    read_rows,
    row_ids,
    write_concat,
)

_UNIT = ("input_id", "iteration")
_UNIT_CASTERS = {"input_id": str, "iteration": int}

# Content ids that pandas would read as numbers: digits around an "e", and a
# leading zero.
_NUMBER_LIKE_IDS = ["12345678901234e5", "0123456789012345"]

# Text as a model writes it: commas, quotes and line breaks of each kind.
_TEXTS = [
    'He said "hi", then left.',
    "line one\nline two",
    "a Windows break\r\nand a lone\rreturn",
    "caf\u00e9, na\u00efve",
]


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def _batch(start):
    """One row per text of ``_TEXTS``, every cell set, ids counting from ``start``."""
    return [
        {"input_id": f"id{start + n}", "iteration": n, "sample": text,
         "accepted": n % 2 == 0, "score": n / 2}
        for n, text in enumerate(_TEXTS)
    ]


def _batch_with_blanks(start):
    """``_batch`` and one row whose cells are empty, ``None`` or NaN."""
    blank = {"input_id": f"id{start + len(_TEXTS)}", "iteration": len(_TEXTS),
             "sample": "", "accepted": None, "score": float("nan")}
    return [*_batch(start), blank]


def _to_csv_append(path, rows):
    """Append with pandas alone: the bytes ``append_rows`` has to write."""
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


@pytest.fixture()
def two_row_chunks(monkeypatch):
    """Read two rows at a time, so a small file spans several chunks."""
    monkeypatch.setattr(rows_module, "_CHUNK_ROWS", 2)


@pytest.fixture()
def held_for_append(monkeypatch):
    """Refuse opens for append as Windows does while another program, such
    as a spreadsheet, holds the file. Returns the attempts made."""

    def hold(refusals):
        attempts = []
        opened = Path.open

        def open_(path, mode="r", *args, **kwargs):
            if mode != "ab":
                return opened(path, mode, *args, **kwargs)
            attempts.append(path.name)
            if len(attempts) <= refusals:
                raise PermissionError("held by another program")
            return opened(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_)
        monkeypatch.setattr(rows_module.time, "sleep", lambda seconds: None)
        return attempts

    return hold


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

    def test_a_new_file_gets_the_bytes_to_csv_writes(self, tmp_path):
        path, reference = tmp_path / "out.csv", tmp_path / "reference.csv"
        append_rows(path, _batch_with_blanks(0))
        _to_csv_append(reference, _batch_with_blanks(0))
        assert path.read_bytes() == reference.read_bytes()

    def test_an_existing_file_gets_the_bytes_to_csv_writes(self, tmp_path):
        path, reference = tmp_path / "out.csv", tmp_path / "reference.csv"
        for start in (0, 10, 20):
            append_rows(path, _batch_with_blanks(start))
            _to_csv_append(reference, _batch_with_blanks(start))
            assert path.read_bytes() == reference.read_bytes()

    def test_a_row_ends_with_the_line_ending_of_the_platform(self, tmp_path):
        """A line break inside a cell is written as it is."""
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "one\ntwo"}])
        end = os.linesep
        assert path.read_bytes() == f'input_id,sample{end}a,"one\ntwo"{end}'.encode()

    def test_keys_in_another_order_land_under_the_right_columns(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x", "model": "m1"}])
        append_rows(path, [{"model": "m2", "input_id": "b", "sample": "y"}])
        assert read_rows(path).to_dict("records") == [
            {"input_id": "a", "sample": "x", "model": "m1"},
            {"input_id": "b", "sample": "y", "model": "m2"}]

    def test_a_row_without_a_header_column_leaves_its_cell_empty(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x", "model": "m1"}])
        append_rows(path, [{"input_id": "b", "model": "m2"}])
        assert _lines(path) == ["input_id,sample,model", "a,x,m1", "b,,m2"]

    def test_a_column_the_header_lacks_raises_and_writes_nothing(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        before = path.read_bytes()
        with pytest.raises(ValueError, match="'note'") as raised:
            append_rows(path, [{"input_id": "b", "sample": "y", "note": "new"}])
        assert "out.csv" in str(raised.value)
        assert path.read_bytes() == before

    def test_returns_the_size_of_the_file(self, tmp_path):
        path = tmp_path / "out.csv"
        assert append_rows(path, _batch(0)) == {path: path.stat().st_size}
        assert append_rows(path, _batch(10)) == {path: path.stat().st_size}

    def test_the_rows_are_synced_to_disk_once_all_are_written(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "out.csv"
        sizes_at_sync = []
        monkeypatch.setattr(
            rows_module.os, "fsync",
            lambda fd: sizes_at_sync.append(os.fstat(fd).st_size),
        )
        (size,) = append_rows(path, _batch(0)).values()
        assert sizes_at_sync == [size]

    def test_an_empty_file_is_given_a_header(self, tmp_path):
        path = tmp_path / "out.csv"
        path.touch()
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        assert _lines(path) == ["input_id,sample", "a,x"]

    def test_a_file_another_program_holds_is_opened_again(
            self, tmp_path, held_for_append):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        attempts = held_for_append(refusals=2)
        append_rows(path, [{"input_id": "b", "sample": "y"}])
        assert len(attempts) == 2 + 1
        assert _lines(path) == ["input_id,sample", "a,x", "b,y"]

    def test_a_file_that_stays_held_raises_and_gets_no_byte(
            self, tmp_path, held_for_append):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        before = path.read_bytes()
        attempts = held_for_append(refusals=10**6)
        with pytest.raises(PermissionError):
            append_rows(path, [{"input_id": "b", "sample": "y"}])
        assert len(attempts) > 1 and path.read_bytes() == before

    def test_a_header_behind_a_byte_order_mark_is_read(self, tmp_path):
        """A spreadsheet saves a CSV with the mark in front of the header."""
        path = tmp_path / "out.csv"
        path.write_bytes(b"\xef\xbb\xbfinput_id,sample\r\na,x\r\n")
        append_rows(path, [{"sample": "y", "input_id": "b"}])
        assert read_rows(path).to_dict("records") == [
            {"input_id": "a", "sample": "x"}, {"input_id": "b", "sample": "y"}]


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


class TestIdsAreReadAsText:
    """A file whose ids all look like numbers gives them back as written."""

    @pytest.mark.parametrize("given", _NUMBER_LIKE_IDS)
    def test_read_rows_and_row_ids(self, tmp_path, given):
        path = tmp_path / "out.csv"
        append_rows(path, [{"sample_id": given, "input_id": given, "sample": "x"}])
        frame = read_rows(path)
        assert list(frame["sample_id"]) == list(frame["input_id"]) == [given]
        assert row_ids(frame, "sample") == [given]
        assert [row["input_id"] for row in iter_rows(path)] == [given]

    @pytest.mark.parametrize("given", _NUMBER_LIKE_IDS)
    def test_keys_in(self, tmp_path, given):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": given, "iteration": 3}])
        assert keys_in(path, _UNIT, _UNIT_CASTERS) == {(given, 3)}


class TestMissingCells:
    def test_an_empty_text_cell_reads_back_empty(self, tmp_path):
        """A number column keeps its gap, so it stays numeric."""
        path = tmp_path / "out.csv"
        append_rows(path, [{"sample": "x", "subcategory": "", "turn": 0},
                           {"sample": "y", "subcategory": "phishing", "turn": None}])
        frame = read_rows(path)
        assert list(frame["subcategory"]) == ["", "phishing"]
        assert pd.api.types.is_numeric_dtype(frame["turn"])
        assert frame["turn"].isna().tolist() == [False, True]

    def test_a_column_without_a_value_reads_back_empty(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"sample": "x", "subcategory": ""},
                           {"sample": "y", "subcategory": None}])
        assert list(read_rows(path)["subcategory"]) == ["", ""]


class TestIterRows:
    def test_gives_the_rows_read_rows_gives(self, tmp_path, two_row_chunks):
        path = tmp_path / "out.csv"
        append_rows(path, [*_batch(0), *_batch(10)[:1]])
        assert list(iter_rows(path)) == read_rows(path).to_dict("records")
        assert [row["sample"] for row in iter_rows(path)] == [*_TEXTS, _TEXTS[0]]

    def test_columns_limits_the_read(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, _batch(0)[:2])
        assert list(iter_rows(path, columns=["input_id", "iteration"])) == [
            {"input_id": "id0", "iteration": 0}, {"input_id": "id1", "iteration": 1}]

    def test_no_file_gives_no_rows(self, tmp_path):
        assert list(iter_rows(tmp_path / "nope.csv")) == []

    def test_a_file_with_only_a_header_gives_no_rows(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("input_id,sample\n", encoding="utf-8")
        assert list(iter_rows(path)) == []


class TestCountRows:
    def test_counts_rows_not_lines(self, tmp_path, two_row_chunks):
        """A line break inside a cell does not start a row."""
        path = tmp_path / "out.csv"
        append_rows(path, [*_batch(0), *_batch(10)])
        assert count_rows(path) == len(read_rows(path)) == 2 * len(_TEXTS)
        assert len(path.read_bytes().split(b"\n")) > count_rows(path) + 2

    def test_no_file_counts_zero(self, tmp_path):
        assert count_rows(tmp_path / "nope.csv") == 0

    def test_a_file_with_only_a_header_counts_zero(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("input_id,sample\n", encoding="utf-8")
        assert count_rows(path) == 0


class TestHasRows:
    def test_a_row_below_the_header(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        assert has_rows(path)

    def test_a_file_with_only_a_header_has_none(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("input_id,sample\n", encoding="utf-8")
        assert not has_rows(path)

    def test_no_file_has_none(self, tmp_path):
        assert not has_rows(tmp_path / "nope.csv")


class TestAnEmptyFile:
    """A save killed between opening the file and writing to it leaves 0 bytes:
    a reader takes that as no rows, as it takes no file."""

    @pytest.mark.parametrize("read, nothing", [
        (lambda path: read_rows(path).empty, True),
        (lambda path: list(iter_rows(path)), []),
        (count_rows, 0),
        (has_rows, False),
        (lambda path: keys_in(path, _UNIT, _UNIT_CASTERS), set()),
    ], ids=["read_rows", "iter_rows", "count_rows", "has_rows", "keys_in"])
    def test_reads_as_no_rows(self, tmp_path, read, nothing):
        path = tmp_path / "out.csv"
        path.touch()
        assert read(path) == nothing

    def test_is_a_part_without_rows(self, tmp_path):
        path, frame = tmp_path / "empty.csv", pd.DataFrame({"sample_id": ["b0"]})
        path.touch()
        out = tmp_path / "out.csv"
        parts = [Part(path, accepted_only=True), Part(frame)]
        assert write_concat(out, parts) == [0, 1]
        assert _lines(out) == ["sample_id", "b0"]


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

    def test_keys_of_a_file_read_in_several_chunks(self, tmp_path, two_row_chunks):
        path = tmp_path / "out.csv"
        append_rows(path, [*_batch(0), *_batch(10)])
        assert keys_in(path, _UNIT, _UNIT_CASTERS) == {
            (f"id{start + n}", n) for start in (0, 10) for n in range(len(_TEXTS))}

    def test_a_row_wider_than_the_header_raises(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("input_id,iteration\na,0\nb,1,extra\n", encoding="utf-8")
        with pytest.raises(pd.errors.ParserError):
            keys_in(path, _UNIT, _UNIT_CASTERS)

    def test_a_file_cut_off_inside_a_cell_raises(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text(
            'input_id,iteration,sample\na,0,"whole"\nb,1,"cut o', encoding="utf-8"
        )
        with pytest.raises(pd.errors.ParserError):
            keys_in(path, _UNIT, _UNIT_CASTERS)


class TestCheckColumns:
    """What ``append_rows`` checks, for a save that writes several files."""

    def test_a_column_the_header_lacks_raises_as_the_append_would(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x"}])
        rows = [{"input_id": "b", "sample": "y"},
                {"input_id": "c", "sample": "z", "note": "new", "more": 1}]
        with pytest.raises(ValueError) as checked:
            check_columns(path, rows)
        with pytest.raises(ValueError) as appended:
            append_rows(path, rows)
        assert str(checked.value) == str(appended.value)
        assert "'note', 'more'" in str(checked.value)

    def test_rows_the_file_can_take_pass_and_write_nothing(self, tmp_path):
        path = tmp_path / "out.csv"
        append_rows(path, [{"input_id": "a", "sample": "x", "model": "m"}])
        before = path.read_bytes()
        check_columns(path, [{"sample": "y", "input_id": "b"}])     # fewer, reordered
        assert path.read_bytes() == before

    def test_a_file_with_no_header_takes_any_columns(self, tmp_path):
        check_columns(tmp_path / "new.csv", [{"anything": 1}])
        assert list(tmp_path.iterdir()) == []


# A verdict cell as the library, a spreadsheet or a hand edit writes it, and
# whether it reads as accepted.
_VERDICTS = [("True", True), ("TRUE", True), ("true", True), ("False", False),
             ("FALSE", False), ("", False), ("yes", False), ("1", False),
             ("0", False)]


class TestAcceptedMask:
    def test_true_in_any_case_and_nothing_else(self):
        cells = pd.Series([cell for cell, _ in _VERDICTS], dtype=str)
        assert accepted_mask(cells).tolist() == [ok for _, ok in _VERDICTS]

    def test_a_column_read_as_booleans(self):
        assert accepted_mask(pd.Series([True, False])).tolist() == [True, False]

    def test_the_tree_and_the_build_keep_the_same_rows(self, tmp_path):
        from redact.dataset import merge_inputs

        path = tmp_path / "Datasets" / "Cyber" / "samples.csv"
        path.parent.mkdir(parents=True)
        path.write_text("sample_id,sample,accepted\n" + "".join(
            f"id{n},t{n},{cell}\n" for n, (cell, _) in enumerate(_VERDICTS)),
            encoding="utf-8")
        kept = [f"id{n}" for n, (_, ok) in enumerate(_VERDICTS) if ok]
        assert list(merge_inputs(tmp_path)["sample_id"]) == kept
        out = tmp_path / "out.csv"
        write_concat(out, [Part(path, accepted_only=True)])
        assert list(read_rows(out)["sample_id"]) == kept


class TestWriteConcat:
    def _parts(self, tmp_path):
        """A file of three rows, one not accepted, and a frame with other columns."""
        path = tmp_path / "a.csv"
        append_rows(path, [{"sample_id": f"a{n}", "sample": text, "accepted": n != 1}
                           for n, text in enumerate(_TEXTS[:3])])
        frame = pd.DataFrame({"sample_id": ["b0"], "score": [1.0], "note": ["x"]})
        return path, frame

    def test_writes_what_concatenating_the_parts_gives(self, tmp_path):
        path, frame = self._parts(tmp_path)
        out = tmp_path / "out.csv"
        assert write_concat(out, [Part(path), Part(frame)]) == [3, 1]
        expected = pd.concat([read_rows(path), frame], ignore_index=True)
        # Read back, a column of text or of verdicts that a part lacks is
        # empty in its rows; one of numbers is missing.
        expected = expected.fillna({"sample": "", "accepted": "", "note": ""})
        pd.testing.assert_frame_equal(read_rows(out), expected)

    def test_a_dataset_type_tags_the_rows_of_its_part(self, tmp_path):
        path, frame = self._parts(tmp_path)
        out = tmp_path / "out.csv"
        write_concat(out, [Part(path, "input"), Part(frame, "output")])
        assert read_rows(out)["dataset_type"].tolist() == ["input"] * 3 + ["output"]

    def test_accepted_only_keeps_the_rows_that_read_true(self, tmp_path):
        path, _ = self._parts(tmp_path)
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("True", "TRUE"), encoding="utf-8")
        out = tmp_path / "out.csv"
        assert write_concat(out, [Part(path, accepted_only=True)]) == [2]
        assert read_rows(out)["sample_id"].tolist() == ["a0", "a2"]

    def test_a_file_without_an_accepted_column_is_copied_whole(self, tmp_path):
        _, frame = self._parts(tmp_path)
        path = tmp_path / "b.csv"
        frame.to_csv(path, index=False)
        out = tmp_path / "out.csv"
        assert write_concat(out, [Part(path, accepted_only=True)]) == [1]

    def test_a_whole_number_is_written_as_it_stands(self, tmp_path):
        """Also where another part leaves its column empty."""
        out = tmp_path / "out.csv"
        write_concat(out, [Part(pd.DataFrame({"turn": [0.0, None]})),
                           Part(pd.DataFrame({"other": ["x"]}))])
        assert _lines(out) == ["turn,other", "0,", ",", ",x"]

    def test_the_size_of_a_chunk_does_not_change_a_byte(self, tmp_path, monkeypatch):
        path, frame = self._parts(tmp_path)
        parts = [Part(path, "input", accepted_only=True), Part(frame, "output")]
        one, many = tmp_path / "one.csv", tmp_path / "many.csv"
        write_concat(one, parts)
        monkeypatch.setattr(rows_module, "_CHUNK_ROWS", 1)
        write_concat(many, parts)
        assert many.read_bytes() == one.read_bytes()

    def test_keep_empty_false_leaves_the_target_when_no_part_has_a_row(self, tmp_path):
        out = tmp_path / "out.csv"
        out.write_text("old\n", encoding="utf-8")
        empty = pd.DataFrame({"sample_id": []})
        assert write_concat(out, [Part(empty)], keep_empty=False) == [0]
        assert _lines(out) == ["old"]
        write_concat(out, [Part(empty)])
        assert _lines(out) == ["sample_id"]

    def test_a_part_that_fails_leaves_the_target_as_it_was(self, tmp_path):
        path, _ = self._parts(tmp_path)
        out = tmp_path / "out.csv"
        write_concat(out, [Part(path)])
        before = out.read_bytes()
        broken = tmp_path / "broken.csv"
        broken.write_text('sample_id,sample\nb0,"a cell that never', encoding="utf-8")
        with pytest.raises(pd.errors.ParserError):
            write_concat(out, [Part(path), Part(broken)])
        assert out.read_bytes() == before
        assert not out.with_name("out.csv.tmp").exists()
