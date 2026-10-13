"""Tests for a row's id."""

import pandas as pd
import pytest

from redact.dataset import content_id, row_id, row_ids

_TEXT = "the prompt"


def _row(value, id_col="sample_id"):
    """One DataFrame row whose id cell holds ``value`` as it is."""
    frame = pd.DataFrame({"prompt": [_TEXT], id_col: pd.Series([value], dtype=object)})
    return frame.iloc[0]


class TestContentId:
    def test_is_the_first_16_hex_chars_of_the_md5(self):
        assert content_id("hello") == "5d41402abc4b2a76"

    def test_same_text_gives_same_id(self):
        assert content_id("a prompt") == content_id("a prompt")

    def test_different_text_gives_different_id(self):
        assert content_id("a prompt") != content_id("another prompt")

    def test_hashes_the_utf8_bytes(self):
        assert content_id("é") == "66ddcd97cfdeabb2"


class TestRowId:
    @pytest.mark.parametrize(
        "value", [None, pd.NA, float("nan"), "", " ", "NaN"],
        ids=["None", "pd.NA", "float nan", "empty", "whitespace", "NaN"],
    )
    def test_a_missing_id_gives_the_content_id_of_the_text(self, value):
        assert row_id(_row(value), "prompt") == content_id(_TEXT)

    def test_a_literal_nan_string_is_a_missing_id(self):
        assert row_id(_row("nan"), "prompt") == content_id(_TEXT)

    @pytest.mark.parametrize(
        "value, expected", [("abc", "abc"), (" abc ", "abc"), (123, "123")],
        ids=["string", "padded string", "integer"],
    )
    def test_a_set_id_is_kept_as_a_stripped_string(self, value, expected):
        assert row_id(_row(value), "prompt") == expected

    def test_a_row_without_the_id_column_gives_the_content_id(self):
        row = pd.DataFrame({"prompt": [_TEXT]}).iloc[0]
        assert row_id(row, "prompt") == content_id(_TEXT)

    def test_id_col_names_the_column_the_id_is_read_from(self):
        row = _row("origin", id_col="input_id")
        assert row_id(row, "prompt", id_col="input_id") == "origin"
        assert row_id(row, "prompt") == content_id(_TEXT)


class TestRowIds:
    def _frame(self, id_col="sample_id"):
        ids = ["abc", None, " padded ", float("nan"), "nan", "", 123, pd.NA, "abc"]
        return pd.DataFrame({
            id_col: pd.Series(ids, dtype=object),
            "prompt": [f"prompt {n}" for n in range(len(ids))],
            "turn": range(len(ids)),
        })

    def test_is_the_row_id_of_every_row_in_order(self):
        frame = self._frame()
        ids = row_ids(frame, "prompt")
        assert ids == [row_id(row, "prompt") for _, row in frame.iterrows()]
        assert ids == ["abc", content_id("prompt 1"), "padded", content_id("prompt 3"),
                       content_id("prompt 4"), content_id("prompt 5"), "123",
                       content_id("prompt 7"), "abc"]

    def test_a_frame_without_the_id_column_gives_content_ids(self):
        frame = pd.DataFrame({"prompt": ["one", "two"]})
        assert row_ids(frame, "prompt") == [content_id("one"), content_id("two")]

    def test_id_col_names_the_column_the_ids_are_read_from(self):
        frame = self._frame(id_col="input_id")
        assert row_ids(frame, "prompt", id_col="input_id")[:3] == [
            "abc", content_id("prompt 1"), "padded"]
        assert row_ids(frame, "prompt")[0] == content_id("prompt 0")

    def test_a_frame_without_rows_gives_no_ids(self):
        assert row_ids(pd.DataFrame({"prompt": []}), "prompt") == []

    def test_the_index_of_the_frame_plays_no_part(self):
        frame = self._frame().iloc[::-1]
        assert row_ids(frame, "prompt")[0] == "abc"
        assert row_ids(frame, "prompt")[-1] == "abc"
        assert row_ids(frame, "prompt")[2] == "123"
