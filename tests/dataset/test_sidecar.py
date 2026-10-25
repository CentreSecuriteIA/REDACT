"""Tests for how a sidecar file is written and read.

A manifest is replaced whole or not at all, and read one row at a time. A
ledger is appended to as bytes, and says with ``_saved`` how much of each
file its units cover.
"""

import json
import os
from pathlib import Path

import pytest

import redact.dataset.rows as rows
import redact.dataset.sidecar as sidecar
from redact.dataset import Ledger, Manifest


def _names(folder):
    return sorted(p.name for p in folder.iterdir())


@pytest.fixture()
def held_ledger(monkeypatch):
    """Refuse opens of a ledger for append as Windows does while another
    program holds the file. Returns the attempts made."""

    def hold(refusals):
        attempts = []
        opened = Path.open

        def open_(path, mode="r", *args, **kwargs):
            if not mode.startswith("a"):
                return opened(path, mode, *args, **kwargs)
            attempts.append(path.name)
            if len(attempts) <= refusals:
                raise PermissionError("held by another program")
            return opened(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_)
        monkeypatch.setattr(rows.time, "sleep", lambda seconds: None)
        return attempts

    return hold


def _manifest(tmp_path, rows=()):
    man = Manifest(tmp_path / "run.manifest.jsonl")
    man.write(rows)
    return man


def _ledger(tmp_path, key_fields=("id",), casters=None):
    return Ledger(tmp_path / "run.state.jsonl", key_fields=key_fields, casters=casters)


def _records(led):
    return [json.loads(line) for line in led.path.read_bytes().split(b"\n") if line]


class TestManifestWrite:
    def test_a_write_streams_its_rows_through_a_temporary_file(self, tmp_path):
        man = _manifest(tmp_path, [{"id": "old"}])
        seen = []

        def rows():
            yield {"id": "1"}
            # The first row is taken, the old manifest still in place.
            seen.append((_names(tmp_path), man.load()))
            yield {"id": "2"}

        man.write(rows())
        assert seen == [
            (["run.manifest.jsonl", "run.manifest.jsonl.tmp"], [{"id": "old"}])
        ]
        assert man.load() == [{"id": "1"}, {"id": "2"}]
        assert _names(tmp_path) == ["run.manifest.jsonl"]

    def test_rows_that_raise_half_way_keep_the_old_manifest(self, tmp_path):
        man = _manifest(tmp_path, [{"id": "1"}, {"id": "2"}])
        before = man.path.read_bytes()

        def rows():
            yield {"id": "3"}
            raise RuntimeError("planning failed")

        with pytest.raises(RuntimeError, match="planning failed"):
            man.write(rows())
        assert man.path.read_bytes() == before
        assert _names(tmp_path) == ["run.manifest.jsonl"]

    def test_rows_that_raise_half_way_leave_no_first_manifest(self, tmp_path):
        man = _manifest(tmp_path)

        def rows():
            yield {"id": "1"}
            raise RuntimeError("planning failed")

        with pytest.raises(RuntimeError, match="planning failed"):
            man.write(rows())
        assert _names(tmp_path) == []

    def test_a_refused_rename_is_tried_again(self, tmp_path, monkeypatch):
        man = _manifest(tmp_path, [{"id": "1"}])
        rename, refused, attempts = os.replace, 2, []

        def replace(source, target):
            attempts.append(source)
            if len(attempts) <= refused:
                raise PermissionError("held by another process")
            rename(source, target)

        monkeypatch.setattr(rows.os, "replace", replace)
        monkeypatch.setattr(rows.time, "sleep", lambda seconds: None)
        man.write([{"id": "2"}])
        assert len(attempts) == refused + 1
        assert man.load() == [{"id": "2"}]
        assert _names(tmp_path) == ["run.manifest.jsonl"]

    def test_a_rename_that_stays_refused_keeps_the_old_manifest(
        self, tmp_path, monkeypatch
    ):
        man = _manifest(tmp_path, [{"id": "1"}])
        attempts = []

        def replace(source, target):
            attempts.append(source)
            raise PermissionError("held by another process")

        monkeypatch.setattr(rows.os, "replace", replace)
        monkeypatch.setattr(rows.time, "sleep", lambda seconds: None)
        with pytest.raises(PermissionError):
            man.write([{"id": "2"}])
        assert len(attempts) > 1
        assert man.load() == [{"id": "1"}]
        assert _names(tmp_path) == ["run.manifest.jsonl"]

    def test_no_rows_from_an_iterator_create_nothing(self, tmp_path):
        man = Manifest(tmp_path / "sub" / "run.manifest.jsonl")
        man.write(row for row in ())
        assert _names(tmp_path) == []

    def test_no_rows_from_an_iterator_leave_the_old_manifest(self, tmp_path):
        man = _manifest(tmp_path, [{"id": "1"}])
        before = man.path.read_bytes()
        man.write(row for row in ())
        assert man.path.read_bytes() == before
        assert _names(tmp_path) == ["run.manifest.jsonl"]

    def test_the_bytes_are_those_of_a_plain_text_write(self, tmp_path):
        """Line endings stay the platform's, as before the temporary file."""
        rows = [{"id": "1", "text": "café"}, {"id": "2", "plan": ["a", "b"]}]
        plain = tmp_path / "plain.jsonl"
        with plain.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        assert _manifest(tmp_path, rows).path.read_bytes() == plain.read_bytes()


class TestManifestIter:
    def test_iterating_gives_the_rows_load_gives(self, tmp_path):
        man = _manifest(tmp_path, [{"id": "1"}, {"id": "2"}, {"id": "3"}])
        with man.path.open("a", encoding="utf-8") as fh:
            fh.write("not json at all\n\n" + json.dumps({"id": "4"}) + "\n")
        assert list(man) == man.load() == [
            {"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]

    def test_rows_come_one_at_a_time(self, tmp_path):
        rows = iter(_manifest(tmp_path, [{"id": "1"}, {"id": "2"}]))
        assert next(rows) == {"id": "1"}
        assert next(rows) == {"id": "2"}
        with pytest.raises(StopIteration):
            next(rows)

    def test_no_file_gives_no_rows(self, tmp_path):
        assert list(Manifest(tmp_path / "nope.manifest.jsonl")) == []


class TestLedgerRecords:
    def test_each_unit_line_comes_in_the_order_written(self, tmp_path):
        led = Ledger(tmp_path / "run.state.jsonl", key_fields=("id",))
        led.record([], saved={})
        led.record([{"id": "1", "extra": [1]}, {"id": "2"}], saved={"f": [1, "x"]})
        with led.path.open("a", encoding="utf-8") as fh:
            fh.write("not json\n" + json.dumps({"other": 1}) + "\n")
        records = list(led.records())
        assert [r["id"] for r in records] == ["1", "2"]
        assert records[0]["extra"] == [1]

    def test_no_file_gives_no_records(self, tmp_path):
        led = Ledger(tmp_path / "x.state.jsonl", key_fields=("id",))
        assert list(led.records()) == []

    def test_the_key_of_a_record_is_cast_as_on_read(self, tmp_path):
        led = Ledger(tmp_path / "x.state.jsonl", key_fields=("id", "n"),
                     casters={"n": int})
        assert led.key({"id": "a", "n": "3", "other": 1}) == ("a", 3)
        with pytest.raises(KeyError):
            led.key({"id": "a"})


class TestLedgerAppend:
    def test_records_that_raise_half_way_append_nothing(self, tmp_path):
        led = Ledger(tmp_path / "run.state.jsonl", key_fields=("id",))
        led.record([{"id": "1"}])
        before = led.path.read_bytes()

        def records():
            yield {"id": "2"}
            raise RuntimeError("no more")

        with pytest.raises(RuntimeError, match="no more"):
            led.record(records())
        assert led.path.read_bytes() == before
        assert _names(tmp_path) == ["run.state.jsonl"]

    def test_records_from_an_iterator_are_appended(self, tmp_path):
        led = Ledger(tmp_path / "run.state.jsonl", key_fields=("id",))
        led.record([{"id": "1"}])
        led.record({"id": str(n)} for n in (2, 3))
        lines = led.path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [
            {"id": "1"}, {"id": "2"}, {"id": "3"}]

    def test_lines_end_in_a_newline_on_every_platform(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1", "text": "café"}, {"id": "2"}])
        assert led.path.read_bytes() == (
            '{"id": "1", "text": "café"}\n{"id": "2"}\n'.encode())

    def test_an_append_is_one_write_synced_to_disk(self, tmp_path, monkeypatch):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}])
        synced = []
        monkeypatch.setattr(
            sidecar.os, "fsync", lambda fd: synced.append(led.path.read_bytes()))
        led.record([{"id": "2"}, {"id": "3"}])
        # Every byte of the append was in the file when it was synced.
        assert synced == [b'{"id": "1"}\n{"id": "2"}\n{"id": "3"}\n']

    @pytest.mark.parametrize("torn, counted", [
        (b'{"id": "2"}', {"1", "2", "3"}),      # all of the line but its newline
        (b'{"id": "2', {"1", "3"}),             # cut inside the line
    ])
    def test_an_append_after_a_torn_line_starts_on_a_new_line(
            self, tmp_path, torn, counted):
        """Written onto the torn line, the first new record would not parse."""
        led = _ledger(tmp_path)
        led.path.write_bytes(b'{"id": "1"}\n' + torn)
        assert led.completed() == {"1"}
        led.record([{"id": "3"}])
        assert led.path.read_bytes() == b'{"id": "1"}\n' + torn + b'\n{"id": "3"}\n'
        assert _ledger(tmp_path).completed() == counted

    def test_an_append_to_an_empty_file_starts_at_its_first_byte(self, tmp_path):
        led = _ledger(tmp_path)
        led.path.touch()
        led.record([{"id": "1"}])
        assert led.path.read_bytes() == b'{"id": "1"}\n'

    def test_a_ledger_another_program_holds_is_opened_again(
            self, tmp_path, held_ledger):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}])
        attempts = held_ledger(refusals=2)
        led.record([{"id": "2"}])
        assert attempts == ["run.state.jsonl"] * 3
        assert led.path.read_bytes() == b'{"id": "1"}\n{"id": "2"}\n'

    def test_a_ledger_that_stays_held_raises_and_gets_no_byte(
            self, tmp_path, held_ledger):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}])
        attempts = held_ledger(refusals=10**6)
        with pytest.raises(PermissionError):
            led.record([{"id": "2"}])
        assert len(attempts) > 1 and led.path.read_bytes() == b'{"id": "1"}\n'


_SAVED = {"out.csv": [120, "0a1b2c3d"]}


class TestLedgerSaved:
    """``_saved``: how much of each file the units recorded so far cover."""

    def test_saved_goes_on_the_last_record(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}, {"id": "2"}], saved=_SAVED)
        assert _records(led) == [{"id": "1"}, {"id": "2", "_saved": _SAVED}]

    def test_the_callers_record_is_left_as_it_was(self, tmp_path):
        last = {"id": "2"}
        _ledger(tmp_path).record([{"id": "1"}, last], saved=_SAVED)
        assert last == {"id": "2"}

    def test_saved_with_no_records_is_a_line_of_its_own(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([], saved={})
        assert led.path.read_bytes() == b'{"_saved": {}}\n'

    def test_the_ledger_knows_what_it_last_recorded(self, tmp_path):
        led = _ledger(tmp_path)
        assert led.saved is None
        led.record([{"id": "1"}], saved=_SAVED)
        assert led.saved == _SAVED
        # Another object on the same file learns it from the file.
        assert _ledger(tmp_path).saved == _SAVED

    def test_a_reset_forgets_what_was_recorded(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}], saved=_SAVED)
        led.reset()
        assert led.saved is None

    def test_records_without_saved_are_refused_once_the_ledger_holds_it(self, tmp_path):
        """They would sit after the last ``_saved`` line, where no reader
        counts them."""
        _ledger(tmp_path).record([{"id": "1"}], saved=_SAVED)
        led = _ledger(tmp_path)
        before = led.path.read_bytes()
        with pytest.raises(ValueError, match="_saved"):
            led.record([{"id": "2"}])
        assert led.path.read_bytes() == before

    def test_no_records_and_no_saved_write_nothing_on_any_ledger(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}], saved=_SAVED)
        before = led.path.read_bytes()
        led.record([])
        assert led.path.read_bytes() == before


class TestLedgerRead:
    def test_no_file_reads_as_an_empty_ledger(self, tmp_path):
        assert _ledger(tmp_path).read() == (set(), None, 0)

    def test_a_ledger_without_saved_counts_every_line(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}, {"id": "2"}])
        done, saved, end = led.read()
        assert (done, saved, end) == ({"1", "2"}, None, led.path.stat().st_size)

    def test_lines_after_the_last_saved_are_not_finished_units(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1"}, {"id": "2"}], saved=_SAVED)
        counted = led.path.stat().st_size
        with led.path.open("ab") as fh:     # a save that lost its last line
            fh.write(b'{"id": "3"}\n')
        done, saved, end = led.read()
        assert (done, saved, end) == ({"1", "2"}, _SAVED, counted)
        assert led.completed() == {"1", "2"}

    def test_saved_is_that_of_the_last_line_that_carries_it(self, tmp_path):
        led = _ledger(tmp_path)
        newer = {"out.csv": [300, "ffffffff"]}
        led.record([{"id": "1"}], saved=_SAVED)
        led.record([{"id": "2"}], saved=newer)
        assert led.read() == ({"1", "2"}, newer, led.path.stat().st_size)

    def test_a_save_with_no_rows_still_marks_its_units(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([], saved={})
        led.record([{"id": "dropped"}], saved={})
        assert led.read().done == {"dropped"}

    @pytest.mark.parametrize("saved", [None, _SAVED])
    def test_a_line_without_its_newline_does_not_count(self, tmp_path, saved):
        """A write that was cut off is not a line, even when it parses."""
        led = _ledger(tmp_path)
        led.record([{"id": "1"}], saved=saved)
        whole = led.path.stat().st_size
        with led.path.open("ab") as fh:
            fh.write(json.dumps({"id": "2", "_saved": {}}).encode("utf-8"))
        assert led.read() == ({"1"}, saved, whole)

    def test_lines_that_end_in_a_carriage_return_are_read(self, tmp_path):
        """A ledger written on Windows before lines ended in a bare newline."""
        led = _ledger(tmp_path)
        led.path.write_bytes(b'{"id": "1"}\r\n{"id": "2"}\r\n')
        led.record([{"id": "3"}])
        assert led.read() == ({"1", "2", "3"}, None, led.path.stat().st_size)

    def test_a_key_with_unicode_line_separators_comes_back_whole(self, tmp_path):
        key = "one two\u0085three"
        led = _ledger(tmp_path)
        led.record([{"id": key}, {"id": "plain"}])
        assert led.completed() == {key, "plain"}

    def test_lines_that_are_not_units_are_skipped(self, tmp_path):
        led = _ledger(tmp_path)
        led.path.write_bytes(
            b'{"id": "1"}\nnot json\n\n[1, 2]\n"text"\n{"other": 1}\n\xff\xfe\n'
            b'{"id": "2"}\n')
        assert led.read() == ({"1", "2"}, None, led.path.stat().st_size)

    def test_composite_keys_are_cast_as_completed_casts_them(self, tmp_path):
        led = _ledger(tmp_path, ("input_id", "iteration"),
                      {"input_id": str, "iteration": int})
        led.record([{"input_id": 7, "iteration": "0"}], saved={})
        assert led.read().done == {("7", 0)}


class TestLedgerIds:
    """``ids``: the content ids of the rows the finished units wrote."""

    def test_the_ids_of_every_finished_unit_are_one_set(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1", "ids": ["a", "b"]}, {"id": "2"}], saved={})
        led.record([{"id": "3", "ids": ["c"]}, {"id": "4", "ids": []}], saved={})
        assert _ledger(tmp_path).ids == {"a", "b", "c"}

    def test_ids_after_the_last_saved_do_not_count(self, tmp_path):
        """Their rows are cut off the output, and may be written again."""
        led = _ledger(tmp_path)
        led.record([{"id": "1", "ids": ["a"]}], saved={})
        with led.path.open("ab") as fh:     # a save that lost its last line
            fh.write(b'{"id": "2", "ids": ["b"]}\n')
        assert _ledger(tmp_path).ids == {"a"}

    def test_a_ledger_without_saved_counts_the_ids_of_every_line(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1", "ids": ["a"]}, {"id": "2", "ids": ["b"]}])
        assert _ledger(tmp_path).ids == {"a", "b"}

    def test_the_ledger_knows_the_ids_it_recorded(self, tmp_path):
        led = _ledger(tmp_path)
        assert led.ids == set()
        led.record([{"id": "1", "ids": ["a"]}], saved={})
        assert led.ids == {"a"}

    def test_a_line_that_is_no_unit_gives_its_ids_too(self, tmp_path):
        led = _ledger(tmp_path)
        led.path.write_bytes(b'{"ids": ["a"]}\n{"id": "1", "ids": 7}\n')
        assert (led.completed(), led.ids) == ({"1"}, {"a"})

    def test_a_reset_forgets_the_ids(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"id": "1", "ids": ["a"]}], saved={})
        led.reset()
        assert led.ids == set()
