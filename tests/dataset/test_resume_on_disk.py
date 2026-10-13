"""What a resume does to the files of a stage that names its output.

Each test leaves a folder as an interrupted save, another program or an older
version of the library would, and resumes it with ``resume_state``.
"""

import csv
import io
import json
import logging
import os
import zlib
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock, Timeout

import redact.dataset.resume as resume_module
import redact.dataset.rows as rows_module
from redact.dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    drop_units,
    release_lock,
    resume_state,
)


def _ledger(folder):
    return Ledger(folder / "out.state.jsonl", key_fields=("input_id",),
                  casters={"input_id": str})


def _rows(keys):
    return [{"input_id": key, "text": f'text of "{key}", two\nlines'} for key in keys]


def _buffer(led, out, every=32):
    return CommitBuffer(
        append=lambda rows: append_rows(out, rows), ledger=led, every=every)


def _run(folder, keys, *, every=2, out=None):
    """Save one row for each of ``keys`` that is not finished, as a stage does."""
    led, out = _ledger(folder), out or folder / "out.csv"
    done = resume_state(led, resume=True, artifact=out)
    with _buffer(led, out, every) as buf:
        for key in keys:
            if key not in done:
                buf.add(_rows([key]), {"input_id": key})
    return led, out


def _resume(folder, out=None):
    """The finished units a run finds that resumes and ends, and its ledger."""
    led = _ledger(folder)
    done = resume_state(led, resume=True, artifact=out or folder / "out.csv")
    release_lock(led)
    return done, led


def _crc(data):
    return f"{zlib.crc32(data[-4096:]):08x}"


def _keys_on_disk(out):
    """The ``input_id`` of every row, from a parse that takes no torn row."""
    text = out.read_bytes().decode("utf-8")
    assert text.endswith("\n")
    header, *rows = csv.reader(io.StringIO(text, newline=""))
    assert all(len(row) == len(header) for row in rows)
    return [row[header.index("input_id")] for row in rows]


def _files(folder):
    return {path.name: path.read_bytes() for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix != ".lock"}


def _ledger_lines(led):
    return [json.loads(line) for line in led.path.read_bytes().split(b"\n") if line]


def _no_read(*args, **kwargs):
    raise AssertionError("the resume read the rows of the output")


@pytest.fixture()
def no_reading(monkeypatch):
    """Fail the test if a resume reads the rows of an output."""
    monkeypatch.setattr(resume_module, "keys_in", _no_read)
    monkeypatch.setattr(pd, "read_csv", _no_read)


class TestResumeFromTheLedger:
    def test_no_row_of_the_output_is_read(self, tmp_path, no_reading):
        _run(tmp_path, "abcd")
        before = _files(tmp_path)
        done, _led = _resume(tmp_path)
        assert done == set("abcd")
        assert _files(tmp_path) == before

    def test_rows_that_no_save_recorded_are_moved_to_cut(
            self, tmp_path, no_reading, caplog):
        _led, out = _run(tmp_path, "ab")
        saved = out.read_bytes()
        tail = b'c,"a whole row, added by hand"\r\n'
        out.write_bytes(saved + tail)

        with caplog.at_level(logging.WARNING, logger="redact.dataset.resume"):
            done, _led = _resume(tmp_path)
        assert done == set("ab")
        assert out.read_bytes() == saved and _keys_on_disk(out) == ["a", "b"]
        assert out.with_name("out.csv.cut").read_bytes() == tail
        (warning,) = caplog.records
        assert str(len(tail)) in warning.getMessage()
        assert "out.csv" in warning.getMessage()

    def test_a_second_cut_keeps_the_tail_of_the_first(self, tmp_path):
        _led, out = _run(tmp_path, "ab")
        saved = out.read_bytes()
        for tail in (b"first tail\n", b"second tail\n"):
            out.write_bytes(saved + tail)
            _resume(tmp_path)
        assert out.with_name("out.csv.cut").read_bytes() == b"first tail\nsecond tail\n"

    def test_the_cut_is_synced_to_disk(self, tmp_path, monkeypatch):
        _led, out = _run(tmp_path, "ab")
        saved = out.read_bytes()
        out.write_bytes(saved + b"tail\n")
        synced = []
        monkeypatch.setattr(os, "fsync", lambda fd: synced.append(
            (out.stat().st_size, out.with_name("out.csv.cut").stat().st_size)))
        _resume(tmp_path)
        # The tail is safe in .cut before the file is cut, and both are synced.
        assert synced == [(len(saved) + 5, 5), (len(saved), 5)]

    def test_a_save_that_stopped_after_its_rows_is_cut_back(self, tmp_path, no_reading):
        led, out = _run(tmp_path, "ab")
        before = _files(tmp_path)
        append_rows(out, _rows("cd"))           # the rows landed, the acks did not

        done, _led = _resume(tmp_path)
        assert done == set("ab")
        after = _files(tmp_path)
        assert after.pop("out.csv.cut") and after == before

    def test_a_save_that_stopped_inside_its_acks_is_cut_out_of_the_ledger(
            self, tmp_path, no_reading):
        led, out = _run(tmp_path, "ab")
        before = _files(tmp_path)
        append_rows(out, _rows("cd"))
        with led.path.open("ab") as fh:
            fh.write(b'{"input_id": "c"}\n{"input_id": "d", "_sav')

        done, _led = _resume(tmp_path)
        assert done == set("ab")
        assert _files(tmp_path)["out.state.jsonl"] == before["out.state.jsonl"]

        led, out = _run(tmp_path, "abcd")       # the next save starts on a new line
        assert led.completed() == set("abcd")
        assert _keys_on_disk(out) == list("abcd")
        assert all("input_id" in line or line == {"_saved": {}}
                   for line in _ledger_lines(led))

    def test_acks_that_lost_their_last_newline_are_not_a_finished_save(self, tmp_path):
        """The line parses, but the next save would be written onto its end."""
        led, out = _run(tmp_path, "abcd")
        led.path.write_bytes(led.path.read_bytes()[:-1])

        done, _led = _resume(tmp_path)
        assert done == set("ab") and _keys_on_disk(out) == ["a", "b"]

        led, out = _run(tmp_path, "abcd")
        assert led.completed() == set("abcd")
        assert _keys_on_disk(out) == list("abcd")

    @pytest.mark.parametrize("landed", [b"", b"input_", None])
    def test_a_first_save_that_stopped_in_its_rows_starts_clean(
            self, tmp_path, no_reading, landed):
        """``None``: every row landed, and the acks did not."""
        led, out = _ledger(tmp_path), tmp_path / "out.csv"
        led.record([], saved={})                # the opening line
        if landed is None:
            append_rows(out, _rows("ab"))
            landed = out.read_bytes()
        else:
            out.write_bytes(landed)

        done, _led = _resume(tmp_path)
        assert done == set() and not out.exists()
        assert _files(tmp_path).get("out.csv.cut", b"") == landed

        led, out = _run(tmp_path, "ab")         # the next save writes the header
        assert _keys_on_disk(out) == ["a", "b"] and led.completed() == set("ab")

    def test_an_opening_line_cut_off_mid_write_starts_clean(self, tmp_path, no_reading):
        led = _ledger(tmp_path)
        led.path.write_bytes(b'{"_sav')
        done, _led = _resume(tmp_path)
        assert done == set()

        led, out = _run(tmp_path, "ab")
        assert _ledger_lines(led)[0] == {"_saved": {}}
        assert led.completed() == set("ab") and _keys_on_disk(out) == ["a", "b"]

    def test_a_last_save_with_no_rows_leaves_every_file_as_it_is(
            self, tmp_path, no_reading):
        led, out = _run(tmp_path, "ab")
        resume_state(led, resume=True, artifact=out)
        with _buffer(led, out) as buf:
            buf.add([], {"input_id": "dropped"})
        before = _files(tmp_path)

        done, _led = _resume(tmp_path)
        assert done == {"a", "b", "dropped"} and _files(tmp_path) == before

    def test_an_artifact_in_a_subfolder_is_found_by_its_recorded_name(
            self, tmp_path, no_reading):
        out = tmp_path / "Cyber Crime" / "samples.csv"
        out.parent.mkdir()
        led, out = _run(tmp_path, "ab", out=out)
        assert list(led.saved) == ["Cyber Crime/samples.csv"]
        saved = out.read_bytes()
        out.write_bytes(saved + b"tail\n")

        done, _led = _resume(tmp_path, out)
        assert done == set("ab") and out.read_bytes() == saved

    def test_an_artifact_under_another_spelling_is_the_file_the_ledger_records(
            self, tmp_path, no_reading):
        """On Windows names differ in case only: ``OUT.csv`` is ``out.csv``,
        and must not be moved aside as a file no save recorded."""
        _led, out = _run(tmp_path, "abcd")
        other = tmp_path / "OUT.csv"
        if not other.exists():
            pytest.skip("this file system tells out.csv and OUT.csv apart")
        before = _files(tmp_path)

        led = Ledger.sidecar(other, ("input_id",), {"input_id": str})
        assert resume_state(led, resume=True, artifact=other) == set("abcd")
        assert _files(tmp_path) == before               # nothing cut, nothing moved

        with _buffer(led, other) as buf:                # and the next save goes on
            buf.add(_rows("e"), {"input_id": "e"})
        assert led.saved == {"out.csv": [out.stat().st_size, _crc(out.read_bytes())]}
        assert _keys_on_disk(out) == list("abcde")
        assert _resume(tmp_path)[0] == set("abcde")     # under the first spelling

    def test_a_resume_keeps_extra_keys(self, tmp_path):
        _run(tmp_path, "ab")
        led = _ledger(tmp_path)
        assert resume_state(led, resume=True, artifact=tmp_path / "out.csv",
                            extra={"z"}) == {"a", "b", "z"}


def _quote_every_cell(out):
    """Save the file again as a spreadsheet may: the same rows, more bytes."""
    rows = list(csv.reader(io.StringIO(out.read_bytes().decode("utf-8"), newline="")))
    text = io.StringIO(newline="")
    csv.writer(text, quoting=csv.QUOTE_ALL).writerows(rows)
    out.write_bytes(text.getvalue().encode("utf-8"))


def _change_a_byte(out):
    """One other letter in the last row: the same size, and still a CSV."""
    data = out.read_bytes()
    at = data.rindex(b"lines")
    out.write_bytes(data[:at] + b"linez" + data[at + 5:])


_DAMAGE = {
    "a changed byte": _change_a_byte,
    "ten bytes shorter": lambda out: out.write_bytes(out.read_bytes()[:-10]),
    "saved again, larger": _quote_every_cell,
    "deleted": lambda out: out.unlink(),
}


class TestAFileTheLedgerCannotVouchFor:
    @pytest.mark.parametrize("damage", _DAMAGE)
    def test_the_resume_stops_and_changes_no_file(self, tmp_path, damage, no_reading):
        _led, out = _run(tmp_path, "abcd")
        _DAMAGE[damage](out)
        before = _files(tmp_path)

        with pytest.raises(RuntimeError) as stopped:
            _resume(tmp_path)
        assert _files(tmp_path) == before
        message = str(stopped.value)
        assert str(out) in message and "out.state.jsonl" in message
        # The three ways out.
        assert "Restore" in message
        assert "delete the ledger" in message
        assert "resume=False" in message

    def test_a_resume_that_stops_holds_no_lock(self, tmp_path):
        led, out = _run(tmp_path, "ab")
        out.unlink()
        with pytest.raises(RuntimeError):
            _resume(tmp_path)
        lock = FileLock(led.path.with_name("out.state.jsonl.lock"), timeout=0)
        lock.acquire()                          # another run could start
        lock.release()

    def test_nothing_is_cut_before_every_file_is_checked(self, tmp_path):
        led, first = _ledger(tmp_path), tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        resume_state(led, resume=True, artifact=first)
        with CommitBuffer(append=lambda rows: {
                **append_rows(first, rows), **append_rows(second, rows)},
                ledger=led) as buf:
            buf.add(_rows("a"), {"input_id": "a"})
        first.write_bytes(first.read_bytes() + b"a tail\n")
        _change_a_byte(second)
        before = _files(tmp_path)

        with pytest.raises(RuntimeError, match="b.csv"):
            resume_state(_ledger(tmp_path), resume=True, artifact=first)
        assert _files(tmp_path) == before

    def test_deleting_the_ledger_takes_over_what_the_file_holds(self, tmp_path):
        """The way out that keeps the rows: units without rows run again."""
        led, out = _run(tmp_path, "abcd")
        _change_a_byte(out)
        led.path.unlink()
        done, led = _resume(tmp_path)
        assert done == set("abcd")
        assert led.saved["out.csv"][0] == out.stat().st_size


class _HandleThatDoesNotCut:
    """A handle on a file another program has mapped, as a cut by path meets
    it on Windows: ``truncate`` returns and the file keeps its size."""

    def __init__(self, handle):
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()

    def truncate(self, size):
        pass

    def fileno(self):
        return self._handle.fileno()


class TestTheCut:
    def _held(self, monkeypatch, refusals, handle=None):
        """Refuse the first ``refusals`` opens for a cut; count every attempt."""
        attempts = []
        opened = Path.open

        def open_(path, mode="r", *args, **kwargs):
            if mode != "r+b":
                return opened(path, mode, *args, **kwargs)
            attempts.append(path.name)
            if len(attempts) <= refusals:
                raise PermissionError("held by another program")
            fh = opened(path, mode, *args, **kwargs)
            return handle(fh) if handle else fh

        monkeypatch.setattr(Path, "open", open_)
        monkeypatch.setattr(rows_module.time, "sleep", lambda seconds: None)
        return attempts

    def test_a_refused_cut_is_tried_again(self, tmp_path, monkeypatch):
        _led, out = _run(tmp_path, "ab")
        saved = out.read_bytes()
        out.write_bytes(saved + b"tail\n")
        attempts = self._held(monkeypatch, refusals=2)
        _resume(tmp_path)
        assert attempts == ["out.csv"] * 3 and out.read_bytes() == saved

    def test_a_cut_that_stays_refused_stops_the_resume(self, tmp_path, monkeypatch):
        _led, out = _run(tmp_path, "ab")
        held = out.read_bytes() + b"tail\n"
        out.write_bytes(held)
        attempts = self._held(monkeypatch, refusals=10**6)
        with pytest.raises(RuntimeError, match="another program holds") as stopped:
            _resume(tmp_path)
        assert str(out) in str(stopped.value)
        assert len(attempts) > 1 and out.read_bytes() == held

    def test_a_removal_that_stays_refused_stops_the_resume_the_same_way(
            self, tmp_path, monkeypatch):
        """A file no save recorded is removed, not cut."""
        led, out = _ledger(tmp_path), tmp_path / "out.csv"
        led.record([], saved={})
        out.write_bytes(b"input_")

        def unlink(path, missing_ok=False):
            raise PermissionError("held by another program")

        monkeypatch.setattr(Path, "unlink", unlink)
        monkeypatch.setattr(rows_module.time, "sleep", lambda seconds: None)
        with pytest.raises(RuntimeError, match="another program holds") as stopped:
            _resume(tmp_path)
        assert str(out) in str(stopped.value) and out.read_bytes() == b"input_"

    def test_a_cut_that_did_not_take_stops_the_resume(self, tmp_path, monkeypatch):
        _led, out = _run(tmp_path, "ab")
        out.write_bytes(out.read_bytes() + b"tail\n")
        self._held(monkeypatch, refusals=0, handle=_HandleThatDoesNotCut)
        with pytest.raises(RuntimeError, match="out.csv"):
            _resume(tmp_path)

    def test_the_ledger_is_cut_the_same_way(self, tmp_path, monkeypatch):
        led, _out = _run(tmp_path, "ab")
        saved = led.path.read_bytes()
        led.path.write_bytes(saved + b'{"input_id": "c"}\n')
        attempts = self._held(monkeypatch, refusals=1)
        _resume(tmp_path)
        assert attempts == ["out.state.jsonl"] * 2 and led.path.read_bytes() == saved


def _old_folder(folder, acked, rows, line_end=b"\n"):
    """A folder as the library wrote it before the ledger recorded sizes."""
    led, out = _ledger(folder), folder / "out.csv"
    if rows:
        append_rows(out, _rows(rows))
    if acked:
        led.path.write_bytes(b"".join(
            json.dumps({"input_id": key}).encode() + line_end for key in acked))
    return led, out


class TestAFolderWrittenBeforeSizesWereRecorded:
    def test_rows_without_acks_are_acked_once_and_never_read_again(
            self, tmp_path, monkeypatch):
        led, out = _old_folder(tmp_path, acked="ab", rows="abcd")
        old_lines = led.path.read_bytes()
        records = []
        record = Ledger.record
        monkeypatch.setattr(Ledger, "record", lambda self, *args, **kwargs: (
            records.append(args), record(self, *args, **kwargs))[1])

        done, led = _resume(tmp_path)
        assert done == set("abcd")
        assert len(records) == 1                # one ledger write
        assert led.path.read_bytes().startswith(old_lines)
        first, last = _ledger_lines(led)[2:]
        assert {first["input_id"], last["input_id"]} == set("cd")
        data = out.read_bytes()
        assert "_saved" not in first
        assert last["_saved"] == {"out.csv": [len(data), _crc(data)]}

        monkeypatch.setattr(resume_module, "keys_in", _no_read)
        monkeypatch.setattr(pd, "read_csv", _no_read)
        assert _resume(tmp_path)[0] == set("abcd")
        assert out.read_bytes() == data

    def test_an_output_with_no_ledger_is_taken_over_as_it_is(self, tmp_path):
        led, out = _old_folder(tmp_path, acked="", rows="abc")
        done, led = _resume(tmp_path)
        assert done == set("abc") and led.completed() == set("abc")
        assert led.saved["out.csv"][0] == out.stat().st_size

    def test_a_ledger_with_no_output_is_kept(self, tmp_path):
        led, _out = _old_folder(tmp_path, acked="ab", rows="")
        done, led = _resume(tmp_path)
        assert done == set("ab")
        assert _ledger_lines(led) == [
            {"input_id": "a"}, {"input_id": "b"}, {"_saved": {}}]

    @pytest.mark.parametrize("acked", ["ab", ""])
    def test_an_empty_output_is_taken_over_as_no_file(self, tmp_path, acked):
        """A first save that died before its first byte left the file."""
        led, out = _old_folder(tmp_path, acked=acked, rows="")
        out.touch()
        done, led = _resume(tmp_path)
        assert done == set(acked) and not out.exists()
        assert not out.with_name("out.csv.cut").exists()
        expected = [{"input_id": key} for key in acked] + [{"_saved": {}}] * bool(acked)
        assert (_ledger_lines(led) if led.exists() else []) == expected

    def test_a_new_folder_is_left_empty(self, tmp_path):
        done, led = _resume(tmp_path)
        assert done == set() and _files(tmp_path) == {}

    def test_acks_after_a_torn_last_line_start_on_a_new_line(self, tmp_path):
        led, out = _old_folder(tmp_path, acked="ab", rows="abc")
        led.path.write_bytes(led.path.read_bytes() + b'{"input_id": "lo')
        done, led = _resume(tmp_path)
        assert done == set("abc")
        assert _ledger_lines(led)[:2] == [{"input_id": "a"}, {"input_id": "b"}]
        assert led.completed() == set("abc")

    def test_a_ledger_with_windows_line_ends_is_taken_over(self, tmp_path):
        led, out = _old_folder(tmp_path, acked="ab", rows="abc", line_end=b"\r\n")
        done, led = _resume(tmp_path)
        assert done == set("abc") and led.completed() == set("abc")

    def test_the_next_saves_follow_the_takeover(self, tmp_path, no_reading):
        _old_folder(tmp_path, acked="ab", rows="")
        led, out = _run(tmp_path, "abcd")
        assert led.completed() == set("abcd") and _keys_on_disk(out) == ["c", "d"]
        assert _resume(tmp_path)[0] == set("abcd")

    @pytest.mark.parametrize(
        "tail", [b'e,"cut o', b"e,a row too many,cells\r\nf,x\r\n"])
    def test_an_output_that_does_not_parse_stops_the_resume(self, tmp_path, tail):
        led, out = _old_folder(tmp_path, acked="ab", rows="abcd")
        out.write_bytes(out.read_bytes() + tail)
        before = _files(tmp_path)
        with pytest.raises(RuntimeError) as stopped:
            _resume(tmp_path)
        assert str(out) in str(stopped.value) and "resume=False" in str(stopped.value)
        assert _files(tmp_path) == before

    def test_an_output_that_ends_inside_a_row_stops_the_resume(self, tmp_path):
        """pandas reads a row that lost its last cells as a short row, so the
        end of the file is checked on its own."""
        led, out = _old_folder(tmp_path, acked="ab", rows="abcd")
        out.write_bytes(out.read_bytes() + b"e")
        before = _files(tmp_path)
        with pytest.raises(RuntimeError, match="line break"):
            _resume(tmp_path)
        assert _files(tmp_path) == before


class TestAFreshRun:
    def test_it_removes_the_artifact_the_ledger_the_manifest_and_the_cut(
            self, tmp_path):
        led, out = _run(tmp_path, "ab")
        out.write_bytes(out.read_bytes() + b"tail\n")
        _resume(tmp_path)                       # leaves out.csv.cut
        manifest = Manifest.sidecar(out)
        manifest.write([{"input_id": "a"}])
        assert set(_files(tmp_path)) == {
            "out.csv", "out.csv.cut", "out.state.jsonl", "out.manifest.jsonl"}

        led = _ledger(tmp_path)
        assert resume_state(led, resume=False, artifact=out, manifest=manifest) == set()
        assert _files(tmp_path) == {}

        with _buffer(led, out) as buf:
            buf.add(_rows("c"), {"input_id": "c"})
        assert _ledger_lines(led)[0] == {"_saved": {}} and led.completed() == {"c"}


class TestAStageThatNamesNoArtifact:
    def test_its_ledger_is_read_and_written_as_before(self, tmp_path):
        """The constitution stage acks by hand; its units must count on the
        next run, which a ``_saved`` line written by a resume would undo."""
        path, found = tmp_path / "constitution.state.jsonl", []
        for units in (["A::harmful"], ["B::harmful"], []):      # three runs
            led = Ledger(path, ("unit",), {"unit": str})
            found.append(resume_state(led, resume=True))
            led.record([{"unit": unit} for unit in units])
        assert found == [set(), {"A::harmful"}, {"A::harmful", "B::harmful"}]
        assert path.read_bytes() == b'{"unit": "A::harmful"}\n{"unit": "B::harmful"}\n'
        assert list(tmp_path.iterdir()) == [path]

    def test_a_unit_whose_ack_lost_its_newline_runs_once_more_and_no_more(
            self, tmp_path):
        """Its next ack must not land on the torn line, where it would not
        parse, and send the unit round again."""
        path = tmp_path / "constitution.state.jsonl"
        path.write_bytes(b'{"unit": "A"}\n{"unit": "B"}')

        def run():
            led = Ledger(path, ("unit",), {"unit": str})
            done = resume_state(led, resume=True)
            led.record([{"unit": unit} for unit in "ABC" if unit not in done])
            return done

        assert [run(), run(), run()] == [{"A"}, set("ABC"), set("ABC")]


_TREE = "*/samples.csv"


def _file(folder, category):
    return folder / category / "samples.csv"


def _tree_rows(units):
    """One row per unit: its key, the category whose file it goes to, its style."""
    return [{"input_id": key, "category": category, "style": style,
             "text": f'text of "{key}", two\nlines', "id": f"id-{key}"}
            for key, category, style in units]


def _tree_buffer(led, folder, every=32):
    def append(rows):
        by_file = {}
        for row in rows:
            by_file.setdefault(_file(folder, row["category"]), []).append(row)
        sizes = {}
        for path, its_rows in by_file.items():
            path.parent.mkdir(exist_ok=True)
            sizes.update(append_rows(path, its_rows))
        return sizes

    return CommitBuffer(append=append, ledger=led, every=every)


def _tree_run(folder, units, *, every=2):
    """Save one row for each unit that is not finished, in its category's file."""
    led = _ledger(folder)
    done = resume_state(led, resume=True, artifact=folder, pattern=_TREE)
    with _tree_buffer(led, folder, every) as buf:
        for row in _tree_rows(units):
            if row["input_id"] not in done:
                buf.add([row], {"input_id": row["input_id"], "style": row["style"],
                                "ids": [row["id"]]})
    return led


def _tree_resume(folder, **kwargs):
    led = _ledger(folder)
    done = resume_state(led, resume=True, artifact=folder, pattern=_TREE, **kwargs)
    release_lock(led)
    return done, led


def _tree_files(folder):
    """Each file by its path from the folder, so two categories do not collide."""
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix != ".lock"}


# Units as (key, category, style): three files, the third first written last.
_UNITS = [("a", "Cyber", "long"), ("b", "Fraud", "long"), ("c", "Cyber", "short"),
          ("d", "Fraud", "short"), ("e", "Privacy", "long"), ("f", "Cyber", "long")]


class TestAStageOfSeveralFiles:
    """The stage is told its folder and the pattern of its files under it."""

    def test_every_file_is_recorded_by_its_path_from_the_ledger(self, tmp_path):
        led = _tree_run(tmp_path, _UNITS)
        assert led.saved == {
            f"{category}/samples.csv": [
                _file(tmp_path, category).stat().st_size,
                _crc(_file(tmp_path, category).read_bytes())]
            for category in ("Cyber", "Fraud", "Privacy")}

    def test_a_resume_reads_no_row_and_changes_no_file(self, tmp_path, no_reading):
        _tree_run(tmp_path, _UNITS)
        before = _tree_files(tmp_path)
        done, led = _tree_resume(tmp_path)
        assert done == set("abcdef") and _tree_files(tmp_path) == before
        assert led.ids == {f"id-{key}" for key in "abcdef"}

    def test_each_recorded_file_is_cut_back_to_its_last_finished_save(
            self, tmp_path, no_reading):
        _tree_run(tmp_path, _UNITS)
        before = _tree_files(tmp_path)
        tails = {"Cyber": b"a tail\r\n", "Privacy": b"another\r\n"}
        for category, tail in tails.items():
            with _file(tmp_path, category).open("ab") as fh:
                fh.write(tail)

        done, _led = _tree_resume(tmp_path)
        after = _tree_files(tmp_path)
        assert done == set("abcdef")
        for category, tail in tails.items():
            assert after.pop(f"{category}/samples.csv.cut") == tail
        assert after == before

    def test_a_file_no_save_recorded_is_moved_aside_whole(self, tmp_path, no_reading):
        """Its first save did not finish: every row in it is a tail."""
        _tree_run(tmp_path, _UNITS[:4])
        before = _tree_files(tmp_path)
        unfinished = _file(tmp_path, "Privacy")
        unfinished.parent.mkdir()
        append_rows(unfinished, _tree_rows(_UNITS[4:5]))
        landed = unfinished.read_bytes()

        done, _led = _tree_resume(tmp_path)
        assert done == set("abcd") and not unfinished.exists()
        after = _tree_files(tmp_path)
        assert after.pop("Privacy/samples.csv.cut") == landed and after == before

        led = _tree_run(tmp_path, _UNITS)       # the next save writes its header
        assert _keys_on_disk(unfinished) == ["e"] and led.completed() == set("abcdef")

    def test_a_file_the_last_save_did_not_write_is_left_alone(
            self, tmp_path, no_reading):
        _tree_run(tmp_path, _UNITS[:5])         # the last save wrote Privacy alone
        led = _ledger(tmp_path)
        resume_state(led, resume=True, artifact=tmp_path, pattern=_TREE)
        with _tree_buffer(led, tmp_path) as buf:    # and one more wrote no row
            buf.add([], {"input_id": "dropped"})
        before = _tree_files(tmp_path)

        done, _led = _tree_resume(tmp_path)
        assert done == {*"abcde", "dropped"} and _tree_files(tmp_path) == before

    def test_a_file_the_pattern_does_not_match_is_not_the_stages(self, tmp_path):
        _tree_run(tmp_path, _UNITS)
        others = [tmp_path / "samples.csv", tmp_path / "Cyber" / "notes.csv",
                  tmp_path / "Cyber" / "deeper" / "samples.csv"]
        for other in others:
            other.parent.mkdir(exist_ok=True)
            other.write_bytes(b"not a file of the stage\n")

        _tree_resume(tmp_path)
        led = _ledger(tmp_path)
        resume_state(led, resume=False, artifact=tmp_path, pattern=_TREE)
        assert sorted(_tree_files(tmp_path)) == sorted(
            other.relative_to(tmp_path).as_posix() for other in others)

    def test_a_file_that_cannot_be_vouched_for_stops_the_resume_before_any_cut(
            self, tmp_path):
        _tree_run(tmp_path, _UNITS)
        with _file(tmp_path, "Cyber").open("ab") as fh:
            fh.write(b"a tail\n")
        _change_a_byte(_file(tmp_path, "Privacy"))
        before = _tree_files(tmp_path)

        with pytest.raises(RuntimeError, match="Privacy"):
            _tree_resume(tmp_path)
        assert _tree_files(tmp_path) == before

    def test_a_category_under_another_spelling_is_the_file_the_ledger_records(
            self, tmp_path, no_reading):
        """On Windows a folder that is there as ``Cyber`` takes the rows of a
        category spelled ``cyber``. The glob then finds the file under one
        name and the ledger records it under the other."""
        (tmp_path / "Cyber").mkdir()
        if not (tmp_path / "cyber").exists():
            pytest.skip("this file system tells Cyber and cyber apart")
        led = _tree_run(tmp_path, [("a", "cyber", "long"), ("b", "Fraud", "long")])
        assert set(led.saved) == {"cyber/samples.csv", "Fraud/samples.csv"}
        before = _tree_files(tmp_path)
        assert set(before) == {
            "Cyber/samples.csv", "Fraud/samples.csv", "out.state.jsonl"}

        done, _led = _tree_resume(tmp_path)
        assert done == set("ab") and _tree_files(tmp_path) == before

        led = _tree_run(tmp_path, [("c", "CYBER", "long")])     # and the next save
        assert set(led.saved) == {"cyber/samples.csv", "Fraud/samples.csv"}
        assert _keys_on_disk(_file(tmp_path, "Cyber")) == ["a", "c"]

    def test_a_fresh_run_removes_the_tree_its_cut_files_the_ledger_and_the_manifest(
            self, tmp_path):
        _tree_run(tmp_path, _UNITS[:4])
        with _file(tmp_path, "Cyber").open("ab") as fh:
            fh.write(b"a tail\n")
        unfinished = _file(tmp_path, "Privacy")
        unfinished.parent.mkdir()
        unfinished.write_bytes(b"input_")
        _tree_resume(tmp_path)      # leaves a .cut beside Cyber, and one with no file
        manifest = Manifest(tmp_path / "out.manifest.jsonl")
        manifest.write([{"input_id": "a"}])
        assert set(_tree_files(tmp_path)) == {
            "Cyber/samples.csv", "Cyber/samples.csv.cut", "Fraud/samples.csv",
            "Privacy/samples.csv.cut", "out.state.jsonl", "out.manifest.jsonl"}

        led = _ledger(tmp_path)
        assert resume_state(led, resume=False, artifact=tmp_path, pattern=_TREE,
                            manifest=manifest) == set()
        assert _tree_files(tmp_path) == {}

        release_lock(led)
        led = _tree_run(tmp_path, _UNITS[:1])
        assert _ledger_lines(led)[0] == {"_saved": {}} and led.completed() == {"a"}


def _old_tree(folder, acked, units):
    """A tree as the library wrote it before the ledger recorded sizes."""
    for row in _tree_rows(units):
        _file(folder, row["category"]).parent.mkdir(exist_ok=True)
        append_rows(_file(folder, row["category"]), [row])
    led = _ledger(folder)
    if acked:
        led.path.write_bytes(b"".join(
            json.dumps({"input_id": key}).encode() + b"\r\n" for key in acked))
    return led


def _units_in(path):
    """Each unit in a file of the tree, with the ids of its rows."""
    found = {}
    for row in pd.read_csv(path, dtype=str).to_dict("records"):
        found.setdefault(row["input_id"], []).append(row["id"])
    return found


class TestATreeWrittenBeforeSizesWereRecorded:
    def test_one_read_and_one_ledger_write_take_it_over(self, tmp_path, monkeypatch):
        led = _old_tree(tmp_path, acked="ab", units=_UNITS[:4])
        old_lines = led.path.read_bytes()
        read, records = [], []
        record = Ledger.record
        monkeypatch.setattr(Ledger, "record", lambda self, *args, **kwargs: (
            records.append(args), record(self, *args, **kwargs))[1])

        def units_in(path):
            read.append(path.relative_to(tmp_path).as_posix())
            return _units_in(path)

        done, led = _tree_resume(tmp_path, units_in=units_in)
        assert done == set("abcd")
        assert read == ["Cyber/samples.csv", "Fraud/samples.csv"]   # each file once
        assert len(records) == 1
        assert led.path.read_bytes().startswith(old_lines)
        # Every unit with rows records their ids, the acked ones too.
        *acks, last = _ledger_lines(led)[2:]
        assert sorted(acks + [{k: v for k, v in last.items() if k != "_saved"}],
                      key=lambda ack: ack["input_id"]) == [
            {"input_id": key, "ids": [f"id-{key}"]} for key in "abcd"]
        assert last["_saved"] == {
            f"{category}/samples.csv": [
                _file(tmp_path, category).stat().st_size,
                _crc(_file(tmp_path, category).read_bytes())]
            for category in ("Cyber", "Fraud")}
        assert led.ids == {f"id-{key}" for key in "abcd"}

        monkeypatch.setattr(pd, "read_csv", _no_read)
        done, led = _tree_resume(tmp_path, units_in=_no_read)
        assert done == set("abcd") and led.ids == {f"id-{key}" for key in "abcd"}

    def test_without_a_reader_the_key_columns_are_read_and_no_id_is_recorded(
            self, tmp_path):
        led = _old_tree(tmp_path, acked="ab", units=_UNITS[:4])
        done, led = _tree_resume(tmp_path)
        assert done == set("abcd") and led.ids == set()
        assert [line["input_id"] for line in _ledger_lines(led)] == list("abcd")

    def test_a_file_that_ends_inside_a_row_stops_the_takeover(self, tmp_path):
        _old_tree(tmp_path, acked="ab", units=_UNITS[:4])
        torn = _file(tmp_path, "Fraud")
        torn.write_bytes(torn.read_bytes()[:-3])
        before = _tree_files(tmp_path)
        with pytest.raises(RuntimeError, match="line break") as stopped:
            _tree_resume(tmp_path, units_in=_units_in)
        assert str(torn) in str(stopped.value) and _tree_files(tmp_path) == before

    def test_an_empty_file_is_taken_over_as_no_file(self, tmp_path):
        _old_tree(tmp_path, acked="a", units=_UNITS[:1])
        empty = _file(tmp_path, "Fraud")
        empty.parent.mkdir()
        empty.touch()
        done, led = _tree_resume(tmp_path, units_in=_units_in)
        assert done == {"a"} and not empty.exists()
        assert list(led.saved) == ["Cyber/samples.csv"]


_SHORT = set("cdgh")     # the units of style short, among those the tests save


def _drop_short(folder, **kwargs):
    """Start the units of style ``short`` over, as a fresh run of that style.

    Their acks are picked by key: an old ledger's lines carry no style.
    """
    led = _ledger(folder)
    drop_units(led, artifact=folder, pattern=_TREE,
               rows=lambda row: row["style"] == "short",
               acks=lambda ack: ack.get("input_id") in _SHORT, **kwargs)
    return led


def _rows_of(data):
    """A CSV's rows as the bytes they are in the file, its header first."""
    text = data.decode("utf-8")
    rows, start, quoted = [], 0, False
    for at, char in enumerate(text):
        quoted ^= char == '"'
        if char == "\n" and not quoted:
            rows.append(text[start:at + 1].encode("utf-8"))
            start = at + 1
    assert start == len(text)
    return rows


def _stage_files(folder):
    """``_tree_files`` without the copy a test keeps under ``whole/``."""
    return {name: data for name, data in _tree_files(folder).items()
            if not name.startswith("whole/")}


class TestStartingSomeUnitsOver:
    """``drop_units``: a fresh run of part of a stage."""

    def test_their_rows_and_acks_go_and_the_rest_stands_byte_for_byte(
            self, tmp_path, monkeypatch):
        # Cyber and Fraud hold both styles, Privacy long alone, Theft short alone.
        led = _tree_run(tmp_path, _UNITS)
        only_short = _file(tmp_path, "Theft")
        resume_state(led, resume=True, artifact=tmp_path, pattern=_TREE)
        with _tree_buffer(led, tmp_path) as buf:
            buf.add(_tree_rows([("g", "Theft", "short")]),
                    {"input_id": "g", "style": "short", "ids": ["id-g"]})
            buf.add([], {"input_id": "h", "style": "short", "status": "rejected"})
        before = _tree_files(tmp_path)
        old_lines = led.path.read_bytes().split(b"\n")
        replaced, replace = [], os.replace
        monkeypatch.setattr(os, "replace", lambda source, target: (
            replaced.append(Path(target).relative_to(tmp_path).as_posix()),
            replace(source, target))[1])

        release_lock(_drop_short(tmp_path))
        after = _tree_files(tmp_path)
        # A file with no row of the style is not written again.
        assert replaced == ["Cyber/samples.csv", "Fraud/samples.csv", "out.state.jsonl"]
        cyber = _rows_of(before["Cyber/samples.csv"])
        fraud = _rows_of(before["Fraud/samples.csv"])
        assert after["Cyber/samples.csv"] == cyber[0] + cyber[1] + cyber[3]    # a, f
        assert after["Fraud/samples.csv"] == fraud[0] + fraud[1]                # b
        assert after["Privacy/samples.csv"] == before["Privacy/samples.csv"]
        assert not only_short.exists()              # no row left: no file
        assert set(after) == {"Cyber/samples.csv", "Fraud/samples.csv",
                              "Privacy/samples.csv", "out.state.jsonl"}

        # The acks that stay are the lines they were; one last line has the sizes.
        *kept, last, end = after["out.state.jsonl"].split(b"\n")
        assert end == b"" and kept == [
            line for line in old_lines if b'"short"' not in line and line]
        assert json.loads(last) == {"_saved": {
            name: [len(data), _crc(data)] for name, data in after.items()
            if name.endswith(".csv")}}

    def test_the_resume_that_follows_knows_only_what_stayed(self, tmp_path, no_reading):
        _tree_run(tmp_path, _UNITS)
        led = _drop_short(tmp_path)
        before = _tree_files(tmp_path)
        assert resume_state(led, resume=True, artifact=tmp_path,
                            pattern=_TREE) == set("abef")
        assert led.ids == {f"id-{key}" for key in "abef"}   # no id of a removed row
        release_lock(led)
        assert _tree_files(tmp_path) == before

        led = _tree_run(tmp_path, _UNITS)                   # and the units run again
        assert led.completed() == set("abcdef")
        assert sorted(_keys_on_disk(_file(tmp_path, "Cyber"))) == ["a", "c", "f"]

    def test_rows_written_elsewhere_keep_their_line_ends(self, tmp_path):
        """A tree copied from a pod ends its rows in a bare newline."""
        led = _tree_run(tmp_path, _UNITS)
        cyber = _file(tmp_path, "Cyber")
        cyber.write_bytes(cyber.read_bytes().replace(b"\r\n", b"\n"))
        led.path.unlink()                   # so the copy is taken over as it is
        rows = _rows_of(cyber.read_bytes())

        release_lock(_drop_short(tmp_path, units_in=_units_in))
        assert cyber.read_bytes() == rows[0] + rows[1] + rows[3]

    def test_it_makes_this_process_the_writer_until_the_stage_ends(self, tmp_path):
        _tree_run(tmp_path, _UNITS)
        led = _drop_short(tmp_path)
        lock = FileLock(led.path.with_name("out.state.jsonl.lock"), timeout=0)
        with pytest.raises(Timeout):
            lock.acquire()
        release_lock(led)
        lock.acquire()
        lock.release()

    def test_rows_of_a_save_that_did_not_finish_are_cut_off_first(self, tmp_path):
        """Left in the file, they would be recorded as saved with no unit."""
        _tree_run(tmp_path, _UNITS)
        before = _tree_files(tmp_path)
        tail = _rows_of(before["Cyber/samples.csv"])[1]     # a row of style long
        with _file(tmp_path, "Cyber").open("ab") as fh:
            fh.write(tail)
        unfinished = _file(tmp_path, "Theft")
        unfinished.parent.mkdir()
        unfinished.write_bytes(b"input_id,category,style,text,id\r\n")

        release_lock(_drop_short(tmp_path))
        cyber = _rows_of(before["Cyber/samples.csv"])
        assert _file(tmp_path, "Cyber").read_bytes() == cyber[0] + cyber[1] + cyber[3]
        assert (tmp_path / "Cyber" / "samples.csv.cut").read_bytes() == tail
        assert not unfinished.exists()

    @pytest.mark.parametrize("damage", ["a changed byte", "saved again, larger"])
    def test_a_file_that_holds_other_bytes_stops_it_and_no_file_is_changed(
            self, tmp_path, damage):
        led = _tree_run(tmp_path, _UNITS)
        _DAMAGE[damage](_file(tmp_path, "Privacy"))
        before = _tree_files(tmp_path)
        with pytest.raises(RuntimeError, match="Privacy"):
            _drop_short(tmp_path)
        assert _tree_files(tmp_path) == before
        lock = FileLock(led.path.with_name("out.state.jsonl.lock"), timeout=0)
        lock.acquire()                          # a stop holds no lock
        lock.release()

    def test_stopped_before_the_ledger_it_is_finished_by_being_made_again(
            self, tmp_path, monkeypatch):
        """In between, the files are shorter than the ledger records."""
        _tree_run(tmp_path, _UNITS)
        whole = tmp_path / "whole"              # the same folder, started over whole
        for name, data in _tree_files(tmp_path).items():
            (whole / name).parent.mkdir(parents=True, exist_ok=True)
            (whole / name).write_bytes(data)
        release_lock(_drop_short(whole))

        with monkeypatch.context() as patch:
            patch.setattr(Ledger, "rewrite", lambda self, drop, saved: 1 / 0)
            with pytest.raises(ZeroDivisionError):
                _drop_short(tmp_path)
        stopped = _stage_files(tmp_path)
        with pytest.raises(RuntimeError, match="resume=False"):     # a resume stops
            _tree_resume(tmp_path)
        assert _stage_files(tmp_path) == stopped

        release_lock(_drop_short(tmp_path))
        assert _stage_files(tmp_path) == _tree_files(whole)

    def test_a_shorter_file_that_ends_inside_a_row_stops_it(self, tmp_path):
        """Shorter than recorded is what this call leaves when it stops half
        way, and such a file ends with a whole row."""
        _tree_run(tmp_path, _UNITS)
        torn = _file(tmp_path, "Fraud")
        torn.write_bytes(torn.read_bytes()[:-9])
        before = _tree_files(tmp_path)
        with pytest.raises(RuntimeError, match="copied incompletely") as stopped:
            _drop_short(tmp_path)
        assert str(torn) in str(stopped.value) and _tree_files(tmp_path) == before

    def test_a_shorter_file_that_ends_inside_a_quoted_cell_stops_it(self, tmp_path):
        """Its last byte is a line break, the one inside the cell."""
        led = _tree_run(tmp_path, _UNITS)
        ledger = led.path.read_bytes()
        torn = _file(tmp_path, "Fraud")
        data = torn.read_bytes()
        torn.write_bytes(data[:data.rindex(b"lines")])
        with pytest.raises(RuntimeError, match="unexpected end of data") as stopped:
            _drop_short(tmp_path)
        assert str(torn) in str(stopped.value)
        assert led.path.read_bytes() == ledger      # and a resume still stops
        with pytest.raises(RuntimeError, match="recorded"):
            _tree_resume(tmp_path)

    def test_a_file_is_whole_on_disk_before_it_replaces_the_old_one(
            self, tmp_path, monkeypatch):
        _tree_run(tmp_path, _UNITS)
        before = _tree_files(tmp_path)
        synced = []

        def fsync(fd):
            files = _tree_files(tmp_path)
            (temporary,) = (name for name in files if name.endswith(".tmp"))
            name = temporary.removesuffix(".tmp")
            synced.append((name, files[name] == before[name]))

        monkeypatch.setattr(os, "fsync", fsync)
        release_lock(_drop_short(tmp_path))
        assert synced == [(name, True) for name in (
            "Cyber/samples.csv", "Fraud/samples.csv", "out.state.jsonl")]

    def test_an_old_tree_is_taken_over_first(self, tmp_path):
        """Its rows without acks would otherwise be recorded as saved, with
        no unit to own them."""
        _old_tree(tmp_path, acked="ac", units=_UNITS[:4])
        led = _drop_short(tmp_path, units_in=_units_in)
        assert resume_state(led, resume=True, artifact=tmp_path,
                            pattern=_TREE) == set("ab")
        assert led.ids == {"id-a", "id-b"}
        assert _keys_on_disk(_file(tmp_path, "Cyber")) == ["a"]
        assert _keys_on_disk(_file(tmp_path, "Fraud")) == ["b"]

    def test_a_folder_that_holds_nothing_is_left_empty(self, tmp_path):
        release_lock(_drop_short(tmp_path))
        assert _tree_files(tmp_path) == {}

    def test_a_stage_of_one_file_starts_some_units_over_too(self, tmp_path):
        led, out = _ledger(tmp_path), tmp_path / "out.csv"
        resume_state(led, resume=True, artifact=out)
        with _buffer(led, out) as buf:
            for key in "abc":
                buf.add(_rows([key]), {"input_id": key})
        drop_units(led, artifact=out, rows=lambda row: row["input_id"] == "b",
                   acks=lambda ack: ack.get("input_id") == "b")
        assert resume_state(led, resume=True, artifact=out) == set("ac")
        release_lock(led)
        assert _keys_on_disk(out) == ["a", "c"]
        assert led.saved == {"out.csv": [out.stat().st_size, _crc(out.read_bytes())]}
