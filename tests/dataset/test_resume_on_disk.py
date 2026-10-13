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
from filelock import FileLock

import redact.dataset.resume as resume_module
import redact.dataset.rows as rows_module
from redact.dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
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
