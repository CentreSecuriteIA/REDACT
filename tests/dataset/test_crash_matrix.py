"""A crash at every point of a save, and at each step of a resume and of a
takeover.

A crash inside a write leaves the bytes before it and some prefix of the
bytes it was adding, so every folder a dying run can leave is rebuilt from
the two files as they are before and after each write of one clean run. Each
such folder is resumed with the real ``resume_state`` and held against the
five guarantees:

1. a unit is finished only if its rows are on disk;
2. no row is on disk that a rerun would write again;
3. what was acked before the crash is still finished, with the same rows;
4. each file is exactly what the ledger records;
5. the resume reads no row of the output.

Then the run is taken to its end, and each unit must have its rows once.
"""

import csv
import io
import os
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path

import pandas as pd
import pytest

import redact
import redact.dataset.resume as resume_module
from redact.dataset import (
    CommitBuffer,
    Ledger,
    append_rows,
    release_lock,
    resume_state,
)

# The texts of each unit's rows, as a model writes them. u2 produces no row.
_UNITS = {
    "u0": ["plain"],
    "u1": ['a comma, and a "quote"', "two\nlines"],
    "u2": [],
    "u3": ["café 中文"],
    "u4": ["a break\r\ninside", "x"],
    "u5": ["last"],
}
_EVERY = 2      # units per save: three saves
_SAVES = [list(_UNITS)[n:n + _EVERY] for n in range(0, len(_UNITS), _EVERY)]


class _Crash(BaseException):
    """The process died here."""


def _ledger(folder):
    return Ledger(folder / "out.state.jsonl", key_fields=("input_id",),
                  casters={"input_id": str})


def _rows(key, attempt):
    return [{"input_id": key, "row": n, "text": text, "attempt": attempt}
            for n, text in enumerate(_UNITS[key])]


def _run(folder, *, attempt, every=_EVERY, on_write=lambda: None):
    """Resume, then save every unit that is not finished, ``every`` to a save."""
    led, out = _ledger(folder), folder / "out.csv"
    done = resume_state(led, resume=True, artifact=out)

    def append(rows):
        sizes = append_rows(out, rows)
        on_write()
        return sizes

    with CommitBuffer(append=append, ledger=led, every=every) as buf:
        for key in _UNITS:
            if key not in done:
                buf.add(_rows(key, attempt), {"input_id": key})


def _files(folder):
    """The bytes of each file of the stage; its lock file holds none."""
    return {path.name: path.read_bytes() for path in folder.iterdir()
            if path.suffix != ".lock"}


def _put(folder, out, ledger):
    """Leave the folder holding these bytes and nothing else; ``None`` is no file."""
    for path in folder.iterdir():
        path.unlink()
    for name, data in (("out.csv", out), ("out.state.jsonl", ledger)):
        if data is not None:
            (folder / name).write_bytes(data)


def _units_in(data):
    """Each unit's rows as ``(row, text, attempt)``, from a parse of the
    output's bytes that takes no torn row."""
    text = data.decode("utf-8")
    assert text.endswith("\n")
    header, *rows = csv.reader(io.StringIO(text, newline=""))
    assert header == ["input_id", "row", "text", "attempt"]
    units = {}
    for row in rows:
        assert len(row) == len(header)
        units.setdefault(row[0], []).append((int(row[1]), row[2], int(row[3])))
    return units


def _check(folder, finished):
    """Hold what is on disk against the guarantees; the rows found.

    ``finished`` is what the finished units must be: the ones acked before
    the crash, no fewer (3) and no more (1).
    """
    out, led = folder / "out.csv", _ledger(folder)
    done, saved, end = led.read()
    data = out.read_bytes() if out.exists() else None
    rows = {} if data is None else _units_in(data)
    assert done == finished
    assert set(rows) == {key for key in done if _UNITS[key]}        # 1 and 2
    for key, found in rows.items():
        assert [(n, text) for n, text, _ in found] == list(enumerate(_UNITS[key]))
    if data is None:                                                # 4
        assert not saved
    else:
        assert saved == {"out.csv": [len(data), f"{zlib.crc32(data[-4096:]):08x}"]}
    assert not led.exists() or led.path.stat().st_size == end
    return rows


def _no_read(*args, **kwargs):
    raise AssertionError("the resume read the rows of the output")


def _resume(folder):
    """The finished units a run finds that resumes the folder and ends."""
    led = _ledger(folder)
    done = resume_state(led, resume=True, artifact=folder / "out.csv")
    release_lock(led)
    return done


def _recover(folder, monkeypatch, finished, *, may_read=False):
    """Resume the folder, check it, take the run to its end, check again.

    Only the first resume of a folder written before sizes were recorded
    ``may_read`` the output. The resume of the run that follows never does.
    """
    with monkeypatch.context() as patch:
        if not may_read:                                            # 5
            patch.setattr(resume_module, "keys_in", _no_read)
            patch.setattr(pd, "read_csv", _no_read)
        assert _resume(folder) == finished
    before = _check(folder, finished)

    with monkeypatch.context() as patch:
        patch.setattr(resume_module, "keys_in", _no_read)           # 5
        _run(folder, attempt=1, every=len(_UNITS))
    after = _check(folder, set(_UNITS))
    for key, rows in after.items():
        # Rows acked before the crash are the same rows; the others ran again.
        assert rows == before.get(key, [(n, text, 1) for n, text, _ in rows])


def _clean_run(folder, monkeypatch):
    """``(output, ledger)`` after each write of one run, starting with none."""
    states = [(None, None)]

    def snapshot():
        files = _files(folder)
        states.append((files.get("out.csv"), files.get("out.state.jsonl")))

    record = Ledger.record
    with monkeypatch.context() as patch:
        patch.setattr(Ledger, "record", lambda self, *args, **kwargs: (
            record(self, *args, **kwargs), snapshot())[0])
        _run(folder, attempt=0, on_write=snapshot)
    # The opening line, then the rows and the acks of each save.
    assert len(states) == 1 + 1 + 2 * len(_SAVES)
    return states


def _cuts(start, end, every_byte=False):
    """The sizes at which a write from ``start`` to ``end`` is cut off: before
    it, after its first byte, in its middle and before its last byte, or
    after every byte but the last."""
    if every_byte:
        return range(start, end)
    return sorted({start, start + 1, (start + end) // 2, end - 1})


def _crashes(states, every_byte_of=None):
    """Every folder a crash of the run leaves: ``(output, ledger, finished)``.

    Between two writes the folder is a snapshot. Inside a write the file
    being written holds a prefix of the added bytes, and a new file may
    exist empty before its first byte. The two writes of the save numbered
    ``every_byte_of`` are cut at every byte, the others as ``_cuts`` says.
    """
    for step, (before, after) in enumerate(zip(states, states[1:])):
        save = max(0, (step - 1) // 2)      # the opening line, then rows and acks
        which = 0 if before[0] != after[0] else 1
        old, new = before[which], after[which]
        assert new.startswith(old or b"")
        sizes = _cuts(len(old or b""), len(new), save == every_byte_of)
        for prefix in dict.fromkeys([old, *(new[:size] for size in sizes)]):
            folder = list(before)
            folder[which] = prefix
            yield *folder, _finished(save)
    yield *states[-1], _finished(len(_SAVES))


def _finished(saves):
    return {key for save in _SAVES[:saves] for key in save}


@pytest.fixture()
def no_sync(monkeypatch):
    """A simulated crash loses nothing that was written, so the syncs would
    only make the matrix slow."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)


def test_a_crash_at_any_byte_of_a_save_keeps_the_guarantees(
        tmp_path, monkeypatch, no_sync):
    """The second save is cut at every byte: it adds rows to a file the
    ledger records, and one of its units has no row."""
    states = _clean_run(tmp_path, monkeypatch)
    points = 0
    for out, ledger, finished in _crashes(states, every_byte_of=1):
        _put(tmp_path, out, ledger)
        _recover(tmp_path, monkeypatch, finished)
        points += 1
    print(f"{points} crash points in a run")
    second_save = sum(map(len, states[5])) - sum(map(len, states[3]))
    assert points > second_save


def _dies_at_sync(folder, monkeypatch, number):
    """Resume, and die at the ``number``-th sync, once the step it follows
    has changed the folder. False when the resume ended first."""
    syncs = 0

    def fsync(fd):
        nonlocal syncs
        syncs += 1
        if syncs == number:
            raise _Crash

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fsync)
        try:
            resume_state(_ledger(folder), resume=True, artifact=folder / "out.csv")
        except _Crash:
            return True
    release_lock(_ledger(folder))
    return False


_RESUME_STEPS = ("the tail into .cut", "the cut of the output", "the cut of the ledger")


def test_a_crash_at_any_step_of_a_resume_keeps_the_guarantees(
        tmp_path, monkeypatch, no_sync):
    """A resume that died after any of its steps is resumed again."""
    states = _clean_run(tmp_path, monkeypatch)
    points, most_steps = 0, 0
    for out, ledger, finished in _crashes(states):
        step = 1
        while True:
            _put(tmp_path, out, ledger)
            if not _dies_at_sync(tmp_path, monkeypatch, step):
                break
            _recover(tmp_path, monkeypatch, finished)
            points, step = points + 1, step + 1
        most_steps = max(most_steps, step - 1)
    print(f"{points} crash points in a resume")
    assert most_steps == len(_RESUME_STEPS)     # some resume had to make them all


# Folders as the library left them before the ledger recorded sizes: the
# units with rows in the output, and the units acked in the ledger.
_OLD = {
    "every row acked": ("u0 u1 u3", "u0 u1 u2 u3"),
    "rows not acked": ("u0 u1 u3", "u0 u1"),
    "no ledger": ("u0 u1 u3", ""),
    "no output": ("", "u2"),
}


def _old_folder(folder, with_rows, acked, torn):
    """Write such a folder, its last ledger line ``torn`` off mid-write."""
    rows = [row for key in with_rows.split() for row in _rows(key, 0)]
    lines = b"".join(b'{"input_id": "%s"}\r\n' % key.encode() for key in acked.split())
    _put(folder, None, lines + torn if lines or torn else None)
    if rows:
        append_rows(folder / "out.csv", rows)
    files = _files(folder)
    return files.get("out.csv"), files.get("out.state.jsonl")


@pytest.mark.parametrize("shape", _OLD)
@pytest.mark.parametrize("torn", [b"", b'{"input_id": "u'], ids=["whole", "torn"])
def test_a_crash_at_any_step_of_a_takeover_keeps_the_guarantees(
        tmp_path, monkeypatch, no_sync, shape, torn):
    """A takeover cuts a torn last line off the ledger and makes one write.
    Stopped before either, or inside the write, it is made again."""
    with_rows, acked = _OLD[shape]
    finished = set(with_rows.split()) | set(acked.split())
    out, old = _old_folder(tmp_path, with_rows, acked, torn)
    _resume(tmp_path)
    taken_over = _files(tmp_path)["out.state.jsonl"]
    whole = len(old or b"") - len(torn)
    assert taken_over.startswith((old or b"")[:whole])

    sizes = _cuts(whole, len(taken_over))
    crashes = dict.fromkeys([old, *(taken_over[:size] for size in sizes), taken_over])
    for ledger in crashes:
        _put(tmp_path, out, ledger)
        _recover(tmp_path, monkeypatch, finished, may_read=ledger != taken_over)
    print(f"{len(crashes)} crash points in the takeover")
    assert len(crashes) > len(sizes)


_SECOND_RUN = """
    import sys
    from pathlib import Path
    from redact.dataset import Ledger, Manifest, resume_state

    out = Path(sys.argv[1])
    for resume in (True, False):
        try:
            resume_state(Ledger.sidecar(out, ("input_id",)), resume=resume,
                         artifact=out, manifest=Manifest.sidecar(out))
        except RuntimeError as refused:
            print(f"resume={resume} refused: {refused}")
        else:
            print(f"resume={resume} went through")
"""


def test_a_second_process_is_refused_while_the_first_holds_the_lock(tmp_path):
    """Whether it resumes or starts over, it must leave the first run's files
    alone: that run is between the rows of a save and their acks."""
    led, out = _ledger(tmp_path), tmp_path / "out.csv"
    manifest = out.with_name("out.manifest.jsonl")
    manifest.write_bytes(b'{"input_id": "u0"}\n')
    _run(tmp_path, attempt=0)
    resume_state(led, resume=True, artifact=out)        # the first run, live
    append_rows(out, _rows("u0", 1))                    # its rows, acks to come
    before = {path.name: path.read_bytes() for path in (out, led.path, manifest)}

    second = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SECOND_RUN), str(out)],
        env={**os.environ, "PYTHONPATH": str(Path(redact.__file__).parents[1])},
        capture_output=True, text=True, timeout=120, check=False)

    assert second.returncode == 0, second.stderr
    answers = second.stdout.splitlines()
    assert [answer.split(":")[0] for answer in answers] == [
        "resume=True refused", "resume=False refused"]
    assert all("out.state.jsonl.lock" in answer for answer in answers)
    assert {path.name: path.read_bytes()
            for path in (out, led.path, manifest)} == before
    assert not out.with_name("out.csv.cut").exists()
