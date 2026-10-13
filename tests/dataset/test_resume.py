"""The two resume invariants, tested directly rather than only through stages.

Both are only observable after a crash, which is exactly why they are worth
testing at this level: the per-stage tests exercise the happy path, and these
pin the failure ordering.
"""

import json
import os
import zlib

import pytest
from filelock import FileLock, Timeout

from redact.dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    commit,
    release_lock,
    resume_state,
)
from redact.llm_pipeline import LLMRequest, drive_units
from tests.conftest import as_resolver


def _ledger(tmp_path, key_fields=("input_id",), casters=None):
    return Ledger(tmp_path / "a.state.jsonl", key_fields=key_fields, casters=casters)


class TestResumeState:
    def test_resume_returns_recorded_units(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"input_id": "a"}, {"input_id": "b"}])
        assert resume_state(led, resume=True) == {"a", "b"}

    def test_fresh_clears_ledger_and_artifact(self, tmp_path):
        """Clearing only the artifact leaves stale acks, so the next run skips
        every unit and silently writes nothing."""
        led = _ledger(tmp_path)
        led.record([{"input_id": "a"}])
        artifact = tmp_path / "out.csv"
        artifact.write_text("x", encoding="utf-8")

        assert resume_state(led, resume=False, artifact=artifact) == set()
        assert not artifact.exists()
        assert led.completed() == set()

    def test_fresh_tolerates_a_missing_artifact(self, tmp_path):
        led = _ledger(tmp_path)
        assert resume_state(led, resume=False, artifact=tmp_path / "nope.csv") == set()

    def test_extra_is_unioned_for_back_compat(self, tmp_path):
        """Jailbreak reads pre-ledger runs back out of the output CSV."""
        led = _ledger(tmp_path)
        led.record([{"input_id": "a"}])
        assert resume_state(led, resume=True, extra={"b"}) == {"a", "b"}

    def test_extra_is_ignored_on_a_fresh_run(self, tmp_path):
        led = _ledger(tmp_path)
        assert resume_state(led, resume=False, extra={"b"}) == set()

    def test_composite_keys_round_trip(self, tmp_path):
        led = _ledger(
            tmp_path, ("input_id", "iteration"), {"input_id": str, "iteration": int}
        )
        led.record([{"input_id": "a", "iteration": 0}])
        assert resume_state(led, resume=True) == {("a", 0)}

    def test_fresh_clears_the_manifest(self, tmp_path):
        led = _ledger(tmp_path)
        manifest = Manifest(tmp_path / "a.manifest.jsonl")
        manifest.write([{"input_id": "a"}])

        assert resume_state(led, resume=False, manifest=manifest) == set()
        assert not manifest.exists()

    def test_resume_keeps_the_manifest(self, tmp_path):
        led = _ledger(tmp_path)
        manifest = Manifest(tmp_path / "a.manifest.jsonl")
        manifest.write([{"input_id": "a"}])

        resume_state(led, resume=True, manifest=manifest)
        assert manifest.load() == [{"input_id": "a"}]

    def test_resume_adds_the_keys_of_the_rows_in_the_artifact(self, tmp_path):
        """A crash between the append and the ack leaves rows with no ack."""
        led = _ledger(
            tmp_path, ("input_id", "iteration"), {"input_id": str, "iteration": int}
        )
        led.record([{"input_id": "a", "iteration": 0}])
        artifact = tmp_path / "out.csv"
        append_rows(artifact, [{"input_id": "a", "iteration": 0, "sample": "x"},
                               {"input_id": "b", "iteration": 1, "sample": "y"}])

        assert resume_state(led, resume=True, artifact=artifact) == {("a", 0), ("b", 1)}

    def test_resume_tolerates_a_missing_artifact(self, tmp_path):
        led = _ledger(tmp_path)
        led.record([{"input_id": "a"}])
        assert resume_state(led, resume=True, artifact=tmp_path / "nope.csv") == {"a"}


class TestCommit:
    def test_appends_then_acks(self, tmp_path):
        led = _ledger(tmp_path)
        seen = []
        commit([{"input_id": "a"}], append=seen.append, ledger=led)
        assert seen and led.completed() == {"a"}

    def test_a_failing_append_acks_nothing(self, tmp_path):
        """The whole point of the ordering: a crash mid-write must leave the
        unit un-acked so it re-runs, rather than recorded as done."""
        led = _ledger(tmp_path)

        def _boom(_rows):
            raise OSError("disk full")

        with pytest.raises(OSError):
            commit([{"input_id": "a"}], append=_boom, ledger=led)
        assert led.completed() == set()

    def test_records_default_to_the_ledger_key_fields(self, tmp_path):
        """Acking whole rows would bloat the ledger with every output column."""
        led = _ledger(tmp_path)
        commit(
            [{"input_id": "a", "sample": "x" * 500, "accepted": True}],
            append=lambda rows: None,
            ledger=led,
        )
        line = (tmp_path / "a.state.jsonl").read_text(encoding="utf-8").strip()
        assert json.loads(line) == {"input_id": "a"}

    def test_units_with_no_rows_are_still_acked(self, tmp_path):
        """Paraphrase drops deduped units; un-acked they would retry forever."""
        led = _ledger(tmp_path)
        calls = []
        commit([], append=calls.append, ledger=led, records=[{"input_id": "dropped"}])
        assert calls == []                      # nothing written
        assert led.completed() == {"dropped"}   # but acked

    def test_rows_with_no_records_are_refused_before_they_are_written(self, tmp_path):
        """Rows on disk that no unit owns could never be told from a torn save."""
        led = _ledger(tmp_path)
        calls = []
        with pytest.raises(ValueError, match="no record"):
            commit([{"input_id": "a"}], append=calls.append, ledger=led, records=[])
        assert calls == [] and not led.exists()

    def test_an_append_that_reports_no_sizes_leaves_plain_lines(self, tmp_path):
        led = _ledger(tmp_path)
        commit([{"input_id": "a"}], append=lambda rows: None, ledger=led)
        assert _lines(led) == [{"input_id": "a"}]


def _lines(led):
    return [json.loads(line) for line in led.path.read_bytes().split(b"\n") if line]


_CHECKSUM_BYTES = 4096


def _crc(data: bytes) -> str:
    return f"{zlib.crc32(data[-_CHECKSUM_BYTES:]):08x}"


def _opened(tmp_path, name="out.csv"):
    """A ledger that ``resume_state`` opened on its artifact, and the artifact."""
    led, out = _ledger(tmp_path), tmp_path / name
    out.parent.mkdir(exist_ok=True)
    resume_state(led, resume=True, artifact=out)
    return led, out


def _save(out, led, key, text="t"):
    """One save of one unit with one row, through ``append_rows``."""
    commit([{"input_id": key, "text": text}],
           append=lambda rows: append_rows(out, rows), ledger=led)


class TestCommitRecordsSizes:
    """The last ledger line of a save says how much of each file is saved."""

    def test_the_last_line_carries_the_size_and_a_checksum(self, tmp_path):
        led, out = _opened(tmp_path)
        commit([{"input_id": "a", "text": "x"}, {"input_id": "b", "text": "y"}],
               append=lambda rows: append_rows(out, rows), ledger=led)
        data = out.read_bytes()
        assert _lines(led) == [
            {"_saved": {}},
            {"input_id": "a"},
            {"input_id": "b", "_saved": {"out.csv": [len(data), _crc(data)]}},
        ]

    def test_the_checksum_covers_the_last_4096_bytes_before_the_size(self, tmp_path):
        led, out = _opened(tmp_path)
        _save(out, led, "a", "x" * 5000)
        data = out.read_bytes()
        assert len(data) > _CHECKSUM_BYTES
        assert led.saved == {"out.csv": [len(data), _crc(data)]}
        assert _crc(data) != f"{zlib.crc32(data):08x}"

    def test_the_size_is_the_files_size_in_bytes(self, tmp_path):
        led, out = _opened(tmp_path)
        _save(out, led, "a", "café 中文\nsecond line\r\nthird")
        assert led.saved["out.csv"][0] == os.path.getsize(out)

    def test_a_file_is_named_by_its_path_from_the_ledgers_folder(self, tmp_path):
        led, out = _opened(tmp_path, "Cyber Crime/samples.csv")
        _save(out, led, "a")
        assert list(led.saved) == ["Cyber Crime/samples.csv"]

    def test_every_saved_line_carries_every_file(self, tmp_path):
        led, first = _opened(tmp_path, "a.csv")
        second = tmp_path / "b.csv"
        _save(first, led, "a")
        recorded = led.saved["a.csv"]
        _save(second, led, "b")
        assert _lines(led)[-1]["_saved"] == {
            "a.csv": recorded,
            "b.csv": [second.stat().st_size, _crc(second.read_bytes())],
        }

    def test_a_save_with_no_rows_repeats_what_was_recorded(self, tmp_path):
        led, out = _opened(tmp_path)
        _save(out, led, "a")
        recorded = dict(led.saved)
        commit([], append=lambda rows: append_rows(out, rows), ledger=led,
               records=[{"input_id": "dropped"}])
        assert _lines(led)[-1] == {"input_id": "dropped", "_saved": recorded}
        assert led.completed() == {"a", "dropped"}

    def test_the_opening_line_is_synced_before_the_first_rows(
            self, tmp_path, monkeypatch):
        """Rows on disk with no ledger would read as a folder to adopt."""
        led, out = _opened(tmp_path)
        assert not led.exists()               # a resume alone writes nothing
        synced, at_append = [], []
        fsync = os.fsync
        monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(
            led.path.read_bytes() if led.exists() else None), fsync(fd))[1])

        def append(rows):
            at_append.append((list(synced), led.path.read_bytes()))
            return append_rows(out, rows)

        commit([{"input_id": "a", "text": "x"}], append=append, ledger=led)
        opening = b'{"_saved": {}}\n'
        assert at_append == [([opening], opening)]

    def test_a_first_save_with_no_rows_writes_the_opening_line_too(self, tmp_path):
        led, out = _opened(tmp_path)
        commit([], append=lambda rows: append_rows(out, rows), ledger=led,
               records=[{"input_id": "dropped"}])
        assert _lines(led) == [{"_saved": {}}, {"input_id": "dropped", "_saved": {}}]

    def test_rows_whose_append_reports_no_size_are_not_acked(self, tmp_path):
        """The ledger would record a size that leaves the rows out, and the
        next resume would cut them off acked units."""
        led, out = _opened(tmp_path)
        with pytest.raises(RuntimeError, match="size"):
            commit([{"input_id": "a"}], append=lambda rows: None, ledger=led)
        assert led.completed() == set()


def _row(key):
    return {"input_id": key, "text": f"t-{key}"}


class TestCommitBuffer:
    def test_appends_before_acking(self, tmp_path):
        led = _ledger(tmp_path)

        def append(rows):
            assert led.completed() == set()     # nothing acked until the rows are down

        buf = CommitBuffer(append=append, ledger=led)
        buf.add([_row("a")], {"input_id": "a"})
        buf.add([_row("b")], {"input_id": "b"})
        buf.flush()
        assert led.completed() == {"a", "b"}

    def test_flushes_every_n_units_and_at_exit(self, tmp_path):
        led = _ledger(tmp_path)
        appends = []
        with CommitBuffer(append=appends.append, ledger=led, every=2) as buf:
            buf.add([_row("a")], {"input_id": "a"})
            assert appends == []
            buf.add([_row("b")], {"input_id": "b"})
            assert appends == [[_row("a"), _row("b")]]
            buf.add([_row("c")], {"input_id": "c"})
        assert appends == [[_row("a"), _row("b")], [_row("c")]]
        assert buf.committed == 3 and led.completed() == {"a", "b", "c"}

    def test_a_failing_append_acks_nothing_and_drops_the_batch(self, tmp_path):
        """A unit whose rows never reached disk must re-run, not be acked, and
        must not be re-appended by a later flush either."""
        led = _ledger(tmp_path)
        appends = []

        def append(rows):
            if any(r["input_id"] == "a" for r in rows):
                raise OSError("disk full")
            appends.append(rows)

        buf = CommitBuffer(append=append, ledger=led)
        buf.add([_row("a")], {"input_id": "a"})
        with pytest.raises(OSError):
            buf.flush()
        assert led.completed() == set() and buf.committed == 0
        buf.flush()                               # nothing queued: no append
        buf.add([_row("c")], {"input_id": "c"})
        buf.flush()
        assert appends == [[_row("c")]] and led.completed() == {"c"}

    def test_a_unit_with_no_rows_is_acked(self, tmp_path):
        led = _ledger(tmp_path)
        appends = []
        with CommitBuffer(append=appends.append, ledger=led) as buf:
            buf.add([], {"input_id": "dropped", "status": "dropped_dedup"})
        assert appends == [] and led.completed() == {"dropped"}

    def test_a_unit_with_several_rows_is_one_ack(self, tmp_path):
        led = _ledger(tmp_path)
        appends = []
        with CommitBuffer(append=appends.append, ledger=led) as buf:
            buf.add([_row("a"), _row("a"), _row("a")], {"input_id": "a"})
        assert len(appends[0]) == 3
        assert len(led.path.read_text(encoding="utf-8").splitlines()) == 1
        assert buf.committed == 1

    def test_finished_units_are_flushed_when_the_block_raises(self, tmp_path):
        """A finished unit is complete whatever stops the driver, so leaving
        on an exception still commits what is queued."""
        led = _ledger(tmp_path)
        appends = []
        with pytest.raises(RuntimeError, match="boom"):
            with CommitBuffer(append=appends.append, ledger=led) as buf:
                buf.add([_row("a")], {"input_id": "a"})
                raise RuntimeError("boom")
        assert appends == [[_row("a")]] and led.completed() == {"a"}

    def test_a_failing_exit_flush_chains_to_the_original_exception(self, tmp_path):
        """Both failures stay visible: the append error is raised with the
        driver's exception as its context, and nothing is acked."""
        led = _ledger(tmp_path)

        def append(rows):
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full") as raised:
            with CommitBuffer(append=append, ledger=led) as buf:
                buf.add([_row("a")], {"input_id": "a"})
                raise RuntimeError("boom")
        assert isinstance(raised.value.__context__, RuntimeError)
        assert led.completed() == set()

    def test_a_unit_is_queued_whole_or_not_at_all(self, tmp_path):
        """An interrupt while a unit is being queued must not leave its rows
        in a save that does not ack it."""
        led = _ledger(tmp_path)
        appends = []

        def interrupted():
            yield _row("b")
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt), CommitBuffer(
                append=appends.append, ledger=led) as buf:
            buf.add([_row("a")], {"input_id": "a"})
            buf.add(interrupted(), {"input_id": "b"})
        assert appends == [[_row("a")]] and led.completed() == {"a"}

    def test_every_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError):
            CommitBuffer(append=lambda rows: None, ledger=_ledger(tmp_path), every=0)

    def test_only_finished_units_are_acked_when_the_driver_raises(self, tmp_path):
        """A dispatch failure propagates through the buffer: the units in that
        call and the deferred ones are left un-acked for the next run, while
        the unit that finished before it is committed on the way out."""
        led = _ledger(tmp_path)
        appends = []

        class _FailsSecondCall:
            calls = 0

            def batch_generate(self, model, messages_list, **kw):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("batch boom")
                return ["r"] * len(messages_list)

        def unit(key, n):
            for _ in range(n):
                yield LLMRequest("m", [{"role": "user", "content": key}])
            return key

        units = [("a", unit("a", 2)), ("b", unit("b", 1)), ("c", unit("c", 1))]
        with pytest.raises(RuntimeError, match="batch boom"):
            with CommitBuffer(append=appends.append, ledger=led) as buf:
                drive_units(iter(units), resolve=as_resolver(_FailsSecondCall()), pool_size=2,
                            on_done=lambda k, v: buf.add([_row(k)], {"input_id": k}))
        assert appends == [[_row("b")]] and led.completed() == {"b"}


def _lock_file(led):
    return led.path.with_name(led.path.name + ".lock")


def _another_run(led):
    """Hold the stage's lock as a second run would: through a lock of its own."""
    lock = FileLock(_lock_file(led), timeout=0)
    lock.acquire()
    return lock


def _is_free(led):
    """Whether a second run could take the stage's lock now."""
    try:
        _another_run(led).release()
    except Timeout:
        return False
    return True


def _buffer(led, out):
    return CommitBuffer(append=lambda rows: append_rows(out, rows), ledger=led)


class TestOneWriter:
    """One process writes a stage's output at a time."""

    def test_a_resume_with_an_artifact_takes_the_lock_beside_the_ledger(self, tmp_path):
        led, _out = _opened(tmp_path)
        assert _lock_file(led).name == "a.state.jsonl.lock"
        assert not _is_free(led)

    def test_a_resume_without_an_artifact_takes_no_lock(self, tmp_path):
        led = _ledger(tmp_path)
        resume_state(led, resume=True)
        assert list(tmp_path.iterdir()) == [] and _is_free(led)

    @pytest.mark.parametrize("resume", [True, False])
    def test_a_second_run_is_refused_before_it_touches_a_file(self, tmp_path, resume):
        led, out = _opened(tmp_path)
        with _buffer(led, out) as buf:
            buf.add([_row("a")], {"input_id": "a"})
        manifest = Manifest(tmp_path / "a.manifest.jsonl")
        manifest.write([{"input_id": "a"}])
        with out.open("ab") as fh:              # a save between its two writes
            fh.write(b"b,t-b\n")
        before = {p.name: p.read_bytes() for p in (out, led.path, manifest.path)}

        holder = _another_run(led)
        try:
            with pytest.raises(RuntimeError) as refused:
                resume_state(_ledger(tmp_path), resume=resume, artifact=out,
                             manifest=manifest)
        finally:
            holder.release()
        assert str(_lock_file(led)) in str(refused.value)
        assert "another run" in str(refused.value)
        assert "this same process" in str(refused.value)
        assert {p.name: p.read_bytes()
                for p in (out, led.path, manifest.path)} == before
        assert not out.with_name("out.csv.cut").exists()

    def test_the_lock_is_free_after_the_buffer_exits(self, tmp_path):
        led, out = _opened(tmp_path)
        with _buffer(led, out) as buf:
            buf.add([_row("a")], {"input_id": "a"})
            assert not _is_free(led)
        assert _is_free(led)

    def test_the_lock_is_free_after_the_block_raises(self, tmp_path):
        led, out = _opened(tmp_path)
        with pytest.raises(RuntimeError, match="boom"), _buffer(led, out):
            raise RuntimeError("boom")
        assert _is_free(led)

    def test_the_lock_is_free_after_the_exit_flush_raises(self, tmp_path):
        led, _out = _opened(tmp_path)

        def append(rows):
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"), CommitBuffer(
                append=append, ledger=led) as buf:
            buf.add([_row("a")], {"input_id": "a"})
        assert _is_free(led)

    def test_a_stage_that_stopped_before_its_buffer_can_be_called_again(self, tmp_path):
        """An interrupted notebook cell leaves the lock with this process."""
        led, out = _opened(tmp_path)
        again = _ledger(tmp_path)
        assert resume_state(again, resume=True, artifact=out) == set()
        with _buffer(again, out) as buf:
            buf.add([_row("a")], {"input_id": "a"})
        assert again.completed() == {"a"} and _is_free(led)

    def test_release_lock_frees_a_stage_that_opens_no_buffer(self, tmp_path):
        led, _out = _opened(tmp_path)
        release_lock(led)
        assert _is_free(led)
        release_lock(led)                       # nothing held: nothing to do

    def test_a_buffer_refuses_a_ledger_that_resume_state_did_not_open(self, tmp_path):
        """Without the resume, what an unfinished save left would stay in the
        file and be saved over."""
        led, out = _opened(tmp_path)
        with _buffer(led, out) as buf:
            buf.add([_row("a")], {"input_id": "a"})
        before = (out.read_bytes(), led.path.read_bytes())

        for unopened in (led, _ledger(tmp_path)):   # after its buffer; a new object
            with pytest.raises(RuntimeError, match="resume_state"), _buffer(
                    unopened, out) as buf:
                buf.add([_row("b")], {"input_id": "b"})
        assert (out.read_bytes(), led.path.read_bytes()) == before

    def test_a_ledger_opened_without_an_artifact_saves_as_before(self, tmp_path):
        led = _ledger(tmp_path)
        resume_state(led, resume=True)
        appends = []
        with CommitBuffer(append=appends.append, ledger=led) as buf:
            buf.add([_row("a")], {"input_id": "a"})
        assert appends == [[_row("a")]] and _lines(led) == [{"input_id": "a"}]

    @pytest.mark.parametrize("change", ["grew", "shrank", "gone"])
    def test_a_file_changed_behind_the_writer_stops_the_next_save(
            self, tmp_path, change):
        led, out = _opened(tmp_path)
        appends = []

        def append(rows):
            appends.append(rows)
            return append_rows(out, rows)

        with pytest.raises(RuntimeError, match="out.csv") as stopped, CommitBuffer(
                append=append, ledger=led, every=1) as buf:
            buf.add([_row("a")], {"input_id": "a"})
            if change == "grew":
                with out.open("ab") as fh:
                    fh.write(b"x,another writer\n")
            elif change == "shrank":
                out.write_bytes(out.read_bytes()[:-3])
            else:
                out.unlink()
            ledger_before = led.path.read_bytes()
            buf.add([_row("b")], {"input_id": "b"})
        assert "recorded" in str(stopped.value)
        assert len(appends) == 1                # the second save wrote no row
        assert led.path.read_bytes() == ledger_before
