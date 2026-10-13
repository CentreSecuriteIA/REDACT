"""The two resume invariants, tested directly rather than only through stages.

Both are only observable after a crash, which is exactly why they are worth
testing at this level: the per-stage tests exercise the happy path, and these
pin the failure ordering.
"""

import json
import os
import zlib

import pytest

from redact.dataset import (
    CommitBuffer,
    Ledger,
    Manifest,
    append_rows,
    commit,
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


def _save(out, led, key, text="t"):
    """One save of one unit with one row, through ``append_rows``."""
    commit([{"input_id": key, "text": text}],
           append=lambda rows: append_rows(out, rows), ledger=led)


class TestCommitRecordsSizes:
    """The last ledger line of a save says how much of each file is saved."""

    def test_the_last_line_carries_the_size_and_a_checksum(self, tmp_path):
        led, out = _ledger(tmp_path), tmp_path / "out.csv"
        commit([{"input_id": "a", "text": "x"}, {"input_id": "b", "text": "y"}],
               append=lambda rows: append_rows(out, rows), ledger=led)
        data = out.read_bytes()
        assert _lines(led) == [
            {"input_id": "a"},
            {"input_id": "b", "_saved": {"out.csv": [len(data), _crc(data)]}},
        ]

    def test_the_checksum_covers_the_last_4096_bytes_before_the_size(self, tmp_path):
        led, out = _ledger(tmp_path), tmp_path / "out.csv"
        _save(out, led, "a", "x" * 5000)
        data = out.read_bytes()
        assert len(data) > _CHECKSUM_BYTES
        assert led.saved == {"out.csv": [len(data), _crc(data)]}
        assert _crc(data) != f"{zlib.crc32(data):08x}"

    def test_the_size_is_the_files_size_in_bytes(self, tmp_path):
        led, out = _ledger(tmp_path), tmp_path / "out.csv"
        _save(out, led, "a", "café 中文\nsecond line\r\nthird")
        assert led.saved["out.csv"][0] == os.path.getsize(out)

    def test_a_file_is_named_by_its_path_from_the_ledgers_folder(self, tmp_path):
        led = _ledger(tmp_path)
        out = tmp_path / "Cyber Crime" / "samples.csv"
        out.parent.mkdir()
        _save(out, led, "a")
        assert list(led.saved) == ["Cyber Crime/samples.csv"]

    def test_every_saved_line_carries_every_file(self, tmp_path):
        led = _ledger(tmp_path)
        first, second = tmp_path / "a.csv", tmp_path / "b.csv"
        _save(first, led, "a")
        recorded = led.saved["a.csv"]
        _save(second, led, "b")
        assert _lines(led)[-1]["_saved"] == {
            "a.csv": recorded,
            "b.csv": [second.stat().st_size, _crc(second.read_bytes())],
        }

    def test_a_save_with_no_rows_repeats_what_was_recorded(self, tmp_path):
        led, out = _ledger(tmp_path), tmp_path / "out.csv"
        _save(out, led, "a")
        recorded = dict(led.saved)
        commit([], append=lambda rows: append_rows(out, rows), ledger=led,
               records=[{"input_id": "dropped"}])
        assert _lines(led)[-1] == {"input_id": "dropped", "_saved": recorded}
        assert led.completed() == {"a", "dropped"}


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
