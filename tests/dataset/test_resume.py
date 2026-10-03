"""The two resume invariants, tested directly rather than only through stages.

Both are only observable after a crash, which is exactly why they are worth
testing at this level: the per-stage tests exercise the happy path, and these
pin the failure ordering.
"""

import json

import pytest

from redact.dataset import Ledger, commit, resume_state


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
