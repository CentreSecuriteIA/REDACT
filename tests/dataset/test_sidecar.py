"""Tests for how a sidecar file is written and streamed.

A manifest is replaced whole or not at all, and read one row at a time.
"""

import json
import os

import pytest

import redact.dataset.sidecar as sidecar
from redact.dataset import Ledger, Manifest


def _names(folder):
    return sorted(p.name for p in folder.iterdir())


def _manifest(tmp_path, rows=()):
    man = Manifest(tmp_path / "run.manifest.jsonl")
    man.write(rows)
    return man


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

        monkeypatch.setattr(sidecar.os, "replace", replace)
        monkeypatch.setattr(sidecar.time, "sleep", lambda seconds: None)
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

        monkeypatch.setattr(sidecar.os, "replace", replace)
        monkeypatch.setattr(sidecar.time, "sleep", lambda seconds: None)
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
