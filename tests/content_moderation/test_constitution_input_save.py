"""Seeded inputs on the save that records sizes: the files it writes, the
repeats it drops, and what a resume, a fresh run and a takeover leave.

``fixtures/seeded_inputs.json`` holds the category files of one run of three
saves as the code at commit 66817a6 wrote them, when an entry was one call
for its two samples. A CSV is compared by its cells as text. Model calls are
stubbed and depend only on the request and how many of its entry's came
before it, so a run writes the same rows every time.
"""

import csv
import json
import re
from pathlib import Path

import pandas as pd
import pytest

import redact.content_moderation.inputs as G
from redact.content_moderation.inputs import (
    _constitution_inputs_ledger,
    _constitution_inputs_manifest,
)
from redact.dataset import content_id
from tests.conftest import (
    Killed,
    MockBackend,
    kill_after_rows,
    lock_is_free,
    write_inputs,
)
from tests.content_moderation.test_constitution_input_ledger import (
    _PROMPT,
    _token_pipeline,
    _TokenBackend,
    one,
)

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "seeded_inputs.json"

# Where an entry's first sample is not its own. Two entries next to each
# other are live together; zq2 and zq4 finished long before zq40 and zq41
# were generated.
_REPEATS = {
    8: "one zq900 text, two entries of one category",
    10: "one zq900 text, two entries of one category",
    20: "one zq901 text, two entries of two categories",
    21: "one zq901 text, two entries of two categories",
    40: "first sample zq2",     # zq2 is of the same category
    41: "first sample zq4",     # zq4 is of the other
}
_NO_SAMPLES = 30    # every reply of this entry is rejected whole
_TWICE = 50         # this entry's reply holds one text twice
_FRAUD_FROM = 66    # the entries from here on are of a third category

_DESCRIPTION, _RETIRED_COLUMN = 9, 12   # columns of a fixture's category file


def _category(i: int) -> str:
    """Fraud holds the last entries only: its file first appears in the last save."""
    return "Fraud" if i >= _FRAUD_FROM else ("Violence", "Privacy")[i % 2]


def _entries(*numbers: int) -> pd.DataFrame:
    """The constitution entries of these numbers; all 70 when none is given."""
    return pd.DataFrame([
        {"source_category": _category(i), "sample_description": f"desc zq{i}",
         "constitution_subcategory": "Sub" if i % 4 else "",
         "constitution_category": "Top",
         "entry_type": "benign" if i % 3 == 0 else "harmful"}
        for i in numbers or range(70)
    ])


def _key(i: int, iteration: int = 0) -> tuple[str, str, int]:
    return f"desc zq{i}", "benign" if i % 3 == 0 else "harmful", iteration


def _keys(*numbers: int) -> set[tuple[str, str, int]]:
    """The keys of both units of each entry."""
    return {_key(i, iteration) for i in numbers for iteration in (0, 1)}


def _write(token: str, attempt: int) -> str:
    """An entry's first unit writes its first sample, its second its second."""
    i = int(token[2:])
    if i == _NO_SAMPLES:
        return "sorry, nothing here"
    first = _REPEATS.get(i, f"first sample {token}")
    second = first if i == _TWICE else f'second "quoted" sample, café {token}'
    return one((first, second)[(attempt - 1) % 2])


def _judge(token: str, attempt: int) -> str:
    """The second sample of every fifth entry is rejected, in two lines."""
    rejected = (int(token[2:]) % 5, attempt) == (3, 2)
    return "No: weak\nsecond line" if rejected else "Yes"


def _run(folder: Path, entries: pd.DataFrame, write=_write, **kwargs):
    """Run the stage on ``entries``; the result and the generating backend."""
    gen = _TokenBackend(write)
    pipe = _token_pipeline(folder, gen, _TokenBackend(_judge))
    result = pipe.run_from_constitution(
        entries, _PROMPT, samples_per_entry=2, batch_size=4, verbose=False, **kwargs)
    return result, gen


def _tree(folder: Path) -> dict[str, list[list[str]]]:
    """Each category file's header and rows, cells as text, by its path."""
    found = {}
    for path in sorted(folder.glob("*/samples.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            found[path.relative_to(folder).as_posix()] = list(csv.reader(fh))
    return found


def _texts(folder: Path) -> dict[str, list[str]]:
    """The sample texts in each category file, by the category."""
    return {name.split("/")[0]: [row[1] for row in rows[1:]]
            for name, rows in _tree(folder).items()}


def _ids_on_disk(folder: Path) -> list[str]:
    return [row[0] for rows in _tree(folder).values() for row in rows[1:]]


def _files(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix != ".lock"}


def _acks(folder: Path) -> list[dict]:
    """The ledger's lines, without the sizes a save records."""
    lines = _constitution_inputs_ledger(folder).path.read_bytes().splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "_saved"}
            for line in lines]


def _no_read(*args, **kwargs):
    raise AssertionError("the stage read the rows of its output")


def _five_saves(folder: Path) -> dict:
    """The files and the counts of one run of 70 entries, 140 units saved 32
    at a time."""
    result, _gen = _run(folder, _entries())
    return {
        "files": _tree(folder),
        "counts": {
            "entries": result.total_entries_processed,
            "skipped": result.skipped_entries,
            "generated": result.total_prompts_generated,
            "accepted": result.total_prompts_accepted,
            "rejected": result.total_prompts_rejected,
        },
    }


class TestTheFilesOfARun:
    def test_five_saves_write_the_rows_of_the_fixture_with_their_iteration(
            self, tmp_path):
        """Same rows, columns and column order, but for the fixture's last
        column, which the stage no longer writes, and last an ``iteration``
        column: 1 for an entry's second sample, else 0. The one row that is
        gone: zq20 and zq21 were live together and gave one text, and each
        wrote it to its own category. Now the first to finish writes it. The
        two rows that moved: the first samples of zq40 and zq41 are repeats,
        which are not checked, so their second units finish with those of
        zq36 to zq39, not a check before them."""
        expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))["files"]
        privacy = expected["Privacy/samples.csv"]
        (second_copy,) = (row for row in privacy if row[1] == _REPEATS[21])
        assert second_copy[_DESCRIPTION] == "desc zq21"
        privacy.remove(second_copy)
        for category, moved, after in (("Violence", 40, 38), ("Privacy", 41, 39)):
            rows = expected[f"{category}/samples.csv"]
            (row,) = (row for row in rows if row[_DESCRIPTION] == f"desc zq{moved}")
            rows.remove(row)
            last = max(i for i, r in enumerate(rows)
                       if r[_DESCRIPTION] == f"desc zq{after}")
            rows.insert(last + 1, row)
        for rows in expected.values():
            rows[0][_RETIRED_COLUMN:] = ["iteration"]
            for row in rows[1:]:
                row[_RETIRED_COLUMN:] = [
                    "1" if row[1].startswith('second "quoted"') else "0"]

        run = _five_saves(tmp_path)
        assert run["files"] == expected
        # Units now: 140, of which zq30's two are rejected and five give a
        # text the folder holds (zq10 or zq8, zq21, zq40, zq41, zq50).
        assert run["counts"] == {"entries": 133, "skipped": 7, "generated": 133,
                                 "accepted": 120, "rejected": 13}
        saves = [line for line in _constitution_inputs_ledger(tmp_path)
                 .path.read_bytes().splitlines() if b'"_saved"' in line]
        assert len(saves) == 1 + 5      # the opening line, then one a save

    def test_a_text_that_is_exactly_n_a_survives_a_second_save(self, tmp_path):
        """The rewrite read ``N/A`` back as a missing value and wrote an
        empty cell."""
        def run(*numbers):
            gen = _TokenBackend(lambda t, n: one("N/A" if t == "zq0" else t))
            pipe = _token_pipeline(tmp_path, gen, MockBackend("Yes", model="tok-check"))
            pipe.run_from_constitution(
                _entries(*numbers), _PROMPT, samples_per_entry=1, verbose=False)

        run(0)
        run(0, 2)
        assert _texts(tmp_path) == {"Violence": ["N/A", "zq2"]}

    def test_a_column_a_file_lacks_raises_before_any_file_of_the_save_is_written(
            self, tmp_path):
        _run(tmp_path, _entries(0, 1))
        privacy = tmp_path / "Privacy" / "samples.csv"
        older = pd.read_csv(privacy).drop(columns="source_group_tag")
        older.to_csv(privacy, index=False)
        _constitution_inputs_ledger(tmp_path).path.unlink()     # an older folder
        before = _files(tmp_path)

        # One save: zq2 goes to Violence, which is written first, zq3 to Privacy.
        with pytest.raises(ValueError, match="source_group_tag"):
            _run(tmp_path, _entries(0, 1, 2, 3))
        after = _files(tmp_path)
        assert after.pop("constitution_inputs.manifest.jsonl")
        assert after.pop("constitution_inputs.plan.json")
        ledger = after.pop("constitution_inputs.state.jsonl")
        assert after == {name: data for name, data in before.items()
                         if name.endswith(".csv")}
        assert b"zq2" not in ledger and b"zq3" not in ledger


class TestRepeatedTexts:
    def test_two_live_entries_that_give_one_text_write_it_once(self, tmp_path):
        result, _gen = _run(tmp_path, _entries(20, 21))
        assert _texts(tmp_path) == {
            "Violence": [_REPEATS[20], 'second "quoted" sample, café zq20'],
            "Privacy": ['second "quoted" sample, café zq21'],
        }
        # Every unit is finished, and each records the row it wrote.
        shared, of_20, of_21 = (content_id(text) for text in (
            _REPEATS[20], 'second "quoted" sample, café zq20',
            'second "quoted" sample, café zq21'))
        fields = ("sample_description", "entry_type", "iteration")
        assert _acks(tmp_path)[1:] == [
            dict(zip(fields, _key(20)), ids=[shared], accepted=True),
            dict(zip(fields, _key(20, iteration=1)), ids=[of_20], accepted=True),
            dict(zip(fields, _key(21)), status="no_sample", attempts=1),
            dict(zip(fields, _key(21, iteration=1)), ids=[of_21], accepted=True),
        ]
        # The counts are of the rows written.
        counts = result.total_entries_processed, result.total_prompts_generated
        assert counts == (3, 3)

    def test_a_unit_whose_text_is_a_repeat_is_finished_with_no_row(self, tmp_path):
        def same(token, attempt):
            return one(("one zq900 text", "another zq900 text")[attempt - 1])

        result, _gen = _run(tmp_path, _entries(0, 1), write=same)
        assert _texts(tmp_path) == {
            "Violence": ["one zq900 text", "another zq900 text"]}
        assert result.total_entries_processed == len(_entries(0, 1))
        assert _constitution_inputs_ledger(tmp_path).completed() == _keys(0, 1)
        assert _acks(tmp_path)[-1]["status"] == "no_sample"

    def test_a_text_repeated_in_another_category_after_a_resume_is_dropped(
            self, tmp_path, monkeypatch):
        """The ids come from the ledger: no row is read to know them."""
        _run(tmp_path, _entries(4))
        monkeypatch.setattr(pd, "read_csv", _no_read)
        _run(tmp_path, _entries(4, 41))         # zq41 gives zq4's first sample
        assert _texts(tmp_path) == {
            "Violence": ["first sample zq4", 'second "quoted" sample, café zq4'],
            "Privacy": ['second "quoted" sample, café zq41'],
        }

    def test_the_ids_of_a_save_that_did_not_finish_are_not_known(
            self, tmp_path, monkeypatch):
        """Its rows are cut off, so the entry that runs again writes them."""
        kill_after_rows(monkeypatch, G, on_call=1)
        with pytest.raises(Killed):
            _run(tmp_path, _entries(4))
        _run(tmp_path, _entries(4))
        assert _texts(tmp_path) == {
            "Violence": ["first sample zq4", 'second "quoted" sample, café zq4']}
        ledger = _constitution_inputs_ledger(tmp_path)
        assert ledger.ids == set(_ids_on_disk(tmp_path))


class TestTheGenerationPrompt:
    def _prompts(self, gen) -> dict[str, list[dict]]:
        """The messages of the first request made for each entry, by its token."""
        prompts = {}
        for call in gen.calls:
            text = call["messages"][-1]["content"]
            prompts.setdefault(re.search(r"zq\d+", text).group(), call["messages"])
        return prompts

    def test_it_is_the_same_before_and_after_a_save(self, tmp_path):
        """No text an earlier unit saved enters a later unit's prompt."""
        _result, gen = _run(tmp_path / "many", _entries(*range(100, 140)))
        _result, alone = _run(tmp_path / "alone", _entries(139))
        assert self._prompts(gen)["zq139"] == self._prompts(alone)["zq139"]


class TestAResume:
    def test_it_reads_no_row_of_the_output(self, tmp_path, monkeypatch):
        _run(tmp_path, _entries(*range(6)))
        before = _files(tmp_path)
        monkeypatch.setattr(pd, "read_csv", _no_read)
        result, gen = _run(tmp_path, _entries(*range(6)))
        assert gen.calls == [] and result.total_entries_processed == 0
        assert _files(tmp_path) == before
        assert lock_is_free(_constitution_inputs_ledger(tmp_path).path)

    def test_a_save_killed_between_its_files_is_cut_back_and_made_again(
            self, tmp_path, monkeypatch):
        """Violence was written, Privacy and the acks were not."""
        _run(tmp_path, _entries(0, 1))
        saved = _files(tmp_path)
        kill_after_rows(monkeypatch, G, on_call=1)
        with pytest.raises(Killed):
            _run(tmp_path, _entries(0, 1, 2, 3))
        killed = _files(tmp_path)
        assert killed["Violence/samples.csv"] != saved["Violence/samples.csv"]
        assert killed["Privacy/samples.csv"] == saved["Privacy/samples.csv"]
        assert lock_is_free(_constitution_inputs_ledger(tmp_path).path)

        _result, gen = _run(tmp_path, _entries(0, 1, 2, 3))
        assert sorted(token for batch in gen.batches for token in batch) == [
            "zq2", "zq2*", "zq3", "zq3*"]
        assert _texts(tmp_path) == {
            "Violence": ["first sample zq0", 'second "quoted" sample, café zq0',
                         "first sample zq2", 'second "quoted" sample, café zq2'],
            "Privacy": ["first sample zq1", 'second "quoted" sample, café zq1',
                        "first sample zq3", 'second "quoted" sample, café zq3'],
        }
        assert (tmp_path / "Violence" / "samples.csv.cut").exists()

    def test_a_category_first_written_by_a_save_that_died_starts_again(
            self, tmp_path, monkeypatch):
        _run(tmp_path, _entries(0))
        kill_after_rows(monkeypatch, G, on_call=1)
        with pytest.raises(Killed):
            _run(tmp_path, _entries(0, 1))      # Privacy's first rows, no acks
        _run(tmp_path, _entries(0, 1))
        assert _tree(tmp_path)["Privacy/samples.csv"][0][0] == "sample_id"
        assert _texts(tmp_path)["Privacy"] == [
            "first sample zq1", 'second "quoted" sample, café zq1']

    @pytest.mark.parametrize("resume", [True, False])
    def test_the_stage_touches_nothing_while_another_run_holds_the_lock(
            self, tmp_path, resume):
        from filelock import FileLock

        _run(tmp_path, _entries(0, 1))
        before = _files(tmp_path)
        ledger = _constitution_inputs_ledger(tmp_path)
        another = FileLock(str(ledger.path) + ".lock", timeout=0)
        another.acquire()
        try:
            with pytest.raises(RuntimeError, match="is locked"):
                _run(tmp_path, _entries(0, 1, 2), resume=resume)
        finally:
            another.release()
        assert _files(tmp_path) == before       # the manifest too


def _again(token: str, attempt: int) -> str:
    """What a second run writes: other texts, and none for the same entry."""
    if token == f"zq{_NO_SAMPLES}":
        return "sorry, nothing here"
    return one(f"{('first', 'second')[(attempt - 1) % 2]} sample again {token}")


class TestAFreshRun:
    def test_it_removes_the_tree_its_cut_files_and_the_ledger(self, tmp_path):
        _run(tmp_path, _entries(0, 1))
        with (tmp_path / "Violence" / "samples.csv").open("ab") as fh:
            fh.write(b"a tail\r\n")
        _run(tmp_path, _entries(0, 1))          # the resume keeps the tail in .cut
        assert (tmp_path / "Violence" / "samples.csv.cut").exists()

        _result, gen = _run(tmp_path, _entries(1), resume=False)
        assert set(_files(tmp_path)) == {
            "Privacy/samples.csv", "constitution_inputs.state.jsonl",
            "constitution_inputs.manifest.jsonl", "constitution_inputs.plan.json"}
        assert _acks(tmp_path)[0] == {}         # a new ledger: its opening line
        assert _constitution_inputs_manifest(tmp_path).load() == [
            {"sample_description": "desc zq1", "entry_type": "harmful",
             "iteration": iteration, "category": "Privacy", "status": "planned"}
            for iteration in (0, 1)]


def _old_folder(folder: Path, with_rows, acked) -> None:
    """A folder as the stage left it before the ledger recorded sizes and ids,
    and before an entry was a unit per sample: rows without an iteration, and
    one plain ledger line per entry."""
    for i in with_rows:
        description, entry_type, _iteration = _key(i)
        write_inputs(
            folder, [f"first sample zq{i}", f"second sample zq{i}"], _category(i),
            source="constitution", extra={
                "entry_type": entry_type, "subcategory": "Sub",
                "source_sample_description": description,
                "constitution_category": "Top", "source_group_tag": "",
                "rejection_reason": ""})
    lines = [dict(zip(("sample_description", "entry_type"), _key(i))) for i in acked]
    _constitution_inputs_ledger(folder).path.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


class TestAFolderWrittenBeforeUnitsHadAnIteration:
    def test_a_resume_stops_before_any_call_and_touches_nothing(self, tmp_path):
        _old_folder(tmp_path, with_rows=(0, 1), acked=(0, 1))
        old = _files(tmp_path)
        with pytest.raises(ValueError, match="without an iteration") as stopped:
            result, gen = _run(tmp_path, _entries(0, 1, 2))
        assert "resume=False" in str(stopped.value)
        assert "new data_dir" in str(stopped.value)
        assert _files(tmp_path) == old
        assert lock_is_free(_constitution_inputs_ledger(tmp_path).path)

    def test_a_file_without_the_column_stops_it_too(self, tmp_path):
        """Also with no ledger: a file of the first format."""
        old_columns = {"entry_type": "benign", "sample_description": "desc zq0"}
        write_inputs(tmp_path, ["first sample zq0", "second sample zq0"], "Violence",
                     source="constitution", extra=old_columns)
        with pytest.raises(ValueError, match="'iteration'"):
            _run(tmp_path, _entries(0))
        assert not _constitution_inputs_ledger(tmp_path).exists()

    def test_a_fresh_run_starts_every_entry_over(self, tmp_path):
        """The acks of a ledger of that time are removed with the rows."""
        _old_folder(tmp_path, with_rows=(0, 1), acked=(0, 1, _NO_SAMPLES))
        result, gen = _run(tmp_path, _entries(0, 1, _NO_SAMPLES), write=_again,
                           resume=False)
        assert (result.total_entries_processed, result.skipped_entries) == (4, 2)
        assert _texts(tmp_path) == {
            "Violence": ["first sample again zq0", "second sample again zq0"],
            "Privacy": ["first sample again zq1", "second sample again zq1"],
        }
        assert _constitution_inputs_ledger(tmp_path).completed() == _keys(
            0, 1, _NO_SAMPLES)
