"""Seeded inputs on the save that records sizes: the files it writes, the
repeats it drops, and what a resume, a fresh run and a takeover leave.

``fixtures/seeded_inputs.json`` holds the category files of one run of three
saves as the code wrote them through ``append_samples`` (commit 66817a6). A
CSV is compared by its cells as text. Model calls are stubbed and depend only
on the request, so a run writes the same rows every time.
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
from redact.dataset import Ledger, append_samples, content_id
from tests.conftest import Killed, MockBackend, kill_after_rows, lock_is_free
from tests.content_moderation.test_constitution_input_ledger import (
    _PROMPT,
    _token_pipeline,
    _TokenBackend,
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

_DESCRIPTION, _STYLE = 9, 12    # columns of a category file


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


def _key(i: int, style: str = "") -> tuple[str, str, str]:
    return f"desc zq{i}", "benign" if i % 3 == 0 else "harmful", style


def _write(token: str, attempt: int) -> str:
    i = int(token[2:])
    if i == _NO_SAMPLES:
        return "sorry, nothing here"
    first = _REPEATS.get(i, f"first sample {token}")
    second = first if i == _TWICE else f'second "quoted" sample, café {token}'
    return f"1. {first}\n2. {second}"


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


def _three_saves(folder: Path) -> dict:
    """The files and the counts of one run of 70 entries, saved 32 at a time."""
    result, _gen = _run(folder, _entries(), style="long")
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
    def test_three_saves_write_the_rows_append_samples_wrote_but_one(self, tmp_path):
        """Same rows, columns and column order. The one row that is gone:
        zq20 and zq21 were live together and gave one text, and each wrote
        it to its own category. Now the first to finish writes it."""
        expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        privacy = expected["files"]["Privacy/samples.csv"]
        (second_copy,) = (row for row in privacy if row[1] == _REPEATS[21])
        assert second_copy[_DESCRIPTION] == "desc zq21"
        privacy.remove(second_copy)

        assert _three_saves(tmp_path) == expected
        saves = [line for line in _constitution_inputs_ledger(tmp_path)
                 .path.read_bytes().splitlines() if b'"_saved"' in line]
        assert len(saves) == 1 + 3      # the opening line, then one a save

    def test_a_file_has_the_columns_append_samples_gives_a_new_file(self, tmp_path):
        _run(tmp_path, _entries(0))
        append_samples(
            ["a sample"], category="Old", turn=0, source="constitution",
            extra_columns=[{
                "entry_type": "harmful", "subcategory": "Sub",
                "source_sample_description": "desc", "constitution_category": "Top",
                "source_group_tag": "", "template_style": "", "rejection_reason": ""}],
            dataset_dir=tmp_path)
        tree = _tree(tmp_path)
        assert tree["Violence/samples.csv"][0] == tree["Old/samples.csv"][0]

    def test_a_text_that_is_exactly_n_a_survives_a_second_save(self, tmp_path):
        """The rewrite read ``N/A`` back as a missing value and wrote an
        empty cell."""
        def run(*numbers):
            gen = _TokenBackend(lambda t, n: "1. N/A" if t == "zq0" else f"1. {t}")
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
        # Both entries are finished, and each records the rows it wrote.
        shared, of_20, of_21 = (content_id(text) for text in (
            _REPEATS[20], 'second "quoted" sample, café zq20',
            'second "quoted" sample, café zq21'))
        assert _acks(tmp_path)[1:] == [
            dict(zip(("sample_description", "entry_type", "style"), _key(20)),
                 ids=[shared, of_20]),
            dict(zip(("sample_description", "entry_type", "style"), _key(21)),
                 ids=[of_21]),
        ]
        # The counts are of what was generated, as before.
        counts = result.total_entries_processed, result.total_prompts_generated
        assert counts == (2, 4)

    def test_an_entry_whose_every_text_is_a_repeat_is_finished_with_no_row(
            self, tmp_path):
        def same(token, attempt):
            return "1. one zq900 text\n2. another zq900 text"

        result, _gen = _run(tmp_path, _entries(0, 1), write=same)
        assert _texts(tmp_path) == {
            "Violence": ["one zq900 text", "another zq900 text"]}
        assert result.total_entries_processed == len(_entries(0, 1))
        assert _constitution_inputs_ledger(tmp_path).completed() == {_key(0), _key(1)}
        assert _acks(tmp_path)[-1]["ids"] == []

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

    def test_a_text_another_style_wrote_is_dropped(self, tmp_path):
        _run(tmp_path, _entries(4), style="long")
        _run(tmp_path, _entries(41), style="short")
        assert _texts(tmp_path)["Privacy"] == ['second "quoted" sample, café zq41']

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


_SHOWN = 20     # the texts a request lists


def _shown(prompt: str) -> tuple[list[str], str]:
    """The texts a request lists as existing, and its last line."""
    listed = prompt.split("Do NOT generate anything similar to these:\n")[1]
    lines = listed.splitlines()
    return [line[2:] for line in lines if line.startswith("- ")], lines[-1]


class TestThePreviewOfEarlierTexts:
    def _prompts(self, gen) -> dict[str, str]:
        """The first request made for each entry, by its token."""
        prompts = {}
        for call in gen.calls:
            text = call["messages"][-1]["content"]
            prompts.setdefault(re.search(r"zq\d+", text).group(), text)
        return prompts

    def test_a_request_lists_the_first_twenty_texts_of_the_run(self, tmp_path):
        _result, gen = _run(tmp_path, _entries(*range(100, 140)))
        prompts = self._prompts(gen)
        assert "already exist" not in prompts["zq100"]

        # The texts of the first ten entries, in the order they were finished.
        ids = [sample_id for ack in _acks(tmp_path)[1:11] for sample_id in ack["ids"]]
        text_of = {content_id(text): text
                   for texts in _texts(tmp_path).values() for text in texts}
        first_twenty = [text_of[sample_id] for sample_id in ids]
        assert len(first_twenty) == _SHOWN
        for late in ("zq132", "zq139"):
            shown, _last = _shown(prompts[late])
            assert shown == first_twenty

    def test_a_request_says_how_many_more_there_are(self, tmp_path):
        _result, gen = _run(tmp_path, _entries(*range(100, 140)))
        shown, last = _shown(self._prompts(gen)["zq139"])
        said = re.fullmatch(
            r"\(\.\.\. and (\d+) more\. Ensure all your outputs are novel\.\)", last)
        # The last four entries are generated once the first 32 are finished,
        # with two texts each.
        assert len(shown) == _SHOWN and int(said[1]) == 2 * 32 - _SHOWN

    def test_after_a_resume_it_starts_empty(self, tmp_path, monkeypatch):
        _run(tmp_path, _entries(*range(12)))
        monkeypatch.setattr(pd, "read_csv", _no_read)
        _result, gen = _run(tmp_path, _entries(*range(13)))
        assert "already exist" not in self._prompts(gen)["zq12"]


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
            "zq2", "zq3"]
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

    @pytest.mark.parametrize("fresh", [False, True])
    def test_the_stage_touches_nothing_while_another_run_holds_the_lock(
            self, tmp_path, fresh):
        from filelock import FileLock

        _run(tmp_path, _entries(0, 1), style="long")
        before = _files(tmp_path)
        ledger = _constitution_inputs_ledger(tmp_path)
        another = FileLock(str(ledger.path) + ".lock", timeout=0)
        another.acquire()
        try:
            with pytest.raises(RuntimeError, match="is locked"):
                _run(tmp_path, _entries(0, 1, 2), style="long", fresh=fresh)
        finally:
            another.release()
        assert _files(tmp_path) == before       # the manifest too

    def test_a_run_that_does_not_save_leaves_the_folder_alone(self, tmp_path):
        _run(tmp_path, _entries(0, 1))
        before = _files(tmp_path)
        _result, gen = _run(tmp_path, _entries(0, 1, 2), save=False, fresh=True)
        # Nothing is resumed from: the two saved entries are generated too.
        assert sorted(token for batch in gen.batches for token in batch) == [
            "zq0", "zq1", "zq2"]
        assert _files(tmp_path) == before
        assert not list(tmp_path.glob("*.lock"))


def _again(token: str, attempt: int) -> str:
    """What a second run writes: other texts, and none for the same entry."""
    if token == f"zq{_NO_SAMPLES}":
        return "sorry, nothing here"
    return f"1. first sample again {token}\n2. second sample again {token}"


class TestAFreshRun:
    def _two_styles(self, folder):
        """Long in Violence and Privacy; short there and alone in Fraud, with
        an entry that has no row."""
        _run(folder, _entries(0, 1, 2), style="long")
        _run(folder, _entries(4, 5, 66, _NO_SAMPLES), style="short")

    def test_of_one_style_keeps_the_rows_and_acks_of_the_others(self, tmp_path):
        _run(tmp_path, _entries(0, 1, 2), style="long")
        long_only = _files(tmp_path)
        _run(tmp_path, _entries(4, 5, 66, _NO_SAMPLES), style="short")
        short_ids = {row[0] for rows in _tree(tmp_path).values()
                     for row in rows[1:] if row[_STYLE] == "short"}

        _result, gen = _run(tmp_path, _entries(4, 5, 66, _NO_SAMPLES), write=_again,
                            style="short", fresh=True)
        # Every entry of the style was generated again, three times for none.
        assert sorted(token for batch in gen.batches for token in batch) == sorted(
            ["zq4", "zq5", "zq66", "zq30", "zq30*", "zq30*"])
        after = _files(tmp_path)
        for name in ("Violence/samples.csv", "Privacy/samples.csv"):
            assert after[name].startswith(long_only[name])      # byte for byte
        assert _texts(tmp_path) == {
            "Violence": ["first sample zq0", 'second "quoted" sample, café zq0',
                         "first sample zq2", 'second "quoted" sample, café zq2',
                         "first sample again zq4", "second sample again zq4"],
            "Privacy": ["first sample zq1", 'second "quoted" sample, café zq1',
                        "first sample again zq5", "second sample again zq5"],
            "Fraud": ["first sample again zq66", "second sample again zq66"],
        }
        ledger = _constitution_inputs_ledger(tmp_path)
        assert ledger.completed() == {
            *(_key(i, "long") for i in (0, 1, 2)),
            *(_key(i, "short") for i in (4, 5, 66, _NO_SAMPLES))}
        # No id of a removed row is known, and the sizes are those of the files.
        assert ledger.ids == set(_ids_on_disk(tmp_path))
        assert not ledger.ids & short_ids
        assert {name: size for name, (size, _crc) in ledger.saved.items()} == {
            name: len(data) for name, data in after.items() if name.endswith(".csv")}

    def test_of_one_style_that_dies_at_its_first_call_is_resumed(self, tmp_path):
        self._two_styles(tmp_path)

        def dies(token, attempt):
            raise RuntimeError("engine died")

        with pytest.raises(RuntimeError, match="engine died"):
            _run(tmp_path, _entries(4, 5, 66, _NO_SAMPLES), write=dies,
                 style="short", fresh=True)
        assert all(row[_STYLE] == "long"
                   for rows in _tree(tmp_path).values() for row in rows[1:])
        assert "Fraud/samples.csv" not in _tree(tmp_path)   # no row left: no file
        ledger = _constitution_inputs_ledger(tmp_path)
        assert ledger.completed() == {_key(i, "long") for i in (0, 1, 2)}
        assert lock_is_free(ledger.path)

        result, _gen = _run(tmp_path, _entries(4, 5, 66, _NO_SAMPLES), style="short")
        assert (result.total_entries_processed, result.skipped_entries) == (3, 1)
        assert _texts(tmp_path)["Fraud"] == [
            "first sample zq66", 'second "quoted" sample, café zq66']
        assert _constitution_inputs_ledger(tmp_path).ids == set(_ids_on_disk(tmp_path))

    def _dies_before_the_ledger(self, folder, monkeypatch, style):
        """A fresh run of ``style`` that wrote the files again and not the
        ledger; the folder as it leaves it."""
        with monkeypatch.context() as patch:
            patch.setattr(Ledger, "rewrite", lambda self, drop, saved: 1 / 0)
            with pytest.raises(ZeroDivisionError):
                _run(folder, _entries(4), style=style, fresh=True)
        return _files(folder)

    def test_of_one_style_that_stopped_before_the_ledger_is_finished_by_another(
            self, tmp_path, monkeypatch):
        """In between, a resume stops: rows are gone whose acks are not."""
        self._two_styles(tmp_path)
        stopped = self._dies_before_the_ledger(tmp_path, monkeypatch, "short")

        with pytest.raises(RuntimeError, match="'short' did not finish"):
            _run(tmp_path, _entries(4), style="short")
        assert _files(tmp_path) == stopped

        _run(tmp_path, _entries(4), write=_again, style="short", fresh=True)
        assert _texts(tmp_path)["Violence"][-2:] == [
            "first sample again zq4", "second sample again zq4"]
        ledger = _constitution_inputs_ledger(tmp_path)
        assert ledger.completed() == {
            *(_key(i, "long") for i in (0, 1, 2)), _key(4, "short")}
        assert ledger.ids == set(_ids_on_disk(tmp_path))

    def test_of_another_style_stops_on_one_that_did_not_finish(
            self, tmp_path, monkeypatch):
        """It would leave a ledger that matches the files, with the acks of
        the first style in it and its rows gone: a resume of that style
        would then generate nothing."""
        self._two_styles(tmp_path)
        stopped = self._dies_before_the_ledger(tmp_path, monkeypatch, "short")

        with pytest.raises(RuntimeError, match="'short' did not finish"):
            _run(tmp_path, _entries(0, 1, 2), write=_again, style="long", fresh=True)
        assert _files(tmp_path) == stopped
        assert lock_is_free(_constitution_inputs_ledger(tmp_path).path)

        # The first is made again and finishes; then the second runs.
        short = _entries(4, 5, 66, _NO_SAMPLES)
        _result, gen = _run(tmp_path, short, style="short", fresh=True)
        again = {token.rstrip("*") for batch in gen.batches for token in batch}
        assert again == {"zq4", "zq5", "zq66", "zq30"}
        _run(tmp_path, _entries(0, 1, 2), write=_again, style="long", fresh=True)
        assert _texts(tmp_path) == {
            "Violence": ["first sample zq4", 'second "quoted" sample, café zq4',
                         "first sample again zq0", "second sample again zq0",
                         "first sample again zq2", "second sample again zq2"],
            "Privacy": ["first sample zq5", 'second "quoted" sample, café zq5',
                        "first sample again zq1", "second sample again zq1"],
            "Fraud": ["first sample zq66", 'second "quoted" sample, café zq66'],
        }
        ledger = _constitution_inputs_ledger(tmp_path)
        assert ledger.completed() == {
            *(_key(i, "long") for i in (0, 1, 2)),
            *(_key(i, "short") for i in (4, 5, 66, _NO_SAMPLES))}
        assert ledger.ids == set(_ids_on_disk(tmp_path))

    def test_of_one_style_stops_on_a_file_that_lost_its_last_rows(self, tmp_path):
        """Cut short at the end of a row, the file is whole to look at. The
        ledger recorded more of it, and no fresh run is unfinished."""
        self._two_styles(tmp_path)
        violence = tmp_path / "Violence" / "samples.csv"
        data = violence.read_bytes()
        # Its last two rows are zq4's, and a row starts with its id.
        first_of_zq4 = content_id("first sample zq4").encode()
        violence.write_bytes(data[:data.index(first_of_zq4)])
        assert violence.read_bytes().endswith(b"\n")
        before = _files(tmp_path)

        with pytest.raises(RuntimeError, match="copied incompletely") as stopped:
            _run(tmp_path, _entries(4), style="short", fresh=True)
        assert str(violence) in str(stopped.value)
        assert "start the whole stage over" in str(stopped.value)
        assert _files(tmp_path) == before

    def test_of_one_style_leaves_a_file_without_a_style_column_alone(self, tmp_path):
        """Its rows are of no style, as a resume reads them."""
        append_samples(["an old sample"], category="Violence", turn=0,
                       source="constitution", dataset_dir=tmp_path)
        old = (tmp_path / "Violence" / "samples.csv").read_bytes()
        _run(tmp_path, _entries(_NO_SAMPLES), style="long", fresh=True)
        assert (tmp_path / "Violence" / "samples.csv").read_bytes() == old

    def test_without_a_style_removes_the_tree_its_cut_files_and_the_ledger(
            self, tmp_path):
        _run(tmp_path, _entries(0, 1))
        with (tmp_path / "Violence" / "samples.csv").open("ab") as fh:
            fh.write(b"a tail\r\n")
        _run(tmp_path, _entries(0, 1))          # the resume keeps the tail in .cut
        assert (tmp_path / "Violence" / "samples.csv.cut").exists()

        _result, gen = _run(tmp_path, _entries(1), fresh=True)
        assert set(_files(tmp_path)) == {
            "Privacy/samples.csv", "constitution_inputs.state.jsonl",
            "constitution_inputs.manifest.jsonl"}
        assert _acks(tmp_path)[0] == {}         # a new ledger: its opening line
        assert _constitution_inputs_manifest(tmp_path).load() == [{
            "sample_description": "desc zq1", "entry_type": "harmful", "style": "",
            "status": "planned"}]


def _old_folder(folder: Path, with_rows, acked) -> None:
    """A folder as the stage left it before the ledger recorded sizes and ids:
    rows through ``append_samples``, and one plain ledger line per entry."""
    for i in with_rows:
        description, entry_type, style = _key(i, "long")
        append_samples(
            [f"first sample zq{i}", f"second sample zq{i}"], category=_category(i),
            turn=0, source="constitution", dataset_dir=folder,
            extra_columns=[{
                "entry_type": entry_type, "subcategory": "Sub",
                "source_sample_description": description,
                "constitution_category": "Top", "source_group_tag": "",
                "template_style": style, "rejection_reason": ""}] * 2)
    lines = [dict(zip(("sample_description", "entry_type", "style"), _key(i, "long")))
             for i in acked]
    _constitution_inputs_ledger(folder).path.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


class TestAFolderWrittenBeforeSizesWereRecorded:
    def test_it_is_taken_over_by_one_read_and_one_ledger_write(
            self, tmp_path, monkeypatch):
        # zq2 and zq3 have rows and no ack; zq30 is acked and has no row.
        _old_folder(tmp_path, with_rows=(0, 1, 2, 3), acked=(0, 1, _NO_SAMPLES))
        old = _files(tmp_path)
        reads, writes = [], []
        read_csv, record = pd.read_csv, Ledger.record
        monkeypatch.setattr(pd, "read_csv", lambda path, *args, **kwargs: (
            reads.append(Path(path).parent.name), read_csv(path, *args, **kwargs))[1])
        monkeypatch.setattr(Ledger, "record", lambda self, *args, **kwargs: (
            writes.append(len(writes)), record(self, *args, **kwargs))[1])

        result, gen = _run(tmp_path, _entries(0, 1, 2, 3, _NO_SAMPLES), style="long")
        assert gen.calls == [] and result.total_entries_processed == 0
        assert reads == ["Privacy", "Violence"] and len(writes) == 1
        after = _files(tmp_path)
        assert {name: data for name, data in after.items()
                if name.endswith(".csv")} == {
            name: data for name, data in old.items() if name.endswith(".csv")}
        ledger = _constitution_inputs_ledger(tmp_path)
        assert after[ledger.path.name].startswith(old[ledger.path.name])
        assert ledger.completed() == {
            _key(i, "long") for i in (0, 1, 2, 3, _NO_SAMPLES)}
        on_disk = _ids_on_disk(tmp_path)
        assert ledger.ids == set(on_disk) and len(ledger.ids) == len(on_disk)

    def test_the_resume_after_reads_the_ledger_alone_and_knows_the_ids(
            self, tmp_path, monkeypatch):
        _old_folder(tmp_path, with_rows=(0, 1, 2), acked=(0,))
        _run(tmp_path, _entries(0, 1, 2), style="long")
        monkeypatch.setattr(pd, "read_csv", _no_read)

        def repeats(token, attempt):
            return "1. first sample zq2\n2. a new sample zq5"

        _run(tmp_path, _entries(0, 1, 2, 5), write=repeats, style="long")
        assert _texts(tmp_path)["Privacy"][-1] == "a new sample zq5"
        assert "first sample zq2" not in _texts(tmp_path)["Privacy"]

    def test_a_file_of_the_first_format_is_read_by_the_columns_it_has(self, tmp_path):
        """It names the description ``sample_description`` and has no style
        column: its rows are of the style ``""``."""
        old_columns = {"entry_type": "benign", "sample_description": "desc zq0"}
        append_samples(
            ["first sample zq0", "second sample zq0"], category="Violence", turn=0,
            source="constitution", dataset_dir=tmp_path,
            extra_columns=[old_columns] * 2)
        _result, gen = _run(tmp_path, _entries(0))
        assert gen.calls == []
        ledger = _constitution_inputs_ledger(tmp_path)
        assert ledger.completed() == {_key(0)}
        assert ledger.ids == set(_ids_on_disk(tmp_path))

    def test_a_fresh_run_of_its_style_starts_every_entry_over(self, tmp_path):
        """The acks of a ledger of that time are removed with the rows."""
        _old_folder(tmp_path, with_rows=(0, 1), acked=(0, 1, _NO_SAMPLES))
        result, gen = _run(tmp_path, _entries(0, 1, _NO_SAMPLES), write=_again,
                           style="long", fresh=True)
        assert (result.total_entries_processed, result.skipped_entries) == (2, 1)
        assert _texts(tmp_path) == {
            "Violence": ["first sample again zq0", "second sample again zq0"],
            "Privacy": ["first sample again zq1", "second sample again zq1"],
        }
