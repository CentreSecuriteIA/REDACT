"""The build's files and a run's counts, against what they were before the
build copied its parts in chunks.

``fixtures/build_files.json`` holds, as the code at commit e8109df gave them:
the files ``build_dataset`` writes from one set of stage files, the merged
inputs file of a training run, and the counts ``run_pipeline`` reports for a
training and an eval run. A CSV is compared by its cells as text.

One difference is meant: ``turn``, ``complexity``, ``num_techniques`` and
``iteration`` used to print as ``0.0`` where a part of the build lacks the
column, and print as integers now.

Two cases of the fixture are not that commit's: without a merged inputs file
(the ``inputs tree`` build) the build takes its inputs from the category files
alone, with the ids they hold; and the rows of ``variations.csv``, which the
jailbreak stage writes beside its jailbreaks, are a part of each build and are
counted in each run.

The build now copies the accepted rows of every part. The rows of the
rejected output ``out-2`` and the jailbreak error row ``jb-3`` are gone from
each build that held them, and the build count of each run is its accepted
rows.

The build leaves out a row whose ``sample_id`` an earlier part wrote. In the
runs the stub model answers a natural typos rewrite with the text it answers
that input's output request with, so those outputs are the variations' rows.

The plan pairs ASCII art with neither the emoji or unicode suffix, braille
nor a translation, whose characters no font draws, so every jailbreak of
the runs is accepted: the build count of the training run is 88.

The eval run is not that commit's: its inputs are seeded from a small
constitution of placeholder entries, as every run's are.

An eval build writes three files of one flat schema, ``eval_inputs.csv``,
``eval_outputs.csv`` and ``eval_jailbreaks.csv``, in place of the complete
dataset and ``paraphrased.csv``: the fixture's two eval builds (``eval`` and
``inputs tree``) hold those files, as the code that first wrote them gave
them, and the eval run's build counts their rows, which have no
``accepted`` column.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from redact import paths, telemetry
from redact.dataset import read_rows
from tests.conftest import MockBackend, make_client, write_inputs
from tests.test_stage_files import _files

_FIXTURE = Path(__file__).parent / "fixtures" / "build_files.json"

_INTEGER_COLUMNS = ("turn", "complexity", "num_techniques", "iteration")

_TEXTS = ["first input, with a comma", "second input\nof two lines",
          'third "quoted" input, é']

_INPUTS = pd.DataFrame({
    "sample_id": ["in-1", "in-2", "in-3"],
    "sample": _TEXTS,
    "category": ["Cyber", "Cyber", "Fraud"],
    "turn": [0, 1, 2],
    "accepted": [True, True, True],
    "source": ["constitution"] * 3,
    "subcategory": ["phishing", "", "cards"],
    "entry_type": ["harmful", "benign", "harmful"],
    "template_style": ["long", "long", "short"],
})

_JAILBREAKS = pd.DataFrame({
    "sample_id": ["jb-1", "jb-2", "jb-3", "jb-4"],
    "prompt": [*_TEXTS, _TEXTS[0]],
    "category": ["Cyber", "Cyber", "Fraud", "Cyber"],
    "turn": [0, 1, 2, 0],
    "accepted": [True, True, False, True],
    "source": ["constitution"] * 4,
    "subcategory": ["phishing", "", "cards", "phishing"],
    "entry_type": ["harmful", "benign", "harmful", "harmful"],
    "template_style": ["long", "long", "short", "long"],
    "input_id": ["in-1", "in-2", "in-3", "in-1"],
    "jailbreak": ["FIRST INPUT, WITH A COMMA", "tupni dnoces", _TEXTS[2],
                  "First input?\nAnswer in a list."],
    "technique": ["to_upper", "to_reversed", "to_broken", "to_upper+ask_for_a_list"],
    "technique_info": ["to_upper", "to_reversed", "ERROR: chain failed", "two"],
    "complexity": [1, 3, 0, 2],
    "num_techniques": [1, 1, 1, 2],
    "is_noop": [False, False, True, False],
    "reasoning": ["", "", "chain failed", ""],
    "iteration": [0, 0, 0, 1],
    "combination_spec_version": ["1.0"] * 4,
})

_VARIATIONS = pd.DataFrame({
    "sample_id": ["var-1", "var-2", "var-3"],
    "prompt": _TEXTS,
    "category": ["Cyber", "Cyber", "Fraud"],
    "turn": [0, 1, 2],
    "accepted": [True, True, True],
    "source": ["constitution"] * 3,
    "subcategory": ["phishing", "", "cards"],
    "entry_type": ["harmful", "benign", "harmful"],
    "template_style": ["long", "long", "short"],
    "input_id": ["in-1", "in-2", "in-3"],
    "jailbreak": ["first input, with a comma!!", "SECOND INPUT\nOF TWO LINES",
                  'Summarize: third "quoted" input, é'],
    "technique": ["to_doubled_punctuation", "to_uppercase", "to_summarize"],
    "technique_info": ["", "", ""],
    "complexity": [0, 0, 0],
    "num_techniques": [1, 1, 1],
    "is_noop": [False, False, False],
    "reasoning": ["", "", ""],
    "iteration": [0, 0, 0],
    "combination_spec_version": ["1.0"] * 3,
})

_OUTPUTS = pd.DataFrame({
    "input_id": ["in-1", "in-2", "in-3"],
    "input_prompt": _TEXTS,
    "category": ["Cyber", "Cyber", "Fraud"],
    "subcategory": ["phishing", "", "cards"],
    "entry_type": ["harmful", "benign", "harmful"],
    "model": ["gen-model"] * 3,
    "source": ["constitution"] * 3,
    "sample_id": ["out-1", "out-2", "out-3"],
    "output_response": ["an answer", "a refusal, politely", "an answer\nof two lines"],
    "accepted": [True, False, True],
    "rejection_reason": ["", "No: a refusal", ""],
})

_PARAPHRASED_INPUTS = pd.DataFrame({
    "sample_id": ["pi-1", "pi-2", "pi-3"],
    "input_id": ["in-1", "in-2", "in-3"],
    "iteration": [0, 0, 1],
    "sample": ["a first input, reworded", "a second input", "a third input"],
    "category": ["Cyber", "Cyber", "Fraud"],
    "entry_type": ["harmful", "benign", "harmful"],
    "paraphrase_model": ["para-a", "para-b", "para-a"],
    "accepted": [True, False, True],
    "reasoning": ["", "meaning changed", ""],
    "source": ["paraphrase_inputs"] * 3,
})

_PARAPHRASED_OUTPUTS = pd.DataFrame({
    "sample_id": ["po-1", "po-2"],
    "input_id": ["in-1", "in-3"],
    "iteration": [0, 0],
    "sample": ["an answer, reworded", "another answer"],
    "category": ["Cyber", "Fraud"],
    "entry_type": ["harmful", "harmful"],
    "paraphrase_model": ["para-a", "para-a"],
    "accepted": [False, True],
    "reasoning": ["meaning changed", ""],
    "source": ["paraphrase_outputs"] * 2,
})


def _write_stage_files(data_dir: Path, *, inputs: str = "merged", rows: bool = True):
    """Write the stage files a build reads under ``data_dir``.

    Args:
        inputs: ``"merged"`` writes the merged inputs file, ``"tree"`` only
            the category files of the inputs tree, one of their rows
            rejected.
        rows: False leaves the jailbreaks, the variations, the outputs and
            the paraphrased inputs with a header and no row.
    """
    datasets = paths.datasets(data_dir)
    datasets.mkdir(parents=True)
    if inputs == "merged":
        _INPUTS.to_csv(datasets / "constitution_inputs_merged.csv", index=False)
    else:
        tree = _INPUTS.drop(columns=["subcategory", "entry_type", "template_style"])
        tree = tree.assign(accepted=[True, False, True])
        for category, part in tree.groupby("category"):
            path = paths.category_csv(category,
                                      root=paths.constitution_inputs_dir(data_dir))
            path.parent.mkdir(parents=True)
            part.to_csv(path, index=False)
    keep = slice(None) if rows else slice(0)
    _JAILBREAKS.iloc[keep].to_csv(paths.jailbreaks_csv(data_dir), index=False)
    _VARIATIONS.iloc[keep].to_csv(paths.variations_csv(data_dir), index=False)
    _OUTPUTS.iloc[keep].to_csv(paths.output_responses_csv(data_dir), index=False)
    paraphrased_inputs = _PARAPHRASED_INPUTS.iloc[keep]
    if not rows:
        # A column of its own, which a file without a row does not add.
        paraphrased_inputs = paraphrased_inputs.assign(only_here="")
    paraphrased_inputs.to_csv(paths.paraphrases_inputs_csv(data_dir), index=False)
    _PARAPHRASED_OUTPUTS.to_csv(paths.paraphrases_outputs_csv(data_dir), index=False)


_BUILDS = {
    "training": dict(mode="training"),
    "parts without rows": dict(mode="training", rows=False),
}
_EVAL_BUILDS = {
    "eval": dict(mode="eval"),
    "inputs tree": dict(mode="eval", inputs="tree"),
}


def _built(data_dir: Path, mode: str, **stage_files) -> dict:
    """The files one build writes: the complete dataset, or the three eval
    files."""
    from redact import build_dataset

    _write_stage_files(data_dir, **stage_files)
    before = set(_files(paths.datasets(data_dir)))
    build_dataset(data_dir=data_dir, mode=mode, verbose=False)
    after = _files(paths.datasets(data_dir))
    return {name: rows for name, rows in after.items() if name not in before}


def _with_integers(rows: list[list[str]]) -> list[list[str]]:
    """A CSV's rows with ``1.0`` in an integer column written as ``1``."""
    header, *body = rows
    whole = [i for i, name in enumerate(header) if name in _INTEGER_COLUMNS]
    return [header] + [
        [str(int(float(cell))) if i in whole and cell else cell
         for i, cell in enumerate(row)]
        for row in body
    ]


@pytest.mark.parametrize("build", sorted(_BUILDS))
def test_a_build_writes_the_files_it_wrote_before_with_integers_as_integers(
        build, tmp_path):
    before = json.loads(_FIXTURE.read_text(encoding="utf-8"))["builds"][build]
    built = _built(tmp_path, **_BUILDS[build])
    assert built == {name: _with_integers(rows) for name, rows in before.items()}
    assert built != before      # the fixture does hold the old decimals


@pytest.mark.parametrize("build", sorted(_EVAL_BUILDS))
def test_an_eval_build_writes_the_files_it_wrote_before(build, tmp_path):
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))["builds"][build]
    assert _built(tmp_path, **_EVAL_BUILDS[build]) == expected


def _complete(data_dir: Path) -> list[list[str]]:
    """The cells of the complete dataset under ``data_dir``, as text."""
    return _files(paths.datasets(data_dir))["complete_dataset.csv"]


def test_a_build_reads_no_part_whole(tmp_path, monkeypatch):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    read_csv = pd.read_csv

    def in_pieces(*args, **kwargs):
        whole = kwargs.get("chunksize") is None and kwargs.get("nrows") is None
        assert not whole, f"read whole: {args[0]}"
        return read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", in_pieces)
    returned = build_dataset(data_dir=tmp_path, verbose=False, read_back=False)
    assert returned.empty and len(returned.columns) == 0
    accepted = sum(int(part["accepted"].sum()) for part in (
        _INPUTS, _JAILBREAKS, _VARIATIONS, _OUTPUTS, _PARAPHRASED_INPUTS,
        _PARAPHRASED_OUTPUTS))
    assert len(_complete(tmp_path)) == 1 + accepted     # the header, then the rows


def test_a_build_returns_the_rows_of_its_file_by_default(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    returned = build_dataset(data_dir=tmp_path, verbose=False)
    on_disk = pd.read_csv(paths.complete_dataset_csv(tmp_path))
    pd.testing.assert_frame_equal(returned, on_disk)
    assert set(returned["dataset_type"]) == {
        "content_moderation_input", "jailbreak", "natural_variation",
        "content_moderation_output", "content_moderation_paraphrase"}


@pytest.mark.parametrize("mode", ["training", "eval"])
def test_the_size_of_a_chunk_does_not_change_a_byte(tmp_path, monkeypatch, mode):
    import redact.dataset.rows as rows
    from redact import build_dataset

    for data_dir, chunk_rows in ((tmp_path / "one", None), (tmp_path / "many", 2)):
        if chunk_rows:
            monkeypatch.setattr(rows, "_CHUNK_ROWS", chunk_rows)
        _write_stage_files(data_dir)
        build_dataset(data_dir=data_dir, mode=mode, verbose=False)
    one, many = (paths.datasets(tmp_path / name) for name in ("one", "many"))
    names = ["complete_dataset.csv"] if mode == "training" else list(_EVAL_HEADERS)
    for name in names:
        assert (many / name).read_bytes() == (one / name).read_bytes()


def test_a_build_that_fails_keeps_the_dataset_it_would_replace(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    out = paths.complete_dataset_csv(tmp_path)
    build_dataset(data_dir=tmp_path, verbose=False)
    before = out.read_bytes()
    # The third part ends inside a cell: the build stops after copying two.
    with paths.output_responses_csv(tmp_path).open("ab") as fh:
        fh.write(b'in-9,"a cell that never ends')
    with pytest.raises(pd.errors.ParserError):
        build_dataset(data_dir=tmp_path, verbose=False)
    assert out.read_bytes() == before
    assert [p.name for p in out.parent.iterdir() if p.suffix == ".tmp"] == []


def test_a_refused_rename_of_the_dataset_is_tried_again(tmp_path, monkeypatch):
    """On Windows a sync client that holds the old file refuses the rename."""
    import redact.dataset.rows as rows
    from redact import build_dataset

    replace, refused = rows.os.replace, []

    def refuse_once(source, target):
        if not refused:
            refused.append(Path(target).name)
            raise PermissionError("held by another program")
        return replace(source, target)

    monkeypatch.setattr(rows.os, "replace", refuse_once)
    monkeypatch.setattr(rows.time, "sleep", lambda seconds: None)
    _write_stage_files(tmp_path)
    build_dataset(data_dir=tmp_path, verbose=False)
    assert refused == ["complete_dataset.csv"]
    assert len(_complete(tmp_path)) > 1


def test_a_cell_that_reads_as_a_missing_value_is_written_empty(tmp_path):
    """As when the parts were read whole. The stages write ``nan`` for a
    category, subcategory or entry type their input lacks."""
    from redact import build_dataset

    _write_stage_files(tmp_path)
    marked = _INPUTS.assign(subcategory=["nan", "N/A", "null"])
    marked.to_csv(paths.datasets(tmp_path) / "constitution_inputs_merged.csv",
                  index=False)
    build_dataset(data_dir=tmp_path, verbose=False)
    header, *rows = _complete(tmp_path)
    at = header.index("subcategory")
    assert [row[at] for row in rows[: len(_INPUTS)]] == ["", "", ""]


def test_an_empty_stage_file_is_a_part_without_rows(tmp_path):
    """What a save killed before its first byte leaves, outside the stage's resume."""
    from redact import build_dataset

    _write_stage_files(tmp_path)
    for stage_file in (paths.jailbreaks_csv, paths.output_responses_csv):
        stage_file(tmp_path).write_bytes(b"")
    built = build_dataset(data_dir=tmp_path, verbose=False)
    assert "jailbreak" not in set(built["dataset_type"])
    assert "content_moderation_output" not in set(built["dataset_type"])


def test_a_row_whose_id_an_earlier_part_wrote_is_left_out(tmp_path, caplog):
    """A variation with a jailbreak's text, and an adversarial no-op, whose
    text is its input's: each is the same row as one written before it."""
    import logging

    from redact import build_dataset

    _write_stage_files(tmp_path)
    noop = _JAILBREAKS.iloc[[0]].assign(sample_id="in-1", jailbreak=_TEXTS[0],
                                        is_noop=True)
    pd.concat([_JAILBREAKS, noop]).to_csv(paths.jailbreaks_csv(tmp_path), index=False)
    _VARIATIONS.assign(sample_id=["jb-1", "var-2", "var-3"]).to_csv(
        paths.variations_csv(tmp_path), index=False)
    with caplog.at_level(logging.INFO, logger="redact"):
        built = build_dataset(data_dir=tmp_path)
    assert built["sample_id"].is_unique
    by_id = dict(zip(built["sample_id"], built["dataset_type"]))
    assert (by_id["in-1"], by_id["jb-1"]) == ("content_moderation_input", "jailbreak")
    left_out = [r.getMessage() for r in caplog.records if "left out" in r.getMessage()]
    assert left_out == [
        "Jailbreaks: left out 1 rows whose sample_id an earlier part wrote.",
        "Variations: left out 1 rows whose sample_id an earlier part wrote."]


def test_the_variations_are_their_accepted_rows_from_the_path_given(tmp_path):
    from redact import build_dataset

    def variations() -> list[str]:
        built = pd.read_csv(paths.complete_dataset_csv(tmp_path))
        return list(built[built["dataset_type"] == "natural_variation"]["sample_id"])

    _write_stage_files(tmp_path)
    elsewhere = tmp_path / "elsewhere.csv"
    _VARIATIONS.assign(accepted=[True, False, True]).to_csv(elsewhere, index=False)
    build_dataset(data_dir=tmp_path, variations_path=elsewhere, verbose=False)
    assert variations() == ["var-1", "var-3"]
    build_dataset(data_dir=tmp_path, include_variations=False, verbose=False)
    assert variations() == []


@pytest.mark.parametrize("left", ["missing", "empty"])
def test_variations_left_out_are_logged_as_the_other_parts_are(
        tmp_path, caplog, left):
    import logging

    from redact import build_dataset

    _write_stage_files(tmp_path)
    variations = paths.variations_csv(tmp_path)
    if left == "missing":
        variations.unlink()
    else:
        _VARIATIONS.iloc[:0].to_csv(variations, index=False)
    with caplog.at_level(logging.INFO, logger="redact"):
        build_dataset(data_dir=tmp_path)
    logged = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("Variations")]
    assert logged == ([f"Variations: file not found ({variations})"]
                      if left == "missing" else ["Variations: none found"])


def test_every_part_adds_its_accepted_rows_only(tmp_path):
    """A rejected or error row keeps its input's ``entry_type``, so in the
    dataset it would carry the wrong label. It stays in its stage file."""
    from redact import build_dataset

    _write_stage_files(tmp_path)
    inputs = tmp_path / "inputs.csv"
    _INPUTS.assign(accepted=[True, False, True]).to_csv(inputs, index=False)
    build_dataset(data_dir=tmp_path, inputs_path=inputs, verbose=False)
    built = pd.read_csv(paths.complete_dataset_csv(tmp_path))
    assert built["accepted"].all()
    assert list(built["sample_id"]) == [
        "in-1", "in-3", "jb-1", "jb-2", "jb-4", "var-1", "var-2", "var-3",
        "out-1", "out-3", "pi-1", "pi-3", "po-2"]
    for stage_file, rejected in ((paths.jailbreaks_csv, "jb-3"),
                                 (paths.output_responses_csv, "out-2")):
        assert rejected in set(pd.read_csv(stage_file(tmp_path))["sample_id"])


def test_a_part_without_an_accepted_column_keeps_every_row(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    _OUTPUTS.drop(columns="accepted").to_csv(
        paths.output_responses_csv(tmp_path), index=False)
    build_dataset(data_dir=tmp_path, verbose=False)
    built = pd.read_csv(paths.complete_dataset_csv(tmp_path))
    outputs = built[built["dataset_type"] == "content_moderation_output"]
    assert list(outputs["sample_id"]) == ["out-1", "out-2", "out-3"]


def test_a_paraphrase_that_a_spreadsheet_marked_true_is_kept(tmp_path):
    """A spreadsheet saves ``True`` as ``TRUE``, which a whole read took too."""
    from redact import build_dataset

    _write_stage_files(tmp_path)
    path = paths.paraphrases_outputs_csv(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("True", "TRUE").replace("False", "FALSE"),
                    encoding="utf-8")
    build_dataset(data_dir=tmp_path, verbose=False)
    ids = [row[0] for row in _complete(tmp_path)]
    assert "po-2" in ids and "po-1" not in ids


# --- an eval build -------------------------------------------------------------

_EVAL_HEADERS = {
    "eval_inputs.csv": [
        "sample_id", "text", "label", "paraphrased", "input_id", "paraphrase_model",
        "entry_type", "category", "subcategory", "source_taxonomy", "dataset_type"],
    "eval_jailbreaks.csv": [
        "sample_id", "text", "label", "is_jailbreak", "technique", "complexity",
        "num_techniques", "iteration", "input_id", "entry_type", "category",
        "subcategory", "source_taxonomy", "dataset_type"],
    "eval_outputs.csv": [
        "sample_id", "text", "label", "paraphrased", "input_id", "model",
        "paraphrase_model", "entry_type", "category", "subcategory",
        "source_taxonomy", "dataset_type"],
}


def _eval_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    """The rows of the three eval files under ``data_dir``."""
    return {name: read_rows(paths.datasets(data_dir) / name) for name in _EVAL_HEADERS}


def test_an_eval_build_writes_three_files_of_one_schema_and_no_other(tmp_path):
    """Not the complete dataset, nor a separate file of paraphrases."""
    built = _built(tmp_path, mode="eval")
    assert {name: rows[0] for name, rows in built.items()} == _EVAL_HEADERS
    # The inputs have no source_taxonomy: the column is there, empty.
    for rows in _eval_files(tmp_path).values():
        assert set(rows["source_taxonomy"]) == {""}


def test_the_input_file_holds_the_inputs_then_their_paraphrases_marked(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
    rows = _eval_files(tmp_path)["eval_inputs.csv"]
    columns = ["sample_id", "text", "label", "paraphrased", "input_id",
               "paraphrase_model", "dataset_type"]
    assert rows[columns].values.tolist() == [
        ["in-1", _TEXTS[0], "harmful", False, "", "", "content_moderation_input"],
        ["in-2", _TEXTS[1], "benign", False, "", "", "content_moderation_input"],
        ["in-3", _TEXTS[2], "harmful", False, "", "", "content_moderation_input"],
        ["pi-1", "a first input, reworded", "harmful", True, "in-1", "para-a",
         "content_moderation_paraphrase"],
        ["pi-3", "a third input", "harmful", True, "in-3", "para-a",
         "content_moderation_paraphrase"],
    ]
    # A paraphrase's file has no subcategory: it is its input's.
    assert rows["subcategory"].tolist() == [
        "phishing", "", "cards", "phishing", "cards"]


def test_the_output_file_holds_the_responses_then_their_paraphrases_marked(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
    rows = _eval_files(tmp_path)["eval_outputs.csv"]
    columns = ["sample_id", "text", "label", "paraphrased", "input_id", "model",
               "paraphrase_model", "subcategory", "dataset_type"]
    assert rows[columns].values.tolist() == [
        ["out-1", "an answer", "harmful", False, "in-1", "gen-model", "", "phishing",
         "content_moderation_output"],
        ["out-3", "an answer\nof two lines", "harmful", False, "in-3", "gen-model", "",
         "cards", "content_moderation_output"],
        ["po-2", "another answer", "harmful", True, "in-3", "", "para-a", "cards",
         "content_moderation_paraphrase"],
    ]


def test_the_jailbreak_file_holds_the_attacks_then_the_bare_benign_inputs(tmp_path):
    """``label`` is the content's, ``is_jailbreak`` the wrapper's: an attack on
    a benign input is a benign jailbreak."""
    from redact import build_dataset

    _write_stage_files(tmp_path)
    build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
    header, *rows = _files(paths.datasets(tmp_path))["eval_jailbreaks.csv"]
    assert header == _EVAL_HEADERS["eval_jailbreaks.csv"]
    assert rows == [
        ["jb-1", "FIRST INPUT, WITH A COMMA", "harmful", "True", "to_upper", "1", "1",
         "0", "in-1", "harmful", "Cyber", "phishing", "", "jailbreak"],
        ["jb-2", "tupni dnoces", "benign", "True", "to_reversed", "3", "1", "0",
         "in-2", "benign", "Cyber", "", "", "jailbreak"],
        ["jb-4", "First input?\nAnswer in a list.", "harmful", "True",
         "to_upper+ask_for_a_list", "2", "2", "1", "in-1", "harmful", "Cyber",
         "phishing", "", "jailbreak"],
        ["in-2", _TEXTS[1], "benign", "False", "", "", "", "", "", "benign", "Cyber",
         "", "", "content_moderation_input"],
    ]


def test_an_attack_that_left_its_input_as_it_was_is_not_a_positive(tmp_path):
    """An accepted no-op has its input's text and id."""
    from redact import build_dataset

    _write_stage_files(tmp_path)
    noop = _JAILBREAKS.iloc[[0]].assign(sample_id="in-1", jailbreak=_TEXTS[0],
                                        is_noop=True)
    pd.concat([_JAILBREAKS, noop]).to_csv(paths.jailbreaks_csv(tmp_path), index=False)
    build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
    rows = _eval_files(tmp_path)["eval_jailbreaks.csv"]
    assert rows["sample_id"].tolist() == ["jb-1", "jb-2", "jb-4", "in-2"]


def test_every_eval_row_carries_the_source_taxonomy_of_its_input(tmp_path):
    """A derived row whose stage file lacks the column takes its input's."""
    from redact import build_dataset

    _write_stage_files(tmp_path)
    general = _INPUTS.assign(source_taxonomy=["harmful", "benign", "harmful"])
    general.to_csv(paths.constitution_inputs_merged_csv(tmp_path), index=False)
    build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
    files = _eval_files(tmp_path)
    assert files["eval_inputs.csv"]["source_taxonomy"].tolist() == [
        "harmful", "benign", "harmful", "harmful", "harmful"]
    assert files["eval_outputs.csv"]["source_taxonomy"].tolist() == ["harmful"] * 3
    assert files["eval_jailbreaks.csv"]["source_taxonomy"].tolist() == [
        "harmful", "benign", "harmful", "benign"]


def test_an_eval_build_returns_the_rows_of_each_file(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    built = build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
    assert list(built) == ["inputs", "outputs", "jailbreaks"]
    for name, rows in built.items():
        on_disk = pd.read_csv(paths.datasets(tmp_path) / f"eval_{name}.csv",
                              dtype={"sample_id": str, "input_id": str})
        pd.testing.assert_frame_equal(rows, on_disk)
    unread = build_dataset(data_dir=tmp_path, mode="eval", verbose=False,
                           read_back=False)
    assert list(unread) == ["inputs", "outputs", "jailbreaks"]
    assert all(rows.empty for rows in unread.values())


def test_an_eval_build_without_inputs_has_no_bare_rows(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    build_dataset(data_dir=tmp_path, mode="eval", include_inputs=False, verbose=False)
    files = _eval_files(tmp_path)
    assert files["eval_inputs.csv"]["sample_id"].tolist() == ["pi-1", "pi-3"]
    attacks = files["eval_jailbreaks.csv"]["sample_id"].tolist()
    assert attacks == ["jb-1", "jb-2", "jb-4"]


def test_an_eval_build_has_no_complete_dataset_to_move(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    with pytest.raises(ValueError, match="output_path"):
        build_dataset(data_dir=tmp_path, mode="eval", output_path=tmp_path / "x.csv")
    assert not any(paths.datasets(tmp_path).glob("eval_*.csv"))


def test_an_eval_build_without_data_writes_nothing(tmp_path):
    from redact import build_dataset

    assert build_dataset(data_dir=tmp_path, mode="eval", verbose=False) == {}
    assert not paths.datasets(tmp_path).exists()


@pytest.mark.parametrize("dataset_type", ["training", "eval"])
def test_the_build_stage_counts_the_files_of_its_dataset_type(tmp_path, dataset_type):
    """And reads no variations for an eval build."""
    from redact.stages import STAGES, RunContext

    ctx = RunContext(
        data_dir=tmp_path, dataset_type=dataset_type, models={}, prompt_dir=None,
        taxonomy="harmful", taxonomy_dir=None, params={}, augmentations={},
        resume=True, verbose=False)
    files = {"training": [paths.complete_dataset_csv(tmp_path)],
             "eval": [paths.eval_inputs_csv(tmp_path), paths.eval_outputs_csv(tmp_path),
                      paths.eval_jailbreaks_csv(tmp_path)]}[dataset_type]
    assert STAGES["build"].files(ctx) == (files, {})
    read = paths.variations_csv(tmp_path) in STAGES["build"].sources(ctx)
    assert read == (dataset_type == "training")

# --- an eval build that leaves out what a training run holds -------------------

def _training_folder(folder: Path, inputs: list[str], descriptions=()) -> Path:
    """A training run's folder: its inputs, one rejected, and its constitution."""
    rows = pd.DataFrame({"sample_id": [f"t-{n}" for n in range(len(inputs))],
                         "sample": inputs, "accepted": True})
    rejected = {"sample_id": "t-x", "sample": _TEXTS[2], "accepted": False}
    rows = pd.concat([rows, pd.DataFrame([rejected])])
    paths.datasets(folder).mkdir(parents=True)
    rows.to_csv(paths.constitution_inputs_merged_csv(folder), index=False)
    paths.constitution_dir(folder).mkdir(parents=True)
    pd.DataFrame({"sample_description": ["an entry", *descriptions]}).to_csv(
        paths.constitution_merged_csv(folder), index=False)
    return folder


def _described(data_dir: Path) -> None:
    """Give the eval inputs a constitution description each."""
    _INPUTS.assign(source_sample_description=["first", "second", "third"]).to_csv(
        paths.constitution_inputs_merged_csv(data_dir), index=False)


def test_an_eval_row_a_training_run_holds_is_left_out_with_its_descendants(
        tmp_path, caplog):
    """By id, or by text up to case, spacing and punctuation. A row whose
    input came from a constitution entry the training run has too is kept,
    and counted and warned about."""
    import logging

    from redact import build_dataset

    data_dir = tmp_path / "eval"
    _write_stage_files(data_dir)
    _described(data_dir)
    training = _training_folder(tmp_path / "training", [
        "SECOND input of two lines!!", "an unrelated input"], descriptions=["third"])
    exact = paths.datasets(training) / "elsewhere.csv"
    pd.DataFrame({"sample_id": ["po-2"], "sample": ["x"]}).to_csv(exact, index=False)

    with caplog.at_level(logging.WARNING, logger="redact"):
        build_dataset(data_dir=data_dir, mode="eval", exclude_from=[training],
                      verbose=False)
    files = _eval_files(data_dir)
    # in-2 by its text, with every row from it; po-2 by its id. in-3 and the
    # rows from it stay, flagged by its description.
    inputs = files["eval_inputs.csv"]["sample_id"].tolist()
    assert inputs == ["in-1", "in-3", "pi-1", "pi-3"]
    assert files["eval_outputs.csv"]["sample_id"].tolist() == ["out-1", "out-3"]
    assert files["eval_jailbreaks.csv"]["sample_id"].tolist() == ["jb-1", "jb-4"]
    assert json.loads(paths.eval_excluded_json(data_dir).read_text(
        encoding="utf-8"))["excluded"] == {
        "exact": 1, "normalised": 3, "descriptions": 3}
    (warned,) = [r.getMessage() for r in caplog.records]
    assert warned.startswith("3 rows") and "kept" in warned


def test_the_rows_of_a_training_dataset_file_are_left_out(tmp_path):
    """A file is read for its ids, texts and descriptions; its rejected rows
    are not training rows."""
    from redact import build_dataset

    _write_stage_files(tmp_path / "eval")
    complete = tmp_path / "complete_dataset.csv"
    pd.DataFrame({"sample_id": ["in-1", "x-2", "x-3"],
                  "jailbreak": ["", "an answer", "a third input"],
                  "accepted": [True, True, False]}).to_csv(complete, index=False)
    build_dataset(data_dir=tmp_path / "eval", mode="eval", exclude_from=[complete],
                  verbose=False)
    files = _eval_files(tmp_path / "eval")
    assert files["eval_inputs.csv"]["sample_id"].tolist() == ["in-2", "in-3", "pi-3"]
    assert files["eval_outputs.csv"]["sample_id"].tolist() == ["out-3", "po-2"]


def test_a_training_run_with_the_same_benign_pairs_is_warned_about(tmp_path, caplog):
    import logging

    from redact import build_dataset

    _write_stage_files(tmp_path / "eval")
    training = _training_folder(tmp_path / "training", ["an unrelated input"])
    for folder in (tmp_path / "eval", training):
        paths.benign_csv(folder).parent.mkdir(parents=True)
        paths.benign_csv(folder).write_text("question,answer\nq,a\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="redact"):
        build_dataset(data_dir=tmp_path / "eval", mode="eval",
                      exclude_from=[training], verbose=False)
    (warned,) = [r.getMessage() for r in caplog.records]
    assert "benign_samples.csv" in warned and "same" in warned


def test_a_missing_training_path_raises_before_any_file_is_written(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    with pytest.raises(FileNotFoundError, match="nowhere"):
        build_dataset(data_dir=tmp_path, mode="eval",
                      exclude_from=[tmp_path / "nowhere"], verbose=False)
    assert not any(paths.datasets(tmp_path).glob("eval_*.csv"))


def test_a_training_build_excludes_nothing(tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path)
    with pytest.raises(ValueError, match="exclude_from"):
        build_dataset(data_dir=tmp_path, mode="training", exclude_from=[tmp_path])


def test_an_eval_build_without_exclude_from_leaves_no_count_of_an_earlier_one(
        tmp_path):
    from redact import build_dataset

    _write_stage_files(tmp_path / "eval")
    training = _training_folder(tmp_path / "training", ["an unrelated input"])
    build_dataset(data_dir=tmp_path / "eval", mode="eval", exclude_from=[training],
                  verbose=False)
    assert paths.eval_excluded_json(tmp_path / "eval").exists()
    build_dataset(data_dir=tmp_path / "eval", mode="eval", verbose=False)
    assert not paths.eval_excluded_json(tmp_path / "eval").exists()


def test_a_run_records_what_its_eval_build_left_out(tmp_path, monkeypatch):
    """In the build's counts, with the training files among its sources."""
    from redact import run_pipeline
    from redact.driver import read_record

    _write_stage_files(tmp_path / "eval")
    training = _training_folder(tmp_path / "training", ["FIRST input, with a comma"])
    try:
        summary = run_pipeline(
            {"dataset_type": "eval", "data_dir": str(tmp_path / "eval"),
             "stages": ["build"]},
            params={"build": {"exclude_from": [str(training)]}}, verbose=False)
    finally:
        telemetry.uninstall()
    excluded = {"exact": 0, "normalised": 5, "descriptions": 0}
    assert summary["stages"]["build"]["counts"]["excluded"] == excluded
    record = read_record("build", tmp_path / "eval")
    assert record["counts"]["excluded"] == excluded
    assert {str(paths.constitution_inputs_merged_csv(training)),
            str(paths.constitution_merged_csv(training))} <= set(record["sources"])


# --- a build without a merged inputs file -------------------------------------

def _input_rows(data_dir: Path) -> pd.DataFrame:
    """The input rows of the complete dataset under ``data_dir``."""
    built = pd.read_csv(paths.complete_dataset_csv(data_dir))
    return built[built["dataset_type"] == "content_moderation_input"]


def test_a_build_takes_its_inputs_from_the_category_files_alone(tmp_path):
    """Not from the paraphrase files, nor from the dataset an earlier build left."""
    from redact import build_dataset

    _write_stage_files(tmp_path, inputs="tree")
    assert paths.paraphrases_inputs_csv(tmp_path).exists()
    for _ in range(2):      # the second build finds the first one's files
        build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
        inputs = read_rows(paths.eval_inputs_csv(tmp_path))
        bare = inputs[~inputs["paraphrased"]]
        assert list(bare["sample_id"]) == ["in-1", "in-3"]


def test_a_build_keeps_every_column_and_every_row_of_the_category_files(tmp_path):
    """Also a text that two categories hold."""
    from redact import build_dataset

    for category, entry_type in (("Cyber", "harmful"), ("Fraud", "benign")):
        write_inputs(paths.constitution_inputs_dir(tmp_path), ["the same text"],
                     category, extra={"entry_type": entry_type})
    build_dataset(data_dir=tmp_path, verbose=False)
    inputs = _input_rows(tmp_path)
    assert list(inputs["category"]) == ["Cyber", "Fraud"]
    assert list(inputs["entry_type"]) == ["harmful", "benign"]


def test_a_build_takes_no_category_folder_outside_the_inputs_tree(tmp_path):
    from redact import build_dataset

    write_inputs(paths.constitution_inputs_dir(tmp_path), ["a seeded input"])
    write_inputs(paths.datasets(tmp_path), ["a folder of Datasets"], "Fraud")
    build_dataset(data_dir=tmp_path, verbose=False)
    assert list(_input_rows(tmp_path)["sample"]) == ["a seeded input"]


# --- a run from the first stage to the build ----------------------------------

# Today's flat form, read as the adversarial kind; the natural kind is the default.
_PURE = {
    "pure_only": True, "include_hacking": False, "include_manipulation": False,
    "include_requests": False,
}
_MODELS = {"gen": "gen-model", "check": "check-model", "paraphraser": "para-model",
           "paraphrase_check": "check-model"}


def _stub_models(monkeypatch) -> None:
    """Stub every model call of a run; a reply depends only on its request.

    A sample reads ``<style> <ordinal> zqN sample``. The third sample of
    ``zq1`` is rejected as an input, the answers to a second sample of
    ``zq0`` are rejected as outputs, and the paraphrases of a first sample of
    ``zq2`` are rejected.
    """
    import redact.content_moderation.paraphrase as CP
    from redact.llms import ModelClient
    from tests.test_pipelines import _TokenBackend

    def write(token, text):
        made = re.search(r"MAKE-INPUTS-(\w+)", text)
        if made:
            style = made.group(1).lower()
            return "\n".join(f"{n}. {style} {ordinal} {token} sample" for n, ordinal
                             in enumerate(("first", "second", "third"), 1))
        return "answer to the " + re.search(r"\w+ \w+ zq\d+ sample", text).group()

    def judge(token, text):
        if "answer to the" in text:
            return "No: a weak answer" if "second zq0" in text else "Yes"
        return "No: off topic" if "third zq1" in text else "Yes"

    backends = {
        "gen-model": _TokenBackend(write, "gen-model"),
        "check-model": _TokenBackend(judge, "check-model"),
        "para-model": MockBackend(model="para-model"),
    }
    for name, backend in backends.items():
        make_client(backend, name)      # registers the name
    monkeypatch.setattr(ModelClient, "create", lambda m: make_client(backends[m], m))
    monkeypatch.setattr(
        CP, "paraphrase_batch",
        lambda client, texts, **kw: [f"reworded: {t}" for t in texts])
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda client, texts, checker, **kw: [
            (False, "meaning changed") if "first zq2" in t else (True, "")
            for t in texts])
    monkeypatch.setattr("redact.llms.resources.estimate._load_config", lambda _id: None)


def _counts(recipe: dict, before=None) -> dict:
    """Run the recipe, after calling ``before``; the counts it reports, per stage."""
    from redact import run_pipeline

    if before:
        before()
    try:
        summary = run_pipeline(recipe, params={}, verbose=False)
    finally:
        telemetry.uninstall()
    return {stage: done["counts"] for stage, done in summary["stages"].items()}


def _overlay(folder: Path) -> Path:
    """A prompt overlay whose generation templates the stub model knows."""
    overlay = folder / "prompts"
    for style in ("long", "short"):
        template = overlay / "input" / "generation" / "from_constitution" / style
        template.mkdir(parents=True)
        (template / "template.json").write_text(json.dumps({
            "system_prompt": "sys",
            "template": f"MAKE-INPUTS-{style.upper()} about {{sample_description}}",
        }), encoding="utf-8")
    return overlay


def _constitution_stage(monkeypatch, constitution: pd.DataFrame) -> None:
    """Stub the constitution stage: it saves ``constitution``, which seeds
    the inputs from there."""
    import redact

    def generate_constitution(**kwargs):
        merged = paths.constitution_merged_csv(kwargs["data_dir"])
        merged.parent.mkdir(parents=True, exist_ok=True)
        constitution.to_csv(merged, index=False)
        return pd.DataFrame()

    monkeypatch.setattr(redact, "generate_constitution", generate_constitution)


def _training_run(folder: Path, monkeypatch, before=None) -> dict:
    """A training run after an earlier call left inputs of another style."""
    from redact import generate_inputs

    overlay = _overlay(folder)
    constitution = pd.DataFrame([
        {"source_category": category, "constitution_subcategory": "Sub",
         "sample_description": f"what zq{n} is about", "entry_type": entry_type}
        for n, (category, entry_type) in enumerate([
            ("Cyber", "harmful"), ("Cyber", "benign"), ("Fraud", "harmful"),
            ("Fraud", "dual_use_benign")])
    ])
    _stub_models(monkeypatch)
    _constitution_stage(monkeypatch, constitution)

    generate_inputs(
        data_dir=folder, constitution_df=constitution.iloc[:2], style="short",
        model="gen-model", check_model="check-model", prompt_dir=overlay, verbose=False)
    counts = _counts({
        "dataset_type": "training", "data_dir": str(folder), "prompt_dir": str(overlay),
        "stages": ["constitution", "inputs", "outputs", "paraphrase", "jailbreaks",
                   "build"],
        "models": _MODELS, "augmentations": _PURE,
    }, before)
    merged = "constitution_inputs_merged.csv"
    return {"counts": counts, merged: _files(paths.datasets(folder))[merged]}


def _eval_run(folder: Path, monkeypatch, before=None) -> dict:
    """An eval run from a small constitution of placeholder entries."""
    overlay = _overlay(folder)
    _stub_models(monkeypatch)
    _constitution_stage(monkeypatch, pd.DataFrame([
        {"source_category": category, "constitution_subcategory": "Subcategory",
         "sample_description": f"placeholder sample description zq{n}",
         "entry_type": entry_type}
        for n, (category, entry_type) in enumerate([
            ("Category A", "harmful"), ("Category B", "benign")])
    ]))
    counts = _counts({
        "dataset_type": "eval", "data_dir": str(folder), "prompt_dir": str(overlay),
        "stages": ["constitution", "inputs", "outputs", "paraphrase", "jailbreaks",
                   "build"],
        "models": _MODELS, "augmentations": _PURE,
    }, before)
    return {"counts": counts}


_RUNS = {"training": _training_run, "eval": _eval_run}


@pytest.mark.parametrize("run", sorted(_RUNS))
def test_a_run_reports_the_counts_it_reported_before(run, tmp_path, monkeypatch):
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))["runs"][run]
    assert _RUNS[run](tmp_path, monkeypatch) == expected


@pytest.mark.parametrize("run", sorted(_RUNS))
def test_a_run_counts_its_rows_without_a_stage_reading_them_back(
        run, tmp_path, monkeypatch):
    import redact.content_moderation.outputs as G
    import redact.jailbreak.pipeline as JPipe
    import redact.stages as P

    def read_back(*args, **kwargs):
        raise AssertionError("a stage read its rows back")

    # What the stages return their rows with: the inputs in stages.py, the
    # jailbreaks in jailbreak/pipeline.py, the outputs in outputs.py.
    def from_the_run_on():
        monkeypatch.setattr(P, "merge_all", read_back)
        monkeypatch.setattr(JPipe, "read_rows", read_back)
        monkeypatch.setattr(G, "read_rows", read_back)

    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))["runs"][run]
    counted = _RUNS[run](tmp_path, monkeypatch, before=from_the_run_on)
    assert counted["counts"] == expected["counts"]


def test_inputs_that_are_all_rejected_leave_no_merged_file(tmp_path, monkeypatch):
    from redact import generate_inputs
    from redact.llms import ModelClient

    backends = {"gen-model": MockBackend("1. one zq0 sample\n2. another zq0 sample"),
                "check-model": MockBackend("No: off topic")}
    monkeypatch.setattr(ModelClient, "create", lambda m: make_client(backends[m], m))
    constitution = pd.DataFrame([{
        "source_category": "Cyber", "constitution_subcategory": "Sub",
        "sample_description": "what zq0 is about", "entry_type": "harmful"}])
    returned = generate_inputs(
        data_dir=tmp_path, constitution_df=constitution, samples_per_entry=2,
        model="gen-model", check_model="check-model", verbose=False)
    tree = paths.constitution_inputs_dir(tmp_path)
    assert len(pd.read_csv(tree / "Cyber" / "samples.csv")) > 0     # all rejected
    assert returned.empty
    assert not (paths.datasets(tmp_path) / "constitution_inputs_merged.csv").exists()


def test_a_fresh_run_that_accepts_nothing_removes_the_last_merged_file(
        tmp_path, monkeypatch):
    """So the build does not take inputs that no category file holds now."""
    from redact import build_dataset, generate_inputs
    from redact.llms import ModelClient

    backends = {"gen-model": MockBackend("1. one zq0 sample\n2. another zq0 sample"),
                "check-model": MockBackend("Yes")}
    monkeypatch.setattr(ModelClient, "create", lambda m: make_client(backends[m], m))
    constitution = pd.DataFrame([{
        "source_category": "Cyber", "constitution_subcategory": "Sub",
        "sample_description": "what zq0 is about", "entry_type": "harmful"}])
    run = dict(data_dir=tmp_path, constitution_df=constitution, samples_per_entry=2,
               model="gen-model", check_model="check-model", verbose=False)
    generate_inputs(**run)
    merged = paths.constitution_inputs_merged_csv(tmp_path)
    assert list(pd.read_csv(merged)["sample"]) == [
        "one zq0 sample", "another zq0 sample"]

    backends["gen-model"] = MockBackend("1. third zq0 sample\n2. fourth zq0 sample")
    backends["check-model"] = MockBackend("No: off topic")
    generate_inputs(**run, resume=False)
    assert not merged.exists()
    built = build_dataset(data_dir=tmp_path, verbose=False)
    assert built.empty


def test_inputs_not_read_back_are_merged_on_disk_and_into_no_frame(
        tmp_path, monkeypatch):
    import redact.stages as P
    from redact import generate_inputs
    from redact.llms import ModelClient

    def merged_frame(*args, **kwargs):
        raise AssertionError("the inputs were merged into a frame")

    backends = {"gen-model": MockBackend("1. one zq0 sample\n2. another zq0 sample"),
                "check-model": MockBackend("Yes")}
    monkeypatch.setattr(ModelClient, "create", lambda m: make_client(backends[m], m))
    monkeypatch.setattr(P, "merge_all", merged_frame)
    constitution = pd.DataFrame([{
        "source_category": "Cyber", "constitution_subcategory": "Sub",
        "sample_description": "what zq0 is about", "entry_type": "harmful"}])
    returned = generate_inputs(
        data_dir=tmp_path, constitution_df=constitution, samples_per_entry=2,
        model="gen-model", check_model="check-model", verbose=False, read_back=False)
    assert returned.empty and len(returned.columns) == 0
    merged = pd.read_csv(paths.datasets(tmp_path) / "constitution_inputs_merged.csv")
    assert list(merged["sample"]) == ["one zq0 sample", "another zq0 sample"]
