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
(the ``standalone tree`` build, and the build count of the eval run) the
build takes its inputs from the category files alone, with the ids they hold;
and the rows of ``variations.csv``, which the jailbreak stage writes beside its
jailbreaks, are a part of each build and are counted in each run.

The build now copies the accepted rows of every part. The rows of the
rejected output ``out-2`` and the jailbreak error row ``jb-3`` are gone from
each build that held them, and the build count of each run is its accepted
rows: 19 of the eval run's 20, 90 of the training run's 92.
"""

import json
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

from redact import paths, telemetry
from redact.dataset import append_samples
from tests.conftest import MockBackend, make_client
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
        inputs: ``"merged"`` writes the merged file of a training run,
            ``"tree"`` the category files of the standalone path, one of
            their rows rejected.
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
            (datasets / category).mkdir()
            part.to_csv(datasets / category / "samples.csv", index=False)
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
    "eval": dict(mode="eval"),
    "standalone tree": dict(mode="eval", inputs="tree"),
    "parts without rows": dict(mode="training", rows=False),
}


def _built(data_dir: Path, mode: str, **stage_files) -> dict:
    """The files one build writes: the complete dataset, and in eval mode the
    paraphrased one."""
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
    for name in ("complete_dataset.csv", "paraphrased.csv")[: 1 + (mode == "eval")]:
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
    for _ in range(2):      # the second build finds the first one's dataset
        build_dataset(data_dir=tmp_path, mode="eval", verbose=False)
        assert list(_input_rows(tmp_path)["sample_id"]) == ["in-1", "in-3"]


def test_a_build_keeps_every_column_and_every_row_of_the_category_files(tmp_path):
    """Also a text that two categories hold."""
    from redact import build_dataset

    for category, entry_type in (("Cyber", "harmful"), ("Fraud", "benign")):
        append_samples(["the same text"], category=category, turn=0,
                       extra_columns=[{"entry_type": entry_type}],
                       dataset_dir=paths.datasets(tmp_path))
    build_dataset(data_dir=tmp_path, verbose=False)
    inputs = _input_rows(tmp_path)
    assert list(inputs["category"]) == ["Cyber", "Fraud"]
    assert list(inputs["entry_type"]) == ["harmful", "benign"]


def test_a_build_takes_the_seeded_inputs_that_were_not_merged(tmp_path):
    from redact import build_dataset

    append_samples(["a seeded input"], category="Cyber", turn=0,
                   dataset_dir=paths.constitution_inputs_dir(tmp_path))
    append_samples(["a standalone input"], category="Fraud", turn=0,
                   dataset_dir=paths.datasets(tmp_path))
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


def _training_run(folder: Path, monkeypatch, before=None) -> dict:
    """A training run after an earlier call left inputs of another style."""
    import redact
    from redact import generate_inputs

    overlay = folder / "prompts"
    for style in ("long", "short"):
        template = overlay / "input" / "generation" / "from_constitution" / style
        template.mkdir(parents=True)
        (template / "template.json").write_text(json.dumps({
            "system_prompt": "sys",
            "template": f"MAKE-INPUTS-{style.upper()} about {{sample_description}}",
        }), encoding="utf-8")
    constitution = pd.DataFrame([
        {"source_category": category, "constitution_subcategory": "Sub",
         "sample_description": f"what zq{n} is about", "entry_type": entry_type}
        for n, (category, entry_type) in enumerate([
            ("Cyber", "harmful"), ("Cyber", "benign"), ("Fraud", "harmful"),
            ("Fraud", "dual_use_benign")])
    ])
    _stub_models(monkeypatch)

    def generate_constitution(**kwargs):
        # As the stage does: the entries are saved, and seed the inputs from there.
        merged = paths.constitution_merged_csv(kwargs["data_dir"])
        merged.parent.mkdir(parents=True, exist_ok=True)
        constitution.to_csv(merged, index=False)
        return pd.DataFrame()

    monkeypatch.setattr(redact, "generate_constitution", generate_constitution)

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
    """An eval run; the meta-prompt generation of its inputs stage is stubbed."""
    from redact.content_moderation import InputPipeline

    def run_standalone(self, categories, **kwargs):
        for n, (category, _info) in enumerate(categories):
            append_samples(
                [f"eval {nth} zq{n} sample" for nth in ("first", "second", "third")],
                category=category, turn=0, accepted=[True, True, n == 0],
                dataset_dir=self.dataset_dir)

    _stub_models(monkeypatch)
    monkeypatch.setattr(InputPipeline, "run_standalone", run_standalone)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)    # the standalone path
        counts = _counts({
            "dataset_type": "eval", "data_dir": str(folder),
            "taxonomy": {"categories": {"Cyber": "computers", "Fraud": "money"}},
            "stages": ["inputs", "outputs", "paraphrase", "jailbreaks", "build"],
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
