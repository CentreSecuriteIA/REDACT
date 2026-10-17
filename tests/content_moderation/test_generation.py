"""Tests for content moderation InputPipeline with mock backend, and the
response axes of the outputs stage."""

import json
import logging
import re
from pathlib import Path

import pandas as pd
import pytest

from redact import paths
from redact.content_moderation import axes as A
from redact.content_moderation.inputs import InputPipeline
from redact.dataset import Manifest, read_rows
from tests.conftest import MockBackend, make_client, write_inputs


def test_the_pipeline_writes_to_the_inputs_tree_by_default():
    client = make_client(MockBackend(), "m")
    pipeline = InputPipeline(gen=client, check=client)
    assert pipeline.dataset_dir == paths.constitution_inputs_dir()


class TestOutputsOfInputsWithoutASubcategory:
    def test_an_input_without_a_subcategory_gives_an_empty_one(
            self, tmp_path, monkeypatch):
        """Not the text ``nan``."""
        import pandas as pd

        import redact.stages as P
        from redact import generate_outputs

        monkeypatch.setattr(P.ModelClient, "create",
                            lambda m: make_client(MockBackend("ans"), m))
        write_inputs(paths.constitution_inputs_dir(tmp_path), ["an input"],
                     extra={"entry_type": "harmful", "subcategory": ""})
        generate_outputs(data_dir=tmp_path, check_outputs=False, verbose=False)
        written = pd.read_csv(paths.output_responses_csv(tmp_path), dtype=str,
                              keep_default_na=False)
        assert list(written["subcategory"]) == [""]


class TestGeneralBenignEntries:
    """An entry of the general-benign part is an input of its benign category."""

    _PROMPT = {"system_prompt": "sys", "template": "about {sample_description}"}

    def _run(self, tmp_path, monkeypatch, **entry):
        import pandas as pd

        import redact.content_moderation.inputs as G

        checked = []
        build = G.build_quality_checker
        monkeypatch.setattr(G, "build_quality_checker",
                            lambda **k: checked.append(k) or build(**k))
        pipeline = InputPipeline(
            gen=make_client(MockBackend("1. a question about bread\n[END]"), "m"),
            check=make_client(MockBackend("Yes"), "m-check"), dataset_dir=tmp_path)
        pipeline.run_from_constitution(
            pd.DataFrame([{"source_category": "Cooking", "sample_description": "desc",
                           "constitution_subcategory": "Bread",
                           "constitution_category": "Baking", "entry_type": "benign",
                           "source_group_tag": "general", **entry}]),
            self._PROMPT, samples_per_entry=1, verbose=False)
        rows = pd.read_csv(tmp_path / "Cooking" / "samples.csv", dtype=str,
                           keep_default_na=False)
        return checked, rows

    def test_it_is_checked_against_its_benign_category_and_keeps_its_taxonomy(
            self, tmp_path, monkeypatch):
        checked, rows = self._run(tmp_path, monkeypatch, source_taxonomy="benign")
        assert checked == [{"category": "Cooking", "entry_type": "benign",
                            "subcategory": "Bread", "prompt_dir": None}]
        assert rows[["category", "entry_type", "subcategory", "source_taxonomy",
                     "accepted"]].to_dict("records") == [
            {"category": "Cooking", "entry_type": "benign", "subcategory": "Bread",
             "source_taxonomy": "benign", "accepted": "True"}]

    def test_a_constitution_without_the_column_gives_inputs_without_it(
            self, tmp_path, monkeypatch):
        _, rows = self._run(tmp_path, monkeypatch)
        assert "source_taxonomy" not in rows.columns


_RESPONSE_COLUMNS = ["response_length", "response_format", "response_register"]
_QUESTIONS = 6


def _questions(n: int = _QUESTIONS) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"in-{i}" for i in range(n)],
        "sample": [f"zq{i} question" for i in range(n)],
        "category": ["Cyber", "Fraud"] * (n // 2),
        "entry_type": ["harmful"] * n,
        "subcategory": ["Sub"] * n,
        "source": ["constitution"] * n,
    })


def _respond(folder, monkeypatch, inputs=None, *, judge=lambda token: "Yes",
             **kwargs):
    """Run the outputs stage on ``inputs``; its generating and checking
    backends."""
    import redact.stages as P
    from redact import generate_outputs
    from tests.test_pipelines import _TokenBackend

    backends = {
        "gen-model": _TokenBackend(lambda t, text: f"answer {t}", "gen-model"),
        "check-model": _TokenBackend(lambda t, text: judge(t), "check-model")}
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
    generate_outputs(inputs=_questions() if inputs is None else inputs,
                     data_dir=folder, model="gen-model", check_model="check-model",
                     verbose=False, **kwargs)
    return backends["gen-model"], backends["check-model"]


def _planned(folder) -> dict:
    """Each input's planned spec, by its id."""
    manifest = Manifest.sidecar(paths.output_responses_csv(folder))
    return {row["input_id"]: row.get("spec") for row in manifest.load()}


def _written(folder) -> pd.DataFrame:
    return read_rows(paths.output_responses_csv(folder))


def _coverage(folder) -> Path:
    return paths.datasets(folder) / "output_responses.coverage.csv"


class TestTheResponseAxes:
    """One spec of the response axes per input row, shown in its prompt,
    recorded on its row and counted in the coverage table."""

    def test_each_input_row_is_dealt_one_spec_shown_and_recorded(
            self, tmp_path, monkeypatch):
        gen, _ = _respond(tmp_path, monkeypatch)
        planned = _planned(tmp_path)
        assert sorted(planned) == [f"in-{i}" for i in range(_QUESTIONS)]
        rows = _written(tmp_path)
        assert list(rows.columns[-3:]) == _RESPONSE_COLUMNS
        for row in rows.to_dict("records"):
            assert {c: row[c] for c in _RESPONSE_COLUMNS} == {
                f"response_{axis}": value
                for axis, value in planned[row["input_id"]].items()}
        axes = A.resolve("outputs")
        for call in gen.calls:
            text = call["messages"][-1]["content"]
            token = re.search(r"zq(\d+)", text).group(1)
            spec = A.render_spec(axes, planned[f"in-{token}"], {})
            assert spec and text.endswith("\n\n" + spec)

    def test_the_checkers_messages_do_not_change(self, tmp_path, monkeypatch):
        _, with_axes = _respond(tmp_path / "with", monkeypatch)
        _, without = _respond(tmp_path / "without", monkeypatch, axes=False)

        def sent(backend):
            return sorted(json.dumps(call["messages"]) for call in backend.calls)

        assert sent(with_axes) == sent(without)

    def test_without_axes_no_spec_is_dealt_shown_or_recorded(
            self, tmp_path, monkeypatch):
        gen, _ = _respond(tmp_path, monkeypatch, axes=False)
        assert all(spec is None for spec in _planned(tmp_path).values())
        assert not set(_RESPONSE_COLUMNS) & set(_written(tmp_path).columns)
        assert all(call["messages"][-1]["content"].endswith("\n\n")
                   for call in gen.calls)
        assert not _coverage(tmp_path).exists()

    def test_a_resume_deals_the_same_specs_and_makes_no_call(
            self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch)
        first = _planned(tmp_path)
        gen, check = _respond(tmp_path, monkeypatch)
        assert gen.calls == check.calls == []
        assert _planned(tmp_path) == first

    def test_a_resume_with_more_inputs_keeps_the_specs_of_those_done(
            self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch)
        first = _written(tmp_path).set_index("input_id")[_RESPONSE_COLUMNS]
        _respond(tmp_path, monkeypatch, inputs=_questions(2 * _QUESTIONS))
        now = _written(tmp_path).set_index("input_id")[_RESPONSE_COLUMNS]
        assert len(now) == 2 * _QUESTIONS
        assert now.loc[first.index].equals(first)

    @pytest.mark.parametrize(("changed", "named"), [
        ({"axes": {"format": {"code": 0}}}, "axes"), ({"seed": 7}, "seed")])
    def test_a_resume_with_another_deal_raises_naming_resume_false(
            self, tmp_path, monkeypatch, changed, named):
        _respond(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, **changed)
        assert f"['{named}']" in str(raised.value)
        assert "resume=False" in str(raised.value)
        gen, _ = _respond(tmp_path, monkeypatch, resume=False, **changed)
        assert len(gen.calls) == _QUESTIONS

    def test_the_coverage_table_counts_each_value_planned_and_accepted(
            self, tmp_path, monkeypatch):
        # The odd inputs' responses are rejected: those of Fraud.
        _respond(tmp_path, monkeypatch,
                 judge=lambda token: "No: weak" if int(token[2:]) % 2 else "Yes")
        table = pd.read_csv(_coverage(tmp_path))
        values = sum(len(entry["values"]) for entry in A.resolve("outputs").values())
        assert len(table) == 2 * values     # every value, in each of two groups
        per_group = table.groupby(["category", "axis"])[["planned", "accepted"]].sum()
        assert set(per_group["planned"]) == {_QUESTIONS // 2}
        assert per_group.loc["Cyber", "accepted"].tolist() == [_QUESTIONS // 2] * 3
        assert per_group.loc["Fraud", "accepted"].tolist() == [0] * 3

    def test_a_file_written_before_the_response_columns_stops_a_resume_first(
            self, tmp_path, monkeypatch):
        before = 2
        _respond(tmp_path, monkeypatch, axes=False, inputs=_questions(before))
        (paths.datasets(tmp_path) / "output_responses.plan.json").unlink()
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, inputs=_questions(2 * before))
        assert "response_length" in str(raised.value)
        assert "resume=False" in str(raised.value)
        assert len(_written(tmp_path)) == before

    def test_a_prompt_without_the_placeholder_is_warned_about(
            self, tmp_path, monkeypatch, caplog):
        overlay = tmp_path / "prompts" / "output" / "generation"
        overlay.mkdir(parents=True)
        (overlay / "template.json").write_text(json.dumps(
            {"system_prompt": "s", "template": "{input_prompt}"}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            _respond(tmp_path, monkeypatch, prompt_dir=tmp_path / "prompts")
        assert "has no {response_spec}" in caplog.text
