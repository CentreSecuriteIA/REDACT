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


def test_the_inputs_check_runs_at_the_check_temperature_and_generation_at_the_default(
        tmp_path):
    gen = MockBackend("1. a question about bread\n[END]")
    check = MockBackend("Yes")
    pipeline = InputPipeline(gen=make_client(gen, "m"),
                             check=make_client(check, "m-check"), dataset_dir=tmp_path)
    pipeline.run_from_constitution(
        pd.DataFrame([{"source_category": "Cooking", "sample_description": "desc",
                       "constitution_subcategory": "Bread",
                       "constitution_category": "Baking", "entry_type": "benign",
                       "source_group_tag": "general"}]),
        {"system_prompt": "sys", "template": "about {sample_description}"},
        samples_per_entry=2, verbose=False)
    assert check.calls
    assert {c.get("temperature") for c in check.calls} == {0.15}
    assert [c.get("temperature") for c in gen.calls] == [None, None]


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
    """Each unit's planned spec, by its ``(input_id, iteration)``."""
    manifest = Manifest.sidecar(paths.output_responses_csv(folder))
    return {(row["input_id"], row["iteration"]): row.get("spec")
            for row in manifest.load()}


def _written(folder) -> pd.DataFrame:
    return read_rows(paths.output_responses_csv(folder))


def _coverage(folder) -> Path:
    return paths.datasets(folder) / "output_responses.coverage.csv"


@pytest.mark.usefixtures("no_length_floor")
class TestTheResponseAxes:
    """One spec of the response axes per input row, shown in its prompt,
    recorded on its row and counted in the coverage table."""

    def test_each_input_row_is_dealt_one_spec_shown_and_recorded(
            self, tmp_path, monkeypatch):
        gen, _ = _respond(tmp_path, monkeypatch)
        planned = _planned(tmp_path)
        assert sorted(planned) == [(f"in-{i}", 0) for i in range(_QUESTIONS)]
        rows = _written(tmp_path)
        assert list(rows.columns[-3:]) == _RESPONSE_COLUMNS
        for row in rows.to_dict("records"):
            assert {c: row[c] for c in _RESPONSE_COLUMNS} == {
                f"response_{axis}": value
                for axis, value in planned[(row["input_id"], 0)].items()}
        axes = A.resolve("outputs")
        for call in gen.calls:
            text = call["messages"][-1]["content"]
            token = re.search(r"zq(\d+)", text).group(1)
            spec = A.render_spec(axes, planned[(f"in-{token}", 0)], {})
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
        # The answer modes are still counted, and nothing else.
        table = pd.read_csv(_coverage(tmp_path))
        assert set(table["axis"]) == {"mode"}
        assert table.groupby("category")["planned"].sum().to_dict() == {
            "Cyber": _QUESTIONS // 2, "Fraud": _QUESTIONS // 2}

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
        # Every value in each of two groups, then the one mode each group dealt.
        assert len(table) == 2 * values + 2
        assert table.tail(2)[["axis", "value"]].values.tolist() == [
            ["mode", "harmful"]] * 2
        per_group = table.groupby(["category", "axis"])[["planned", "accepted"]].sum()
        assert set(per_group["planned"]) == {_QUESTIONS // 2}
        assert per_group.loc["Cyber", "accepted"].tolist() == [_QUESTIONS // 2] * 4
        assert per_group.loc["Fraud", "accepted"].tolist() == [0] * 4

    def test_a_file_written_before_the_response_columns_stops_a_resume_first(
            self, tmp_path, monkeypatch):
        before = 2
        _respond(tmp_path, monkeypatch, axes=False, inputs=_questions(before))
        (paths.datasets(tmp_path) / "output_responses.plan.json").unlink()
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, inputs=_questions(2 * before))
        assert "response_length" in str(raised.value)
        assert "resume=False" in str(raised.value)
        assert "outputs.axes: false" in str(raised.value)
        assert len(_written(tmp_path)) == before

    def test_a_file_written_before_the_answer_modes_stops_a_resume_first(
            self, tmp_path, monkeypatch):
        """Before the plan file, whose keys no setting of this run matches."""
        before = 2
        _respond(tmp_path, monkeypatch, inputs=_questions(before))
        out = paths.output_responses_csv(tmp_path)
        # A folder as the stage left it before the modes: no mode column, a
        # plan file without modes, and no ledger, which a resume takes over.
        old = _written(tmp_path).drop(columns=["iteration", "response_mode",
                                               "output_label", "response_label"])
        old.to_csv(out, index=False)
        for sidecar in ("state.jsonl", "manifest.jsonl"):
            (paths.datasets(tmp_path) / f"output_responses.{sidecar}").unlink()
        plan = paths.datasets(tmp_path) / "output_responses.plan.json"
        record = json.loads(plan.read_text(encoding="utf-8"))
        del record["modes"]
        plan.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, inputs=_questions(2 * before))
        message = str(raised.value)
        assert "'iteration', 'response_mode'" in message
        assert "each in an answer mode" in message
        assert "resume=False" in message
        assert "outputs.axes" not in message and "['modes']" not in message
        assert len(_written(tmp_path)) == before

    def test_a_prompt_without_the_placeholder_is_warned_about(
            self, tmp_path, monkeypatch, caplog):
        overlay = tmp_path / "prompts" / "output" / "generation" / "harmful"
        overlay.mkdir(parents=True)
        (overlay / "template.json").write_text(json.dumps(
            {"system_prompt": "s", "template": "{input_prompt}"}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            _respond(tmp_path, monkeypatch, prompt_dir=tmp_path / "prompts")
        assert "output/generation/harmful has no {response_spec}" in caplog.text


def _overlay(folder: Path) -> Path:
    """A prompt overlay whose output prompts name their mode in the user turn."""
    root = folder / "prompts" / "output"
    for mode in ("harmful", "helpful"):
        for step, template in (
                ("generation",
                 f"GEN-{mode.upper()} {{input_prompt}}\n\n{{response_spec}}"),
                ("quality_check",
                 f"CHECK-{mode.upper()} {{Category}}: {{input_prompt}} "
                 "=> {output_response}")):
            path = root / step / mode
            path.mkdir(parents=True)
            (path / "template.json").write_text(json.dumps(
                {"system_prompt": "s", "template": template}), encoding="utf-8")
    return folder / "prompts"


def _questions_of(entry_types: list, category: str = "Cyber") -> pd.DataFrame:
    n = len(entry_types)
    return pd.DataFrame({
        "sample_id": [f"in-{i}" for i in range(n)],
        "sample": [f"zq{i} question" for i in range(n)],
        "category": [category] * n,
        "entry_type": entry_types,
        "subcategory": ["Sub"] * n,
        "source": ["constitution"] * n,
    })


def _marker(call: dict) -> str:
    """The GEN- or CHECK- marker of a recorded call's user turn."""
    return re.search(r"(GEN|CHECK)-(HARMFUL|HELPFUL)",
                     call["messages"][-1]["content"]).group()


def _token(call: dict) -> str:
    return re.search(r"zq\d+", call["messages"][-1]["content"]).group()


@pytest.mark.usefixtures("no_length_floor")
class TestTheAnswerModes:
    """Each response is dealt a mode its input's entry type allows; the mode
    picks the prompt and the checker and gives the row its label."""

    _FOUR = ["harmful", "harmful", "dual_use_harmful", "dual_use_harmful",
             "dual_use_benign", "dual_use_benign", "benign", "benign"]

    def test_the_mode_picks_the_generation_prompt_and_the_checker(
            self, tmp_path, monkeypatch):
        inputs = _questions_of(self._FOUR)
        gen, check = _respond(tmp_path, monkeypatch, inputs=inputs,
                              prompt_dir=_overlay(tmp_path))
        rows = _written(tmp_path).set_index("input_prompt")
        mode_of = {f"zq{i}": rows.loc[f"zq{i} question", "response_mode"]
                   for i in range(len(self._FOUR))}
        assert {_token(c): _marker(c) for c in gen.calls} == {
            token: f"GEN-{mode.upper()}" for token, mode in mode_of.items()}
        assert {_token(c): _marker(c) for c in check.calls} == {
            token: f"CHECK-{mode.upper()}" for token, mode in mode_of.items()}
        # The checker is shown the category, never the entry type.
        for call in check.calls:
            text = call["messages"][-1]["content"]
            assert "Cyber:" in text and "dual_use" not in text
        assert (mode_of["zq0"], mode_of["zq6"]) == ("harmful", "helpful")

    def test_the_rows_carry_the_mode_its_label_and_the_compound_label(
            self, tmp_path, monkeypatch):
        from redact.content_moderation.modes import DEFAULT_MODES, OUTPUT_LABEL

        _respond(tmp_path, monkeypatch, inputs=_questions_of(self._FOUR))
        rows = _written(tmp_path)
        at = list(rows.columns).index("rejection_reason") + 1
        assert list(rows.columns[at:at + 4]) == [
            "iteration", "response_mode", "output_label", "response_label"]
        assert set(rows["iteration"]) == {0}
        assert sorted(rows["entry_type"]) == sorted(self._FOUR)
        for row in rows.to_dict("records"):
            assert row["response_mode"] in DEFAULT_MODES[row["entry_type"]]
            assert row["output_label"] == OUTPUT_LABEL[row["response_mode"]]
            assert row["response_label"] == (
                f"{row['entry_type']}_{row['output_label']}")
        assert sorted(rows["response_label"]) == sorted([
            "harmful_harmful", "harmful_harmful",
            "dual_use_harmful_harmful", "dual_use_harmful_benign",
            "dual_use_benign_harmful", "dual_use_benign_benign",
            "benign_benign", "benign_benign"])

    def test_a_dual_use_group_of_six_gives_three_of_each_mode(
            self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch,
                 inputs=_questions_of(["dual_use_benign"] * 6))
        assert _written(tmp_path)["response_mode"].value_counts().to_dict() == {
            "harmful": 3, "helpful": 3}
        planned = Manifest.sidecar(paths.output_responses_csv(tmp_path)).load()
        assert sorted(row["mode"] for row in planned) == (
            ["harmful"] * 3 + ["helpful"] * 3)

    def test_a_custom_mapping_gives_its_modes(self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch, inputs=_questions_of(["benign"] * 2),
                 modes={"benign": ["harmful"]})
        rows = _written(tmp_path)
        assert set(rows["response_label"]) == {"benign_harmful"}
        assert set(rows["output_label"]) == {"harmful"}

    def test_a_resume_with_another_mapping_raises_naming_resume_false(
            self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, modes={"harmful": ["helpful"]})
        assert "['modes']" in str(raised.value)
        assert "resume=False" in str(raised.value)
        gen, _ = _respond(tmp_path, monkeypatch, resume=False,
                          modes={"harmful": ["helpful"]})
        assert len(gen.calls) == _QUESTIONS
        assert set(_written(tmp_path)["response_label"]) == {"harmful_benign"}

    def test_the_plan_file_records_the_whole_mapping(self, tmp_path, monkeypatch):
        from redact.content_moderation.modes import resolve_modes

        _respond(tmp_path, monkeypatch)
        plan = json.loads((paths.datasets(tmp_path) / "output_responses.plan.json")
                          .read_text(encoding="utf-8"))
        assert list(plan) == ["seed", "outputs_per_input", "modes", "axes"]
        assert plan["outputs_per_input"] == 1
        assert plan["modes"] == resolve_modes()

    def test_an_unknown_entry_type_raises_before_anything_is_cleared(
            self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch)
        out = paths.output_responses_csv(tmp_path)
        files = {p.name: p.read_bytes() for p in out.parent.iterdir()
                 if p.suffix != ".lock"}
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, resume=False,
                     inputs=_questions_of(["harmful", "general"]))
        assert "outputs.modes" in str(raised.value)
        assert "'general'" in str(raised.value)
        assert {p.name: p.read_bytes() for p in out.parent.iterdir()
                if p.suffix != ".lock"} == files

    @pytest.mark.parametrize("cell", [float("nan"), None, "", "  "])
    def test_an_input_without_an_entry_type_raises_before_anything_is_cleared(
            self, tmp_path, monkeypatch, cell):
        """A mode and a label are never guessed for it."""
        _respond(tmp_path, monkeypatch)
        out = paths.output_responses_csv(tmp_path)
        files = {p.name: p.read_bytes() for p in out.parent.iterdir()
                 if p.suffix != ".lock"}
        inputs = _questions_of(["benign"] * 3).astype({"entry_type": object})
        inputs.loc[1, "entry_type"] = cell
        inputs.loc[2, "entry_type"] = cell
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, inputs=inputs, resume=False)
        message = str(raised.value)
        assert "2 input(s) have no entry type" in message
        assert "['in-1', 'in-2']" in message
        assert {p.name: p.read_bytes() for p in out.parent.iterdir()
                if p.suffix != ".lock"} == files

    def test_a_frame_without_an_entry_type_column_is_harmful_with_a_warning(
            self, tmp_path, monkeypatch, caplog):
        inputs = _questions_of(["benign"] * 2).drop(columns="entry_type")
        with caplog.at_level(logging.WARNING):
            _respond(tmp_path, monkeypatch, inputs=inputs)
        rows = _written(tmp_path)
        assert set(rows["entry_type"]) == {"harmful"}
        assert set(rows["response_label"]) == {"harmful_harmful"}
        assert "no entry_type column" in caplog.text

    def test_an_entry_type_member_is_read_as_its_value(self, tmp_path, monkeypatch):
        from redact.types import EntryType

        inputs = _questions_of([EntryType.BENIGN, EntryType.HARMFUL])
        _respond(tmp_path, monkeypatch, inputs=inputs)
        rows = _written(tmp_path).set_index("input_id")
        assert rows.loc[["in-0", "in-1"], "response_label"].tolist() == [
            "benign_benign", "harmful_harmful"]

    # --- k responses per input ---------------------------------------------

    def test_each_input_is_k_units_whose_responses_differ_on_every_axis(
            self, tmp_path, monkeypatch):
        gen, _ = _respond(tmp_path, monkeypatch, outputs_per_input=2)
        assert len(gen.calls) == 2 * _QUESTIONS
        planned = _planned(tmp_path)
        assert sorted(planned) == [(f"in-{i}", it) for i in range(_QUESTIONS)
                                   for it in (0, 1)]
        rows = _written(tmp_path)
        assert sorted(zip(rows["input_id"], rows["iteration"])) == sorted(planned)
        for i in range(_QUESTIONS):
            first, second = planned[(f"in-{i}", 0)], planned[(f"in-{i}", 1)]
            assert all(first[axis] != second[axis] for axis in first)

    def test_at_two_per_input_a_dual_use_input_gets_one_answer_of_each_mode(
            self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch, outputs_per_input=2,
                 inputs=_questions_of(["dual_use_harmful"] * 3 + ["benign"] * 2))
        rows = _written(tmp_path)
        modes = rows.groupby("input_id")["response_mode"].apply(sorted).to_dict()
        assert modes == {"in-0": ["harmful", "helpful"], "in-1": ["harmful", "helpful"],
                         "in-2": ["harmful", "helpful"], "in-3": ["helpful", "helpful"],
                         "in-4": ["helpful", "helpful"]}
        assert set(rows[rows["entry_type"] == "dual_use_harmful"]["response_label"]
                   ) == {"dual_use_harmful_harmful", "dual_use_harmful_benign"}

    def test_a_resume_makes_no_call_and_keeps_the_modes_and_specs(
            self, tmp_path, monkeypatch):
        inputs = _questions_of(["dual_use_benign"] * 4)
        _respond(tmp_path, monkeypatch, inputs=inputs, outputs_per_input=2)
        manifest = Manifest.sidecar(paths.output_responses_csv(tmp_path))
        first = manifest.load()
        gen, check = _respond(tmp_path, monkeypatch, inputs=inputs,
                              outputs_per_input=2)
        assert gen.calls == check.calls == []
        assert manifest.load() == first

    def test_a_resume_with_another_count_per_input_raises_naming_resume_false(
            self, tmp_path, monkeypatch):
        _respond(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as raised:
            _respond(tmp_path, monkeypatch, outputs_per_input=2)
        assert "['outputs_per_input']" in str(raised.value)
        assert "resume=False" in str(raised.value)
        gen, _ = _respond(tmp_path, monkeypatch, resume=False, outputs_per_input=2)
        assert len(gen.calls) == 2 * _QUESTIONS

    @pytest.mark.parametrize("value", [0, 1.5, True, "2"])
    def test_a_count_per_input_that_is_not_a_whole_number_raises(
            self, tmp_path, monkeypatch, value):
        with pytest.raises(ValueError, match="outputs_per_input"):
            _respond(tmp_path, monkeypatch, outputs_per_input=value)
        assert not paths.datasets(tmp_path).exists()

    @pytest.mark.parametrize("with_csv", [True, False])
    def test_a_ledger_of_units_without_an_iteration_stops_before_any_call(
            self, tmp_path, monkeypatch, with_csv):
        """Read through a ledger keyed on input_id: the stage's own skips an
        ack without its key, and would run every unit again onto the rows."""
        _respond(tmp_path, monkeypatch, inputs=_questions(2))
        out = paths.output_responses_csv(tmp_path)
        state = out.with_name("output_responses.state.jsonl")
        state.write_text("".join(
            json.dumps({"input_id": f"in-{i}", "accepted": True}) + "\n"
            for i in range(2)), encoding="utf-8")
        if not with_csv:
            out.unlink()

        def no_model(source, **kwargs):
            raise AssertionError("a unit reached a model")

        import redact.content_moderation.outputs as G
        monkeypatch.setattr(G, "drive_units", no_model)
        with pytest.raises(ValueError, match="without an iteration") as raised:
            _respond(tmp_path, monkeypatch, inputs=_questions(2))
        assert "resume=False" in str(raised.value)
        assert "new data_dir" in str(raised.value)


class TestTheLengthFloor:
    """With the checker on, a response with almost no text, once its
    code-fence marker lines are removed, is rejected without a check call."""

    def _run(self, folder, monkeypatch, replies, **kwargs):
        import redact.stages as P
        from redact import generate_outputs

        gen = MockBackend(replies, "gen-model")
        check = MockBackend("Yes", "check-model")
        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create",
                            lambda m: make_client(backends[m], m))
        generate_outputs(inputs=_questions_of(["harmful"]), data_dir=folder,
                         model="gen-model", check_model="check-model",
                         verbose=False, **kwargs)
        return gen, check, _written(folder).iloc[0]

    @pytest.mark.parametrize("reply", [
        "Here is a response:",                  # 19 characters
        "```\n```",                             # an empty code block
        "```python\n\n```\n",
        "  [RESPONSE]  ",
    ])
    def test_a_response_with_almost_no_text_is_rejected_without_a_check(
            self, tmp_path, monkeypatch, reply):
        from redact.content_moderation.feedback import TOO_SHORT

        _, check, row = self._run(tmp_path, monkeypatch, reply)
        assert check.calls == []
        assert not bool(row["accepted"])
        assert row["rejection_reason"] == TOO_SHORT
        assert row["rejection_reason"].startswith("too short")

    @pytest.mark.parametrize("reply", [
        "```python\nprint('forty characters of code')\n```",
        "Here is a response with some content.",
    ])
    def test_a_response_with_twenty_characters_or_more_is_checked(
            self, tmp_path, monkeypatch, reply):
        _, check, row = self._run(tmp_path, monkeypatch, reply)
        assert len(check.calls) == 1
        assert bool(row["accepted"]) and row["output_response"] == reply

    def test_a_short_response_is_generated_again_with_the_reason(
            self, tmp_path, monkeypatch):
        from redact.content_moderation.feedback import TOO_SHORT

        gen, check, row = self._run(
            tmp_path, monkeypatch,
            ["Here is a response:", "A response long enough to be checked."],
            max_attempts=2)
        assert TOO_SHORT not in gen.calls[0]["messages"][-1]["content"]
        assert TOO_SHORT in gen.calls[1]["messages"][-1]["content"]
        assert len(check.calls) == 1
        assert bool(row["accepted"])

    def test_without_the_checker_a_short_response_is_accepted(
            self, tmp_path, monkeypatch):
        _, check, row = self._run(tmp_path, monkeypatch, "short",
                                  check_outputs=False)
        assert check.calls == []
        assert bool(row["accepted"]) and row["rejection_reason"] == ""
