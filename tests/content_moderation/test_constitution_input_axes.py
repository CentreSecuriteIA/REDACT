"""The generation axes in the inputs stage: one unit per ``(entry,
iteration)``, each dealt one spec, shown in its prompt, recorded on its row
and counted in the coverage table; the plan file a resume is checked
against; and the units run apart on the pool. Fully offline."""

import asyncio
import csv
import json
import logging
import re
from collections import Counter

import pandas as pd
import pytest

from redact.content_moderation import axes as A
from redact.content_moderation.inputs import (
    InputPipeline,
    _constitution_inputs_ledger,
    _constitution_inputs_manifest,
)
from redact.llms.backends import ComputeConfig
from tests.conftest import MockBackend, lock_is_free, make_client
from tests.content_moderation.test_constitution_input_ledger import (
    _NTH,
    _rows,
    _token_df,
    _token_pipeline,
    _TokenBackend,
    one,
)

_PROMPT = {"system_prompt": "sys",
           "template": "Generate items about {sample_description}\n\n{sample_spec}"}
_K = 3
_ENTRIES = 4
_AXIS_COLUMNS = ["length", "register", "writer", "framing", "directness", "focus"]
_SECOND = 2         # a token's second request
_SEED, _OTHER_SEED = 42, 7      # the core's default seed, and another


def _run(folder, *, axes=True, weighting="table", k=_K, entries=_ENTRIES,
         judge=None, **kwargs):
    """Run the stage; the generating backend."""
    gen = _TokenBackend(lambda t, n: one(f"{_NTH.get(n, n)} sample {t}"))
    check = _TokenBackend(judge or (lambda t, n: "Yes"))
    _token_pipeline(folder, gen, check).run_from_constitution(
        _token_df(entries), _PROMPT, samples_per_entry=k, batch_size=4,
        verbose=False, axes=A.resolve("inputs", axes, weighting=weighting), **kwargs)
    return gen


def _planned(folder) -> dict[tuple[str, int], dict]:
    """Each unit's planned spec, by description and iteration."""
    return {(row["sample_description"], row["iteration"]): row.get("spec")
            for row in _constitution_inputs_manifest(folder).load()}


class TestTheUnits:
    def test_each_entry_is_k_units_and_k_rows_with_their_manifest_specs(
            self, tmp_path):
        _run(tmp_path)
        planned = _planned(tmp_path)
        assert len(planned) == _ENTRIES * _K
        rows = _rows(tmp_path)
        assert len(rows) == _ENTRIES * _K
        assert list(rows.columns[-7:]) == ["iteration", *_AXIS_COLUMNS]
        for row in rows.to_dict("records"):
            spec = planned[row["source_sample_description"], int(row["iteration"])]
            assert {axis: row[axis] for axis in _AXIS_COLUMNS} == spec

    def test_the_prompt_of_a_unit_shows_its_spec(self, tmp_path):
        gen = _run(tmp_path)
        axes = A.resolve("inputs")
        shown = Counter()
        for call in gen.calls:
            text = call["messages"][-1]["content"]
            token = re.search(r"zq\d+", text).group()
            shown[token, text.split("\n\n", 1)[1]] += 1
        fields = {"subcategory": "Sub"}
        assert shown == Counter({
            (description.split()[-1], A.render_spec(axes, spec, fields)): 1
            for (description, _), spec in _planned(tmp_path).items()})

    def test_without_axes_no_spec_is_dealt_shown_or_recorded(self, tmp_path):
        gen = _run(tmp_path, axes=False)
        assert all(spec is None for spec in _planned(tmp_path).values())
        assert not set(_AXIS_COLUMNS) & set(_rows(tmp_path).columns)
        assert all(call["messages"][-1]["content"].endswith("\n\n")
                   for call in gen.calls)
        assert not (tmp_path / "constitution_inputs.coverage.csv").exists()

    def test_a_prompt_without_the_placeholder_is_warned_about(self, tmp_path, caplog):
        pipe = _token_pipeline(tmp_path, _TokenBackend(lambda t, n: one(t)),
                               _TokenBackend(lambda t, n: "Yes"))
        unshown = {"system_prompt": "s", "template": "{sample_description}"}
        with caplog.at_level(logging.WARNING):
            pipe.run_from_constitution(
                _token_df(1), unshown, samples_per_entry=1, verbose=False,
                axes=A.resolve("inputs"))
        assert "has no {sample_spec}" in caplog.text


class TestAResume:
    def test_it_deals_the_same_specs_and_makes_no_call(self, tmp_path):
        _run(tmp_path)
        first = _planned(tmp_path)
        gen = _run(tmp_path)
        assert gen.calls == [] and _planned(tmp_path) == first

    @pytest.mark.parametrize("changed, named", [
        ({"axes": {"length": {"fragment": 1, "one_liner": 0, "short": 0, "long": 0,
                              "very_long": 0}}}, "axes"),
        ({"weighting": "log"}, "axes"),
        ({"k": 2}, "samples_per_entry"),
        ({"seed": 7}, "seed"),
    ])
    def test_with_another_deal_it_raises_naming_resume_false(
            self, tmp_path, changed, named):
        _run(tmp_path)
        with pytest.raises(ValueError, match="resume=False") as stopped:
            _run(tmp_path, **changed)
        assert named in str(stopped.value)
        assert lock_is_free(_constitution_inputs_ledger(tmp_path).path)
        _run(tmp_path, resume=False, **changed)     # a fresh run deals it

    def test_the_plan_file_holds_the_weights_dealt(self, tmp_path):
        _run(tmp_path, weighting="log")
        made_with = json.loads((tmp_path / "constitution_inputs.plan.json").read_text(
            encoding="utf-8"))
        scaled = A.deal_weights(A.resolve("inputs", weighting="log"))
        assert made_with == {"seed": _SEED, "samples_per_entry": _K, "axes": scaled}
        assert scaled != A.deal_weights(A.resolve("inputs"))

    @pytest.mark.parametrize("k", [0, -1, 2.5, True, "3"])
    def test_samples_per_entry_that_is_not_a_whole_number_of_at_least_1_raises_first(
            self, tmp_path, k):
        _run(tmp_path)
        before = sorted(p.name for p in tmp_path.rglob("*"))
        plan = (tmp_path / "constitution_inputs.plan.json").read_bytes()
        with pytest.raises(ValueError, match="inputs.samples_per_entry"):
            _run(tmp_path, k=k, resume=False)
        assert sorted(p.name for p in tmp_path.rglob("*")) == before
        assert (tmp_path / "constitution_inputs.plan.json").read_bytes() == plan
        assert len(_rows(tmp_path)) == _ENTRIES * _K

    def test_a_folder_with_no_done_unit_is_not_compared(self, tmp_path):
        """Nothing dealt before has to be dealt again: a resume of a folder
        whose plan file holds other settings, and no done unit (a run that
        stopped before its first save), runs and writes its own record."""
        path = tmp_path / "constitution_inputs.plan.json"
        path.write_text(json.dumps(
            {"seed": _OTHER_SEED, "samples_per_entry": _K,
             "axes": A.deal_weights(A.resolve("inputs"))}), encoding="utf-8")
        gen = _run(tmp_path)        # at the default seed
        assert len(gen.calls) == _ENTRIES * _K
        assert json.loads(path.read_text(encoding="utf-8"))["seed"] == _SEED
        with pytest.raises(ValueError, match="'seed'"):
            _run(tmp_path, seed=_OTHER_SEED)    # now it has done units

    def test_a_folder_without_its_plan_file_is_warned_about(self, tmp_path, caplog):
        _run(tmp_path)
        (tmp_path / "constitution_inputs.plan.json").unlink()
        with caplog.at_level(logging.WARNING):
            _run(tmp_path)
        assert "no constitution_inputs.plan.json" in caplog.text
        assert (tmp_path / "constitution_inputs.plan.json").exists()

    @staticmethod
    def _entries(tokens) -> pd.DataFrame:
        return pd.DataFrame([
            {"source_category": "Violence", "sample_description": f"desc zq{t}",
             "constitution_subcategory": "Sub", "entry_type": "harmful"}
            for t in tokens])

    def _resume(self, folder, tokens):
        gen = _TokenBackend(lambda t, n: one(f"{_NTH.get(n, n)} sample {t}"))
        _token_pipeline(folder, gen, _TokenBackend(lambda t, n: "Yes")) \
            .run_from_constitution(self._entries(tokens), _PROMPT,
                                   samples_per_entry=_K, batch_size=4,
                                   verbose=False, axes=A.resolve("inputs"))
        return gen

    def test_entries_that_join_a_group_leave_the_done_units_specs_to_their_rows(
            self, tmp_path):
        """zq1 sorts before the done zq5 and zq7 of its group: their units
        keep, in the manifest and the coverage table, the specs their rows
        were written with, and zq1's are dealt around them."""
        self._resume(tmp_path, [5, 7])
        gen = self._resume(tmp_path, [1, 5, 7])
        assert len(gen.calls) == _K
        planned = _planned(tmp_path)
        rows = _rows(tmp_path)
        assert len(rows) == 3 * _K
        for row in rows.to_dict("records"):
            spec = planned[row["source_sample_description"], int(row["iteration"])]
            assert {axis: row[axis] for axis in _AXIS_COLUMNS} == spec
        with (tmp_path / "constitution_inputs.coverage.csv").open(
                encoding="utf-8", newline="") as fh:
            accepted = {(r["axis"], r["value"]): int(r["accepted"])
                        for r in csv.DictReader(fh) if int(r["accepted"])}
        assert accepted == {(axis, value): n for axis in _AXIS_COLUMNS
                            for value, n in rows[axis].value_counts().items()}
        # zq1's siblings still differ on length.
        assert rows[rows["source_sample_description"] == "desc zq1"][
            "length"].nunique() == _K

    def test_acks_written_before_they_recorded_a_spec_are_dealt_again(
            self, tmp_path):
        self._resume(tmp_path, [5, 7])
        ledger = _constitution_inputs_ledger(tmp_path).path
        lines = [json.loads(line) for line in ledger.read_text(
            encoding="utf-8").splitlines()]
        assert all("spec" in ack for ack in lines if "sample_description" in ack)
        ledger.write_text("".join(
            json.dumps({k: v for k, v in ack.items() if k != "spec"}) + "\n"
            for ack in lines), encoding="utf-8")
        gen = self._resume(tmp_path, [1, 5, 7])
        assert len(gen.calls) == _K and len(_rows(tmp_path)) == 3 * _K

    def test_a_broken_plan_file_stops_it_naming_the_remedies(self, tmp_path):
        _run(tmp_path)
        plan = tmp_path / "constitution_inputs.plan.json"
        plan.write_text(plan.read_text(encoding="utf-8")[:20], encoding="utf-8")
        with pytest.raises(ValueError, match="resume=False") as stopped:
            _run(tmp_path)
        assert plan.name in str(stopped.value)
        assert lock_is_free(_constitution_inputs_ledger(tmp_path).path)
        plan.unlink()
        assert _run(tmp_path).calls == []   # deleted, the resume runs unchecked

    def test_a_fresh_run_replaces_a_broken_plan_file(self, tmp_path):
        _run(tmp_path)
        plan = tmp_path / "constitution_inputs.plan.json"
        plan.write_text('{"seed": 4', encoding="utf-8")
        gen = _run(tmp_path, seed=_OTHER_SEED, resume=False)
        assert len(gen.calls) == _ENTRIES * _K
        assert json.loads(plan.read_text(encoding="utf-8"))["seed"] == _OTHER_SEED

    def test_a_folder_written_without_the_axes_stops_a_run_with_them(self, tmp_path):
        _run(tmp_path, axes=False)
        with pytest.raises(ValueError, match="'length'") as stopped:
            _run(tmp_path)
        assert "inputs.axes: false" in str(stopped.value)


class TestTheCoverageTable:
    def test_it_counts_the_planned_specs_and_the_accepted_ones(self, tmp_path):
        """Read from the manifest and the acks, it agrees with the rows."""
        _run(tmp_path, judge=lambda t, n: "No: weak" if n == _SECOND else "Yes")
        with (tmp_path / "constitution_inputs.coverage.csv").open(
                encoding="utf-8", newline="") as fh:
            table = list(csv.DictReader(fh))
        rows = _rows(tmp_path)
        accepted = rows[rows["accepted"] == "True"]
        assert len(accepted) == _ENTRIES * (_K - 1)
        for axis in _AXIS_COLUMNS:
            counted = {
                (r["value"], r["category"]): (int(r["planned"]), int(r["accepted"]))
                for r in table if r["axis"] == axis}
            on_disk = {key: (n, int(accepted.groupby([axis, "category"]).size()
                                    .get(key, 0)))
                       for key, n in rows.groupby([axis, "category"]).size().items()}
            assert {key: c for key, c in counted.items() if c[0]} == on_disk


class _Lane(MockBackend):
    """An async fake; ``reply(text)`` is a coroutine answering each request."""

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self, reply):
        super().__init__()
        self._reply = reply

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None,
                        **kwargs):
        return await self._reply(" ".join(m["content"] for m in messages))


def test_a_unit_checks_its_sample_while_its_sibling_still_generates(tmp_path):
    """Each ``(entry, iteration)`` is its own unit on the pool: the second
    unit's generation is answered only once the first unit's check arrived,
    which an entry that waits for all its samples would never send."""
    checked = asyncio.Event()
    generated = Counter()

    async def write(text):
        token = re.search(r"zq\d+", text).group()
        generated[token] += 1
        if generated[token] == _SECOND:
            await asyncio.wait_for(checked.wait(), timeout=5)
        return one(f"{_NTH[generated[token]]} sample {token}")

    async def judge(text):
        checked.set()
        return "Yes"

    pipe = InputPipeline(gen=make_client(_Lane(write), "lane-gen"),
                         check=make_client(_Lane(judge), "lane-check"),
                         dataset_dir=tmp_path)
    pipe.run_from_constitution(_token_df(1), _PROMPT, samples_per_entry=2,
                               verbose=False, axes=A.resolve("inputs"))
    assert sorted(_rows(tmp_path)["sample"]) == ["first sample zq0",
                                                  "second sample zq0"]


def test_the_front_resolves_the_axes_before_any_model(tmp_path, monkeypatch):
    import redact.stages as P
    from redact import generate_inputs

    def no_model(name):
        raise AssertionError("a model was made")

    monkeypatch.setattr(P.ModelClient, "create", no_model)
    with pytest.raises(ValueError, match="no axis 'tone'"):
        generate_inputs(data_dir=tmp_path, constitution_df=pd.DataFrame(),
                        axes={"tone": {"dry": 1}}, verbose=False)
