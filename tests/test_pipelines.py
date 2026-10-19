"""Tests for high-level pipeline functions."""

import json
import logging
import re
from functools import partial

import pandas as pd
import pytest

from redact.exceptions import PromptTooLongError
from redact.llms.backends import ComputeConfig
from tests.conftest import MockBackend


def test_the_former_module_name_still_resolves():
    from redact import pipelines, stages

    for name in pipelines.__all__:
        assert getattr(pipelines, name) is getattr(stages, name)
    assert set(pipelines.__all__) == {
        "STAGES", "RunContext", "Stage", "build_dataset", "generate_constitution",
        "generate_inputs", "generate_jailbreaks", "generate_outputs",
        "generate_paraphrases"}


class TestGenerateOutputsManifest:
    """generate_outputs writes a plan manifest before the batch loop (offline)."""

    def test_manifest_written_with_one_row_per_input(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs, paths
        from redact.dataset import Manifest
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("a plain answer"), m))
        inputs = pd.DataFrame({
            "sample": ["prompt one", "prompt two", "prompt three"],
            "category": ["Cyber"] * 3,
            "entry_type": ["harmful"] * 3,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        out = paths.output_responses_csv(tmp_path)
        rows = Manifest.sidecar(out).load()
        assert len(rows) == 3 and all(r["status"] == "planned" for r in rows)
        # planned input_ids match the output CSV's input_ids
        df = pd.read_csv(out)
        assert {r["input_id"] for r in rows} == set(df["input_id"].astype(str))

    def test_manifest_is_full_plan_even_on_partial_resume(self, tmp_path, monkeypatch):
        # The manifest is the full intended scope, written before the resume
        # filter — so a resume run that only generates the pending units still
        # records every planned unit in the manifest.
        import redact.stages as P
        from redact import generate_outputs, paths
        from redact.dataset import Ledger, Manifest, content_id
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        out = paths.output_responses_csv(tmp_path)
        # Pre-seed the ledger as if "p one" was already completed.
        Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                       casters={"input_id": str, "iteration": int}).record(
            [{"input_id": content_id("p one"), "iteration": 0}]
        )

        generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                         resume=True, verbose=False)

        assert len(Manifest.sidecar(out).load()) == 3   # full plan (all 3 units)
        assert len(pd.read_csv(out)) == 2               # only the 2 pending generated

    def test_a_fresh_run_stopped_before_it_plans_leaves_no_old_plan(
            self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs, paths
        from redact.dataset import Manifest
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"],
            "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                         verbose=False)
        manifest = Manifest.sidecar(paths.output_responses_csv(tmp_path))
        assert manifest.path.exists()

        def stop(self, rows):
            raise KeyboardInterrupt

        monkeypatch.setattr(Manifest, "write", stop)
        with pytest.raises(KeyboardInterrupt):
            generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                             resume=False, verbose=False)
        assert not manifest.path.exists()

    def test_the_callers_frame_is_left_as_it_was(self, tmp_path, monkeypatch):
        """The stage adds its id column to a frame of its own."""
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        }, index=[7, 3, 5])
        before = inputs.copy()
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                              verbose=False)
        pd.testing.assert_frame_equal(inputs, before)
        assert list(df["input_prompt"]) == list(inputs["sample"])


def _seed_input_tree(data_dir):
    """One accepted input row in the run's inputs tree. Returns its text."""
    from redact import paths
    from tests.conftest import write_inputs

    write_inputs(paths.constitution_inputs_dir(data_dir), ["zq0 the one input"],
                 source="constitution",
                 extra={"subcategory": "phishing", "entry_type": "benign"})
    return "zq0 the one input"


class TestGenerateOutputsFindsItsInputs:
    """Called without ``inputs=``, the stage reads the inputs of its run."""

    def test_the_inputs_of_the_run_are_found(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from redact.dataset import content_id
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        text = _seed_input_tree(tmp_path)
        df = generate_outputs(data_dir=tmp_path, check_outputs=False, verbose=False)
        assert list(df["input_prompt"]) == [text]
        assert list(df["input_id"]) == [content_id(text)]
        assert list(df["entry_type"]) == ["benign"]

    def test_no_inputs_raises(self, tmp_path):
        from redact import generate_outputs

        with pytest.raises(ValueError, match="No input samples found"):
            generate_outputs(data_dir=tmp_path, check_outputs=False, verbose=False)


def _seed_rejected_tree(data_dir):
    """Two input rows, both rejected, in the run's inputs tree."""
    from redact import paths
    from tests.conftest import write_inputs

    write_inputs(paths.constitution_inputs_dir(data_dir),
                 ["zq0 one input", "zq1 another input"], accepted=[False, False])


def _no_model(monkeypatch):
    """Fail any model the stage builds: a skipped stage needs none."""
    import redact.stages as P

    def create(*args, **kwargs):
        raise AssertionError("a skipped stage built a model client")

    monkeypatch.setattr(P.ModelClient, "create", create)


def _skip_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "skipped" in r.getMessage()]


def _left_by_an_earlier_run(out) -> None:
    """The files an earlier run of a stage leaves beside its output ``out``;
    those of a kind of the jailbreak stage hold its coverage and plan too."""
    stem = out.name.removesuffix(".csv")
    kind = ((f"{stem}.coverage.csv", f"{stem}.plan.json")
            if stem in ("jailbreaks", "variations") else ())
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in (out.name, f"{out.name}.cut", f"{out.name}.tmp", f"{stem}.state.jsonl",
                 f"{stem}.manifest.jsonl", *kind):
        out.with_name(name).write_text("left by an earlier run\n", encoding="utf-8")


def _files_beside(out) -> list[str]:
    """The files of the stage whose output is ``out``, its lock aside."""
    stem = out.name.removesuffix(".csv")
    return sorted(p.name for p in out.parent.glob(f"{stem}.*") if p.suffix != ".lock")


class TestAStageWithNoAcceptedInput:
    """Inputs that were all rejected skip a stage with a warning; no inputs at
    all is the caller's mistake and raises."""

    def test_outputs_warn_and_write_nothing(
            self, tmp_path, monkeypatch, caplog):
        from redact import generate_outputs, paths

        _seed_rejected_tree(tmp_path)
        _no_model(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="redact"):
            df = generate_outputs(data_dir=tmp_path, verbose=False)
        assert df.empty
        assert len(_skip_warnings(caplog)) == 1
        out = paths.output_responses_csv(tmp_path)
        assert not list(out.parent.glob(f"{out.stem}*"))

    def test_an_empty_frame_given_is_skipped_the_same_way(
            self, tmp_path, monkeypatch, caplog):
        from redact import generate_outputs, paths

        _no_model(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="redact"):
            df = generate_outputs(inputs=pd.DataFrame(), data_dir=tmp_path,
                                  verbose=False)
        assert df.empty
        assert len(_skip_warnings(caplog)) == 1
        assert not paths.datasets(tmp_path).exists()

    def test_jailbreaks_warn_and_write_nothing(self, tmp_path, monkeypatch, caplog):
        from redact import generate_jailbreaks, paths

        _seed_rejected_tree(tmp_path)
        _no_model(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="redact"):
            df = generate_jailbreaks(data_dir=tmp_path, verbose=False)
        assert df.empty
        assert len(_skip_warnings(caplog)) == 1
        for out in (paths.jailbreaks_csv(tmp_path), paths.variations_csv(tmp_path)):
            assert not list(out.parent.glob(f"{out.stem}*"))

    def test_paraphrases_of_the_inputs_warn_and_write_nothing(
            self, tmp_path, monkeypatch, caplog):
        import redact.stages as P
        from redact import generate_paraphrases, paths
        from tests.conftest import make_client

        _seed_rejected_tree(tmp_path)
        backend = MockBackend("unused")
        monkeypatch.setattr(P.ModelClient, "create",
                            lambda m: make_client(backend, m))
        with caplog.at_level(logging.WARNING, logger="redact"):
            out = generate_paraphrases(data_dir=tmp_path, target="inputs",
                                       verbose=False)
        assert out["inputs"].empty
        assert len(_skip_warnings(caplog)) == 1
        assert backend.calls == []
        para = paths.paraphrases_inputs_csv(tmp_path)
        assert not list(para.parent.glob(f"{para.stem}*"))

    def test_paraphrases_of_outputs_none_accepted_warn(
            self, tmp_path, monkeypatch, caplog):
        import redact.stages as P
        from redact import generate_paraphrases, paths
        from tests.conftest import make_client

        out = paths.output_responses_csv(tmp_path)
        out.parent.mkdir(parents=True)
        pd.DataFrame({"input_id": ["a"], "output_response": ["r"],
                      "accepted": [False]}).to_csv(out, index=False)
        monkeypatch.setattr(P.ModelClient, "create",
                            lambda m: make_client(MockBackend("unused"), m))
        with caplog.at_level(logging.WARNING, logger="redact"):
            generate_paraphrases(data_dir=tmp_path, target="outputs", verbose=False)
        assert len(_skip_warnings(caplog)) == 1
        assert not paths.paraphrases_outputs_csv(tmp_path).exists()

    @pytest.mark.parametrize("stage", ["outputs", "jailbreaks", "paraphrase"])
    def test_a_fresh_run_clears_what_an_earlier_run_left(
            self, tmp_path, monkeypatch, caplog, stage):
        """Else the build merges rows whose inputs are now rejected."""
        import redact.stages as P
        from redact import (
            generate_jailbreaks,
            generate_outputs,
            generate_paraphrases,
            paths,
        )
        from tests.conftest import make_client

        _seed_rejected_tree(tmp_path)
        backend = MockBackend("unused")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        run, outs = {
            "outputs": (generate_outputs, [paths.output_responses_csv(tmp_path)]),
            "jailbreaks": (generate_jailbreaks, [paths.jailbreaks_csv(tmp_path),
                                                 paths.variations_csv(tmp_path)]),
            "paraphrase": (partial(generate_paraphrases, target="inputs"),
                           [paths.paraphrases_inputs_csv(tmp_path)]),
        }[stage]
        for out in outs:
            _left_by_an_earlier_run(out)
        with caplog.at_level(logging.WARNING, logger="redact"):
            run(data_dir=tmp_path, resume=False, verbose=False)
        assert len(_skip_warnings(caplog)) == 1
        assert backend.calls == []
        assert [_files_beside(out) for out in outs] == [[] for _ in outs]

    def test_a_resumed_run_keeps_what_an_earlier_run_left(
            self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths

        _seed_rejected_tree(tmp_path)
        _no_model(monkeypatch)
        out = paths.output_responses_csv(tmp_path)
        _left_by_an_earlier_run(out)
        before = _files_beside(out)
        generate_outputs(data_dir=tmp_path, verbose=False)
        assert _files_beside(out) == before

    def test_jailbreaks_without_any_inputs_raise(self, tmp_path):
        from redact import generate_jailbreaks

        with pytest.raises(ValueError, match="No input samples found"):
            generate_jailbreaks(data_dir=tmp_path, verbose=False)


class TestAKindWhoseEntryTypesMatchNoInput:
    _KIND = {"adversarial": {"entry_types": ["benign"]}, "natural": None}

    def _run(self, tmp_path, monkeypatch, **kwargs):
        import redact.stages as P
        from redact import generate_jailbreaks
        from tests.conftest import make_client

        backend = MockBackend("unused")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        inputs = pd.DataFrame({"prompt": ["zq0 one", "zq1 two"],
                               "category": ["Cyber"] * 2,
                               "entry_type": ["harmful", "dual_use_harmful"]})
        df = generate_jailbreaks(inputs=inputs, data_dir=tmp_path,
                                 augmentation=self._KIND, verbose=False, **kwargs)
        assert df.empty and backend.calls == []

    def test_it_warns_naming_both_lists_and_writes_nothing(
            self, tmp_path, monkeypatch, caplog):
        from redact import paths

        with caplog.at_level(logging.WARNING, logger="redact"):
            self._run(tmp_path, monkeypatch)
        (warned,) = [r.getMessage() for r in caplog.records
                     if r.levelno == logging.WARNING]
        assert "adversarial" in warned and "['benign']" in warned
        assert "['dual_use_harmful', 'harmful']" in warned
        assert _files_beside(paths.jailbreaks_csv(tmp_path)) == []

    def test_a_fresh_run_clears_what_an_earlier_run_left(self, tmp_path, monkeypatch):
        from redact import paths

        out = paths.jailbreaks_csv(tmp_path)
        _left_by_an_earlier_run(out)
        self._run(tmp_path, monkeypatch, resume=False)
        assert _files_beside(out) == []


class _JudgeByText(MockBackend):
    """Accepts a sample whose check request holds one of ``accept``."""

    def __init__(self, accept):
        super().__init__("No: weak")
        self.accept = accept

    def generate(self, messages_list, **kwargs):
        return ["Yes" if any(w in m[-1]["content"] for w in self.accept)
                else "No: weak" for m in messages_list]


class TestInputsWarnOnLowAcceptance:
    """The inputs stage warns, with the counts, when it accepts under half."""

    @pytest.mark.parametrize("accept, warned", [
        ((), "0 of 4"), (("zqa",), "1 of 4"), (("zqa", "zqc"), None),
    ], ids=["none", "one", "half"])
    def test_a_warning_under_half(self, tmp_path, monkeypatch, caplog, accept, warned):
        import redact.stages as P
        from redact import generate_inputs
        from tests.conftest import make_client

        backends = {
            "gen-model": MockBackend([f"1. {text} sample\n[END]"
                                      for text in ("zqa", "zqb", "zqc", "zqd")]),
            "check-model": _JudgeByText(accept),
        }
        monkeypatch.setattr(P.ModelClient, "create",
                            lambda m: make_client(backends[m], m))
        constitution = pd.DataFrame([
            {"source_category": "Cyber", "constitution_subcategory": "Sub",
             "sample_description": f"what zq{n} is about", "entry_type": "harmful"}
            for n in range(2)])
        with caplog.at_level(logging.WARNING, logger="redact"):
            generate_inputs(data_dir=tmp_path, constitution_df=constitution,
                            samples_per_entry=2, model="gen-model",
                            check_model="check-model", verbose=False)
        found = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.WARNING and "accepted" in r.getMessage()]
        if warned is None:
            assert found == []
        else:
            assert len(found) == 1 and warned in found[0]


class TestInputsWithoutAConstitutionGiven:
    """Without ``constitution_df`` the inputs stage reads the constitution
    saved under ``data_dir``, and without one it raises."""

    def test_the_saved_constitution_seeds_the_inputs(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_inputs, paths
        from tests.conftest import make_client

        backends = {"gen-model": MockBackend(["1. zqa sample\n[END]",
                                              "1. zqb sample\n[END]"]),
                    "check-model": MockBackend("Yes")}
        monkeypatch.setattr(P.ModelClient, "create",
                            lambda m: make_client(backends[m], m))
        merged = paths.constitution_merged_csv(tmp_path)
        merged.parent.mkdir(parents=True)
        pd.DataFrame([{
            "source_category": "Category A", "constitution_subcategory": "Sub",
            "sample_description": "placeholder sample description",
            "entry_type": "harmful"}]).to_csv(merged, index=False)
        df = generate_inputs(data_dir=tmp_path, samples_per_entry=2,
                             model="gen-model", check_model="check-model",
                             verbose=False)
        assert list(df["sample"]) == ["zqa sample", "zqb sample"]
        assert set(df["category"]) == {"Category A"}

    def test_without_one_it_raises_naming_both_ways_to_have_one(
            self, tmp_path, monkeypatch):
        from redact import generate_inputs

        _no_model(monkeypatch)
        with pytest.raises(ValueError, match="No constitution at") as raised:
            generate_inputs(data_dir=tmp_path, verbose=False)
        assert "generate_constitution" in str(raised.value)
        assert "an earlier run on the same data_dir" in str(raised.value)


class TestGenerateOutputsReadBack:
    def test_without_read_back_the_output_is_written_and_not_read(
            self, tmp_path, monkeypatch):
        import redact.content_moderation.outputs as G
        import redact.stages as P
        from redact import generate_outputs, paths
        from tests.conftest import MockBackend, make_client

        def read_rows(path):
            raise AssertionError("the output was read back")

        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        monkeypatch.setattr(G, "read_rows", read_rows)
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2,
            "entry_type": ["harmful"] * 2,
        })
        returned = generate_outputs(
            data_dir=tmp_path, inputs=inputs, check_outputs=False, read_back=False)
        assert returned.empty and len(returned.columns) == 0
        written = pd.read_csv(paths.output_responses_csv(tmp_path))
        assert sorted(written["input_prompt"]) == ["p one", "p two"]


class TestGenerateOutputsResume:
    def test_rows_already_in_the_csv_are_not_regenerated(self, tmp_path, monkeypatch):
        """A crash between the CSV append and the ledger ack leaves a row with no
        ack. Without the CSV union, resume regenerates that input and appends a
        *second* row for it — two rows, two sample_ids, no dedup downstream."""
        import redact.stages as P
        from redact import generate_outputs, paths
        from redact.dataset import content_id
        from tests.conftest import MockBackend, make_client

        out = paths.output_responses_csv(tmp_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Same column order the stage appends in, since the append is headerless;
        # a run without the response axes, whose columns the file lacks.
        pd.DataFrame([{
            "input_id": content_id("p one"), "input_prompt": "p one",
            "category": "Cyber", "subcategory": "", "entry_type": "harmful",
            "model": "mock-model", "source": "", "sample_id": "prior",
            "output_response": "written before the crash",
            "accepted": True, "rejection_reason": "", "iteration": 0,
            "response_mode": "harmful", "output_label": "harmful",
            "response_label": "harmful_harmful",
        }]).to_csv(out, index=False)

        backend = MockBackend("ans")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"],
            "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                              resume=True, verbose=False, axes=False)

        assert len(backend.calls) == 1                      # only "p two" generated
        assert len(df) == 2                                 # prior row kept, one added
        assert list(df["input_id"]).count(content_id("p one")) == 1


class TestGenerateOutputsInternals:
    """Internals capture is a folder-per-input_id side channel — no CSV column,
    ever (see .claude/introspection_backend_plan.md's non-interference principle).
    """

    def test_no_internals_kwarg_for_default_backend(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        backend = MockBackend("ans")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)
        assert "internals_id" not in backend.calls[0]
        assert "internals_id" not in df.columns and "internals_ids" not in df.columns

    def test_gen_internals_id_is_input_id_slash_output(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        class _InternalsBackend(MockBackend):
            @property
            def compute_config(self) -> ComputeConfig:
                return ComputeConfig(supports_internals=True)

        backend = _InternalsBackend("ans")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        seen = {c["internals_id"] for c in backend.calls}
        expected = {f'{row["input_id"]}/0/output' for _, row in df.iterrows()}
        assert seen == expected
        # never a CSV column, regardless of capture
        assert "internals_id" not in df.columns

    @pytest.mark.usefixtures("no_length_floor")
    def test_check_internals_id_is_input_id_slash_val_out(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        class _InternalsBackend(MockBackend):
            @property
            def compute_config(self) -> ComputeConfig:
                return ComputeConfig(supports_internals=True)

        gen_backend = _InternalsBackend("the answer")
        check_backend = _InternalsBackend("Yes, fine")
        backends = {"gen-model": gen_backend, "check-model": check_backend}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        generate_outputs(
            data_dir=tmp_path, inputs=inputs, model="gen-model", check_model="check-model",
            verbose=False,
        )
        assert check_backend.calls[0]["internals_id"].endswith("/0/val_out")
        assert gen_backend.calls[0]["internals_id"].endswith("/0/output")
        # same input_id root for both stages
        gen_root = gen_backend.calls[0]["internals_id"].split("/")[0]
        check_root = check_backend.calls[0]["internals_id"].split("/")[0]
        assert gen_root == check_root

    @pytest.mark.usefixtures("no_length_floor")
    def test_check_internals_not_requested_when_check_backend_lacks_support(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        class _InternalsBackend(MockBackend):
            @property
            def compute_config(self) -> ComputeConfig:
                return ComputeConfig(supports_internals=True)

        gen_backend = _InternalsBackend("the answer")
        check_backend = MockBackend("Yes, fine")  # no internals support
        backends = {"gen-model": gen_backend, "check-model": check_backend}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        generate_outputs(
            data_dir=tmp_path, inputs=inputs, model="gen-model", check_model="check-model",
            verbose=False,
        )
        assert "internals_id" not in check_backend.calls[0]
        assert gen_backend.calls[0]["internals_id"].endswith("/output")


class TestGenerateOutputsSampleId:
    """sample_id is this row's own content-hash identity, distinct from input_id
    (which points back to the origin sample) — a library-wide convention, not
    specific to internals logging."""

    def test_sample_id_matches_hash_of_output_response(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from redact.dataset import content_id
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("a plain answer"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        assert "sample_id" in df.columns
        for _, row in df.iterrows():
            assert row["sample_id"] == content_id(row["output_response"])
            assert row["sample_id"] != row["input_id"]


@pytest.mark.usefixtures("no_length_floor")
class TestGenerateOutputsRetry:
    """max_attempts: a rejected response is regenerated with the verdict."""

    def _run(self, tmp_path, monkeypatch, gen_replies, check_replies, **kwargs):
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        gen, check = MockBackend(gen_replies), MockBackend(check_replies)
        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        df = generate_outputs(
            data_dir=tmp_path, inputs=inputs, model="gen-model",
            check_model="check-model", verbose=False, **kwargs,
        )
        return gen, df.iloc[0]

    def test_a_rejected_response_is_regenerated_with_the_verdict(self, tmp_path, monkeypatch):
        gen, row = self._run(tmp_path, monkeypatch, ["first answer", "second answer"],
                             ["No: too vague", "Yes"], max_attempts=2)
        assert len(gen.calls) == 2
        assert "No: too vague" not in gen.calls[0]["messages"][-1]["content"]
        assert "No: too vague" in gen.calls[1]["messages"][-1]["content"]
        assert row["output_response"] == "second answer" and bool(row["accepted"])

    def test_one_attempt_keeps_the_rejected_response(self, tmp_path, monkeypatch):
        gen, row = self._run(tmp_path, monkeypatch, ["first answer"], ["No: too vague"])
        assert len(gen.calls) == 1
        assert not bool(row["accepted"])
        assert row["rejection_reason"] == "No: too vague"

    def test_every_attempt_rejected_keeps_the_last_one(self, tmp_path, monkeypatch):
        gen, row = self._run(tmp_path, monkeypatch, ["first answer", "second answer"],
                             ["No: one", "No: two"], max_attempts=2)
        assert row["output_response"] == "second answer"
        assert not bool(row["accepted"]) and row["rejection_reason"] == "No: two"
        # The feedback is not stacked: the second prompt carries one verdict.
        assert gen.calls[1]["messages"][-1]["content"].count("[FEEDBACK") == 1

    def test_check_calls_carry_the_limit_and_generation_keeps_the_default(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        gen, check = MockBackend("an answer"), MockBackend("Yes")
        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({"sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"]})
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", verbose=False, check_max_tokens=64)
        assert check.calls[0]["max_tokens"] == 64
        assert gen.calls[0]["max_tokens"] == gen.default_max_tokens

    def test_a_chunk_is_one_generation_call_and_one_check_call(self, tmp_path, monkeypatch):
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import make_client
        from tests.llms.test_router import _CountingBackend

        gen = _CountingBackend("an answer", native=True)
        check = _CountingBackend("Yes", native=True)
        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", batch_size=3, verbose=False)
        assert (gen.generate_call_count, check.generate_call_count) == (1, 1)


class _TokenBackend(MockBackend):
    """Native-batching fake: ``reply(token, text)`` answers each request, where
    ``token`` is the ``zqN`` in its prompt and ``text`` all its content.

    Replies depend only on the request, never on call order, so rows can be
    compared across runs whose units finish in different orders. ``batches``
    holds one list per ``generate()`` call: the token of each request, with
    ``*`` for a request carrying the checker's feedback.
    """

    def __init__(self, reply, model):
        super().__init__(model=model)
        self._reply = reply
        self.batches: list[list[str]] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=True)

    def generate(self, messages_list, **kwargs):
        super().generate(messages_list, **kwargs)
        batch, replies = [], []
        for messages in messages_list:
            text = " ".join(m["content"] for m in messages)
            token = re.search(r"zq\d+", text).group()
            batch.append(token + "*" * ("[FEEDBACK" in text))
            replies.append(self._reply(token, text))
        self.batches.append(batch)
        return replies


class _LaneBackend(MockBackend):
    """Async fake: answers ``answer <token>``, the ``zqN`` in the prompt, and
    rejects a prompt whose token is in ``too_long`` as over the context.

    ``sent`` holds the token of every request.
    """

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self, model, too_long=()):
        super().__init__(model=model)
        self.too_long = set(too_long)
        self.sent: list[str] = []

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None,
                        **kwargs):
        token = re.search(r"zq\d+", " ".join(m["content"] for m in messages)).group()
        self.sent.append(token)
        if token in self.too_long:
            raise PromptTooLongError(f"{token} is over the context")
        return f"answer {token}"


class TestPoolSize:
    """A stage keeps as many units live as its models' async lanes hold."""

    def test_the_outputs_stage_drives_with_it(self, tmp_path, monkeypatch):
        import redact.content_moderation.outputs as G
        import redact.stages as P
        from redact import generate_outputs
        from tests.conftest import make_client

        backends = {"gen-model": MockBackend("answer"), "check-model": MockBackend("Yes")}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        seen = {}

        def stop(source, **kwargs):
            seen.update(kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(G, "drive_units", stop)
        monkeypatch.setattr(G, "pool_size", lambda batch_size, *clients: 7 * batch_size)
        inputs = pd.DataFrame({"sample": ["zq0"], "category": ["Cyber"],
                               "entry_type": ["harmful"]})
        with pytest.raises(KeyboardInterrupt):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_model="check-model", batch_size=2, verbose=False)
        assert (seen["pool_size"], seen["cap"]) == (14, 2)


class TestGenerateOutputsPool:
    """Outputs run through drive_units: full calls, rows as units finish."""

    def _clients(self, monkeypatch, gen, check):
        import redact.stages as P
        from tests.conftest import make_client

        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))

    @pytest.mark.usefixtures("no_length_floor")
    def test_a_run_without_retries_writes_the_same_rows(self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths
        from redact.dataset import content_id

        gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        # Odd tokens rejected; one attempt, so no retries.
        check = _TokenBackend(
            lambda t, text: f"No: weak {t}" if int(t[2:]) % 2 else "Yes", "check-model")
        self._clients(monkeypatch, gen, check)
        tokens = [f"zq{i}" for i in range(6)]
        inputs = pd.DataFrame({
            "sample": tokens, "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6,
        })
        # Without the response axes: the row of each input, as before them.
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", batch_size=2, verbose=False,
                         axes=False)

        df = pd.read_csv(paths.output_responses_csv(tmp_path), dtype=str,
                         keep_default_na=False)
        assert list(df.columns) == [
            "input_id", "input_prompt", "category", "subcategory", "entry_type",
            "model", "source", "sample_id", "output_response", "accepted",
            "rejection_reason", "iteration", "response_mode", "output_label",
            "response_label",
        ]
        expected = {
            content_id(t): {
                "input_id": content_id(t), "input_prompt": t, "category": "Cyber",
                "subcategory": "", "entry_type": "harmful", "model": "gen-model",
                "source": "", "sample_id": content_id(f"answer {t}"),
                "output_response": f"answer {t}", "accepted": str(i % 2 == 0),
                "rejection_reason": f"No: weak {t}" if i % 2 else "",
                "iteration": "0", "response_mode": "harmful", "output_label": "harmful",
                "response_label": "harmful_harmful",
            }
            for i, t in enumerate(tokens)
        }
        assert len(df) == 6
        assert {r["input_id"]: r for r in df.to_dict("records")} == expected

    def test_the_ledger_records_how_much_of_the_output_is_saved(
            self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths
        from redact.dataset import Ledger

        gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, gen, None)
        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(3)],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_outputs=False, verbose=False)
        out = paths.output_responses_csv(tmp_path)
        size, _checksum = Ledger.sidecar(out).saved["output_responses.csv"]
        assert size == out.stat().st_size

    def _six_inputs(self):
        return pd.DataFrame({
            "sample": [f"zq{i}" for i in range(6)],
            "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6,
        })

    def test_a_run_killed_between_rows_and_acks_leaves_each_input_once(
            self, tmp_path, monkeypatch):
        import redact.content_moderation.outputs as G
        from redact import generate_outputs, paths
        from redact.dataset import Ledger, content_id
        from tests.conftest import Killed, kill_after_rows, lock_is_free

        inputs = self._six_inputs()

        def run():
            gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
            self._clients(monkeypatch, gen, None)
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, batch_size=2, verbose=False)
            return gen

        out, prompts = paths.output_responses_csv(tmp_path), list(inputs["sample"])
        state = out.with_name("output_responses.state.jsonl")
        kill_after_rows(monkeypatch, G, on_call=1)
        with pytest.raises(Killed):
            run()
        landed = out.read_bytes()
        assert sorted(pd.read_csv(out)["input_prompt"]) == prompts  # the rows landed
        assert state.read_bytes() == b'{"_saved": {}}\n'            # their acks did not
        assert lock_is_free(state)

        second = run()
        assert sorted(sum(second.batches, [])) == prompts   # every input ran again
        assert sorted(pd.read_csv(out)["input_prompt"]) == prompts
        assert out.with_name("output_responses.csv.cut").read_bytes() == landed
        assert Ledger.sidecar(out, ("input_id",)).completed() == {
            content_id(prompt) for prompt in prompts}
        assert lock_is_free(state)

    def test_a_second_run_is_refused_before_it_writes_its_manifest(
            self, tmp_path, monkeypatch):
        from filelock import FileLock

        from redact import generate_outputs, paths

        gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, gen, None)
        out = paths.output_responses_csv(tmp_path)
        out.parent.mkdir(parents=True)
        lock = str(out.with_name("output_responses.state.jsonl")) + ".lock"
        with FileLock(lock), pytest.raises(RuntimeError, match="locked"):
            generate_outputs(data_dir=tmp_path, inputs=self._six_inputs(),
                             model="gen-model", check_outputs=False, verbose=False)
        assert gen.batches == []
        assert [p for p in out.parent.iterdir() if p.suffix != ".lock"] == []

    @pytest.mark.usefixtures("no_length_floor")
    def test_a_retry_shares_a_call_with_a_new_unit(self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths

        gen = _TokenBackend(
            lambda t, text: f"better {t}" if "[FEEDBACK" in text else f"answer {t}",
            "gen-model")
        # zq1's first answer is rejected, everything else accepted.
        check = _TokenBackend(
            lambda t, text: "No: weak" if "answer zq1" in text else "Yes", "check-model")
        self._clients(monkeypatch, gen, check)
        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(6)],
            "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", batch_size=2, max_attempts=2,
                         verbose=False)

        # A chunk loop sends the retry alone ([2, 1, 2, 2]); the pool fills its
        # call with the next input.
        assert [len(b) for b in gen.batches] == [2, 2, 2, 1]
        assert ["zq1*", "zq4"] in gen.batches
        df = pd.read_csv(paths.output_responses_csv(tmp_path))
        assert len(df) == 6 and df["accepted"].all()
        assert df.set_index("input_prompt").loc["zq1", "output_response"] == "better zq1"

    def test_a_failed_call_keeps_finished_rows_and_resumes_the_rest(self, tmp_path, monkeypatch):
        """A dispatch failure stops the stage; the units already finished are
        flushed and acked on the way out, and only the rest run again."""
        from redact import generate_outputs, paths

        class _DiesOnThirdCall(_TokenBackend):
            def generate(self, messages_list, **kwargs):
                if len(self.batches) == 2:
                    raise RuntimeError("engine died")
                return super().generate(messages_list, **kwargs)

        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(10)],
            "category": ["Cyber"] * 10, "entry_type": ["harmful"] * 10,
        })
        self._clients(monkeypatch, _DiesOnThirdCall(lambda t, text: f"answer {t}", "gen-model"), None)
        with pytest.raises(RuntimeError, match="engine died"):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, batch_size=2, verbose=False)
        out = paths.output_responses_csv(tmp_path)
        ledger = out.with_name("output_responses.state.jsonl")
        assert len(pd.read_csv(out)) == 4                       # two calls of two finished
        opening, *acks = ledger.read_text(encoding="utf-8").splitlines()
        assert json.loads(opening) == {"_saved": {}} and len(acks) == 4

        second = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, second, None)
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_outputs=False, batch_size=2, verbose=False)
        assert sum(len(b) for b in second.batches) == 6
        df = pd.read_csv(out)
        assert sorted(df["input_prompt"]) == sorted(inputs["sample"])

    def test_resume_after_a_failed_append_runs_only_unfinished_units(
            self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths

        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(70)],
            "category": ["Cyber"] * 70, "entry_type": ["harmful"] * 70,
        })
        real_to_csv = pd.DataFrame.to_csv
        appends = []

        def failing_second_append(self, *args, **kwargs):
            appends.append(len(self))
            if len(appends) == 2:
                raise OSError("disk full")
            return real_to_csv(self, *args, **kwargs)

        first = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, first, None)
        monkeypatch.setattr(pd.DataFrame, "to_csv", failing_second_append)
        with pytest.raises(OSError, match="disk full"):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, batch_size=4, verbose=False)
        monkeypatch.setattr(pd.DataFrame, "to_csv", real_to_csv)
        out = paths.output_responses_csv(tmp_path)
        assert appends == [32, 32] and len(pd.read_csv(out)) == 32

        second = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, second, None)
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_outputs=False, batch_size=4, verbose=False)

        df = pd.read_csv(out)
        assert sum(len(b) for b in second.batches) == 70 - 32
        assert sorted(df["input_prompt"]) == sorted(inputs["sample"])

    def test_a_prompt_too_long_is_a_rejected_row_and_is_not_sent_again(
            self, tmp_path, monkeypatch, caplog):
        from redact import generate_outputs, paths
        from redact.dataset import Ledger, content_id

        tokens = ["zq0", "zq1", "zq2"]
        inputs = pd.DataFrame({
            "sample": tokens, "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })

        def run():
            gen = _LaneBackend("gen-model", too_long={"zq1"})
            self._clients(monkeypatch, gen, None)
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, verbose=False, axes=False)
            return gen

        with caplog.at_level(logging.WARNING):
            first = run()
        assert sorted(first.sent) == tokens
        assert [r.getMessage() for r in caplog.records] == [
            f"Input {content_id('zq1')} rejected, its prompt is too long: "
            "zq1 is over the context"]

        out = paths.output_responses_csv(tmp_path)
        df = pd.read_csv(out, dtype=str, keep_default_na=False)
        rows = {r["input_prompt"]: r for r in df.to_dict("records")}
        assert rows["zq1"] == {
            "input_id": content_id("zq1"), "input_prompt": "zq1", "category": "Cyber",
            "subcategory": "", "entry_type": "harmful", "model": "gen-model",
            "source": "",
            "sample_id": content_id(f"{content_id('zq1')}:0:prompt too long"),
            "output_response": "",
            "accepted": "False",
            "rejection_reason": "prompt too long: zq1 is over the context",
            "iteration": "0", "response_mode": "harmful", "output_label": "harmful",
            "response_label": "harmful_harmful",
        }
        assert len(df) == len(tokens)
        assert {t: (rows[t]["output_response"], rows[t]["accepted"])
                for t in ("zq0", "zq2")} == {
            "zq0": ("answer zq0", "True"), "zq2": ("answer zq2", "True")}
        ledger = Ledger.sidecar(out, key_fields=("input_id",),
                                casters={"input_id": str})
        assert ledger.completed() == {content_id(t) for t in tokens}

        assert run().sent == []
        assert len(pd.read_csv(out)) == len(tokens)

    def test_rows_of_prompts_too_long_each_have_their_own_stable_sample_id(
            self, tmp_path, monkeypatch):
        """No response to hash, so the id comes from the input's: the build,
        which drops a row whose sample_id an earlier part wrote, keeps them all."""
        from redact import generate_outputs, paths
        from redact.dataset import content_id

        tokens = ["zq0", "zq1"]
        inputs = pd.DataFrame({
            "sample": tokens, "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })

        def ids(data_dir):
            self._clients(monkeypatch, _LaneBackend("gen-model", too_long=tokens), None)
            generate_outputs(data_dir=data_dir, inputs=inputs, model="gen-model",
                             check_outputs=False, verbose=False)
            df = pd.read_csv(paths.output_responses_csv(data_dir), dtype=str)
            return dict(zip(df["input_prompt"], df["sample_id"], strict=True))

        first = ids(tmp_path / "a")
        assert len(set(first.values())) == len(tokens)
        assert not set(first.values()) & {content_id(""), *map(content_id, tokens)}
        assert ids(tmp_path / "b") == first

    @pytest.mark.usefixtures("no_length_floor")
    def test_another_unit_error_still_stops_the_stage(self, tmp_path, monkeypatch):
        """Only a prompt that is too long becomes a row. This unit fails
        building its check, after its response arrived."""
        import redact.content_moderation.outputs as G
        from redact import generate_outputs, paths
        from redact.dataset import Ledger

        def no_checker(**kwargs):
            def build(original, text):
                raise KeyError("no checker")
            return build

        monkeypatch.setattr(G, "build_output_quality_checker", no_checker)
        gen = _LaneBackend("gen-model")
        self._clients(monkeypatch, gen, MockBackend("Yes"))
        inputs = pd.DataFrame({
            "sample": ["zq0"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        with pytest.raises(KeyError, match="no checker"):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_model="check-model", verbose=False)
        assert gen.sent == ["zq0"]
        out = paths.output_responses_csv(tmp_path)
        assert not out.exists()
        ledger = Ledger.sidecar(out, key_fields=("input_id",),
                                casters={"input_id": str})
        assert ledger.completed() == set()
