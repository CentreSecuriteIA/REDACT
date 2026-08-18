"""Tests for high-level pipeline functions."""

import json
import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from redact.pipelines import (
    create_taxonomy,
    _get_backend,
    _output_state_path,
    _read_output_state,
    _append_output_state,
)


class TestCreateTaxonomy:
    def test_creates_json(self, tmp_path):
        tax = create_taxonomy(
            "test_tax",
            categories={"CatA": "Description A", "CatB": {"description": "B"}},
            description="Test taxonomy",
            taxonomy_dir=tmp_path,
        )
        path = tmp_path / "test_tax.json"
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["name"] == "test_tax"
        assert "CatA" in loaded["categories"]
        assert loaded["categories"]["CatA"]["description"] == "Description A"

    def test_string_categories_normalized(self, tmp_path):
        tax = create_taxonomy(
            "norm_test",
            categories={"Cat": "just a string"},
            taxonomy_dir=tmp_path,
        )
        assert tax["categories"]["Cat"] == {"description": "just a string"}

    def test_aliases_and_groups(self, tmp_path):
        tax = create_taxonomy(
            "full_test",
            categories={"A": "desc"},
            aliases={"B": "A"},
            groups={"g1": ["A"]},
            taxonomy_dir=tmp_path,
        )
        assert tax["aliases"] == {"B": "A"}
        assert tax["groups"] == {"g1": ["A"]}

    def test_returns_dict(self, tmp_path):
        result = create_taxonomy("ret_test", {"X": "Y"}, taxonomy_dir=tmp_path)
        assert isinstance(result, dict)
        assert result["name"] == "ret_test"


class TestGetBackendHelper:
    def test_passthrough(self):
        mock_backend = MagicMock()
        backend, limiter = _get_backend(backend=mock_backend, model="m")
        assert backend is mock_backend

    def test_auto_resolve(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test"}):
            with patch("redact.pipelines.get_backend") as mock_get:
                mock_get.return_value = MagicMock()
                backend, limiter = _get_backend(backend=None, model="venice-uncensored")
                mock_get.assert_called_once_with("venice-uncensored")


class TestOutputResumeState:
    """Resume-state ledger used by generate_outputs() (#5)."""

    def test_state_path_is_sidecar(self, tmp_path):
        out = tmp_path / "output_responses.csv"
        sp = _output_state_path(out)
        assert sp.name == "output_responses.state.jsonl"
        assert sp.parent == out.parent

    def test_read_missing_returns_empty(self, tmp_path):
        assert _read_output_state(tmp_path / "nope.state.jsonl") == set()

    def test_append_then_read_roundtrip(self, tmp_path):
        sp = tmp_path / "s.state.jsonl"
        _append_output_state(sp, ["a", "b"])
        _append_output_state(sp, ["c"])
        assert _read_output_state(sp) == {"a", "b", "c"}

    def test_read_ignores_malformed_lines(self, tmp_path):
        sp = tmp_path / "s.state.jsonl"
        sp.write_text(
            '{"input_id": "x"}\nnot json\n{"no_id": 1}\n{"input_id": "y"}\n',
            encoding="utf-8",
        )
        assert _read_output_state(sp) == {"x", "y"}


class TestStandaloneInputsDeprecation:
    def test_standalone_generate_inputs_emits_deprecation_warning(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_inputs
        from tests.conftest import MockBackend

        monkeypatch.setattr(P, "get_backend", lambda m: MockBackend())
        # empty taxonomy -> no categories -> warning fires, then a fast empty return
        with pytest.warns(DeprecationWarning, match="constitution-seeded"):
            generate_inputs(
                data_dir=tmp_path, taxonomy={"categories": {}},
                num_categories=0, verbose=False,
            )


class TestGenerateOutputsManifest:
    """generate_outputs writes a plan manifest before the batch loop (offline)."""

    def test_manifest_written_with_one_row_per_input(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs, paths
        from redact.dataset import Manifest
        from tests.conftest import MockBackend

        monkeypatch.setattr(P, "get_backend", lambda m: MockBackend("a plain answer"))
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
        import redact.pipelines as P
        from redact import generate_outputs, paths
        from redact.dataset import Manifest
        from redact.dataset.io import _hash_text
        from redact.pipelines import _output_state_path, _append_output_state
        from tests.conftest import MockBackend

        monkeypatch.setattr(P, "get_backend", lambda m: MockBackend("ans"))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        out = paths.output_responses_csv(tmp_path)
        # Pre-seed the ledger as if "p one" was already completed.
        _append_output_state(_output_state_path(out), [_hash_text("p one")])

        generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                         resume=True, verbose=False)

        assert len(Manifest.sidecar(out).load()) == 3   # full plan (all 3 units)
        assert len(pd.read_csv(out)) == 2               # only the 2 pending generated


class TestGenerateOutputsInternals:
    """Internals capture is a folder-per-input_id side channel — no CSV column,
    ever (see .claude/introspection_backend_plan.md's non-interference principle).
    """

    def test_no_internals_kwarg_for_default_backend(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend

        backend = MockBackend("ans")
        monkeypatch.setattr(P, "get_backend", lambda m: backend)
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)
        assert "internals_id" not in backend.calls[0]
        assert "internals_id" not in df.columns and "internals_ids" not in df.columns

    def test_gen_internals_id_is_input_id_slash_output(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend

        class _InternalsBackend(MockBackend):
            @property
            def supports_internals(self) -> bool:
                return True

        backend = _InternalsBackend("ans")
        monkeypatch.setattr(P, "get_backend", lambda m: backend)
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        seen = {c["internals_id"] for c in backend.calls}
        expected = {f'{row["input_id"]}/output' for _, row in df.iterrows()}
        assert seen == expected
        # never a CSV column, regardless of capture
        assert "internals_id" not in df.columns

    def test_check_internals_id_is_input_id_slash_val_out(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend

        class _InternalsBackend(MockBackend):
            @property
            def supports_internals(self) -> bool:
                return True

        gen_backend = _InternalsBackend("the answer")
        check_backend = _InternalsBackend("Yes, fine")
        backends = {"gen-model": gen_backend, "check-model": check_backend}
        monkeypatch.setattr(P, "get_backend", lambda m: backends[m])
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        generate_outputs(
            data_dir=tmp_path, inputs=inputs, model="gen-model", check_model="check-model",
            verbose=False,
        )
        assert check_backend.calls[0]["internals_id"].endswith("/val_out")
        assert gen_backend.calls[0]["internals_id"].endswith("/output")
        # same input_id root for both stages
        gen_root = gen_backend.calls[0]["internals_id"].split("/")[0]
        check_root = check_backend.calls[0]["internals_id"].split("/")[0]
        assert gen_root == check_root

    def test_check_internals_not_requested_when_check_backend_lacks_support(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend

        class _InternalsBackend(MockBackend):
            @property
            def supports_internals(self) -> bool:
                return True

        gen_backend = _InternalsBackend("the answer")
        check_backend = MockBackend("Yes, fine")  # no internals support
        backends = {"gen-model": gen_backend, "check-model": check_backend}
        monkeypatch.setattr(P, "get_backend", lambda m: backends[m])
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
        import redact.pipelines as P
        from redact import generate_outputs
        from redact.dataset.io import _hash_text
        from tests.conftest import MockBackend

        monkeypatch.setattr(P, "get_backend", lambda m: MockBackend("a plain answer"))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        assert "sample_id" in df.columns
        for _, row in df.iterrows():
            assert row["sample_id"] == _hash_text(row["output_response"])
            assert row["sample_id"] != row["input_id"]
