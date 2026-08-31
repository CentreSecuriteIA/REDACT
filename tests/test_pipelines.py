"""Tests for high-level pipeline functions."""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from redact.llms.backends import ComputeConfig
from redact.pipelines import (
    _get_client,
    create_taxonomy,
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


class TestGetClientHelper:
    def test_returns_ready_client(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test"}):
            client = _get_client(model="venice-uncensored")
        assert client.model == "venice-uncensored"
        # Fully wired: the budget is bound to the backend, and the limiter
        # lives inside the client rather than alongside it.
        assert client.backend.rpm == 75

    def test_resolves_via_get_client(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test"}):
            with patch("redact.pipelines.ModelClient.create") as mock_get:
                mock_get.return_value = MagicMock()
                _get_client(model="venice-uncensored")
                mock_get.assert_called_once_with("venice-uncensored")


class TestStandaloneInputsDeprecation:
    def test_standalone_generate_inputs_emits_deprecation_warning(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_inputs
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend(), m))
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
        import redact.pipelines as P
        from redact import generate_outputs, paths
        from redact.dataset import Ledger, Manifest
        from redact.dataset.io import _hash_text
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        out = paths.output_responses_csv(tmp_path)
        # Pre-seed the ledger as if "p one" was already completed.
        Ledger.sidecar(out, key_fields=("input_id",), casters={"input_id": str}).record(
            [{"input_id": _hash_text("p one")}]
        )

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
        import redact.pipelines as P
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
        expected = {f'{row["input_id"]}/output' for _, row in df.iterrows()}
        assert seen == expected
        # never a CSV column, regardless of capture
        assert "internals_id" not in df.columns

    def test_check_internals_id_is_input_id_slash_val_out(self, tmp_path, monkeypatch):
        import redact.pipelines as P
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
        assert check_backend.calls[0]["internals_id"].endswith("/val_out")
        assert gen_backend.calls[0]["internals_id"].endswith("/output")
        # same input_id root for both stages
        gen_root = gen_backend.calls[0]["internals_id"].split("/")[0]
        check_root = check_backend.calls[0]["internals_id"].split("/")[0]
        assert gen_root == check_root

    def test_check_internals_not_requested_when_check_backend_lacks_support(self, tmp_path, monkeypatch):
        import redact.pipelines as P
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
        import redact.pipelines as P
        from redact import generate_outputs
        from redact.dataset.io import _hash_text
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("a plain answer"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        assert "sample_id" in df.columns
        for _, row in df.iterrows():
            assert row["sample_id"] == _hash_text(row["output_response"])
            assert row["sample_id"] != row["input_id"]
