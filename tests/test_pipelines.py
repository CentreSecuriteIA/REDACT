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
            config_dir=tmp_path,
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
            config_dir=tmp_path,
        )
        assert tax["categories"]["Cat"] == {"description": "just a string"}

    def test_aliases_and_groups(self, tmp_path):
        tax = create_taxonomy(
            "full_test",
            categories={"A": "desc"},
            aliases={"B": "A"},
            groups={"g1": ["A"]},
            config_dir=tmp_path,
        )
        assert tax["aliases"] == {"B": "A"}
        assert tax["groups"] == {"g1": ["A"]}

    def test_returns_dict(self, tmp_path):
        result = create_taxonomy("ret_test", {"X": "Y"}, config_dir=tmp_path)
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
