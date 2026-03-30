"""Tests for high-level pipeline functions."""

import json
import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from redact.pipelines import (
    create_taxonomy,
    _get_backend,
    _fn_needs_backend,
    _fn_needs_benign,
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


class TestFnInspection:
    def test_fn_needs_backend_true(self):
        def func(prompt, backend, model):
            pass
        assert _fn_needs_backend(func) is True

    def test_fn_needs_backend_false(self):
        def func(prompt):
            pass
        assert _fn_needs_backend(func) is False

    def test_fn_needs_benign_true(self):
        def func(prompt, benign_data):
            pass
        assert _fn_needs_benign(func) is True

    def test_fn_needs_benign_false(self):
        def func(prompt):
            pass
        assert _fn_needs_benign(func) is False
