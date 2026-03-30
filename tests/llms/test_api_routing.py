"""Tests for backend routing and caching."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from redact.llms.api import get_backend, clear_backend_cache, _infer_backend_type, _backend_cache
from redact.llms.model_config import MODEL_REGISTRY, register_model


class TestInferBackendType:
    def test_claude_to_anthropic(self):
        assert _infer_backend_type("claude-opus-4-6") == "anthropic"
        assert _infer_backend_type("claude-sonnet-4-6") == "anthropic"

    def test_unknown_to_venice(self):
        assert _infer_backend_type("some-random-model") == "venice"
        assert _infer_backend_type("deepseek-v3.2") == "venice"


class TestGetBackend:
    def setup_method(self):
        clear_backend_cache()

    def teardown_method(self):
        clear_backend_cache()
        for name in ["test-routing-vllm"]:
            MODEL_REGISTRY.pop(name, None)

    def test_venice_backend(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}):
            with patch("redact.llms.venice_backend.openai.OpenAI"):
                backend = get_backend("venice-uncensored")
                assert backend.backend_name == "venice"

    def test_anthropic_backend(self):
        mock_anthropic = MagicMock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                backend = get_backend("claude-opus-4-6")
                assert backend.backend_name == "anthropic"

    def test_cache_by_type(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}):
            with patch("redact.llms.venice_backend.openai.OpenAI"):
                b1 = get_backend("venice-uncensored")
                b2 = get_backend("deepseek-v3.2")
                assert b1 is b2  # same API type = same cached instance

    def test_vllm_missing_hf_model_id_raises(self):
        register_model("test-routing-vllm", rpm=999, backend_type="vllm")
        with pytest.raises(ValueError, match="hf_model_id"):
            get_backend("test-routing-vllm")


class TestClearBackendCache:
    def test_clears(self):
        _backend_cache["test"] = MagicMock()
        assert len(_backend_cache) > 0
        clear_backend_cache()
        assert len(_backend_cache) == 0
