"""Tests for VeniceBackend with mocked OpenAI client."""

import os
from unittest.mock import patch, MagicMock

import pytest

from redact.llms.venice_backend import VeniceBackend
from redact.llms.model_config import MODEL_REGISTRY, register_model


@pytest.fixture()
def mock_openai_client():
    """Create a mocked OpenAI client that returns a configurable response."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "venice response"
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture()
def venice_backend(mock_openai_client):
    """Create a VeniceBackend with a mocked client."""
    backend = VeniceBackend(api_key="test-key", base_url="https://test.api")
    backend._client = mock_openai_client
    return backend


class TestVeniceBackendInit:
    def test_from_env(self):
        with patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}):
            with patch("redact.llms.venice_backend.openai.OpenAI") as mock_cls:
                backend = VeniceBackend.from_env()
                mock_cls.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://api.venice.ai/api/v1",
                )

    def test_from_env_missing_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("VENICE_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(KeyError):
                    VeniceBackend.from_env()

    def test_backend_name(self, venice_backend):
        assert venice_backend.backend_name == "venice"

    def test_repr(self, venice_backend):
        assert "test.api" in repr(venice_backend)


class TestVeniceGenerate:
    def test_basic_generate(self, venice_backend, mock_openai_client):
        result = venice_backend.generate(
            [{"role": "user", "content": "hi"}], "venice-uncensored"
        )
        assert result == "venice response"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_resolves_defaults(self, venice_backend, mock_openai_client):
        venice_backend.generate(
            [{"role": "user", "content": "hi"}], "venice-uncensored"
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2000  # from model config
        assert "extra_body" in call_kwargs  # venice_parameters from config

    def test_explicit_overrides_defaults(self, venice_backend, mock_openai_client):
        venice_backend.generate(
            [{"role": "user", "content": "hi"}],
            "venice-uncensored",
            max_tokens=500,
            temperature=0.5,
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.5

    def test_merges_extra_body(self, venice_backend, mock_openai_client):
        venice_backend.generate(
            [{"role": "user", "content": "hi"}],
            "venice-uncensored",
            extra_body={"custom_param": True},
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        extra = call_kwargs["extra_body"]
        assert "venice_parameters" in extra  # from model config defaults
        assert extra["custom_param"] is True  # from explicit arg

    def test_unknown_model_no_extra_body(self, venice_backend, mock_openai_client):
        venice_backend.generate(
            [{"role": "user", "content": "hi"}],
            "unknown-model-xyz",
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert "extra_body" not in call_kwargs

    def test_passes_kwargs(self, venice_backend, mock_openai_client):
        venice_backend.generate(
            [{"role": "user", "content": "hi"}],
            "unknown-model-xyz",
            top_p=0.9,
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    def test_default_model_keeps_system_message_separate(self, venice_backend, mock_openai_client):
        # supports_system_prompt defaults to True: no folding.
        venice_backend.generate(
            [{"role": "system", "content": "Sys."}, {"role": "user", "content": "hi"}],
            "unknown-model-xyz",
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "hi"},
        ]

    def test_supports_system_prompt_false_folds_system_message(self, venice_backend, mock_openai_client):
        name = "_test_no_system_prompt_model"
        try:
            register_model(name, rpm=999, backend_type="venice", supports_system_prompt=False)
            venice_backend.generate(
                [{"role": "system", "content": "Sys."}, {"role": "user", "content": "hi"}],
                name,
            )
            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert call_kwargs["messages"] == [{"role": "user", "content": "Sys.\n\nhi"}]
        finally:
            MODEL_REGISTRY.pop(name, None)
