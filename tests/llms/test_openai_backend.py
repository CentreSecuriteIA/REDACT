"""Tests for OpenAIBackend with mocked OpenAI client."""

import os
from unittest.mock import MagicMock, patch

import pytest

from redact.llms.backends import OpenAIBackend
from redact.llms.model_config import get_model_config


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
def openai_backend(mock_openai_client):
    """Create a configured OpenAIBackend with a mocked SDK client."""
    backend = OpenAIBackend(
        "venice-uncensored", api_key="test-key", base_url="https://test.api",
        default_max_tokens=2000,
    )
    backend._client = mock_openai_client
    return backend


class TestOpenAIBackendInit:
    def test_from_config_uses_the_entrys_endpoint(self):
        """The registry entry's credentials and base_url reach the SDK."""
        config = get_model_config("venice-uncensored")
        with patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}):
            with patch("redact.llms.backends.openai.openai.OpenAI") as mock_cls:
                OpenAIBackend.clear_cache()
                backend = OpenAIBackend.from_config(config)
                mock_cls.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://api.venice.ai/api/v1",
                )
        assert backend.rpm == config.api.rpm
        assert backend.max_workers == config.api.recommended_max_workers
        OpenAIBackend.clear_cache()

    def test_sdk_client_shared_between_models_on_one_endpoint(self):
        """Backends are per-model; the HTTP pool underneath them is not."""
        with patch("redact.llms.backends.openai.openai.OpenAI"):
            OpenAIBackend.clear_cache()
            a = OpenAIBackend("model-a", api_key="k", base_url="https://one.api/v1")
            b = OpenAIBackend("model-b", api_key="k", base_url="https://one.api/v1")
            assert a.model != b.model
            assert a._client is b._client
        OpenAIBackend.clear_cache()

    def test_from_config_missing_api_key_raises(self):
        config = get_model_config("venice-uncensored")
        env = os.environ.copy()
        env.pop("VENICE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(KeyError):
                OpenAIBackend.from_config(config)

    def test_backend_name(self, openai_backend):
        assert openai_backend.backend_name == "openai"

    def test_repr(self, openai_backend):
        assert "test.api" in repr(openai_backend)


class TestOpenAIGenerate:
    """The backend is a configured model: the model name and extra_body come
    from construction, and only messages/system prompt/overrides are per call.
    """

    def test_basic_generate(self, openai_backend, mock_openai_client):
        result = openai_backend.generate(
            [[{"role": "user", "content": "hi"}]]
        )
        assert result == ["venice response"]
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_batch_preserves_order(self, openai_backend, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="first"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="second"))]),
        ]
        result = openai_backend.generate(
            [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ]
        )
        assert result == ["first", "second"]

    def test_explicit_overrides_beat_the_models_defaults(self, openai_backend, mock_openai_client):
        openai_backend.generate(
            [[{"role": "user", "content": "hi"}]],
            max_tokens=500,
            temperature=0.5,
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.5

    def test_bound_extra_body_sent_on_every_call(self, mock_openai_client):
        """extra_body is a fact of the model, bound once, not a call argument."""
        backend = OpenAIBackend(
            "venice-uncensored", api_key="k", base_url="https://test.api",
            extra_body={"custom_param": True},
        )
        backend._client = mock_openai_client
        backend.generate([[{"role": "user", "content": "hi"}]])
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"] == {"custom_param": True}

    def test_no_extra_body_key_when_model_has_none(self, openai_backend, mock_openai_client):
        openai_backend.generate([[{"role": "user", "content": "hi"}]])
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert "extra_body" not in call_kwargs

    def test_passes_kwargs(self, openai_backend, mock_openai_client):
        openai_backend.generate(
            [[{"role": "user", "content": "hi"}]],
            top_p=0.9,
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    def test_system_prompt_prepended_as_system_message(self, openai_backend, mock_openai_client):
        openai_backend.generate(
            [[{"role": "user", "content": "hi"}]],
            system_prompts=["Sys."],
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "hi"},
        ]

    def test_no_system_prompt_leaves_messages_unchanged(self, openai_backend, mock_openai_client):
        openai_backend.generate(
            [[{"role": "user", "content": "hi"}]],
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
