"""Tests for AnthropicBackend with mocked Anthropic client."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_anthropic_module():
    """Mock the anthropic module so it can be imported inside __init__."""
    mock_mod = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "claude response"
    mock_mod.Anthropic.return_value.messages.create.return_value = mock_response
    with patch.dict(sys.modules, {"anthropic": mock_mod}):
        yield mock_mod


@pytest.fixture()
def anthropic_backend(mock_anthropic_module):
    """Create a configured AnthropicBackend with a mocked SDK client."""
    from redact.llms.backends import AnthropicBackend
    AnthropicBackend.clear_cache()
    backend = AnthropicBackend(
        "claude-opus-4-6", api_key="test-key", default_max_tokens=300,
    )
    yield backend
    AnthropicBackend.clear_cache()


class TestAnthropicBackendInit:
    def test_from_config_uses_the_entrys_credentials(self, mock_anthropic_module):
        """The registry entry's key env var and budget reach the backend."""
        from redact.llms.backends import AnthropicBackend
        from redact.llms.model_config import get_model_config

        config = get_model_config("claude-opus-4-6")
        AnthropicBackend.clear_cache()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            backend = AnthropicBackend.from_config(config)
            mock_anthropic_module.Anthropic.assert_called_with(api_key="test-key")
        assert backend.rpm == config.api.rpm
        assert backend.max_workers == config.api.recommended_max_workers
        AnthropicBackend.clear_cache()

    def test_from_config_missing_api_key_raises(self, mock_anthropic_module):
        from redact.llms.backends import AnthropicBackend
        from redact.llms.model_config import get_model_config

        config = get_model_config("claude-opus-4-6")
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(KeyError):
                AnthropicBackend.from_config(config)

    def test_backend_name(self, anthropic_backend):
        assert anthropic_backend.backend_name == "anthropic"

    def test_repr(self, anthropic_backend):
        assert "AnthropicBackend" in repr(anthropic_backend)


class TestAnthropicGenerate:
    """The backend is a configured model: the model name and max_tokens
    default come from construction, and only messages/system prompt/overrides
    are per call.
    """

    def test_models_default_max_tokens_used_when_not_overridden(self, anthropic_backend):
        anthropic_backend.generate([[{"role": "user", "content": "hi"}]])
        assert anthropic_backend._client.messages.create.call_args[1]["max_tokens"] == 300

    def test_basic_generate(self, anthropic_backend):
        result = anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]]
        )
        assert result == ["claude response"]

    def test_batch_preserves_order(self, anthropic_backend):
        anthropic_backend._client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(text="first")]),
            MagicMock(content=[MagicMock(text="second")]),
        ]
        result = anthropic_backend.generate(
            [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ]
        )
        assert result == ["first", "second"]

    def test_system_prompt_passed_as_system_param(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]],
            system_prompts=["You are helpful."],
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_no_system_prompt_omits_system_key(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]],
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_explicit_overrides_forwarded(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]],
            max_tokens=1000,
            temperature=0.3,
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.3

    def test_omits_temperature_when_not_given(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]],
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs

    def test_strips_extra_message_keys(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi", "extra_field": "drop me"}]],
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
