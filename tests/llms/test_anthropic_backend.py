"""Tests for AnthropicBackend with mocked Anthropic client."""

import os
import sys
from unittest.mock import patch, MagicMock

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
    """Create an AnthropicBackend with a mocked client."""
    from redact.llms.anthropic_backend import AnthropicBackend
    backend = AnthropicBackend(api_key="test-key")
    return backend


class TestAnthropicBackendInit:
    def test_from_env(self, mock_anthropic_module):
        from redact.llms.anthropic_backend import AnthropicBackend
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            backend = AnthropicBackend.from_env()
            mock_anthropic_module.Anthropic.assert_called_with(api_key="test-key")

    def test_from_env_missing_raises(self, mock_anthropic_module):
        from redact.llms.anthropic_backend import AnthropicBackend
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(KeyError):
                AnthropicBackend.from_env()

    def test_backend_name(self, anthropic_backend):
        assert anthropic_backend.backend_name == "anthropic"

    def test_repr(self, anthropic_backend):
        assert "AnthropicBackend" in repr(anthropic_backend)


class TestAnthropicGenerate:
    def test_basic_generate(self, anthropic_backend):
        result = anthropic_backend.generate(
            [{"role": "user", "content": "hi"}], "claude-opus-4-6"
        )
        assert result == "claude response"

    def test_system_message_extraction(self, anthropic_backend):
        anthropic_backend.generate(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            "claude-opus-4-6",
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_no_system_message(self, anthropic_backend):
        anthropic_backend.generate(
            [{"role": "user", "content": "hi"}],
            "claude-opus-4-6",
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_multiple_system_messages_joined(self, anthropic_backend):
        anthropic_backend.generate(
            [
                {"role": "system", "content": "Part 1."},
                {"role": "system", "content": "Part 2."},
                {"role": "user", "content": "hi"},
            ],
            "claude-opus-4-6",
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Part 1.\n\nPart 2."

    def test_resolves_defaults(self, anthropic_backend):
        anthropic_backend.generate(
            [{"role": "user", "content": "hi"}], "claude-opus-4-6"
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 300  # from model config

    def test_explicit_overrides(self, anthropic_backend):
        anthropic_backend.generate(
            [{"role": "user", "content": "hi"}],
            "claude-opus-4-6",
            max_tokens=1000,
            temperature=0.3,
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.3

    def test_supports_system_prompt_false_folds_into_user_message(self, anthropic_backend):
        from redact.llms.model_config import MODEL_REGISTRY, register_model

        name = "_test_no_system_prompt_claude"
        try:
            register_model(name, rpm=5, backend_type="anthropic", supports_system_prompt=False)
            anthropic_backend.generate(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ],
                name,
            )
            call_kwargs = anthropic_backend._client.messages.create.call_args[1]
            # folded away before the system/conversation split runs, so there's
            # no separate `system` param — it's part of the one user message.
            assert "system" not in call_kwargs
            assert call_kwargs["messages"] == [
                {"role": "user", "content": "You are helpful.\n\nhi"}
            ]
        finally:
            MODEL_REGISTRY.pop(name, None)
