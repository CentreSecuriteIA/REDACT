"""Tests for AnthropicBackend with mocked Anthropic client."""

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from redact.llms import observe
from tests.conftest import first_cut_warning


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
            mock_anthropic_module.Anthropic.assert_called_with(
                api_key="test-key", max_retries=2)
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

    def test_a_cache_cleared_during_the_lookup_does_not_raise(self):
        """A concurrent clear_cache() can empty the cache right after a
        lookup finds the client."""
        import redact.llms.backends.anthropic as anthropic_module

        class _ClearedAfterContains(dict):
            def __contains__(self, key):
                found = super().__contains__(key)
                self.clear()
                return found

        client = object()
        cache = _ClearedAfterContains({("k", 2, None): client})
        with patch.object(anthropic_module, "_sdk_clients", cache):
            assert anthropic_module._sdk_client(
                "k", max_retries=2, timeout_s=None) is client

    def test_backend_name(self, anthropic_backend):
        assert anthropic_backend.backend_name == "anthropic"

    def test_repr(self, anthropic_backend):
        assert "AnthropicBackend" in repr(anthropic_backend)


class TestAnthropicSdkSettings:
    """The setup's retries and timeout reach the SDK client."""

    def _built_with(self, mock_anthropic_module, monkeypatch, **api):
        from redact.llms.backends import AnthropicBackend
        from redact.llms.model_config import APIConfig, ModelConfig

        monkeypatch.setenv("TEST_API_KEY", "k")
        config = ModelConfig(name="m", api=APIConfig(
            backend_type="anthropic", api_key_env="TEST_API_KEY", rpm=10, **api))
        AnthropicBackend.clear_cache()
        AnthropicBackend.from_config(config)
        AnthropicBackend.clear_cache()
        return mock_anthropic_module.Anthropic.call_args.kwargs

    def test_the_setups_retries_and_timeout_reach_the_client(
        self, mock_anthropic_module, monkeypatch
    ):
        assert self._built_with(mock_anthropic_module, monkeypatch,
                                max_retries=5, timeout_s=30) == {
            "api_key": "k", "max_retries": 5, "timeout": 30}

    def test_the_default_setup_keeps_the_sdks_retries_and_timeout(
        self, mock_anthropic_module, monkeypatch
    ):
        assert self._built_with(mock_anthropic_module, monkeypatch) == {
            "api_key": "k", "max_retries": 2}

    def test_models_share_a_client_only_with_one_setting(self, mock_anthropic_module):
        from redact.llms.backends import AnthropicBackend

        mock_anthropic_module.Anthropic.side_effect = lambda **kwargs: MagicMock()
        AnthropicBackend.clear_cache()
        a = AnthropicBackend("model-a", api_key="k")
        b = AnthropicBackend("model-b", api_key="k")
        c = AnthropicBackend("model-c", api_key="k", max_retries=5)
        assert a._client is b._client
        assert c._client is not a._client
        AnthropicBackend.clear_cache()


def test_clear_cache_closes_the_sdk_clients(mock_anthropic_module):
    from redact.llms.backends import AnthropicBackend

    mock_anthropic_module.Anthropic.side_effect = lambda **kwargs: MagicMock()
    AnthropicBackend.clear_cache()
    a = AnthropicBackend("model-a", api_key="k")
    b = AnthropicBackend("model-b", api_key="other")
    a._client.close.side_effect = RuntimeError("already gone")
    AnthropicBackend.clear_cache()
    a._client.close.assert_called_once_with()
    b._client.close.assert_called_once_with()
    assert AnthropicBackend("model-a", api_key="k")._client is not a._client
    AnthropicBackend.clear_cache()


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
        anthropic_backend.generate([[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]])
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
        # The SDK's create() has no temperature keyword from 1.0.
        assert "temperature" not in call_kwargs
        assert call_kwargs["extra_body"] == {"temperature": 0.3}

    def test_a_temperature_joins_the_callers_extra_body(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]], temperature=0.3,
            extra_body={"metadata": {"user_id": "u"}})
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["extra_body"] == {
            "metadata": {"user_id": "u"}, "temperature": 0.3}

    def test_omits_temperature_when_not_given(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]],
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs and "extra_body" not in call_kwargs

    def test_a_model_without_temperature_sends_none(self, mock_anthropic_module):
        """Claude Opus 4.7 and later refuse any sampling setting."""
        from redact.llms.backends import AnthropicBackend

        AnthropicBackend.clear_cache()
        backend = AnthropicBackend(
            "claude-new", api_key="k", supports_temperature=False)
        backend.generate([[{"role": "user", "content": "hi"}]], temperature=0.15)
        call_kwargs = backend._client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs and "extra_body" not in call_kwargs
        AnthropicBackend.clear_cache()

    def test_api_version_2026_10_21_takes_no_temperature(
            self, anthropic_backend, monkeypatch, caplog):
        """The SDK sends it from 2.0, and it refuses a temperature on every
        model: none is sent, with one warning."""
        import redact.llms.backends.anthropic as anthropic_module

        monkeypatch.setattr(anthropic_module, "_sampling_warned", False)
        anthropic_backend._client.default_headers = {"anthropic-version": "2026-10-21"}
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends.anthropic"):
            for _ in range(2):
                anthropic_backend.generate(
                    [[{"role": "user", "content": "hi"}]], temperature=0.15)
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs and "extra_body" not in call_kwargs
        (line,) = [r.getMessage() for r in caplog.records]
        assert line.startswith("claude-opus-4-6: the anthropic SDK sends API "
                               "version 2026-10-21 or later")

    def test_api_version_2023_06_01_takes_one(self, anthropic_backend):
        anthropic_backend._client.default_headers = {"anthropic-version": "2023-06-01"}
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]], temperature=0.15)
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["extra_body"] == {"temperature": 0.15}

    def test_the_reply_is_its_text_blocks(self, anthropic_backend):
        """A model that thinks by default puts a thinking block, which has
        no text, before its answer."""
        thinking = MagicMock(spec=["type", "thinking"], type="thinking", thinking="")
        anthropic_backend._client.messages.create.return_value = MagicMock(
            content=[thinking, MagicMock(text="the "), MagicMock(text="answer")],
            stop_reason="end_turn", usage=None)
        assert anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]]) == ["the answer"]

    def test_strips_extra_message_keys(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi", "extra_field": "drop me"}]],
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_kwargs_cannot_retarget_the_call(self, anthropic_backend):
        anthropic_backend.generate(
            [[{"role": "user", "content": "hi"}]],
            model="another-model", messages=[{"role": "user", "content": "other"}],
        )
        call_kwargs = anthropic_backend._client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


class TestAnthropicEmptyReply:
    """An empty reply is returned as "" with its stop reason on record."""

    HI = [[{"role": "user", "content": "hi"}]]
    LOGGER = "redact.llms.backends.anthropic"

    @pytest.fixture()
    def events(self):
        seen = []
        observe.set_emitter(seen.append)
        yield seen
        observe.set_emitter(None)

    @pytest.mark.parametrize("content", [[], None, [MagicMock(text="")]],
                             ids=["no-blocks", "content-none", "empty-text"])
    def test_empty_reply_is_returned_and_warned_about(
        self, anthropic_backend, events, caplog, content
    ):
        anthropic_backend._client.messages.create.return_value = MagicMock(
            content=content, stop_reason="refusal", usage=None)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            assert anthropic_backend.generate(self.HI) == [""]
        assert [r.getMessage() for r in caplog.records] == [
            "claude-opus-4-6 returned an empty reply (stop_reason=refusal)."
        ]
        assert events[0]["finish_reason"] == "refusal"

    def test_stop_reason_is_recorded_without_a_warning(
        self, anthropic_backend, events, caplog
    ):
        anthropic_backend._client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")], stop_reason="end_turn", usage=None)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            assert anthropic_backend.generate(self.HI) == ["ok"]
        assert caplog.records == []
        assert events[0]["finish_reason"] == "end_turn"

    def test_usage_is_recorded_for_the_one_item(self, anthropic_backend, events):
        anthropic_backend._client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")], stop_reason="end_turn",
            usage=MagicMock(input_tokens=5, output_tokens=7))
        anthropic_backend.generate(self.HI)
        assert (events[0]["in_tok"], events[0]["out_tok"], events[0]["out_tok_max"]) == (5, 7, 7)


class TestAnthropicCutReply:
    """A reply cut at max_tokens or the context window is returned as it is,
    with one warning."""

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
    def test_a_cut_reply_warns_once(self, anthropic_backend, caplog, stop_reason):
        anthropic_backend._client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="partial")], stop_reason=stop_reason, usage=None)
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert anthropic_backend.generate(
                [[{"role": "user", "content": "hi"}]], max_tokens=50) == ["partial"]
        assert [r.getMessage() for r in caplog.records] == [
            first_cut_warning("claude-opus-4-6", 50)]


class _BadRequest(Exception):
    """What the SDK raises for a 400: its message holds the response body."""

    status_code = 400

    def __init__(self, message: str):
        super().__init__(
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            f"'invalid_request_error', 'message': '{message}'}}}}")


class TestAnthropicPromptTooLong:
    """A prompt the API rejects as over the model's context raises
    PromptTooLongError; any other rejection is raised as it is."""

    HI = [[{"role": "user", "content": "hi"}]]

    @pytest.fixture()
    def events(self):
        seen = []
        observe.set_emitter(seen.append)
        yield seen
        observe.set_emitter(None)

    @pytest.mark.parametrize("message", [
        "prompt is too long: 210000 tokens > 200000 maximum",
        "input length and `max_tokens` exceed context limit: 190000 + 20000 > "
        "200000, decrease input length or `max_tokens` and try again",
    ], ids=["prompt", "prompt-and-reply"])
    def test_a_context_rejection_raises_prompt_too_long(
        self, anthropic_backend, events, message
    ):
        from redact.exceptions import PromptTooLongError

        error = _BadRequest(message)
        anthropic_backend._client.messages.create.side_effect = error
        with pytest.raises(PromptTooLongError) as raised:
            anthropic_backend.generate(self.HI)
        assert str(raised.value) == str(error) and raised.value.__cause__ is error
        assert [event["error"] for event in events] == [f"_BadRequest: {error}"]

    def test_another_bad_request_stays_as_it_is(self, anthropic_backend):
        error = _BadRequest("messages: roles must alternate")
        anthropic_backend._client.messages.create.side_effect = error
        with pytest.raises(_BadRequest) as raised:
            anthropic_backend.generate(self.HI)
        assert raised.value is error
