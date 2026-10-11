"""Tests for OpenAIBackend with mocked OpenAI client."""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redact.llms import aio, observe
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

    def test_a_cache_cleared_during_the_lookup_does_not_raise(self):
        """A concurrent clear_cache() can empty the cache right after a
        lookup finds the client."""
        import redact.llms.backends.openai as openai_module

        class _ClearedAfterContains(dict):
            def __contains__(self, key):
                found = super().__contains__(key)
                self.clear()
                return found

        client = object()
        cache = _ClearedAfterContains({("https://one.api/v1", "k"): client})
        with patch.object(openai_module, "_sdk_clients", cache):
            assert openai_module._sdk_client("k", "https://one.api/v1") is client

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
        openai_backend.generate([[
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Sys."},
        ]])
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

    def test_kwargs_cannot_retarget_the_call(self, openai_backend, mock_openai_client):
        openai_backend.generate(
            [[{"role": "user", "content": "hi"}]],
            model="another-model", messages=[{"role": "user", "content": "other"}],
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "venice-uncensored"
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


_HI = [[{"role": "user", "content": "hi"}]]


def _response(content, finish_reason):
    choice = MagicMock(finish_reason=finish_reason)
    choice.message.content = content
    return MagicMock(choices=[choice], usage=None)


@pytest.fixture()
def events():
    seen = []
    observe.set_emitter(seen.append)
    yield seen
    observe.set_emitter(None)


class TestOpenAIEmptyReply:
    """An empty reply is returned as "" with its finish reason on record."""

    @pytest.mark.parametrize("response, reason", [
        (_response(None, "content_filter"), "content_filter"),
        (_response("", "stop"), "stop"),
        (MagicMock(choices=[], usage=None), None),
        (MagicMock(choices=None, usage=None), None),
    ], ids=["no-content", "empty-content", "no-choices", "choices-none"])
    def test_empty_reply_is_returned_and_warned_about(
        self, openai_backend, mock_openai_client, events, caplog, response, reason
    ):
        mock_openai_client.chat.completions.create.return_value = response
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends.openai"):
            assert openai_backend.generate(_HI) == [""]
        assert [r.getMessage() for r in caplog.records] == [
            f"venice-uncensored returned an empty reply (finish_reason={reason})."
        ]
        assert events[0].get("finish_reason") == reason

    def test_finish_reason_is_recorded_without_a_warning(
        self, openai_backend, mock_openai_client, events, caplog
    ):
        create = mock_openai_client.chat.completions.create
        create.return_value = _response("ok", "stop")
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends.openai"):
            assert openai_backend.generate(_HI) == ["ok"]
        assert caplog.records == []
        assert events[0]["finish_reason"] == "stop"


class TestOpenAICutReply:
    """A reply cut at max_tokens is returned as it is, with one warning."""

    CUT = ("venice-uncensored: 1 of 1 replies were cut at the token limit "
           "(max_tokens=50) and are incomplete; raise max_tokens or the context length (max_model_len), or ask for shorter replies.")

    def test_a_cut_reply_warns_once(self, openai_backend, mock_openai_client, caplog):
        mock_openai_client.chat.completions.create.return_value = _response("partial", "length")
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert openai_backend.generate(_HI, max_tokens=50) == ["partial"]
        assert [r.getMessage() for r in caplog.records] == [self.CUT]

    def test_an_empty_cut_reply_warns_about_both(self, openai_backend, mock_openai_client, caplog):
        mock_openai_client.chat.completions.create.return_value = _response("", "length")
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert openai_backend.generate(_HI, max_tokens=50) == [""]
        assert [r.getMessage() for r in caplog.records] == [
            "venice-uncensored returned an empty reply (finish_reason=length).", self.CUT,
        ]


def _async_sdk(response=None, error=None):
    """A stand-in for ``openai.AsyncOpenAI`` whose call answers at once."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response, side_effect=error)
    return client


class TestOpenAIAsync:
    """agenerate() is generate() for one item, on the async SDK client."""

    def test_same_request_text_and_event_as_generate(
        self, openai_backend, mock_openai_client, events
    ):
        response = _response("ok", "stop")
        response.usage = MagicMock(prompt_tokens=812, completion_tokens=40)
        mock_openai_client.chat.completions.create.return_value = response
        openai_backend._async_client = sdk = _async_sdk(response)
        messages = [{"role": "system", "content": "Sys."},
                    {"role": "user", "content": "hi"}]

        assert openai_backend.generate([messages], max_tokens=50, top_p=0.9) == ["ok"]
        assert aio.run(
            openai_backend.agenerate(messages, max_tokens=50, top_p=0.9)) == "ok"

        sent = mock_openai_client.chat.completions.create.call_args
        assert sdk.chat.completions.create.await_args == sent
        sync_event, async_event = events          # one event per call
        del sync_event["ms"], async_event["ms"]
        assert async_event == sync_event
        assert (async_event["in_tok"], async_event["out_tok"]) == (812, 40)

    def test_an_empty_cut_reply_warns_as_generate_does(self, openai_backend, caplog):
        openai_backend._async_client = _async_sdk(_response("", "length"))
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert aio.run(openai_backend.agenerate(_HI[0], max_tokens=50)) == ""
        assert [r.getMessage() for r in caplog.records] == [
            "venice-uncensored returned an empty reply (finish_reason=length).",
            TestOpenAICutReply.CUT,
        ]

    def test_a_failed_call_is_recorded_then_re_raised(self, openai_backend, events):
        openai_backend._async_client = _async_sdk(error=RuntimeError("502"))
        with pytest.raises(RuntimeError, match="502"):
            aio.run(openai_backend.agenerate(_HI[0]))
        (event,) = events
        assert event["error"] == "RuntimeError: 502" and event["n_items"] == 1

    def test_the_async_sdk_client_is_cached_per_endpoint(self):
        """Built by the first agenerate(), shared by the backends on one
        endpoint, and dropped by clear_cache()."""
        built = []

        def build(**kwargs):
            built.append(kwargs)
            return _async_sdk(_response("ok", "stop"))

        OpenAIBackend.clear_cache()
        with patch("redact.llms.backends.openai.openai.OpenAI"), \
             patch("redact.llms.backends.openai.openai.AsyncOpenAI", side_effect=build):
            a = OpenAIBackend("model-a", api_key="k", base_url="https://one.api/v1")
            b = OpenAIBackend("model-b", api_key="k", base_url="https://one.api/v1")
            other = OpenAIBackend("model-c", api_key="k", base_url="https://two.api/v1")
            assert built == []
            for backend in (a, b, other):
                assert aio.run(backend.agenerate(_HI[0])) == "ok"
            assert a._async_client is b._async_client
            assert other._async_client is not a._async_client
            assert built == [{"api_key": "k", "base_url": "https://one.api/v1"},
                             {"api_key": "k", "base_url": "https://two.api/v1"}]

            OpenAIBackend.clear_cache()
            again = OpenAIBackend("model-a", api_key="k", base_url="https://one.api/v1")
            aio.run(again.agenerate(_HI[0]))
            assert again._async_client is not a._async_client
        OpenAIBackend.clear_cache()
