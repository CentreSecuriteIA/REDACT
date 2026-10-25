"""Tests for OpenAIBackend with mocked OpenAI client."""

import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from redact.exceptions import PromptTooLongError
from redact.llms import aio, observe
from redact.llms.backends import OpenAIBackend
from redact.llms.model_config import APIConfig, ModelConfig, get_model_config
from tests.conftest import first_cut_warning


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
                    max_retries=2,
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
        cache = _ClearedAfterContains({("https://one.api/v1", "k", 2, None): client})
        with patch.object(openai_module, "_sdk_clients", cache):
            assert openai_module._sdk_client(
                "k", "https://one.api/v1", max_retries=2, timeout_s=None) is client

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


def _config(**api):
    """An entry on one OpenAI-compatible endpoint, with ``api`` on its setup."""
    return ModelConfig(name="m", api=APIConfig(
        backend_type="openai", api_key_env="TEST_API_KEY",
        base_url="https://one.api/v1", rpm=10, **api))


class TestOpenAISdkSettings:
    """The setup's retries and timeout reach both SDK clients."""

    def _built(self, monkeypatch, config):
        """The kwargs each SDK class was built with, for one backend's calls."""
        built = {"sync": [], "async": []}

        def recorder(kind, response):
            def build(**kwargs):
                built[kind].append(kwargs)
                client = MagicMock()
                client.chat.completions.create = response
                return client
            return build

        monkeypatch.setenv("TEST_API_KEY", "k")
        OpenAIBackend.clear_cache()
        answer = AsyncMock(return_value=_response("ok", "stop"))
        with (
            patch("redact.llms.backends.openai.openai.OpenAI",
                  side_effect=recorder("sync", MagicMock())),
            patch("redact.llms.backends.openai.openai.AsyncOpenAI",
                  side_effect=recorder("async", answer)),
        ):
            backend = OpenAIBackend.from_config(config)
            aio.run(backend.agenerate(_HI[0]))
        OpenAIBackend.clear_cache()
        return built

    def test_the_setups_retries_and_timeout_reach_both_clients(self, monkeypatch):
        built = self._built(monkeypatch, _config(max_retries=5, timeout_s=30))
        expected = {"api_key": "k", "base_url": "https://one.api/v1",
                    "max_retries": 5, "timeout": 30}
        assert built == {"sync": [expected], "async": [expected]}

    def test_the_default_setup_keeps_the_sdks_retries_and_timeout(self, monkeypatch):
        built = self._built(monkeypatch, _config())
        expected = {"api_key": "k", "base_url": "https://one.api/v1", "max_retries": 2}
        assert built == {"sync": [expected], "async": [expected]}

    def test_models_on_one_endpoint_share_a_client_only_with_one_setting(self):
        with patch("redact.llms.backends.openai.openai.OpenAI",
                   side_effect=lambda **kwargs: MagicMock()):
            OpenAIBackend.clear_cache()
            a = OpenAIBackend("model-a", api_key="k", base_url="https://one.api/v1")
            b = OpenAIBackend("model-b", api_key="k", base_url="https://one.api/v1")
            c = OpenAIBackend("model-c", api_key="k", base_url="https://one.api/v1",
                              max_retries=5)
            assert a._client is b._client
            assert c._client is not a._client
        OpenAIBackend.clear_cache()


class _Closing:
    """An SDK client that records its close, or fails it."""

    def __init__(self, closed, fails=False):
        self._closed, self._fails = closed, fails

    def close(self):
        if self._fails:
            raise RuntimeError("already gone")
        self._closed.append(self)


class TestOpenAIClearCache:
    """clear_cache() closes the SDK clients it drops."""

    def test_the_sync_clients_are_closed(self):
        closed = []
        with patch("redact.llms.backends.openai.openai.OpenAI",
                   side_effect=lambda **kwargs: _Closing(closed)):
            OpenAIBackend.clear_cache()
            a = OpenAIBackend("model-a", api_key="k", base_url="https://one.api/v1")
            b = OpenAIBackend("model-b", api_key="k", base_url="https://two.api/v1")
            assert closed == []
            OpenAIBackend.clear_cache()
        assert closed == [a._client, b._client]

    def test_a_client_that_fails_to_close_is_dropped_all_the_same(self):
        closed = []
        with patch("redact.llms.backends.openai.openai.OpenAI",
                   side_effect=lambda **kwargs: _Closing(closed, fails=True)):
            OpenAIBackend.clear_cache()
            a = OpenAIBackend("model-a", api_key="k", base_url="https://one.api/v1")
            OpenAIBackend.clear_cache()
            b = OpenAIBackend("model-b", api_key="k", base_url="https://one.api/v1")
        assert b._client is not a._client
        OpenAIBackend.clear_cache()


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

    def test_a_model_that_rejects_sampling_gets_no_temperature(
            self, mock_openai_client, events):
        """claude-opus-5-5 as a checker: the check's 0.15 is not sent, on
        either path, and the event records none."""
        backend = OpenAIBackend.from_config(get_model_config("claude-opus-5-5"))
        backend._client = mock_openai_client
        backend._async_client = sdk = _async_sdk(_response("Yes", "stop"))
        messages = [{"role": "user", "content": "hi"}]
        backend.generate([messages], temperature=0.15)
        aio.run(backend.agenerate(messages, temperature=0.15))
        sent = mock_openai_client.chat.completions.create.call_args[1]
        assert "temperature" not in sent
        assert "temperature" not in sdk.chat.completions.create.await_args[1]
        assert [event["temperature"] for event in events] == [None, None]

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

    def test_a_reply_stopped_by_the_content_filter_is_returned_and_warned_about(
        self, openai_backend, mock_openai_client, events, caplog
    ):
        create = mock_openai_client.chat.completions.create
        create.return_value = _response("part", "content_filter")
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends.openai"):
            assert openai_backend.generate(_HI) == ["part"]
        assert [r.getMessage() for r in caplog.records] == [
            "venice-uncensored: the provider's content filter stopped a reply; it is incomplete."
        ]
        assert events[0]["finish_reason"] == "content_filter"


class TestOpenAICutReply:
    """A reply cut at max_tokens is returned as it is, with one warning."""

    CUT = first_cut_warning("venice-uncensored", 50)

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
            self.CUT, "venice-uncensored returned an empty reply (finish_reason=length).",
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

    def test_defaults_and_system_prompt_policy_apply_as_in_generate(self, events):
        """No overrides, on a model without a system role: both paths fold the
        system prompt into the user message and record the default limit."""
        response = _response("ok", "stop")
        with patch("redact.llms.backends.openai.openai.OpenAI"):
            OpenAIBackend.clear_cache()
            backend = OpenAIBackend(
                "no-system-role", api_key="k", base_url="https://one.api/v1",
                default_max_tokens=123, supports_system_prompt=False)
        backend._client.chat.completions.create.return_value = response
        backend._async_client = sdk = _async_sdk(response)
        messages = [{"role": "system", "content": "Sys."},
                    {"role": "user", "content": "hi"}]

        assert backend.generate([messages]) == ["ok"]
        assert aio.run(backend.agenerate(messages)) == "ok"

        sent = backend._client.chat.completions.create.call_args
        assert sdk.chat.completions.create.await_args == sent
        assert [m["role"] for m in sent.kwargs["messages"]] == ["user"]
        sync_event, async_event = events
        del sync_event["ms"], async_event["ms"]
        assert async_event == sync_event
        assert async_event["max_tokens"] == backend.default_max_tokens
        OpenAIBackend.clear_cache()

    def test_an_empty_cut_reply_warns_as_generate_does(self, openai_backend, caplog):
        openai_backend._async_client = _async_sdk(_response("", "length"))
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert aio.run(openai_backend.agenerate(_HI[0], max_tokens=50)) == ""
        assert [r.getMessage() for r in caplog.records] == [
            TestOpenAICutReply.CUT,
            "venice-uncensored returned an empty reply (finish_reason=length).",
        ]

    def test_a_failed_call_is_recorded_then_re_raised(self, openai_backend, events):
        openai_backend._async_client = _async_sdk(error=RuntimeError("502"))
        with pytest.raises(RuntimeError, match="502"):
            aio.run(openai_backend.agenerate(_HI[0]))
        (event,) = events
        assert event["error"] == "RuntimeError: 502" and event["n_items"] == 1

    def test_a_cancelled_call_is_recorded_then_re_raised(self, openai_backend, events):
        """The request was sent, so the trace has to show it."""
        sent = asyncio.Event()

        async def never_answered(**kwargs):
            sent.set()
            await asyncio.Event().wait()

        async def cancel_in_flight():
            task = asyncio.ensure_future(openai_backend.agenerate(_HI[0], max_tokens=50))
            await sent.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        openai_backend._async_client = MagicMock()
        openai_backend._async_client.chat.completions.create = never_answered
        aio.run(cancel_in_flight())
        (event,) = events
        assert event["error"] == "CancelledError: "
        assert (event["n_items"], event["max_tokens"]) == (1, 50)

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
            assert built == [
                {"api_key": "k", "base_url": "https://one.api/v1", "max_retries": 2},
                {"api_key": "k", "base_url": "https://two.api/v1", "max_retries": 2},
            ]

            OpenAIBackend.clear_cache()
            again = OpenAIBackend("model-a", api_key="k", base_url="https://one.api/v1")
            aio.run(again.agenerate(_HI[0]))
            assert again._async_client is not a._async_client
        OpenAIBackend.clear_cache()


def _bad_request(message, code=None):
    """An HTTP 400 as the SDK raises it, with the endpoint's error ``code``."""
    request = httpx.Request("POST", "https://test.api/chat/completions")
    return openai.BadRequestError(
        message, response=httpx.Response(400, request=request),
        body={"message": message, "code": code})


class TestOpenAIPromptTooLong:
    """A prompt the endpoint rejects as over the model's context raises
    PromptTooLongError, from generate() and from agenerate()."""

    def _both(self, backend, sdk, error, expected):
        """Fail a call on each path with ``error``; return what each raised."""
        sdk.chat.completions.create.side_effect = error
        backend._async_client = _async_sdk(error=error)
        with pytest.raises(expected) as blocking:
            backend.generate(_HI)
        with pytest.raises(expected) as on_the_loop:
            aio.run(backend.agenerate(_HI[0]))
        return blocking.value, on_the_loop.value

    @pytest.mark.parametrize("error", [
        _bad_request("The input is too long.", code="context_length_exceeded"),
        _bad_request("This model's maximum context length is 8192 tokens. However, "
                     "you requested 0 output tokens and your prompt contains at "
                     "least 10036 input tokens"),
    ], ids=["by-code", "by-message"])
    def test_both_spellings_raise_prompt_too_long(
        self, openai_backend, mock_openai_client, events, error
    ):
        for raised in self._both(openai_backend, mock_openai_client, error,
                                 PromptTooLongError):
            assert str(raised) == str(error) and raised.__cause__ is error
        assert [event["error"] for event in events] == [f"BadRequestError: {error}"] * 2

    def test_another_bad_request_stays_as_it_is(
        self, openai_backend, mock_openai_client, events
    ):
        error = _bad_request("Unsupported parameter: top_k.",
                             code="unsupported_parameter")
        blocking, on_the_loop = self._both(
            openai_backend, mock_openai_client, error, openai.BadRequestError)
        assert blocking is error and on_the_loop is error
        assert [event["error"] for event in events] == [f"BadRequestError: {error}"] * 2
