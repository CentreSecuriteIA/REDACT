"""Backend for OpenAI-compatible chat completion APIs.

Works with any compatible endpoint through ``base_url`` (e.g. Venice AI). An
SDK client holds an HTTP connection pool, so it is cached per endpoint and
shared between the backends that use it: ``openai.OpenAI`` for ``generate()``
and ``openai.AsyncOpenAI`` for ``agenerate()``.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, ClassVar

import openai

from .base import ComputeConfig, LLMBackend

if TYPE_CHECKING:
    from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

# SDK clients shared between backends, keyed on (base_url, api_key).
_sdk_clients: dict[tuple[str, str], "openai.OpenAI"] = {}
_async_sdk_clients: dict[tuple[str, str], "openai.AsyncOpenAI"] = {}
_sdk_clients_lock = threading.Lock()


def _sdk_client(api_key: str, base_url: str, *, async_calls: bool = False):
    cache = _async_sdk_clients if async_calls else _sdk_clients
    key = (base_url, api_key)
    # One read: clear_cache() may drop the key between a test and a lookup.
    client = cache.get(key)
    if client is not None:
        return client
    with _sdk_clients_lock:
        # Re-check under the lock so each endpoint gets one client.
        client = cache.get(key)
        if client is None:
            sdk_class = openai.AsyncOpenAI if async_calls else openai.OpenAI
            # The SDK retries a failed call itself (2 by default), unseen by the rate limiter and the trace.
            client = sdk_class(api_key=api_key, base_url=base_url)
            cache[key] = client
        return client


class OpenAIBackend(LLMBackend):
    """One model on an OpenAI-compatible endpoint."""

    # Every backend class sets each flag explicitly.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,  # no batch endpoint; BatchCaller fans out
        supports_parallel_calls=True,    # independent HTTP calls are safe
        supports_internals=False,        # a hosted endpoint exposes no activations
        supports_async_calls=True,       # agenerate() on the async SDK client
    )

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        *,
        api_model_id: str | None = None,
        extra_body: dict | None = None,
        **identity,
    ):
        """Bind one model to an OpenAI-compatible endpoint.

        Args:
            model: Registry name of the model (e.g. "venice-uncensored").
            api_key: API key.
            base_url: Base URL of the API (e.g. "https://api.venice.ai/api/v1").
            api_model_id: Model identifier sent to the provider. Defaults to
                ``model``.
            extra_body: Provider-specific parameters sent with each call. A
                call that passes its own ``extra_body`` replaces them whole.
            **identity: Forwarded to :meth:`LLMBackend.__init__` (generation
                defaults, system-prompt policy, rpm, max_workers).
        """
        super().__init__(model, **identity)
        # self.model stays the registry name, which telemetry and pricing key
        # on. Requests use the provider's identifier.
        self._api_model_id = api_model_id or model
        self._client = _sdk_client(api_key, base_url)
        # Built by the first agenerate(): an SDK client is slow to construct,
        # and a backend that never makes an async call needs none.
        self._async_client: openai.AsyncOpenAI | None = None
        self._endpoint = (api_key, base_url)
        self._base_url = base_url
        self._extra_body = extra_body or None

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "OpenAIBackend":
        """Build from a registry entry's ``.api`` setup.

        Raises:
            ValueError: The entry has no ``.api`` setup.
            KeyError: The setup's API-key env var is not set.
        """
        api = config.api
        if api is None:
            raise ValueError(
                f"Model {config.name!r} has no api setup. Register it with "
                f"register_model(..., api=APIConfig(backend_type='openai', "
                f"api_key_env='...', base_url='...', rpm=...))."
            )
        return cls(
            api_key=cls._api_key(config),
            base_url=api.base_url,
            api_model_id=api.api_model_id,
            extra_body=api.default_extra_body,
            rpm=api.rpm,
            max_workers=api.recommended_max_workers,
            # Supplies model=config.name and the other shared identity fields.
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the shared SDK clients (for tests, or after env vars change)."""
        _sdk_clients.clear()
        _async_sdk_clients.clear()

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate replies, one SDK call per item.

        Args:
            messages_list: One chat message list per item.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: Ignored. This backend does not capture internals.
            **kwargs: Passed to ``chat.completions.create()``.

        Returns:
            Generated text, one per item, in the order of ``messages_list``.
        """
        prompts, resolved = self._prepare(messages_list)
        max_tok, temp = self._resolve(max_tokens, temperature)

        results = []
        # Runs once via ModelClient (one item per call); loops only for direct callers.
        for messages, system_prompt in zip(resolved, prompts):
            call_kwargs = self._call_kwargs(
                messages, system_prompt, max_tok, temp, kwargs)
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**call_kwargs)
            except Exception as exc:
                self._record_failure(exc, started, max_tok, temp)
                raise
            results.append(self._reply(response, started, max_tok, temp))
        return results

    async def agenerate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_id: str | None = None,
        **kwargs,
    ) -> str:
        """Generate one reply with one SDK call, without blocking the loop.

        Runs on the shared loop (``llms/aio.py``): the async SDK client's
        connections belong to it.

        Args:
            messages: Chat messages of the one item.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_id: Ignored. This backend does not capture internals.
            **kwargs: Passed to ``chat.completions.create()``.

        Returns:
            The generated text.
        """
        (system_prompt,), (messages,) = self._prepare([messages])
        max_tok, temp = self._resolve(max_tokens, temperature)
        call_kwargs = self._call_kwargs(messages, system_prompt, max_tok, temp, kwargs)
        if self._async_client is None:
            self._async_client = _sdk_client(*self._endpoint, async_calls=True)
        create = self._async_client.chat.completions.create

        started = time.perf_counter()
        try:
            response = await create(**call_kwargs)
        except Exception as exc:
            self._record_failure(exc, started, max_tok, temp)
            raise
        return self._reply(response, started, max_tok, temp)

    def _call_kwargs(
        self,
        messages: list[dict],
        system_prompt: str | None,
        max_tok: int,
        temp: float | None,
        kwargs: dict,
    ) -> dict:
        """Arguments of one ``chat.completions.create()`` call."""
        call_messages = messages
        if system_prompt:
            call_messages = [{"role": "system", "content": system_prompt}, *messages]

        # kwargs first: a model= or messages= in them must not re-target the call.
        call_kwargs: dict = {
            **kwargs,
            "model": self._api_model_id,
            "messages": call_messages,
            "max_tokens": max_tok,
        }
        if temp is not None:
            call_kwargs["temperature"] = temp
        if self._extra_body:
            call_kwargs.setdefault("extra_body", self._extra_body)
        return call_kwargs

    def _record_failure(
        self, exc: Exception, started: float, max_tok: int, temp: float | None
    ) -> None:
        """Record the ``call`` event of a call that raised."""
        self._record_call(
            n_items=1, started=started, max_tokens=max_tok,
            temperature=temp, error=f"{type(exc).__name__}: {exc}",
        )

    def _reply(self, response, started: float, max_tok: int, temp: float | None) -> str:
        """Record the ``call`` event of an answered call and return its text."""
        # A response with no choices counts as an empty reply.
        choice = response.choices[0] if response.choices else None
        content = (choice.message.content if choice else None) or ""
        finish_reason = getattr(choice, "finish_reason", None)
        # Some OpenAI-compatible endpoints omit usage.
        usage = getattr(response, "usage", None)
        out_tok = getattr(usage, "completion_tokens", None)
        self._record_call(
            n_items=1, started=started,
            in_tok=getattr(usage, "prompt_tokens", None),
            out_tok=out_tok, out_tok_max=out_tok,
            max_tokens=max_tok, temperature=temp, finish_reason=finish_reason,
        )
        if not content:
            logger.warning("%s returned an empty reply (finish_reason=%s).",
                           self.model, finish_reason)
        self._warn_cut(int(finish_reason == "length"), 1, max_tok)
        return content

    @property
    def backend_name(self) -> str:
        return "openai"

    def __repr__(self) -> str:
        return f"OpenAIBackend(model={self.model!r}, base_url={self._base_url!r})"
