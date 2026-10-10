"""Backend for the Anthropic Messages API (Claude).

Uses the native Anthropic SDK, which takes the system prompt as a separate
parameter. The SDK client is cached per API key and shared between the
backends that use it.

Calls are made in series: this backend declares
``supports_parallel_calls=False``, so ``max_workers`` is always 1.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, ClassVar

from .base import ComputeConfig, LLMBackend

if TYPE_CHECKING:
    from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

# SDK clients shared between backends, keyed by API key.
_sdk_clients: dict[str, object] = {}
_sdk_clients_lock = threading.Lock()


def _sdk_client(api_key: str):
    # One read: clear_cache() may drop the key between a test and a lookup.
    client = _sdk_clients.get(api_key)
    if client is not None:
        return client
    with _sdk_clients_lock:
        # Re-check under the lock so each key gets one client.
        client = _sdk_clients.get(api_key)
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package could not be imported. It is a "
                    "required dependency: reinstall redact (pip install -e .)."
                )
            # The SDK retries a failed call itself (2 by default), unseen by the rate limiter and the trace.
            client = anthropic.Anthropic(api_key=api_key)
            _sdk_clients[api_key] = client
        return client


class AnthropicBackend(LLMBackend):
    """One model on the Anthropic Messages API (Claude)."""

    # Series-only: concurrent requests within the RPM budget can still exceed
    # the tokens-per-minute limit.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,
        supports_parallel_calls=False,
        supports_internals=False,
    )

    def __init__(
        self, model: str, api_key: str, *, api_model_id: str | None = None, **identity
    ):
        """Bind one Claude model to the Anthropic API.

        Args:
            model: Registry name of the model (e.g. "claude-opus-4-6").
            api_key: Anthropic API key.
            api_model_id: Model identifier sent to Anthropic. Defaults to
                ``model``.
            **identity: Forwarded to :meth:`LLMBackend.__init__`.
        """
        super().__init__(model, **identity)
        # self.model stays the registry name, which telemetry and pricing key
        # on. Requests use the provider's identifier.
        self._api_model_id = api_model_id or model
        self._client = _sdk_client(api_key)

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "AnthropicBackend":
        """Build from a registry entry's ``.api`` setup.

        Raises:
            ValueError: The entry has no ``.api`` setup.
            KeyError: The setup's API-key env var is not set.
        """
        api = config.api
        if api is None:
            raise ValueError(
                f"Model {config.name!r} has no api setup. Register it with "
                f"register_model(..., api=APIConfig(backend_type='anthropic', "
                f"api_key_env='...', rpm=...))."
            )
        return cls(
            api_key=cls._api_key(config),
            api_model_id=api.api_model_id,
            rpm=api.rpm,
            max_workers=api.recommended_max_workers,
            # Supplies model=config.name and the other shared identity fields.
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the shared SDK clients (for tests, or after env vars change)."""
        _sdk_clients.clear()

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate replies, one Messages API call per item.

        Args:
            messages_list: One chat message list per item. Only ``role`` and
                ``content`` of each message are sent.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: Ignored. This backend does not capture internals.
            **kwargs: Passed to ``messages.create()``.

        Returns:
            Generated text, one per item, in the order of ``messages_list``.
        """
        prompts, resolved = self._prepare(messages_list)
        max_tok, temp = self._resolve(max_tokens, temperature)

        results = []
        # Runs once via ModelClient (one item per call); loops only for direct callers.
        for messages, system_prompt in zip(resolved, prompts):
            conv_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            # kwargs first: a model= or messages= in them must not re-target the call.
            call_kwargs: dict = {
                **kwargs,
                "model": self._api_model_id,
                "messages": conv_messages,
                "max_tokens": max_tok,
            }
            if system_prompt:
                call_kwargs["system"] = system_prompt
            if temp is not None:
                call_kwargs["temperature"] = temp

            started = time.perf_counter()
            try:
                response = self._client.messages.create(**call_kwargs)
            except Exception as exc:
                self._record_call(
                    n_items=1, started=started, max_tokens=max_tok,
                    temperature=temp, error=f"{type(exc).__name__}: {exc}",
                )
                raise
            usage = getattr(response, "usage", None)
            stop_reason = getattr(response, "stop_reason", None)
            self._record_call(
                n_items=1, started=started,
                in_tok=getattr(usage, "input_tokens", None),
                out_tok=getattr(usage, "output_tokens", None),
                max_tokens=max_tok, temperature=temp, finish_reason=stop_reason,
            )
            # A response with no content blocks counts as an empty reply.
            text = (response.content[0].text or "") if response.content else ""
            if not text:
                logger.warning("%s returned an empty reply (stop_reason=%s).",
                               self.model, stop_reason)
            results.append(text)
        return results

    @property
    def backend_name(self) -> str:
        return "anthropic"

    def __repr__(self) -> str:
        return f"AnthropicBackend(model={self.model!r})"
