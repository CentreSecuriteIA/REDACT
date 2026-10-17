"""Backend for the Anthropic Messages API (Claude).

Uses the native Anthropic SDK, which takes the system prompt as a separate
parameter. The SDK client is cached per API key and retry settings, and
shared between the backends that use it.

Calls are made in series: this backend declares
``supports_parallel_calls=False``, so ``max_workers`` is always 1.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, ClassVar

from ...exceptions import PromptTooLongError
from .base import (
    DEFAULT_MAX_RETRIES,
    ComputeConfig,
    LLMBackend,
    close_quietly,
    sdk_client_kwargs,
)

if TYPE_CHECKING:
    from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

# SDK clients shared between backends, keyed on (api_key, max_retries, timeout_s).
_sdk_clients: dict[tuple, object] = {}
_sdk_clients_lock = threading.Lock()


def _sdk_client(api_key: str, *, max_retries: int, timeout_s: float | None):
    key = (api_key, max_retries, timeout_s)
    # One read: clear_cache() may drop the key between a test and a lookup.
    client = _sdk_clients.get(key)
    if client is not None:
        return client
    with _sdk_clients_lock:
        # Re-check under the lock so each key gets one client.
        client = _sdk_clients.get(key)
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package could not be imported. It is a "
                    "required dependency: reinstall redact (pip install -e .)."
                )
            # The SDK retries a failed call itself, unseen by the rate limiter and the trace.
            client = anthropic.Anthropic(
                api_key=api_key, **sdk_client_kwargs(max_retries, timeout_s))
            _sdk_clients[key] = client
        return client


# The status of the SDK's BadRequestError.
_BAD_REQUEST = 400


def _is_prompt_too_long(exc: BaseException) -> bool:
    """Whether the API rejected a prompt as over the model's context window.

    A 400 (``BadRequestError``) whose message says the prompt is too long, or
    that the prompt and ``max_tokens`` together exceed the context. Read off
    the status code, so the SDK need not be imported.
    """
    message = str(exc)
    return getattr(exc, "status_code", None) == _BAD_REQUEST and (
        "prompt is too long" in message or "exceed context limit" in message)


class AnthropicBackend(LLMBackend):
    """One model on the Anthropic Messages API (Claude)."""

    # Series-only: concurrent requests within the RPM budget can still exceed
    # the tokens-per-minute limit.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,
        supports_parallel_calls=False,
        supports_internals=False,
        supports_async_calls=False,
    )

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        api_model_id: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float | None = None,
        **identity,
    ):
        """Bind one Claude model to the Anthropic API.

        Args:
            model: Registry name of the model (e.g. "claude-opus-4-6").
            api_key: Anthropic API key.
            api_model_id: Model identifier sent to Anthropic. Defaults to
                ``model``.
            max_retries: Retries the SDK makes of a failed call.
            timeout_s: Seconds before the SDK gives up on a call; ``None``
                keeps the SDK's default.
            **identity: Forwarded to :meth:`LLMBackend.__init__`.
        """
        super().__init__(model, **identity)
        # self.model stays the registry name, which telemetry and pricing key
        # on. Requests use the provider's identifier.
        self._api_model_id = api_model_id or model
        self._client = _sdk_client(
            api_key, max_retries=max_retries, timeout_s=timeout_s)

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
            max_retries=api.max_retries,
            timeout_s=api.timeout_s,
            rpm=api.rpm,
            max_workers=api.recommended_max_workers,
            # Supplies model=config.name and the other shared identity fields.
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Close and drop the shared SDK clients (for tests, or after env vars change).

        A backend built before keeps its client, closed, so its next call fails.
        """
        with _sdk_clients_lock:
            clients = [*_sdk_clients.values()]
            _sdk_clients.clear()
        for client in clients:
            close_quietly(client.close)

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

        Raises:
            PromptTooLongError: The API rejected an item's prompt as over
                the model's context.
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
                if _is_prompt_too_long(exc):
                    raise PromptTooLongError(str(exc)) from exc
                raise
            usage = getattr(response, "usage", None)
            stop_reason = getattr(response, "stop_reason", None)
            out_tok = getattr(usage, "output_tokens", None)
            self._record_call(
                n_items=1, started=started,
                in_tok=getattr(usage, "input_tokens", None),
                out_tok=out_tok, out_tok_max=out_tok,
                max_tokens=max_tok, temperature=temp, finish_reason=stop_reason,
                # Two stop reasons cut a reply: the requested cap and the
                # context window.
                cut=stop_reason in ("max_tokens", "model_context_window_exceeded"),
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
