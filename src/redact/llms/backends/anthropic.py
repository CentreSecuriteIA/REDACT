"""Anthropic (Claude) API backend.

Uses the native Anthropic SDK rather than the OpenAI compatibility layer,
since Claude models use a different message format (separate system param,
content blocks, etc.).

One instance **is one registered model** — name, generation defaults and rate
budget are bound in :meth:`AnthropicBackend.from_config` and never re-passed
per call. The ``anthropic.Anthropic`` SDK object (an HTTP connection pool) is
cached by API key and shared between the per-model backends using it.

Note: Claude models have strict rate limits (typically 5 RPM on free tier),
and concurrent requests can exceed TPM even while within RPM. Callers don't
have to remember that — this backend declares
``supports_parallel_calls=False``, so ``max_workers`` is clamped to 1 at
construction, ``BatchCaller`` refuses more, and ``register_model()`` rejects a
registry entry that asks for it.
"""

import os
import threading
import time
from typing import ClassVar

from .base import ComputeConfig, LLMBackend

# Shared SDK clients, keyed by API key — the endpoint is the SDK's own business.
_sdk_clients: dict[str, object] = {}
_sdk_clients_lock = threading.Lock()


def _sdk_client(api_key: str):
    if api_key in _sdk_clients:
        return _sdk_clients[api_key]
    with _sdk_clients_lock:
        # Re-check under the lock — see openai.py for why this is cheap here.
        if api_key not in _sdk_clients:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicBackend. "
                    "Install it with: pip install anthropic"
                )
            _sdk_clients[api_key] = anthropic.Anthropic(api_key=api_key)
        return _sdk_clients[api_key]


class AnthropicBackend(LLMBackend):
    """One model on the Anthropic Messages API (Claude)."""

    # Claude has tight TPM/RPM limits — concurrent requests inside the
    # RPM budget can still trip TPM, so this transport is series-only.
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
            model: Model identifier (e.g. "claude-opus-4-6").
            api_key: Anthropic API key.
            api_model_id: Identifier to send upstream when Anthropic's slug
                differs from ``model``. Defaults to ``model``; ``self.model``
                stays the registry name, which keys telemetry and pricing.
            **identity: Forwarded to :meth:`LLMBackend.__init__`.
        """
        super().__init__(model, **identity)
        # Registry name stays self.model (telemetry key, price lookup); only
        # the request payload uses the provider's own slug.
        self._api_model_id = api_model_id or model
        self._client = _sdk_client(api_key)

    @classmethod
    def from_config(cls, config) -> "AnthropicBackend":
        """Build from a registry entry's ``.api`` setup.

        Raises:
            ValueError: If the entry has no ``.api`` setup.
            KeyError: If the setup's API-key env var is not set.
        """
        api = config.api
        if api is None:
            raise ValueError(
                f"Model {config.name!r} has no api setup. Register it with "
                f"register_model(..., api=APIConfig(backend_type='anthropic', "
                f"api_key_env='...', rpm=...))."
            )
        return cls(
            api_key=os.environ[api.api_key_env],
            api_model_id=api.api_model_id,
            rpm=api.rpm,
            max_workers=api.recommended_max_workers,
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the shared SDK clients (env vars changed, or tests)."""
        _sdk_clients.clear()

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        system_prompts: str | list[str | None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate responses via the Anthropic Messages API.

        One real SDK call per batch item — no native batching, and series-only
        (``compute_config.supports_parallel_calls=False``) means ``BatchCaller``
        never fans this out across threads either. ``internals_ids`` is
        accepted for signature parity but never used — this backend never
        declares ``compute_config.supports_internals``.

        Args:
            messages_list: One chat message list per batch item. Reduced to
                plain ``{"role", "content"}`` pairs (dropping any other keys)
                for the Anthropic ``messages`` param.
            system_prompts: Optional system prompt(s) — passed as that call's
                ``system`` param.
            max_tokens: Overrides this model's default.
            temperature: Overrides this model's default.
            internals_ids: Unused — see above.
            **kwargs: Passed through to the Anthropic client.

        Returns:
            Generated text, one per batch item, same order as ``messages_list``.
        """
        prompts, resolved = self._prepare(messages_list, system_prompts)
        max_tok, temp = self._resolve(max_tokens, temperature)

        results = []
        for messages, system_prompt in zip(resolved, prompts):
            conv_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            call_kwargs: dict = {
                "model": self._api_model_id,
                "messages": conv_messages,
                "max_tokens": max_tok,
                **kwargs,
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
            self._record_call(
                n_items=1, started=started,
                in_tok=getattr(usage, "input_tokens", None),
                out_tok=getattr(usage, "output_tokens", None),
                max_tokens=max_tok, temperature=temp,
            )
            if not response.content:
                results.append("")
            else:
                results.append(response.content[0].text or "")
        return results

    @property
    def backend_name(self) -> str:
        return "anthropic"

    def __repr__(self) -> str:
        return f"AnthropicBackend(model={self.model!r})"
