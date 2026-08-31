"""Generic OpenAI-compatible API backend.

Uses the OpenAI SDK against the chat completions API — works with Venice AI
and any other OpenAI-compatible endpoint by changing base_url. Venice is the
only provider currently registered against it, but the class itself has no
Venice-specific logic — the endpoint identity lives entirely in the registry
entry's ``APIConfig``.

One instance **is one registered model**: the model name and the endpoint's
``extra_body``/rpm/worker budget are bound in :meth:`OpenAIBackend.from_config`
and never re-passed per call. The underlying ``openai.OpenAI`` object is what's
actually expensive (an HTTP connection pool), so *that* is cached by endpoint
identity and shared between the per-model backends pointing at it.

A registry entry can describe the same model both here and as local weights;
``ModelClient.create(name, backend_type="vllm")`` binds the local setup
instead. See ``model_config.py``.
"""

import os
import threading
import time
from typing import ClassVar

import openai

from .base import ComputeConfig, LLMBackend

# Shared SDK clients, keyed on endpoint identity — two OpenAI-compatible
# providers are different transports even though both are "openai", so the
# base_url is part of the key.
_sdk_clients: dict[tuple[str, str], "openai.OpenAI"] = {}
_sdk_clients_lock = threading.Lock()


def _sdk_client(api_key: str, base_url: str) -> "openai.OpenAI":
    key = (base_url, api_key)
    if key in _sdk_clients:
        return _sdk_clients[key]
    with _sdk_clients_lock:
        # Re-check under the lock: a duplicate here only wastes a connection
        # pool rather than GPU memory, but the cache should still hold one
        # object per endpoint so the pool is genuinely shared.
        if key not in _sdk_clients:
            _sdk_clients[key] = openai.OpenAI(api_key=api_key, base_url=base_url)
        return _sdk_clients[key]


class OpenAIBackend(LLMBackend):
    """One model on an OpenAI-compatible endpoint (Venice AI and friends)."""

    # Spelled out rather than inherited from the ABC: reading this class
    # should tell you its whole dispatch profile.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,  # no batch endpoint; BatchCaller fans out
        supports_parallel_calls=True,    # independent HTTP calls are safe
        supports_internals=False,        # a hosted endpoint exposes no activations
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
            model: Model identifier sent upstream (e.g. "venice-uncensored").
            api_key: API key for authentication.
            base_url: Base URL of the API (e.g. "https://api.venice.ai/api/v1").
            api_model_id: Identifier to send upstream when the provider's slug
                differs from ``model``. Defaults to ``model``. ``self.model``
                stays the registry name, because it keys telemetry rows and the
                price lookup in ``telemetry.summary()``.
            extra_body: Provider-specific parameters sent with every call.
            **identity: Forwarded to :meth:`LLMBackend.__init__` —
                generation defaults, system-prompt policy, rpm, max_workers.
        """
        super().__init__(model, **identity)
        # Registry name stays self.model (telemetry key, price lookup); only
        # the request payload uses the provider's own slug.
        self._api_model_id = api_model_id or model
        self._client = _sdk_client(api_key, base_url)
        self._base_url = base_url
        self._extra_body = extra_body or None

    @classmethod
    def from_config(cls, config) -> "OpenAIBackend":
        """Build from a registry entry's ``.api`` setup.

        Raises:
            ValueError: If the entry has no ``.api`` setup.
            KeyError: If the setup's API-key env var is not set.
        """
        api = config.api
        if api is None:
            raise ValueError(
                f"Model {config.name!r} has no api setup. Register it with "
                f"register_model(..., api=APIConfig(backend_type='openai', "
                f"api_key_env='...', base_url='...', rpm=...))."
            )
        return cls(
            api_key=os.environ[api.api_key_env],
            base_url=api.base_url,
            api_model_id=api.api_model_id,
            extra_body=api.default_extra_body,
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
        """Generate responses via the API — one real SDK call per batch item.

        No native batching (the chat completions API has no batch endpoint);
        real concurrency comes from ``BatchCaller`` calling this in parallel
        with a batch of one each, not from looping here. ``internals_ids`` is
        accepted for signature parity but never used — this backend never
        declares ``compute_config.supports_internals``, so the client
        guarantees it's always None by the time it reaches here.

        Args:
            messages_list: One chat message list per batch item.
            system_prompts: Optional system prompt(s) — re-inserted as a
                system-role message per call.
            max_tokens: Overrides this model's default.
            temperature: Overrides this model's default.
            internals_ids: Unused — see above.
            **kwargs: Passed through to the OpenAI client.

        Returns:
            Generated text, one per batch item, same order as ``messages_list``.
        """
        prompts, resolved = self._prepare(messages_list, system_prompts)
        max_tok, temp = self._resolve(max_tokens, temperature)

        results = []
        for messages, system_prompt in zip(resolved, prompts):
            call_messages = messages
            if system_prompt:
                call_messages = [{"role": "system", "content": system_prompt}, *messages]

            call_kwargs: dict = {
                "model": self._api_model_id,
                "messages": call_messages,
                **kwargs,
            }
            if max_tok is not None:
                call_kwargs["max_tokens"] = max_tok
            if temp is not None:
                call_kwargs["temperature"] = temp
            if self._extra_body:
                call_kwargs.setdefault("extra_body", self._extra_body)

            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**call_kwargs)
            except Exception as exc:
                self._record_call(
                    n_items=1, started=started, max_tokens=max_tok,
                    temperature=temp, error=f"{type(exc).__name__}: {exc}",
                )
                raise
            content = response.choices[0].message.content
            # usage is optional in the OpenAI-compatible spec — plenty of
            # endpoints omit it, so this must degrade to None, not raise.
            usage = getattr(response, "usage", None)
            self._record_call(
                n_items=1, started=started,
                in_tok=getattr(usage, "prompt_tokens", None),
                out_tok=getattr(usage, "completion_tokens", None),
                max_tokens=max_tok, temperature=temp,
            )
            results.append(content if content is not None else "")
        return results

    @property
    def backend_name(self) -> str:
        return "openai"

    def __repr__(self) -> str:
        return f"OpenAIBackend(model={self.model!r}, base_url={self._base_url!r})"
