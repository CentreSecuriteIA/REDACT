"""Backend for OpenAI-compatible chat completion APIs.

Works with any compatible endpoint through ``base_url`` (e.g. Venice AI). The
``openai.OpenAI`` SDK client holds an HTTP connection pool, so it is cached
per endpoint and shared between the backends that use it.
"""

import threading
import time
from typing import TYPE_CHECKING, ClassVar

import openai

from .base import ComputeConfig, LLMBackend

if TYPE_CHECKING:
    from ..model_config import ModelConfig

# SDK clients shared between backends, keyed on (base_url, api_key).
_sdk_clients: dict[tuple[str, str], "openai.OpenAI"] = {}
_sdk_clients_lock = threading.Lock()


def _sdk_client(api_key: str, base_url: str) -> "openai.OpenAI":
    key = (base_url, api_key)
    if key in _sdk_clients:
        return _sdk_clients[key]
    with _sdk_clients_lock:
        # Re-check under the lock so each endpoint gets one client.
        if key not in _sdk_clients:
            _sdk_clients[key] = openai.OpenAI(api_key=api_key, base_url=base_url)
        return _sdk_clients[key]


class OpenAIBackend(LLMBackend):
    """One model on an OpenAI-compatible endpoint."""

    # Every backend class sets each flag explicitly.
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
            model: Registry name of the model (e.g. "venice-uncensored").
            api_key: API key.
            base_url: Base URL of the API (e.g. "https://api.venice.ai/api/v1").
            api_model_id: Model identifier sent to the provider. Defaults to
                ``model``.
            extra_body: Provider-specific parameters sent with every call.
            **identity: Forwarded to :meth:`LLMBackend.__init__` (generation
                defaults, system-prompt policy, rpm, max_workers).
        """
        super().__init__(model, **identity)
        # self.model stays the registry name, which telemetry and pricing key
        # on. Requests use the provider's identifier.
        self._api_model_id = api_model_id or model
        self._client = _sdk_client(api_key, base_url)
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
        """Generate replies, one SDK call per item.

        Args:
            messages_list: One chat message list per item.
            system_prompts: Sent as a leading system-role message.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: Ignored. This backend does not capture internals.
            **kwargs: Passed to ``chat.completions.create()``.

        Returns:
            Generated text, one per item, in the order of ``messages_list``.
        """
        prompts, resolved = self._prepare(messages_list, system_prompts)
        max_tok, temp = self._resolve(max_tokens, temperature)

        results = []
        # Runs once via ModelClient (one item per call); loops only for direct callers.
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
            # Some OpenAI-compatible endpoints omit usage.
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
