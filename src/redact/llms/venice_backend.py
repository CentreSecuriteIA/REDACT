"""Venice AI backend (OpenAI-compatible).

Uses the OpenAI SDK since Venice implements the chat completions API.
Also works with any other OpenAI-compatible endpoint by changing base_url.

Model-specific defaults (extra_body, max_tokens, temperature) are
auto-merged from model_config so callers don't need to repeat them.

NOTE: Venice models can also run locally via vLLM for faster generation.
To use local inference, create a VLLMBackend instance manually and pass
it directly to pipeline functions as the ``backend`` parameter.
"""

import os

import openai

from .base import LLMBackend
from .model_config import get_model_config


class VeniceBackend(LLMBackend):
    """Backend for Venice AI and other OpenAI-compatible APIs."""

    def __init__(self, api_key: str, base_url: str):
        """Create a Venice backend.

        Args:
            api_key: API key for authentication.
            base_url: Base URL of the API (e.g. "https://api.venice.ai/api/v1").
        """
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._base_url = base_url

    def generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict | None = None,
        **kwargs,
    ) -> str:
        """Generate a response via the API.

        Parameters are resolved in priority order:
        explicit arg > model config default > omitted.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model identifier (e.g. "venice-uncensored").
            max_tokens: Max tokens to generate. Falls back to model config default.
            temperature: Sampling temperature. Falls back to model config default.
            extra_body: Provider-specific parameters. Merged on top of model
                        config defaults (per-call values override config values).
            **kwargs: Passed through to the OpenAI client.

        Returns:
            The generated text content.
        """
        config = get_model_config(model)

        # Resolve parameters: explicit > model default > omit
        resolved_max_tokens = max_tokens if max_tokens is not None else config.default_max_tokens
        resolved_temperature = temperature if temperature is not None else config.default_temperature

        # Merge extra_body: model defaults as base, per-call overrides on top
        merged_extra = {}
        if config.default_extra_body:
            merged_extra.update(config.default_extra_body)
        if extra_body:
            merged_extra.update(extra_body)

        # Build call kwargs
        call_kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": resolved_max_tokens,
            **kwargs,
        }
        if resolved_temperature is not None:
            call_kwargs["temperature"] = resolved_temperature
        if merged_extra:
            call_kwargs["extra_body"] = merged_extra

        response = self._client.chat.completions.create(**call_kwargs)
        return response.choices[0].message.content

    @classmethod
    def from_env(
        cls,
        env_var: str = "VENICE_API_KEY",
        base_url: str = "https://api.venice.ai/api/v1",
    ) -> "VeniceBackend":
        """Create a backend from an environment variable.

        Args:
            env_var: Name of the env var holding the API key.
            base_url: Base URL of the API endpoint.

        Raises:
            KeyError: If the environment variable is not set.
        """
        api_key = os.environ[env_var]
        return cls(api_key=api_key, base_url=base_url)

    @property
    def backend_name(self) -> str:
        return "venice"

    def __repr__(self) -> str:
        return f"VeniceBackend(base_url={self._base_url!r})"
