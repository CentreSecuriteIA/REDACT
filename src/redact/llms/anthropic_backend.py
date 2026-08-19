"""Anthropic (Claude) API backend.

Uses the native Anthropic SDK rather than the OpenAI compatibility layer,
since Claude models use a different message format (separate system param,
content blocks, etc.).

Note: Claude models have strict rate limits (typically 5 RPM on free tier).
Set max_workers=1 in BatchCaller to avoid hitting token-per-minute limits,
as even within RPM limits concurrent requests can exceed TPM.
"""

import os

from .base import LLMBackend, fold_system_into_first_message
from .model_config import get_model_config


class AnthropicBackend(LLMBackend):
    """Backend for the Anthropic Messages API (Claude models)."""

    def __init__(self, api_key: str):
        """Create an Anthropic backend.

        Args:
            api_key: Anthropic API key.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicBackend. "
                "Install it with: pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        """Generate a response via the Anthropic Messages API.

        Accepts messages in OpenAI format and converts them:
        - System messages are extracted and passed as the ``system`` param.
        - Remaining messages are passed as the ``messages`` list.

        If ``model``'s config has ``supports_system_prompt=False``, system
        messages are folded into the first remaining message *before* the
        above split, so no ``system`` param is sent at all (see
        :func:`redact.llms.base.fold_system_into_first_message`).

        Args:
            messages: Chat messages in OpenAI format.
            model: Model identifier (e.g. "claude-opus-4-6").
            max_tokens: Max tokens to generate. Falls back to model config default.
            temperature: Sampling temperature. Falls back to model config default.
            **kwargs: Passed through to the Anthropic client.

        Returns:
            The generated text content.
        """
        config = get_model_config(model)

        if not config.supports_system_prompt:
            messages = fold_system_into_first_message(messages)

        resolved_max_tokens = max_tokens if max_tokens is not None else config.default_max_tokens
        resolved_temperature = temperature if temperature is not None else config.default_temperature

        # Split system messages from conversation messages
        system_parts: list[str] = []
        conv_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conv_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build call kwargs
        call_kwargs: dict = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": resolved_max_tokens,
            **kwargs,
        }
        if system_parts:
            call_kwargs["system"] = "\n\n".join(system_parts)
        if resolved_temperature is not None:
            call_kwargs["temperature"] = resolved_temperature

        response = self._client.messages.create(**call_kwargs)
        if not response.content:
            return ""
        return response.content[0].text or ""

    @classmethod
    def from_env(
        cls,
        env_var: str = "ANTHROPIC_API_KEY",
    ) -> "AnthropicBackend":
        """Create a backend from an environment variable.

        Args:
            env_var: Name of the env var holding the API key.

        Raises:
            KeyError: If the environment variable is not set.
        """
        api_key = os.environ[env_var]
        return cls(api_key=api_key)

    @property
    def backend_name(self) -> str:
        return "anthropic"

    @property
    def supports_parallel_calls(self) -> bool:
        # Claude has tight TPM/RPM limits — concurrent requests inside the
        # RPM budget can still trip TPM. Router enforces series-only.
        return False

    def __repr__(self) -> str:
        return "AnthropicBackend()"
