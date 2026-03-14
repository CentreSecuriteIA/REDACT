"""Abstract base class for LLM backends.

Every backend (API, vLLM, etc.) implements this interface.
Model is always a required parameter on every call — the backend only
knows *how* to reach an endpoint, not *which* model to use.
"""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, messages: list[dict], model: str, **kwargs) -> str:
        """Generate a single response.

        Args:
            messages: Chat messages in OpenAI format
                      [{"role": "system"|"user"|"assistant", "content": "..."}].
            model: Model identifier (e.g. "venice-uncensored").
            **kwargs: Backend-specific parameters (max_tokens, temperature, etc.).

        Returns:
            The generated text content.
        """

    def batch_generate(
        self, messages_list: list[list[dict]], model: str, **kwargs
    ) -> list[str]:
        """Generate responses for multiple message lists.

        Default implementation loops over generate(). Backends like vLLM
        override this for native batching.
        """
        return [self.generate(msgs, model, **kwargs) for msgs in messages_list]

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for the backend type (e.g. 'api', 'vllm')."""
