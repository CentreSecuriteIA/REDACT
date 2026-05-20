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

    @property
    def supports_native_batching(self) -> bool:
        """True if ``batch_generate()`` is a true single engine pass.

        Default False (API backends loop sequentially). vLLM overrides to True.
        Read by BatchCaller and ModelRouter to choose the right batching path.
        """
        return False

    @property
    def supports_parallel_calls(self) -> bool:
        """True if ``generate()`` is safe to call concurrently from threads.

        Default True (most API backends are fine). Override to False for
        backends that require series execution — e.g. Anthropic (TPM/RPM are
        tight) or vLLM (concurrent generate() calls compete for GPU memory).
        BatchCaller raises ValueError if a caller sets ``max_workers>1`` on
        a backend that returns False here.
        """
        return True
