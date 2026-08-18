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

    @property
    def supports_internals(self) -> bool:
        """True if this backend can capture and persist model internals
        (hidden states / attention / logprobs) alongside generation.

        Default False. Only an introspection-capable backend (e.g.
        ``TransformersIntrospectionBackend``) overrides this to True.

        Callers request capture by passing ``internals_id`` (single call) /
        ``internals_ids`` (one per item, batch call) — an id the *caller*
        computes, rooted at the stage's own ``input_id`` (e.g.
        ``f"{input_id}/output"``), that the backend uses to name where it
        persists captured internals. These are not part of this class's
        method signatures — individual backends declare them explicitly if
        they consume them. Backends that don't support this should never
        receive them: :class:`~redact.llms.wrappers.BatchCaller` is the
        enforcement point — it checks this flag before dispatch and raises
        ``ValueError`` rather than letting an unsupported kwarg reach (and
        crash inside) a backend's underlying SDK call.
        """
        return False

    def rename_capture(self, old_internals_id: str, new_internals_id: str) -> None:
        """Relabel a previously-captured internals folder once its final id
        is known.

        Default no-op — only meaningful for backends that actually persist
        captures (``supports_internals=True``). Some stages can't know a
        unit's real identity until *after* generation (e.g. paraphrase's own
        output text only exists once the call returns, but ``internals_id``
        has to be supplied before it): the caller captures under a pre-call-
        known provisional id (e.g. an iteration index) and calls this once
        the real id is computed, without needing to know whether the call
        was capturing anything at all — safe to call unconditionally.
        """
        return
