"""Process-wide router for LLM access.

Owns one shared ``RateLimiter`` and one ``BatchCaller`` per model (lazily
created, cached). Provides role-based lookup so pipelines don't hardcode
model strings — e.g. ``router.for_role("translation")`` returns the
canonical (backend, model) pair for translation work.

Why a router: the LLM layer has three special-use models (Anthropic for
constitution gen, DeepSeek for translation, vLLM for local inference) plus
the uncensored generation models. Each has different concurrency rules.
The router is the single place that knows which model gets which executor,
so callers can ask for "translation" or "venice-uncensored" and always get
the right batching / parallelism / rate-limit behavior.

Usage::

    from redact.llms import get_router

    router = get_router()

    # Single call (rate-limited, routed through the right executor)
    result = router.generate("venice-uncensored", messages)

    # Batch — picks the right path (vLLM native batch, API thread pool,
    # or Anthropic sequential) based on backend capabilities
    results = router.batch_generate("venice-uncensored", messages_list)

    # Role lookup
    backend, model = router.for_role("translation")  # → DeepSeek
    backend, model = router.for_role("constitution_gen")  # → Claude
"""

from __future__ import annotations

from .api import get_backend as _get_backend_module
from .base import LLMBackend
from .model_config import default_model_for_role
from .wrappers import BatchCaller, RateLimiter


class ModelRouter:
    """Per-process LLM routing layer.

    One instance per process via :func:`get_router`. Holds:
    - a single shared :class:`RateLimiter` (all callers share the same RPM budget)
    - a lazy cache of :class:`BatchCaller` instances, one per model name

    The router does not own backends — those remain cached at module level
    in ``llms.api``. The router owns *executors* (BatchCallers) and rate
    limiting state. Backend instances themselves are stateless wrappers
    that can be shared across executors.
    """

    def __init__(self) -> None:
        self._rate_limiter = RateLimiter()
        self._executors: dict[str, BatchCaller] = {}

    @property
    def rate_limiter(self) -> RateLimiter:
        """The shared rate limiter used by every executor this router owns."""
        return self._rate_limiter

    def get_backend(self, model: str) -> LLMBackend:
        """Return the backend instance for a model (cached at module level)."""
        return _get_backend_module(model)

    def get_executor(self, model: str) -> BatchCaller:
        """Return the cached BatchCaller for a model, creating it if needed.

        Concurrency is set from ``ModelConfig.recommended_max_workers`` via
        :meth:`BatchCaller.from_model`. The executor shares this router's
        rate limiter so per-model RPM is enforced globally.
        """
        if model not in self._executors:
            backend = self.get_backend(model)
            self._executors[model] = BatchCaller.from_model(
                backend, model, rate_limiter=self._rate_limiter,
            )
        return self._executors[model]

    def for_role(self, role: str) -> tuple[LLMBackend, str]:
        """Return ``(backend, model_name)`` for the canonical model of a role.

        Roles defined in the default registry:
        - ``"constitution_gen"`` → ``claude-opus-4-6``
        - ``"translation"``      → ``deepseek-v3.2``
        - ``"uncensored_gen"``   → ``venice-uncensored``
        - ``"uncensored_local"`` → ``venice-uncensored-vllm``

        Raises:
            KeyError: If no model in the registry has this role.
        """
        model = default_model_for_role(role)
        return self.get_backend(model), model

    def generate(
        self,
        model: str,
        messages: list[dict],
        **kwargs,
    ) -> str:
        """Single rate-limited generation routed through the model's executor."""
        return self.get_executor(model).batch_generate(
            [messages], model, **kwargs
        )[0]

    def batch_generate(
        self,
        model: str,
        messages_list: list[list[dict]],
        **kwargs,
    ) -> list[str]:
        """Batch generation routed through the model's executor.

        Picks the right execution mode (native batch / parallel / serial)
        based on backend capabilities. See :meth:`BatchCaller.batch_generate`
        for the dispatch rules.
        """
        return self.get_executor(model).batch_generate(
            messages_list, model, **kwargs
        )

    def clear_executors(self) -> None:
        """Drop cached executors so the next call recreates them.

        Useful when ``recommended_max_workers`` is changed at runtime via
        :func:`register_model`.
        """
        self._executors.clear()


_default_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    """Return the process-wide ModelRouter, creating it on first call."""
    global _default_router
    if _default_router is None:
        _default_router = ModelRouter()
    return _default_router


def clear_router() -> None:
    """Drop the process-wide router (e.g. for tests or env changes)."""
    global _default_router
    _default_router = None
