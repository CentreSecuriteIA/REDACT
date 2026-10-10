"""ModelClient: a configured backend plus how its batches are dispatched.

A backend already holds everything about one model. The client adds a rate
limiter and, for backends without native batching, a
:class:`~redact.llms.wrappers.BatchCaller`. Which of these apply is decided
once, at construction, from the backend's ``rpm`` and ``compute_config``.

Usage::

    client = ModelClient.create("venice-uncensored")
    replies = client.generate(messages_list)
"""


from .backends import ComputeConfig, LLMBackend, backend_for, resolve_setup
from .model_config import get_model_config
from .progress import ProgressReporter
from .wrappers import BatchCaller, RateLimiter, clear_shared_limiters, shared_limiter

# Clients are not cached. They are cheap to build, and what they use that is
# expensive (SDK connection pools, vLLM engines, rate-limit windows) is cached
# elsewhere under its own key.

# Budget claimed per endpoint under "endpoint" scope, as
# {endpoint_id: (rpm, first model that declared it)}. It exists to reject two
# models on one endpoint that declare different rpm.
_endpoint_rpm: dict[str, tuple[int, str]] = {}


def clear_client_cache() -> None:
    """Drop every rate-limit window and endpoint budget claim.

    For tests, or after changing the registry. Clients built before the call
    keep their old window.
    """
    _endpoint_rpm.clear()
    clear_shared_limiters()


def _resolve_rate_limit(model, config, setup) -> tuple[RateLimiter | None, str | None]:
    """Pick the rate-limit window this model's calls count against.

    The window key is the model name. Under ``rate_limit_scope="endpoint"``
    it is the endpoint id, so every model on that endpoint shares one window.
    The limiter comes from ``wrappers.shared_limiter`` and outlives the client.

    Returns:
        ``(limiter, key)``, or ``(None, None)`` for a local setup, which has
        no budget.

    Raises:
        ValueError: Two models sharing an endpoint declare different ``rpm``.
    """
    api = getattr(config, setup, None) if setup == "api" else None
    if api is None:
        return None, None
    if api.rate_limit_scope != "endpoint":
        return shared_limiter(model), model

    endpoint = api.endpoint_id
    claimed = _endpoint_rpm.get(endpoint)
    # A claim by the same model is a refresh: it was registered again.
    if claimed is None or claimed[1] == model:
        _endpoint_rpm[endpoint] = (api.rpm, model)
    elif claimed[0] != api.rpm:
        raise ValueError(
            f"Models {claimed[1]!r} and {model!r} share endpoint {endpoint} "
            f"with rate_limit_scope='endpoint' but declare different rpm "
            f"({claimed[0]} vs {api.rpm}). An account-wide budget is one "
            f"number: make every model on this endpoint declare the same rpm."
        )
    return shared_limiter(endpoint), endpoint


class ModelClient:
    """One configured model and the way its batches are dispatched."""

    def __init__(
        self,
        backend: LLMBackend,
        rate_limiter: RateLimiter | None = None,
        limit_key: str | None = None,
    ):
        """Wrap a finished backend for batch dispatch.

        Prefer :meth:`create`, which reads the rate-limit scope from the
        registry. Built by hand, a client gets one window per model, so for an
        account-metered provider (Anthropic) pass
        ``limit_key=config.api.endpoint_id``. Nothing here checks that models
        sharing an endpoint declare the same ``rpm``.

        Args:
            backend: The configured model.
            rate_limiter: Window to count calls against. Defaults to the
                shared window for ``limit_key``. Ignored when ``backend.rpm``
                is ``None``.
            limit_key: Which shared window to join. Defaults to the model
                name.
        """
        self._backend = backend
        self._limit_key = limit_key or backend.model
        # A backend with no rpm gets no limiter. A native-batching backend is
        # called directly; any other goes through a BatchCaller.
        self._rate_limiter = (
            None if backend.rpm is None
            else (rate_limiter or shared_limiter(self._limit_key))
        )
        self._native = backend.compute_config.supports_native_batching
        self._caller = (
            None if self._native
            else BatchCaller(backend, self._rate_limiter)
        )

    @classmethod
    def create(cls, model: str, backend_type: str | None = None) -> "ModelClient":
        """Build a client for a registered model.

        Cheap to call repeatedly: engines, SDK connection pools and rate-limit
        windows are cached elsewhere and shared between clients.

        Args:
            model: Registered model name.
            backend_type: Which setup to bind ("api", "vllm" or "introspect")
                when the entry has more than one. Defaults to the entry's own.

        Raises:
            KeyError: Unknown model, or its API-key env var is not set.
            ValueError: The entry has no such setup, or models sharing its
                endpoint declare different ``rpm``.
        """
        config = get_model_config(model)
        setup = resolve_setup(config, backend_type)
        limiter, limit_key = _resolve_rate_limit(model, config, setup)
        return cls(backend_for(config, setup), limiter, limit_key=limit_key)

    @property
    def model(self) -> str:
        """Registry name of the model this client calls."""
        return self._backend.model

    @property
    def backend(self) -> LLMBackend:
        """The configured backend."""
        return self._backend

    @property
    def compute_config(self) -> ComputeConfig:
        """The backend's capability flags."""
        return self._backend.compute_config

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        progress: str | None = None,
        **kwargs,
    ) -> list[str]:
        """Send a batch of message lists and return one reply per item.

        Args:
            messages_list: One chat message list per item. An empty list
                returns ``[]``.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: One capture id (or ``None``) per item. Only valid
                on a backend that supports internals capture.
            progress: Label for progress logging. ``None`` logs nothing.
            **kwargs: Backend-specific extras.

        Returns:
            Generated text, one per item, in input order.

        Raises:
            ValueError: ``internals_ids`` is passed to a backend that cannot
                capture, or its length differs from ``messages_list``.
        """
        if not messages_list:
            return []
        if internals_ids is not None:
            if not self.compute_config.supports_internals:
                raise ValueError(
                    f"{type(self._backend).__name__} does not support internals "
                    f"capture (supports_internals=False); remove internals_ids/"
                    f"internals_id or use an internals-capable backend."
                )
            if len(internals_ids) != len(messages_list):
                raise ValueError(
                    f"internals_ids must be the same length as messages_list "
                    f"({len(internals_ids)} != {len(messages_list)})."
                )

        on_complete = (
            ProgressReporter(progress, len(messages_list)).on_complete
            if progress is not None else None
        )

        if self._native:
            # One engine pass handles the whole list. A limiter, if there is
            # one, is charged once per batch.
            if self._rate_limiter is not None:
                self._rate_limiter.wait_if_needed(self._backend)
            results = self._backend.generate(
                messages_list,
                max_tokens=max_tokens, temperature=temperature,
                internals_ids=internals_ids, **kwargs,
            )
            if on_complete:
                for i, r in enumerate(results):
                    on_complete(i, r)
            return results

        return self._caller.run(
            messages_list, on_complete=on_complete,
            max_tokens=max_tokens, temperature=temperature,
            internals_ids=internals_ids, **kwargs,
        )

    def rename_capture(self, old_internals_id: str, new_internals_id: str) -> None:
        """Rename a captured-internals folder once its final id is known.

        For stages whose id is a hash of the generated text: they capture
        under a provisional id and rename afterwards. Does nothing on a
        backend that does not capture.
        """
        self._backend.rename_capture(old_internals_id, new_internals_id)

    def __repr__(self) -> str:
        return (
            f"ModelClient(model={self.model!r}, "
            f"backend={type(self._backend).__name__})"
        )
