"""A configured model plus its dispatch strategy — the unit callers pass around.

A :class:`~redact.llms.backends.base.LLMBackend` is already one fully
configured model: name, generation defaults, provider params, rate budget and
worker budget are all bound when it is built (see ``backends/base.py``).
What it does *not* know is how a whole batch should be run.

That is all :class:`ModelClient` adds. It wraps a finished backend with the
two things that turn "make one call" into "run a batch": a rate limiter and,
for transports without native batching, a :class:`~redact.llms.wrappers.BatchCaller`.
Which of those applies is read once, at construction, off the backend's
:attr:`~redact.llms.backends.base.ComputeConfig` — never per call, and never
from the registry. The client does no resolution: there is nothing left to
resolve.

Build one with :meth:`ModelClient.create`::

    client = ModelClient.create("venice-uncensored")
    replies = client.generate(messages_list)
"""

from collections.abc import Callable

from .backends import ComputeConfig, LLMBackend, backend_for, resolve_setup
from .model_config import get_model_config
from .progress import ProgressReporter
from .wrappers import BatchCaller, RateLimiter, clear_shared_limiters, shared_limiter

# One client per (model, resolved setup) — see ModelClient.create().
_client_cache: dict[tuple[str, str], "ModelClient"] = {}


# Budget already claimed for an endpoint under "endpoint" scope, as
# {endpoint_id: (rpm, first model that declared it)}. Only used to reject
# disagreement — see _resolve_rate_limit().
_endpoint_rpm: dict[str, tuple[int, str]] = {}


def clear_client_cache() -> None:
    """Drop every cached client. For tests, or after changing the registry."""
    _client_cache.clear()
    _endpoint_rpm.clear()
    clear_shared_limiters()


def _resolve_rate_limit(model, config, setup) -> tuple[RateLimiter | None, str | None]:
    """Decide which window this model's calls count against.

    ``(None, None)`` means "private limiter keyed on the model" — the default,
    correct wherever the provider meters per model, and for every local setup
    (whose ``rpm`` is ``None``, making the limiter inert anyway).

    Under ``rate_limit_scope="endpoint"`` the model instead shares one limiter
    *and* one key with every other model behind the same provider+URL+key, so
    an account-metered provider is not handed N times its budget.

    Raises:
        ValueError: If two models sharing an endpoint disagree about ``rpm``.
            One window cannot honour two budgets, and silently picking either
            would over- or under-spend with nothing to point at. Checked here
            rather than at load time so it covers runtime ``register_model``
            too, and it still fires before any request goes out.
    """
    api = getattr(config, setup, None) if setup == "api" else None
    if api is None or api.rate_limit_scope != "endpoint":
        return None, None

    endpoint = api.endpoint_id
    claimed = _endpoint_rpm.get(endpoint)
    if claimed is not None and claimed[0] != api.rpm:
        raise ValueError(
            f"Models {claimed[1]!r} and {model!r} share endpoint {endpoint} "
            f"with rate_limit_scope='endpoint' but declare different rpm "
            f"({claimed[0]} vs {api.rpm}). An account-wide budget is one "
            f"number: make every model on this endpoint declare the same rpm."
        )
    _endpoint_rpm.setdefault(endpoint, (api.rpm, model))
    return shared_limiter(endpoint), endpoint


class ModelClient:
    """One configured model, with its batch strategy bound."""

    def __init__(
        self,
        backend: LLMBackend,
        rate_limiter: RateLimiter | None = None,
        limit_key: str | None = None,
    ):
        """Wire a finished backend for batch dispatch.

        Args:
            backend: The configured model. Everything about *what* to call is
                already on it; this only decides *how* a batch runs.
            rate_limiter: Optional shared limiter, so several clients can
                enforce one window. Defaults to a private one.
            limit_key: Identity that window belongs to; defaults to the
                model's own name. :meth:`create` supplies the endpoint id
                instead for a provider that meters the account — and supplies
                the matching shared limiter with it, since one without the
                other still yields independent windows.
        """
        self._backend = backend
        self._limit_key = limit_key or backend.model
        # Execution strategy is decided ONCE, here — never per call. A
        # native-batching transport takes the whole list in one pass and needs
        # no wrapper; everything else fans out through a BatchCaller, whose
        # worker count the backend already clamped against its own capability.
        # wait_if_needed() is itself a no-op when backend.rpm is None, so a
        # local client is genuinely inert rather than "limited at 999".
        self._rate_limiter = rate_limiter or RateLimiter()
        self._native = backend.compute_config.supports_native_batching
        self._caller = (
            None if self._native
            else BatchCaller(backend, self._rate_limiter, limit_key=self._limit_key)
        )

    @classmethod
    def create(cls, model: str, backend_type: str | None = None) -> "ModelClient":
        """Build a ready-to-call model — the one factory callers should use.

        Cached per ``(model, resolved setup)`` so repeated calls share one
        client — and therefore one rate-limit window — rather than each
        starting a fresh budget.

        Args:
            model: Registered model name.
            backend_type: Optional override selecting which setup to bind on an
                entry with more than one, e.g. ``"vllm"`` to run locally a model
                that defaults to its hosted endpoint.

        Returns:
            The cached client for that model and setup.

        Raises:
            KeyError: If the model isn't registered, or the setup's API-key
                env var is not set.
            ValueError: If the setup can't be resolved or the entry lacks it.
        """
        config = get_model_config(model)
        setup = resolve_setup(config, backend_type)
        key = (model, setup)
        if key not in _client_cache:
            limiter, limit_key = _resolve_rate_limit(model, config, setup)
            _client_cache[key] = cls(
                backend_for(config, setup), limiter, limit_key=limit_key
            )
        return _client_cache[key]

    @property
    def model(self) -> str:
        """Registry name of the model this client calls."""
        return self._backend.model

    @property
    def backend(self) -> LLMBackend:
        """The configured backend. Escape hatch; prefer :meth:`generate`."""
        return self._backend

    @property
    def compute_config(self) -> ComputeConfig:
        """The transport's dispatch-relevant capability flags."""
        return self._backend.compute_config

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        system_prompts: str | list[str | None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        on_complete: Callable[[int, str], None] | None = None,
        progress: str | None = None,
        **kwargs,
    ) -> list[str]:
        """Send a batch of message lists, get a batch of replies.

        Rate limiting and the native-pass-vs-fan-out choice are already wired
        into this client; the model's own defaults are already on its backend.
        Callers pass messages and get text.

        Args:
            messages_list: One chat message list per item. Empty list → ``[]``.
            system_prompts: Optional system prompt — one string for the whole
                batch, or one (or ``None``) per item. Merged with any
                system-role message already in ``messages_list``.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: One capture id (or ``None``) per item. Only valid
                on an internals-capable transport.
            on_complete: ``(index, result)`` callback per completion.
            progress: Label enabling throttled progress ticks.
            **kwargs: Transport-specific extras.

        Returns:
            Generated text, one per item, in input order.

        Raises:
            ValueError: If ``internals_ids`` is passed to a transport that
                can't capture, or a per-item list's length doesn't match
                ``messages_list``.
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

        if progress is not None:
            reporter = ProgressReporter(progress, len(messages_list))
            if on_complete is None:
                on_complete = reporter.on_complete
            else:
                _user_cb = on_complete

                def on_complete(i, r, _cb=_user_cb, _rep=reporter):
                    _cb(i, r)
                    _rep.on_complete(i, r)

        if self._native:
            # One engine pass over the whole list already IS the batch —
            # neither the limiter's fan-out nor the BatchCaller adds anything.
            self._rate_limiter.wait_if_needed(self._backend, self._limit_key)
            results = self._backend.generate(
                messages_list,
                system_prompts=system_prompts,
                max_tokens=max_tokens, temperature=temperature,
                internals_ids=internals_ids, **kwargs,
            )
            if on_complete:
                for i, r in enumerate(results):
                    on_complete(i, r)
            return results

        # BatchCaller dispatches item by item, so a batch-wide system prompt
        # is expanded to one per item here — the only shape it accepts.
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(messages_list)
        return self._caller.run(
            messages_list, on_complete=on_complete,
            system_prompts=system_prompts,
            max_tokens=max_tokens, temperature=temperature,
            internals_ids=internals_ids, **kwargs,
        )

    def rename_capture(self, old_internals_id: str, new_internals_id: str) -> None:
        """Relabel a captured-internals folder once its final id is known.

        For stages whose real id only exists after generation (paraphrase's
        is a hash of the output text), capture under a provisional id and
        rename after. A no-op on any backend that captures nothing.
        """
        self._backend.rename_capture(old_internals_id, new_internals_id)

    def __repr__(self) -> str:
        return (
            f"ModelClient(model={self.model!r}, "
            f"backend={type(self._backend).__name__})"
        )
