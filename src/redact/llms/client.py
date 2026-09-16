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


from .backends import ComputeConfig, LLMBackend, backend_for, resolve_setup
from .model_config import get_model_config
from .progress import ProgressReporter
from .wrappers import BatchCaller, RateLimiter, clear_shared_limiters, shared_limiter

# There is deliberately **no client cache**. A client is a backend plus two
# bound choices — microseconds to build — while everything expensive it
# reaches is already cached on that thing's own identity: SDK connection pools
# per endpoint and vLLM engines per checkpoint (in ``backends/``), rate-limit
# windows per key (``wrappers._shared_limiters``). Caching the cheap wrapper
# on top bought nothing and cost correctness: the cached client held the only
# other reference to an engine, so ``clear_transport_caches()`` freed no GPU
# memory and a rotated API key never took effect. Rebuilding per ``create()``
# also means a runtime ``register_model()`` is picked up immediately.


# Budget already claimed for an endpoint under "endpoint" scope, as
# {endpoint_id: (rpm, first model that declared it)}. Only used to reject
# disagreement — see _resolve_rate_limit().
_endpoint_rpm: dict[str, tuple[int, str]] = {}


def clear_client_cache() -> None:
    """Drop every rate-limit window and endpoint budget claim.

    For tests, or after changing the registry. Named for history: there is no
    client cache any more (see above), but this is still the call that resets
    the per-key state a rebuilt client picks straight back up.
    """
    _endpoint_rpm.clear()
    clear_shared_limiters()


def _resolve_rate_limit(model, config, setup) -> tuple[RateLimiter | None, str | None]:
    """Decide which window this model's calls count against.

    Returns the ``(limiter, key)`` pair for this model. Both halves matter:
    the same key in two limiter instances is still two windows, and one
    instance under two keys likewise. They always come from
    ``wrappers.shared_limiter``, so the window survives this client — clients
    are rebuilt per :meth:`ModelClient.create`, windows must not be.

    The key is the *model* by default, which is right wherever the provider
    meters per model (Venice: three models on one API key, three independent
    budgets). Under ``rate_limit_scope="endpoint"`` it is the endpoint id
    instead, so every model behind the same provider+URL+key shares one
    window and an account-metered provider is not handed N times its budget.

    ``(None, None)`` means "no budget, so no window" — every local setup,
    whose ``rpm`` is ``None``.

    Raises:
        ValueError: If two models sharing an endpoint disagree about ``rpm``.
            One window cannot honour two budgets, and silently picking either
            would over- or under-spend with nothing to point at. Checked on
            every call (there is no client cache to skip it), so a runtime
            ``register_model`` is covered and it fires before any request.
    """
    api = getattr(config, setup, None) if setup == "api" else None
    if api is None:
        # No API setup means no budget — a local binding carries rpm=None.
        return None, None
    if api.rate_limit_scope != "endpoint":
        return shared_limiter(model), model

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

        **Prefer :meth:`create`.** This constructor is the escape hatch for a
        backend the registry doesn't describe, and it can only see what is on
        the backend — which leaves three things it cannot get right on its own:

        - **Account-metered providers.** ``rate_limit_scope`` lives in the
          registry, so this defaults the window to one per *model*. For an
          endpoint-scoped provider (Anthropic) two models on one key then hold
          two windows and collectively exceed the account budget. Pass
          ``limit_key=config.api.endpoint_id`` for every client sharing that
          account, or use :meth:`create`.
        - **The rpm-agreement check.** ``_resolve_rate_limit``'s ValueError for
          two models on one endpoint declaring different ``rpm`` is raised in
          :meth:`create`; nothing checks it here.
        - **Whatever the backend was built with.** A backend constructed
          directly rather than through ``from_config`` falls back to
          ``LLMBackend``'s defaults — notably ``rpm=None``, i.e. no rate
          limiting at all, and ``max_workers=1``. That is silent: an unmetered
          client looks identical to a metered one until you hit 429s.

        Args:
            backend: The configured model. Everything about *what* to call is
                already on it; this only decides *how* a batch runs.
            rate_limiter: The window this client's calls count against.
                Defaults to the shared window for ``limit_key``, the same one
                :meth:`create` uses — so a directly-constructed client and a
                created one for the same model count against *one* budget
                rather than two. Pass an explicit limiter only to deliberately
                isolate a window. Ignored entirely for a backend with no
                budget (see below).
            limit_key: Which shared window to join; defaults to the model's
                own name. :meth:`create` supplies the endpoint id instead for
                a provider that meters the account. Used *only* to select the
                limiter instance — a limiter is one window, so the key is not
                passed any further down.
        """
        self._backend = backend
        self._limit_key = limit_key or backend.model
        # Execution strategy is decided ONCE, here — never per call. A
        # native-batching transport takes the whole list in one pass and needs
        # no wrapper; everything else fans out through a BatchCaller, whose
        # worker count the backend already clamped against its own capability.
        #
        # No budget, no limiter — a local binding carries rpm=None (neither
        # VLLMConfig nor IntrospectConfig has an rpm field), so it gets None
        # rather than a live object that no-ops on every call. Keyed on rpm
        # and NOT on _native below: "does this have a budget" and "how does a
        # batch run" are independent facts that merely coincide today, so a
        # metered native-batching transport would still be limited correctly.
        #
        # Defaulting to the SHARED window, not a private one: with no client
        # cache, `create()` and direct construction are otherwise two paths to
        # two independent windows on one budget. Same key, same window, however
        # the client was built — the caller has to ask for isolation.
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
        """Build a ready-to-call model — the one factory callers should use.

        Builds a **fresh** client every call (~3us: a registry lookup, a setup
        resolution and an object). Nothing expensive is rebuilt — the SDK
        connection pool or vLLM engine comes from its own cache in
        ``backends/``, and the rate-limit window from ``shared_limiter``, so
        two clients for one model still share one budget. ``create(m) is
        create(m)`` is therefore False, deliberately: identity was never the
        contract, the shared window is.

        Args:
            model: Registered model name.
            backend_type: Optional override selecting which setup to bind on an
                entry with more than one, e.g. ``"vllm"`` to run locally a model
                that defaults to its hosted endpoint.

        Returns:
            A client for that model and setup.

        Raises:
            KeyError: If the model isn't registered, or the setup's API-key
                env var is not set.
            ValueError: If the setup can't be resolved or the entry lacks it.
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

        # Progress is the only per-item hook: the public on_complete callback
        # this used to chain with is gone (resume is ledger-driven per chunk,
        # so nothing ever checkpointed off it). BatchCaller keeps its own
        # on_complete param — that is how these ticks reach it.
        on_complete = (
            ProgressReporter(progress, len(messages_list)).on_complete
            if progress is not None else None
        )

        if self._native:
            # One engine pass over the whole list already IS the batch —
            # neither the limiter's fan-out nor the BatchCaller adds anything.
            # Dead for vLLM (rpm=None → no limiter); live only if a metered
            # native-batching transport is ever added, and then per batch.
            if self._rate_limiter is not None:
                self._rate_limiter.wait_if_needed(self._backend)
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
