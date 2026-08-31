"""Rate limiting and the parallel/sequential fan-out primitive that backends
without native batching dispatch through.

Both wrappers take a **finished backend** — one configured model — and read
what they need straight off it: ``backend.rpm``, ``backend.max_workers``,
``backend.compute_config``. They do no registry lookups and no param
resolution (that is settled at backend construction, see
``backends/base.py``), and they never call back into
:class:`~redact.llms.client.ModelClient`, which is what *holds* them.

- ``RateLimiter`` — thread-safe sliding-window RPM enforcement, keyed per
  model. Inert for a backend with ``rpm=None``, i.e. anything local.
- ``BatchCaller`` — parallel-or-sequential fan-out for backends that do no
  native batching. It has no notion of native batching at all; that decision
  belongs to ``ModelClient``, which calls a native-batching backend directly
  and never constructs a ``BatchCaller`` for one. Raises rather than silently
  degrading on a misconfigured combination.

Generalized from the jailbreak reference library:
- RateLimiter:         from _RateLimitedClient (obfuscation.py:47-119)
- BatchCaller:         from runner script ThreadPoolExecutor patterns
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends import LLMBackend
    from .client import ModelClient

logger = logging.getLogger(__name__)

# RPM is requests-per-minute, so the sliding window RateLimiter prunes/limits
# on a 60-second window.
_RATE_LIMIT_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Thread-safe per-model RPM enforcement using a sliding window.

    The budget comes from the backend passed to :meth:`wait_if_needed` — a
    backend built from a model's *local* setup carries ``rpm=None`` and is
    correctly exempt, even when the same registry entry also describes a
    rate-limited hosted endpoint.

    **Scope in practice: one limiter per (model, setup).**
    :meth:`ModelClient.create` passes no limiter, so each client builds its
    own, and clients are cached per ``(model, setup)``. The per-model keying
    below is therefore vestigial — each instance only ever holds one key — but
    it is kept because :class:`ModelClient` still accepts a shared limiter,
    which is the hook for the case below.

    **That scope is right for a per-model cap and wrong for a per-account
    one.** Venice prices and limits each model separately (75 / 20 / 20 RPM
    across three models on one key), so independent windows enforce exactly
    what is declared. Anthropic caps the *account*, so several Anthropic
    models would each get their own full budget and collectively exceed it —
    two models at ``rpm=5`` would issue 10/min against a 5/min account. This
    is latent, not live: ``claude-opus-4-6`` is currently the only Anthropic
    entry, so its per-model limiter *is* the account limiter. Registering a
    second one is what makes it real, and the fix then is to hand both clients
    one shared limiter keyed on endpoint identity (the ``base_url`` +
    ``api_key_env`` pair the SDK client already caches on) rather than to
    change anything here.

    Algorithm: same as jailbreak _RateLimitedClient — track request
    timestamps per model in a 60-second sliding window, sleep if at
    capacity.
    """

    def __init__(self):
        # One dict keyed by model, each value the (lock, timestamps) pair —
        # not two parallel dicts. _get_model_state()'s fast path below reads
        # this dict without the global lock, so the pair must come into
        # existence as a single atomic assignment; two separate dict
        # assignments (self._locks[m]=...; self._timestamps[m]=...) would
        # leave a narrow window where a concurrent fast-path reader could
        # observe the lock but not yet the timestamps list.
        self._state: dict[str, tuple[threading.Lock, list[float]]] = {}
        self._global_lock = threading.Lock()

    def _get_model_state(self, model: str) -> tuple[threading.Lock, list[float]]:
        """Get or create the lock and timestamp list for a model."""
        if model not in self._state:
            with self._global_lock:
                # Double-check after acquiring global lock
                if model not in self._state:
                    self._state[model] = (threading.Lock(), [])
        return self._state[model]

    def wait_if_needed(self, backend: "LLMBackend", key: str | None = None) -> None:
        """Block until making a request against this backend is safe.

        Args:
            backend: The configured model whose ``rpm`` budget to enforce.
                A no-op when ``backend.rpm`` is ``None`` (any local
                transport).
            key: Identity the window belongs to. Defaults to
                ``backend.model`` — one window per model, which is what a
                per-model cap needs. A caller enforcing a per-account cap
                passes the endpoint id instead, so every model behind that
                key shares one window. Sharing the *instance* is not enough
                on its own: two models in one limiter still get two windows
                unless they also agree on this key.
        """
        rpm = backend.rpm
        if rpm is None:
            return
        model = key or backend.model
        lock, timestamps = self._get_model_state(model)

        with lock:
            now = time.time()
            # Prune timestamps older than the window
            timestamps[:] = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW_SECONDS]

            if len(timestamps) >= rpm:
                sleep_for = _RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0]) + 0.1
                if sleep_for > 0:
                    logger.info("[rate-limit] %s: pausing %.1fs (%d RPM)",
                                model, sleep_for, rpm)
                    # Release lock while sleeping so other models aren't blocked
                    lock.release()
                    try:
                        time.sleep(sleep_for)
                    finally:
                        lock.acquire()
                    # Re-prune after sleeping
                    now = time.time()
                    timestamps[:] = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW_SECONDS]

            timestamps.append(time.time())


# Limiters shared by every model behind one endpoint, for providers that meter
# the account rather than the model. Keyed on APIConfig.endpoint_id. Module
# level because the sharing has to outlive any one client — that is the whole
# point — and double-checked like every other cache here, since clients can be
# built from a preload thread.
_shared_limiters: dict[str, RateLimiter] = {}
_shared_limiters_lock = threading.Lock()


def shared_limiter(endpoint_id: str) -> RateLimiter:
    """Get (or create) the one limiter every model on this endpoint shares."""
    if endpoint_id not in _shared_limiters:
        with _shared_limiters_lock:
            if endpoint_id not in _shared_limiters:
                _shared_limiters[endpoint_id] = RateLimiter()
    return _shared_limiters[endpoint_id]


def clear_shared_limiters() -> None:
    """Drop the endpoint-shared limiters. For tests, or a registry change."""
    with _shared_limiters_lock:
        _shared_limiters.clear()


class BatchCaller:
    """Parallel-or-sequential fan-out, with rate limiting, for backends whose
    transport does no native batching.

    Reads only ``supports_parallel_calls`` and ``supports_internals`` off the
    backend's ``compute_config`` — never ``supports_native_batching``, which
    ``ModelClient`` owns: it calls a native-batching backend directly and
    never constructs a ``BatchCaller`` for one.

    Concurrency comes from ``backend.max_workers``, which the backend already
    clamped at construction against its own parallelism capability. Passing
    ``max_workers`` here overrides that — ``run()`` re-checks the invariant, so
    an override that a series-only transport can't honour raises rather than
    silently over-dispatching.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        rate_limiter: RateLimiter | None = None,
        max_workers: int | None = None,
        limit_key: str | None = None,
    ):
        """Wire fan-out around one configured backend.

        Args:
            backend: The configured model to dispatch to.
            rate_limiter: Optional shared limiter; without one, no throttling.
            max_workers: Override for ``backend.max_workers``. Omit in normal
                use — the backend's value is already reconciled with its
                transport's capability.
            limit_key: Identity the rate window belongs to, forwarded to
                :meth:`RateLimiter.wait_if_needed`. ``None`` means per model.
        """
        self._backend = backend
        self._rate_limiter = rate_limiter
        self._limit_key = limit_key
        self._max_workers = (
            backend.max_workers if max_workers is None else max_workers
        )

    @property
    def backend(self) -> "LLMBackend":
        return self._backend

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def _call_one(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        internals_id: str | None = None,
        **kwargs,
    ) -> str:
        """Make a single rate-limited call.

        Wraps the item into a batch of one for the backend's (always
        batch-shaped) ``generate()`` and unwraps the one result.

        Every ``run()`` dispatch, sequential or thread-pooled, goes through
        here, so exactly one item reaches each underlying ``generate()`` call.
        That is a property of this fan-out, not a constraint any backend
        imposes: ``TransformersIntrospectionBackend.generate()`` is fully
        list-shaped and loops internally, it simply never receives more than
        one item by this route. (Not to be confused with
        :func:`assert_single_sample_per_call`, which is about one *prompt*
        yielding several samples in one completion — a different problem.)
        """
        if self._rate_limiter:
            self._rate_limiter.wait_if_needed(self._backend, self._limit_key)
        results = self._backend.generate(
            [messages],
            system_prompts=[system_prompt],
            internals_ids=[internals_id] if internals_id is not None else None,
            **kwargs,
        )
        return results[0]

    def _check_internals_support(self, internals_ids: list | None) -> None:
        """Raise a clear error if internals capture is requested but unsupported.

        Guards the one place ``internals_id(s)`` cross from generic pipeline
        kwargs into an actual backend call. Without this, an unsupported
        kwarg would silently ride ``**kwargs`` into e.g. Venice/Anthropic's
        SDK client call and crash there instead — this fails fast, before
        dispatch, with a message that says what happened.
        """
        if internals_ids is not None and not self._backend.compute_config.supports_internals:
            raise ValueError(
                f"{type(self._backend).__name__} does not support internals "
                f"capture (supports_internals=False); remove internals_ids/"
                f"internals_id or use an internals-capable backend."
            )

    def _check_concurrency(self) -> None:
        """Raise if ``max_workers>1`` on a backend that requires series calls.

        Checks at dispatch time, not only against the backend's own
        construction-time clamp, so a caller who overrode ``max_workers`` (or
        mutated it afterwards) still hits a hard error rather than silently
        over-concurrent dispatch.
        """
        if not self._backend.compute_config.supports_parallel_calls and self._max_workers > 1:
            raise ValueError(
                f"{type(self._backend).__name__} requires series calls; "
                f"max_workers must be 1 (got {self._max_workers}). "
                f"Omit max_workers so the backend's own clamped value is used."
            )

    def run(
        self,
        messages_list: list[list[dict]],
        on_complete: Callable[[int, str], None] | None = None,
        system_prompts: list[str | None] | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Fan out generation for a list of message sets — sequential if
        ``max_workers<=1``, thread-pooled otherwise.

        Args:
            messages_list: List of chat message lists.
            on_complete: Optional callback(index, result) called after each
                         completion. Useful for incremental checkpointing.
            system_prompts: Optional per-item system prompts (same length as
                messages_list) — each forwarded as that item's own
                ``system_prompts`` to ``backend.generate()``. ``None`` (the
                whole param) means no item has one.
            internals_ids: Optional per-item ids (same length as
                messages_list) requesting internals capture — only valid
                when ``backend.compute_config.supports_internals`` is True.
                Raises ``ValueError`` if passed to a backend without it.
            **kwargs: Passed through to ``backend.generate()`` (e.g.
                ``max_tokens``/``temperature`` overrides).

        Returns:
            List of generated texts, in the same order as messages_list.
        """
        self._check_concurrency()
        self._check_internals_support(internals_ids)
        if internals_ids is not None and len(internals_ids) != len(messages_list):
            raise ValueError(
                f"internals_ids must be the same length as messages_list "
                f"({len(internals_ids)} != {len(messages_list)})."
            )
        if system_prompts is not None and len(system_prompts) != len(messages_list):
            raise ValueError(
                f"system_prompts must be the same length as messages_list "
                f"({len(system_prompts)} != {len(messages_list)})."
            )

        if not messages_list:
            return []

        results: list[str | None] = [None] * len(messages_list)

        def item_kwargs(i: int) -> dict:
            call_kwargs = dict(kwargs)
            if internals_ids is not None:
                call_kwargs["internals_id"] = internals_ids[i]
            if system_prompts is not None:
                call_kwargs["system_prompt"] = system_prompts[i]
            return call_kwargs

        if self._max_workers <= 1:
            # Sequential
            for i, msgs in enumerate(messages_list):
                result = self._call_one(msgs, **item_kwargs(i))
                results[i] = result
                if on_complete:
                    on_complete(i, result)
        else:
            # Concurrent
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._call_one, msgs, **item_kwargs(i)): i
                    for i, msgs in enumerate(messages_list)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()  # Propagates exceptions
                    results[idx] = result
                    if on_complete:
                        on_complete(idx, result)

        return results  # type: ignore[return-value]


def assert_single_sample_per_call(client: "ModelClient", samples_per_call: int) -> None:
    """Guard against internals capture on a multi-sample-per-call request.

    A transport has no visibility into whether the *prompt* it's given asks for
    one sample or several in one completion (e.g. content-moderation input
    generation's ``samples_per_entry``) — that's business logic only the caller
    knows. When ``samples_per_call > 1``, one forward pass produces several
    logical samples at once, so its captured internals can't be attributed to
    any one of them; raise rather than silently capturing something meaningless.

    Callers with a multi-sample-per-call shape (e.g.
    ``InputPipeline.run_from_constitution``) should call this before dispatch,
    passing their own ``samples_per_entry``/``samples_per_request``.
    """
    if client.compute_config.supports_internals and samples_per_call != 1:
        raise ValueError(
            f"{type(client.backend).__name__} supports internals capture, but this call "
            f"requests {samples_per_call} samples per LLM call — one forward pass "
            f"can't be attributed to more than one resulting sample. Set the "
            f"samples-per-call parameter to 1, or don't request internals capture."
        )
