"""Rate limiting and the parallel/sequential fan-out primitive that backends
without native batching dispatch through.

Both wrappers take a **finished backend** — one configured model — and read
what they need straight off it: ``backend.rpm``, ``backend.max_workers``,
``backend.compute_config``. They do no registry lookups and no param
resolution (that is settled at backend construction, see
``backends/base.py``), and they never call back into
:class:`~redact.llms.client.ModelClient`, which is what *holds* them.

- ``RateLimiter`` — thread-safe sliding-window RPM enforcement, one window per
  key (a model name, or an endpoint id where the provider meters the account).
  A backend with no budget — ``rpm=None``, i.e. anything local — gets **no
  limiter at all**: ``ModelClient`` keys that off ``backend.rpm`` and stores
  ``None``, rather than attaching a live object that no-ops on every call.
- ``BatchCaller`` — parallel-or-sequential fan-out for backends that do no
  native batching. It has no notion of native batching at all; that decision
  belongs to ``ModelClient``, which calls a native-batching backend directly
  and never constructs a ``BatchCaller`` for one. Raises rather than silently
  degrading on a misconfigured combination.
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
    """Thread-safe RPM enforcement over a 60-second sliding window.

    **One instance is one window.** Which window — a model name for a
    per-model budget, an endpoint id for a per-account one — is decided
    entirely by :func:`shared_limiter` before this object is handed to anyone;
    ``client._resolve_rate_limit()` picks the key from the setup's
    ``rate_limit_scope``. So this class never sees a key and never chooses a
    scope: it counts requests against the one window it is.

    **Instances outlive clients, deliberately.** ``ModelClient`` is rebuilt on
    every ``create()`` (there is no client cache), so a window anchored to a
    client would reset constantly and two call sites on one model would each
    start a fresh 60 seconds. :func:`shared_limiter` is what anchors it
    instead — which is also why :class:`ModelClient`'s own default is the
    shared window for its key, not a private limiter.

    **When there is no limiter at all.** ``backend.rpm is None`` means no
    budget, and ``ModelClient`` then stores ``None`` rather than attaching a
    live object that no-ops on every call. Every local setup lands there by
    construction: neither ``VLLMConfig`` nor the introspection setup has an
    ``rpm`` field, so those backends fall back to ``LLMBackend``'s default —
    and a dual-setup entry is exempt on its local binding even though its
    ``.api`` declares an rpm. :meth:`wait_if_needed` keeps its own ``rpm``
    guard regardless, since it is public and takes any backend.

    """

    def __init__(self):
        # ONE window per instance, not a dict keyed by model. Which key an
        # instance serves is decided by shared_limiter() before this object is
        # ever handed out, so an internal per-key dict was a second lookup of
        # the same key that could only ever hold one entry.
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self, backend: "LLMBackend") -> None:
        """Block until making a request against this backend is safe.

        Args:
            backend: The configured model whose ``rpm`` budget to enforce.
                Returns immediately when its ``rpm`` is ``None``. Only the
                budget is read from it — *which* window this is was settled
                when the instance was chosen.
        """
        rpm = backend.rpm
        if rpm is None:
            return

        with self._lock:
            # A loop, not a single check: the lock is released while sleeping,
            # so N workers can all find the window full, all sleep on the same
            # oldest timestamp, and all wake together. Appending unconditionally
            # after one sleep would let every one of them through at once and
            # overshoot the cap. Re-pruning and re-testing on each pass is what
            # actually holds it.
            while True:
                now = time.time()
                # Drops everything older than the window, in place. This is the
                # only thing that trims the list — and it runs on every call,
                # not just when full, so the list cannot grow without bound:
                # the loop below exits only while len < rpm and appends exactly
                # once, so it holds at most `rpm` entries. (Nothing prunes while
                # idle, but the same bound applies — tens of floats.)
                self._timestamps[:] = [
                    t for t in self._timestamps
                    if now - t < _RATE_LIMIT_WINDOW_SECONDS
                ]
                if len(self._timestamps) < rpm:
                    break

                sleep_for = (
                    _RATE_LIMIT_WINDOW_SECONDS - (now - self._timestamps[0]) + 0.1
                )
                if sleep_for <= 0:
                    continue  # the oldest already aged out; re-prune and retry
                logger.info("[rate-limit] %s: pausing %.1fs (%d RPM)",
                            backend.model, sleep_for, rpm)
                # Released while sleeping so other workers on this same window
                # can re-check rather than queueing behind the sleeper — they
                # re-test above, so this costs nothing in correctness.
                #
                # Waiters race for the lock rather than queueing, so which one
                # goes next is arbitrary. That decides scheduling order within
                # a batch and nothing else: BatchCaller reassembles by index
                # and returns only once every item is done.
                self._lock.release()
                try:
                    time.sleep(sleep_for)
                finally:
                    self._lock.acquire()

            self._timestamps.append(time.time())


# One limiter per window key — a model name where the provider meters per
# model, an APIConfig.endpoint_id where it meters the account. **This cache is
# what makes a rate-limit window outlive any one client**, which matters
# because clients are cheap and rebuilt per ``ModelClient.create()``: without
# it, two call sites generating against the same model would each start a
# fresh 60-second window and collectively exceed the budget. Double-checked
# like every other cache here, since clients can be built from a preload
# thread.
_shared_limiters: dict[str, RateLimiter] = {}
_shared_limiters_lock = threading.Lock()


def shared_limiter(key: str) -> RateLimiter:
    """Get (or create) the one limiter every caller on this key shares.

    Args:
        key: The window's identity — ``backend.model`` for a per-model budget,
            ``APIConfig.endpoint_id`` for a per-account one. Callers that pass
            the same key share one window; that is the whole mechanism.
    """
    if key not in _shared_limiters:
        with _shared_limiters_lock:
            if key not in _shared_limiters:
                _shared_limiters[key] = RateLimiter()
    return _shared_limiters[key]


def clear_shared_limiters() -> None:
    """Drop every rate-limit window. For tests, or a registry change."""
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
    ):
        """Wire fan-out around one configured backend.

        Args:
            backend: The configured model to dispatch to.
            rate_limiter: The window to count against, or ``None`` for no
                throttling at all. ``ModelClient`` passes ``None`` only for a
                backend with no budget (``rpm=None``, i.e. any local one) —
                everything else gets the shared window for its key.
            max_workers: Override for ``backend.max_workers``. Omit in normal
                use — the backend's value is already reconciled with its
                transport's capability.
        """
        self._backend = backend
        self._rate_limiter = rate_limiter
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
            self._rate_limiter.wait_if_needed(self._backend)
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
