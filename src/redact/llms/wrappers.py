"""Rate limiting and per-item fan-out for backends without native batching.

- ``RateLimiter``: thread-safe requests-per-minute limit over a sliding
  60-second window.
- ``BatchCaller``: runs a batch one item per call, in sequence or on a thread
  pool.

Both take a finished backend and read ``rpm``, ``max_workers`` and
``compute_config`` from it.
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

_RATE_LIMIT_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Thread-safe RPM limit over a 60-second sliding window.

    One instance is one window. :func:`shared_limiter` hands out one instance
    per key (a model name, or an endpoint id for an account-wide budget), and
    ``client._resolve_rate_limit()`` picks the key.

    Instances outlive clients. A client is rebuilt on every
    ``ModelClient.create()``, so a window tied to a client would reset each
    time and two call sites on one model would exceed the budget together.
    """

    def __init__(self):
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self, backend: "LLMBackend") -> None:
        """Block until a request against this backend fits in the window.

        Args:
            backend: The model whose ``rpm`` to enforce. Returns immediately
                when its ``rpm`` is ``None``.
        """
        rpm = backend.rpm
        if rpm is None:
            return

        with self._lock:
            # Loop and re-check: the lock is released while sleeping, so
            # several workers can wake together. Each must prune and test
            # again before appending, or they would all pass at once and
            # exceed the limit.
            while True:
                now = time.time()
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
                # Sleep without the lock so other workers are not blocked
                # behind this one.
                self._lock.release()
                try:
                    time.sleep(sleep_for)
                finally:
                    self._lock.acquire()

            self._timestamps.append(time.time())


# One limiter per window key, kept here so that windows outlive clients.
_shared_limiters: dict[str, RateLimiter] = {}
_shared_limiters_lock = threading.Lock()


def shared_limiter(key: str) -> RateLimiter:
    """Return the limiter shared by every caller on this key.

    Args:
        key: ``backend.model`` for a per-model budget, or
            ``APIConfig.endpoint_id`` for a per-account one.
    """
    if key not in _shared_limiters:
        with _shared_limiters_lock:
            if key not in _shared_limiters:
                _shared_limiters[key] = RateLimiter()
    return _shared_limiters[key]


def clear_shared_limiters() -> None:
    """Drop every rate-limit window. For tests, or after a registry change."""
    with _shared_limiters_lock:
        _shared_limiters.clear()


class BatchCaller:
    """Run a batch one item per call, in sequence or on a thread pool.

    For backends without native batching. Concurrency defaults to
    ``backend.max_workers``, which the backend already clamped to 1 if it
    cannot take parallel calls. An explicit ``max_workers`` above 1 on such a
    backend raises in :meth:`run`.
    """

    def __init__(
        self,
        backend: "LLMBackend",
        rate_limiter: RateLimiter | None = None,
        max_workers: int | None = None,
    ):
        """Set up fan-out around one configured backend.

        Args:
            backend: The configured model to call.
            rate_limiter: Window to count calls against, or ``None`` for no
                throttling.
            max_workers: Override for ``backend.max_workers``.
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
        """Make one rate-limited call for a single item (a batch of one)."""
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
        """Raise if ``internals_ids`` is passed to a backend that cannot capture.

        Backends without capture would accept and ignore the ids, so the
        request is rejected here. ``ModelClient.generate()`` makes the same
        check.
        """
        if internals_ids is not None and not self._backend.compute_config.supports_internals:
            raise ValueError(
                f"{type(self._backend).__name__} does not support internals "
                f"capture (supports_internals=False); remove internals_ids/"
                f"internals_id or use an internals-capable backend."
            )

    def _check_concurrency(self) -> None:
        """Raise if ``max_workers > 1`` on a backend that needs series calls.

        The backend clamps its own value at construction, so in normal use
        this fires only for an explicit ``max_workers`` override.
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
        """Generate one reply per message list.

        Sequential when ``max_workers <= 1``, thread-pooled otherwise.

        Args:
            messages_list: One chat message list per item.
            on_complete: Called as ``on_complete(index, result)`` after each
                item finishes. Drives progress ticks.
            system_prompts: One system prompt (or ``None``) per item.
            internals_ids: One capture id (or ``None``) per item. Only valid
                when the backend supports internals capture.
            **kwargs: Passed to ``backend.generate()`` (e.g. ``max_tokens``,
                ``temperature``).

        Returns:
            Generated text, in the same order as ``messages_list``.

        Raises:
            ValueError: A per-item list has the wrong length, the backend
                cannot capture internals, or ``max_workers > 1`` on a
                series-only backend.
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
            for i, msgs in enumerate(messages_list):
                result = self._call_one(msgs, **item_kwargs(i))
                results[i] = result
                if on_complete:
                    on_complete(i, result)
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._call_one, msgs, **item_kwargs(i)): i
                    for i, msgs in enumerate(messages_list)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()  # re-raises a worker's exception
                    results[idx] = result
                    if on_complete:
                        on_complete(idx, result)

        return results  # type: ignore[return-value]


def assert_single_sample_per_call(client: "ModelClient", samples_per_call: int) -> None:
    """Raise if internals capture is combined with several samples per call.

    When one prompt asks for several samples in one completion, the captured
    internals of that forward pass cannot be attributed to any single sample.
    Pipelines with that shape (e.g. ``InputPipeline.run_from_constitution``)
    call this before dispatch with their own samples-per-call setting.

    Args:
        client: The generation client.
        samples_per_call: Samples one LLM call is asked to produce.

    Raises:
        ValueError: The client's backend supports internals capture and
            ``samples_per_call`` is not 1.
    """
    if client.compute_config.supports_internals and samples_per_call != 1:
        raise ValueError(
            f"{type(client.backend).__name__} supports internals capture, but this call "
            f"requests {samples_per_call} samples per LLM call — one forward pass "
            f"can't be attributed to more than one resulting sample. Set the "
            f"samples-per-call parameter to 1, or don't request internals capture."
        )
