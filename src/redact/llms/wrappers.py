"""Rate limiting, retry logic, and multithreaded batch execution.

Generalized from the jailbreak reference library:
- RateLimiter:         from _RateLimitedClient (obfuscation.py:47-119)
- with_retries:        from utils.py:52-82
- with_feedback_retries: from utils.py:85-115
- BatchCaller:         from runner script ThreadPoolExecutor patterns
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .base import LLMBackend
from .model_config import get_model_config


class RateLimiter:
    """Thread-safe per-model RPM enforcement using a sliding window.

    Standalone — not tied to any backend. A single instance should be
    shared across all pipeline modules that hit the same API endpoint.

    Algorithm: same as jailbreak _RateLimitedClient — track request
    timestamps per model in a 60-second sliding window, sleep if at
    capacity.
    """

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._timestamps: dict[str, list[float]] = {}
        self._global_lock = threading.Lock()

    def _get_model_state(self, model: str) -> tuple[threading.Lock, list[float]]:
        """Get or create the lock and timestamp list for a model."""
        if model not in self._locks:
            with self._global_lock:
                # Double-check after acquiring global lock
                if model not in self._locks:
                    self._locks[model] = threading.Lock()
                    self._timestamps[model] = []
        return self._locks[model], self._timestamps[model]

    def wait_if_needed(self, model: str) -> None:
        """Block until making a request for this model is safe.

        Enforces the RPM limit from model_config for the given model.
        """
        config = get_model_config(model)
        rpm = config.rpm
        lock, timestamps = self._get_model_state(model)

        with lock:
            now = time.time()
            # Prune timestamps older than 60 seconds
            timestamps[:] = [t for t in timestamps if now - t < 60.0]

            if len(timestamps) >= rpm:
                sleep_for = 60.0 - (now - timestamps[0]) + 0.1
                if sleep_for > 0:
                    print(
                        f"    [rate-limit] {model}: pausing {sleep_for:.1f}s "
                        f"({rpm} RPM)"
                    )
                    # Release lock while sleeping so other models aren't blocked
                    lock.release()
                    try:
                        time.sleep(sleep_for)
                    finally:
                        lock.acquire()
                    # Re-prune after sleeping
                    now = time.time()
                    timestamps[:] = [t for t in timestamps if now - t < 60.0]

            timestamps.append(time.time())


# ---------------------------------------------------------------------------
# Retry wrappers
# ---------------------------------------------------------------------------


def with_retries(
    generate_fn: Callable[[str], str],
    check_fn: Callable[[str, str], tuple[bool, str]],
    num_retries: int = 5,
) -> Callable[[str], tuple[str, str]]:
    """Wrap a generate+check pair with a simple retry loop.

    Each attempt calls generate_fn fresh with no feedback from the
    previous attempt. Use this for non-translation LLM steps where the
    generator does not accept corrective feedback.

    Args:
        generate_fn: callable(text) -> generated_text
        check_fn: callable(original, generated) -> (accepted, reasoning)
        num_retries: Maximum number of attempts.

    Returns:
        callable(text) -> (result, info)
            On success: (result, "")
            On exhaustion: (last_result, "DISCARDED")
    """

    def wrapper(text: str) -> tuple[str, str]:
        last_result = text
        for _ in range(num_retries):
            result = generate_fn(text)
            accepted, _ = check_fn(text, result)
            if accepted:
                return result, ""
            last_result = result
        return last_result, "DISCARDED"

    return wrapper


def with_feedback_retries(
    generate_fn: Callable[..., str],
    check_fn: Callable[[str, str], tuple[bool, str]],
    num_retries: int = 2,
) -> Callable[[str], tuple[str, str]]:
    """Wrap a generate+check pair with a feedback-aware retry loop.

    On each failed attempt, the feedback from check_fn is forwarded to the
    next generate_fn call so the generator can correct its mistakes.

    Args:
        generate_fn: callable(text, feedback="") -> generated_text
        check_fn: callable(original, generated) -> (accepted, feedback)
        num_retries: Maximum number of attempts.

    Returns:
        callable(text) -> (result, info)
            On success: (result, "")
            On exhaustion: (last_result, "DISCARDED; feedback=<feedback>")
    """

    def wrapper(text: str) -> tuple[str, str]:
        feedback = ""
        last_result = text
        for _ in range(num_retries):
            result = generate_fn(text, feedback=feedback)
            accepted, feedback = check_fn(text, result)
            if accepted:
                return result, ""
            last_result = result
        return last_result, f"DISCARDED; feedback={feedback}"

    return wrapper


# ---------------------------------------------------------------------------
# Batch caller
# ---------------------------------------------------------------------------


class BatchCaller:
    """Run multiple LLM calls with optional concurrency and rate limiting.

    When max_workers=1 (default), runs sequentially for easy debugging.
    When max_workers>1, uses ThreadPoolExecutor for parallel execution.

    NOTE: When using a VLLMBackend, keep max_workers=1 — vLLM manages
    GPU memory internally and its batch_generate() is the correct way to
    parallelize. Multiple concurrent generate() calls from threads would
    compete for GPU RAM and likely OOM or deadlock.
    """

    def __init__(
        self,
        backend: LLMBackend,
        rate_limiter: RateLimiter | None = None,
        max_workers: int = 1,
    ):
        self._backend = backend
        self._rate_limiter = rate_limiter
        self._max_workers = max_workers

    def _call_one(
        self, messages: list[dict], model: str, **kwargs
    ) -> str:
        """Make a single rate-limited call."""
        if self._rate_limiter:
            self._rate_limiter.wait_if_needed(model)
        return self._backend.generate(messages, model, **kwargs)

    def run(
        self,
        messages_list: list[list[dict]],
        model: str,
        on_complete: Callable[[int, str], None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Run generation for a list of message sets.

        Args:
            messages_list: List of chat message lists.
            model: Model identifier.
            on_complete: Optional callback(index, result) called after each
                         completion. Useful for incremental checkpointing.
            **kwargs: Passed through to backend.generate().

        Returns:
            List of generated texts, in the same order as messages_list.
        """
        results: list[str | None] = [None] * len(messages_list)

        if self._max_workers <= 1:
            # Sequential
            for i, msgs in enumerate(messages_list):
                result = self._call_one(msgs, model, **kwargs)
                results[i] = result
                if on_complete:
                    on_complete(i, result)
        else:
            # Concurrent
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._call_one, msgs, model, **kwargs): i
                    for i, msgs in enumerate(messages_list)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()  # Propagates exceptions
                    results[idx] = result
                    if on_complete:
                        on_complete(idx, result)

        return results  # type: ignore[return-value]
