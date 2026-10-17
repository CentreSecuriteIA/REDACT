"""One event loop on one daemon thread, for every async model call.

An async SDK client's connection pool belongs to the loop that opened its
connections, so every async client in the process runs on this loop.
Synchronous code hands it coroutines through :func:`run` and :func:`submit`.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")

# Started on first use and never stopped.
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_start_lock = threading.Lock()


def _serve(started: "concurrent.futures.Future[asyncio.AbstractEventLoop]") -> None:
    """Build the loop on this thread, report it, and run it."""
    try:
        # Built here, not by the starter: on Windows a loop built on the main
        # thread takes over the process's signal wakeup socket.
        loop = asyncio.new_event_loop()
    except BaseException as exc:  # noqa: BLE001 — the starter is waiting on this
        started.set_exception(exc)
        return
    loop.call_soon(started.set_result, loop)
    loop.run_forever()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    if _loop is None:
        with _start_lock:
            # Re-check under the lock so the process gets one loop.
            if _loop is None:
                started: concurrent.futures.Future[asyncio.AbstractEventLoop] = (
                    concurrent.futures.Future()
                )
                # A daemon: the loop never keeps the interpreter from exiting.
                _thread = threading.Thread(
                    target=_serve, args=(started,), name="redact-aio", daemon=True
                )
                _thread.start()
                _loop = started.result()
    return _loop


def submit(coro: Coroutine[Any, Any, _T]) -> "concurrent.futures.Future[_T]":
    """Schedule a coroutine on the loop and return at once.

    Args:
        coro: The coroutine to run.

    Returns:
        A future for its result.
    """
    return asyncio.run_coroutine_threadsafe(coro, _get_loop())


def run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine on the loop and block for its result.

    Args:
        coro: The coroutine to run.

    Returns:
        What the coroutine returns. Its exception propagates. When the wait
        itself is interrupted (Ctrl-C, an exit from a signal handler), the
        coroutine is cancelled and the interrupt propagates.

    Raises:
        RuntimeError: Called from the loop's own thread, where blocking for
            the result would stop the loop that has to produce it.
    """
    if threading.current_thread() is _thread:
        coro.close()
        raise RuntimeError(
            "aio.run() was called on the event loop's thread; await the "
            "coroutine there instead."
        )
    future = submit(coro)
    try:
        return future.result()
    except BaseException:
        # The caller is gone: the coroutine must not run on without it. This
        # does nothing when the exception is the coroutine's own.
        future.cancel()
        raise
