"""Tests for the shared event-loop thread (llms/aio.py)."""

import asyncio
import threading

import pytest

from redact.llms import aio
from tests.conftest import interrupt_next_run


async def _where():
    return asyncio.get_running_loop(), threading.current_thread()


def test_run_returns_the_result_and_raises_the_exception():
    async def ok():
        return "ok"

    async def boom():
        raise ValueError("boom")

    assert aio.run(ok()) == "ok"
    with pytest.raises(ValueError, match="boom"):
        aio.run(boom())


def test_every_coroutine_runs_on_the_same_daemon_thread():
    loop_a, thread_a = aio.run(_where())
    loop_b, thread_b = aio.submit(_where()).result()
    assert loop_a is loop_b and thread_a is thread_b
    assert thread_a is not threading.current_thread()
    assert thread_a.daemon, "a non-daemon loop thread would block interpreter exit"


def test_submit_returns_while_the_coroutine_is_still_running():
    async def new_event():
        return asyncio.Event()

    release = aio.run(new_event())

    async def wait():
        await release.wait()
        return "released"

    async def let_go():
        release.set()

    waiting = aio.submit(wait())
    assert not waiting.done()
    aio.run(let_go())
    assert waiting.result(timeout=10) == "released"


def test_run_on_the_loop_thread_raises_instead_of_deadlocking():
    async def inner():
        return 1

    async def outer():
        return aio.run(inner())

    with pytest.raises(RuntimeError, match="event loop's thread"):
        aio.run(outer())


def test_an_interrupted_run_cancels_its_coroutine(monkeypatch):
    """Ctrl-C reaches the caller blocked in run(). The coroutine it waited for
    must not run on without it."""
    started, cancelled = threading.Event(), threading.Event()

    async def never_returns():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    # The interrupt finds the coroutine running.
    interrupt_next_run(monkeypatch, when=lambda: started.wait(10))
    with pytest.raises(KeyboardInterrupt):
        aio.run(never_returns())
    assert cancelled.wait(10)


def test_concurrent_first_use_starts_one_loop(monkeypatch):
    """Two loops would split one SDK client's connections between them."""
    process_thread = aio._thread
    monkeypatch.setattr(aio, "_loop", None)
    monkeypatch.setattr(aio, "_thread", None)
    built = []
    new_event_loop = asyncio.new_event_loop

    def counting():
        built.append(new_event_loop())
        return built[-1]

    monkeypatch.setattr(aio.asyncio, "new_event_loop", counting)
    barrier = threading.Barrier(4)
    seen = []

    def first_use():
        barrier.wait()
        seen.append(aio._get_loop())

    threads = [threading.Thread(target=first_use) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert len(built) == 1
        assert seen == built * 4
    finally:
        # This test's loop is not the process's: stop it before it is forgotten.
        for loop in built:
            loop.call_soon_threadsafe(loop.stop)
        for thread in threading.enumerate():
            if thread.name == "redact-aio" and thread is not process_thread:
                thread.join()
        for loop in built:
            loop.close()


def test_clearing_the_openai_cache_closes_its_async_clients_on_the_loop():
    from unittest.mock import patch

    import redact.llms.backends.openai as openai_module

    closed_on = []

    class _AsyncClient:
        async def aclose(self):
            closed_on.append(threading.current_thread())

    _, loop_thread = aio.run(_where())
    openai_module.OpenAIBackend.clear_cache()
    with patch.object(openai_module.openai, "AsyncOpenAI",
                      side_effect=lambda **kwargs: _AsyncClient()):
        openai_module._sdk_client("k", "https://one.api/v1", max_retries=2,
                                  timeout_s=None, async_calls=True)
    openai_module.OpenAIBackend.clear_cache()
    assert closed_on == [loop_thread]


def test_clearing_the_openai_cache_starts_no_loop(monkeypatch):
    import redact.llms.backends.openai as openai_module

    monkeypatch.setattr(aio, "_loop", None)
    monkeypatch.setattr(openai_module, "_async_sdk_clients",
                        {("https://one.api/v1", "k", 2, None): object()})
    openai_module.OpenAIBackend.clear_cache()
    assert aio._loop is None and openai_module._async_sdk_clients == {}
