"""Tests for drive_units' async lane (llm_pipeline/drive.py).

The backends here end their requests on the event loop in an order the test
scripts, so every assertion is exact: nothing sleeps and nothing is timed. A
script the driver cannot follow fails its request after ``_PATIENCE`` seconds
instead of hanging the suite.
"""

import asyncio
import logging
import queue
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from redact.llm_pipeline import LLMRequest, drive_units
from redact.llm_pipeline import drive as drive_module
from redact.llms import ModelClient, aio, observe
from redact.llms import client as client_module
from redact.llms.backends import ComputeConfig, OpenAIBackend
from redact.llms.model_config import MODEL_REGISTRY, APIConfig, register_model
from redact.llms.wrappers import RateLimiter
from tests.conftest import MockBackend, make_client

_PATIENCE = 20.0


class _Script:
    """The order in which scripted requests end, and what the loop saw.

    A request is named by the content of its last message. It ends when every
    name before it in ``order`` has ended, however early it started. A name
    missing from ``order`` ends only on :meth:`release`; ``order=None`` ends
    every request at once. With ``closed=True`` nothing ends before
    :meth:`open`.
    """

    def __init__(self, order=None, *, fail=(), closed=False):
        self.order = order
        self.fail = set(fail)
        self.log = []    # ("start" | "end" | "cancel", name), in loop order
        self.sent = {}   # name -> (max_tokens, internals_id)
        self._turn = {name: asyncio.Event() for name in order or ()}
        self._awaited = {}   # log entry -> asyncio.Event, for wait_until()
        if order and not closed:
            self._turn[order[0]].set()

    def open(self):
        """Let the first request end. Not for the loop's thread."""
        self.release(self.order[0])

    def release(self, name):
        """Let ``name`` end, whatever the order. Not for the loop's thread."""
        async def _release():
            self._turn.setdefault(name, asyncio.Event()).set()

        aio.run(_release())

    def wait_until(self, *entry):
        """Block the caller until ``entry`` is in the log. Not for the loop's thread."""
        async def _wait():
            if entry not in self.log:
                logged = self._awaited.setdefault(entry, asyncio.Event())
                await asyncio.wait_for(logged.wait(), _PATIENCE)

        aio.run(_wait())

    def _note(self, *entry):
        self.log.append(entry)
        if entry in self._awaited:
            self._awaited[entry].set()

    async def play(self, name, max_tokens, internals_id):
        self._note("start", name)
        self.sent[name] = (max_tokens, internals_id)
        if self.order is not None:
            turn = self._turn.setdefault(name, asyncio.Event())
            try:
                await asyncio.wait_for(turn.wait(), _PATIENCE)
            except asyncio.CancelledError:
                self._note("cancel", name)
                raise
            following = self.order[self.order.index(name) + 1:] if name in self.order else ()
            if following:
                self._turn[following[0]].set()
        self._note("end", name)
        if name in self.fail:
            raise RuntimeError(f"{name} boom")


class _Scripted(MockBackend):
    """Async backend whose requests follow a :class:`_Script`."""

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self, script, model):
        super().__init__(model=model)
        self.script = script

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None, **kwargs):
        name = messages[-1]["content"]
        await self.script.play(name, max_tokens, internals_id)
        return f"{self.model}:{name}"


def _lane_client(script, model="lane-m", workers=8, backend_cls=_Scripted):
    backend = backend_cls(script, model)
    client = make_client(backend, model)
    backend.max_workers = workers    # make_client bound the registry's 1
    return client


def _unit(client, name, rounds=1, **request_kw):
    """A unit making ``rounds`` calls to ``client`` in sequence; returns the replies."""
    replies = []
    for i in range(1, rounds + 1):
        messages = [{"role": "user", "content": f"{name}{i}"}]
        replies.append((yield LLMRequest.for_client(client, messages, **request_kw)))
    return replies


def _unit_by_name(model, name):
    """A unit making one call to ``model`` by name; returns the reply."""
    return (yield LLMRequest(model, [{"role": "user", "content": f"{name}1"}]))


def _drive(units, **kw):
    done, failed = [], []
    n = drive_units(
        iter(units),
        on_done=lambda k, v: done.append(k),
        on_error=lambda k, e: failed.append((k, type(e).__name__)),
        **kw,
    )
    return n, done, failed


# --- Replies as they arrive ----------------------------------------------------

def test_replies_are_handed_back_in_arrival_order():
    script = _Script(["c1", "a1", "b1", "a2"])
    client = _lane_client(script)
    results = {}
    units = [("a", _unit(client, "a", 2)), ("b", _unit(client, "b")), ("c", _unit(client, "c"))]
    n = drive_units(iter(units), pool_size=3, on_done=results.__setitem__)
    assert list(results) == ["c", "b", "a"]
    assert results == {"c": ["lane-m:c1"], "b": ["lane-m:b1"], "a": ["lane-m:a1", "lane-m:a2"]}
    assert n == 3


def test_a_freed_slot_is_refilled_before_the_slowest_reply_arrives():
    """With two slots, the third unit's request goes out as soon as the fast
    one finishes, while the slow one is still in flight."""
    script = _Script(["b1", "c1", "a1"])
    client = _lane_client(script)
    _, done, _ = _drive([(k, _unit(client, k)) for k in "abc"], pool_size=2)
    assert script.log == [("start", "a1"), ("start", "b1"), ("end", "b1"),
                          ("start", "c1"), ("end", "c1"), ("end", "a1")]
    assert done == ["b", "c", "a"]


def test_a_freed_slot_is_refilled_before_the_next_reply_is_handed_back():
    """Two replies have arrived. Handing back the first frees a slot, and the
    next unit's request is out before the second is handed back."""
    script = _Script(["a1", "b1", "c1"])
    client = _lane_client(script)
    done = []

    def on_done(key, value):
        if key == "a":
            script.wait_until("end", "b1")      # b's reply has arrived as well
        if key == "b":
            script.wait_until("start", "c1")    # never, if c still waits for its slot
        done.append(key)

    drive_units(iter([(k, _unit(client, k)) for k in "abc"]), pool_size=2, on_done=on_done)
    assert done == ["a", "b", "c"]


def _then_open(units, script):
    """The units, then the script opens: by then the loop has stepped every
    request that was free to start."""
    yield from units
    script.open()


def test_a_model_has_at_most_max_workers_requests_in_flight():
    """Above the limit a request waits for the first one to finish, not for
    the slowest."""
    script = _Script(["b1", "c1", "d1", "a1"], closed=True)
    client = _lane_client(script, workers=2)
    units = [(k, _unit(client, k)) for k in "abcd"]
    _, done, _ = _drive(_then_open(units, script), pool_size=5)
    assert script.log == [("start", "a1"), ("start", "b1"), ("end", "b1"),
                          ("start", "c1"), ("end", "c1"),
                          ("start", "d1"), ("end", "d1"), ("end", "a1")]
    assert done == ["b", "c", "d", "a"]


def test_two_clients_for_one_model_and_setup_share_the_limit():
    script = _Script(["b1", "c1", "d1", "a1"], closed=True)
    first, second = _lane_client(script, workers=2), _lane_client(script, workers=2)
    units = [("a", _unit(first, "a")), ("b", _unit(first, "b")),
             ("c", _unit(second, "c")), ("d", _unit(second, "d"))]
    _drive(_then_open(units, script), pool_size=5)
    assert script.log[:4] == [("start", "a1"), ("start", "b1"), ("end", "b1"), ("start", "c1")]


def test_different_models_do_not_share_a_limit():
    """x cannot end before y does, and each model allows one request: both
    are in flight together."""
    script = _Script(["y1", "x1"])
    x = _lane_client(script, "lane-x", workers=1)
    y = _lane_client(script, "lane-y", workers=1)
    _, done, _ = _drive([("x", _unit(x, "x")), ("y", _unit(y, "y"))], pool_size=2)
    assert script.log == [("start", "x1"), ("start", "y1"), ("end", "y1"), ("end", "x1")]
    assert done == ["y", "x"]


def test_each_request_carries_its_own_limit_and_capture_id():
    class _Capturing(_Scripted):
        compute_config = ComputeConfig(supports_async_calls=True, supports_internals=True)

    script = _Script()
    client = _lane_client(script, "lane-capture", backend_cls=_Capturing)
    units = [("a", _unit(client, "a", max_tokens=50, internals_id="root/0")),
             ("b", _unit(client, "b")), ("c", _unit(client, "c", max_tokens=7))]
    _drive(units, pool_size=3)
    assert script.sent == {"a1": (50, "root/0"), "b1": (None, None), "c1": (7, None)}


def test_a_request_by_name_is_resolved_once_and_sent_on_the_lane():
    script = _Script(["b1", "a1"])
    client = _lane_client(script, "lane-named")
    resolved = []

    def resolve(name):
        resolved.append(name)
        return client

    done = {}
    units = [(k, _unit_by_name("lane-named", k)) for k in "ab"]
    drive_units(iter(units), pool_size=2, resolve=resolve, on_done=done.__setitem__)
    assert done == {"b": "lane-named:b1", "a": "lane-named:a1"}
    assert list(done) == ["b", "a"]
    assert resolved == ["lane-named"]


def test_a_registered_openai_compatible_model_takes_the_lane(monkeypatch):
    """By name and end to end: the registry entry, ModelClient.create, the
    lane and the async SDK client, with one ``call`` event per request."""
    async def create(**kwargs):
        choice = MagicMock(finish_reason="stop")
        choice.message.content = "re: " + kwargs["messages"][-1]["content"]
        return MagicMock(choices=[choice], usage=None)

    sdk = MagicMock()
    sdk.chat.completions.create = create
    events = []
    monkeypatch.setenv("LANE_TEST_KEY", "k")
    register_model("lane-served", backend_type="api", api=APIConfig(
        backend_type="openai", api_key_env="LANE_TEST_KEY",
        base_url="http://127.0.0.1:8000/v1", rpm=6000, recommended_max_workers=4))
    OpenAIBackend.clear_cache()
    observe.set_emitter(events.append)
    try:
        with patch("redact.llms.backends.openai.openai.OpenAI"), \
             patch("redact.llms.backends.openai.openai.AsyncOpenAI",
                   return_value=sdk) as built:
            results = {}
            n = drive_units(((k, _unit_by_name("lane-served", k)) for k in "abc"),
                            pool_size=2, on_done=results.__setitem__)
    finally:
        observe.set_emitter(None)
        OpenAIBackend.clear_cache()
        del MODEL_REGISTRY["lane-served"]

    assert n == len(results) and results == {k: f"re: {k}1" for k in "abc"}
    built.assert_called_once_with(api_key="k", base_url="http://127.0.0.1:8000/v1")
    assert [(e["ev"], e["model"], e["backend"], e["n_items"]) for e in events] == [
        ("call", "lane-served", "openai", 1)] * len(results)


def test_the_lane_waits_for_the_rate_limiter(monkeypatch):
    rpm = 2
    backend = _Scripted(_Script(), "lane-limited")
    make_client(backend, "lane-limited")
    backend.rpm, backend.max_workers = rpm, 8
    limiter = RateLimiter()
    client = ModelClient(backend, limiter)
    slept = []

    async def _sleep(seconds):
        slept.append(seconds)
        if len(slept) == 2:        # only the SECOND sleep frees a slot
            limiter._timestamps[0] -= 120.0

    monkeypatch.setattr(client_module, "asyncio", SimpleNamespace(sleep=_sleep))
    n, done, _ = _drive([(k, _unit(client, k)) for k in "abc"], pool_size=3)
    assert n == rpm + 1 and sorted(done) == ["a", "b", "c"]
    assert len(slept) == 2, "the third request went out through a full window"
    assert len(limiter._timestamps) == rpm, "cap exceeded"


# --- Beside the synchronous path -------------------------------------------------

def test_a_run_mixing_synchronous_and_async_clients_finishes_every_unit():
    from tests.llms.test_router import _CountingBackend

    lane = _lane_client(_Script())
    sync_backend = _CountingBackend("ok", native=True)
    sync = make_client(sync_backend, "lane-sync")

    def ask(client, text):
        return LLMRequest.for_client(client, [{"role": "user", "content": text}])

    def lane_then_sync(name):
        answer = yield ask(lane, name)
        return answer, (yield ask(sync, f"check {answer}"))

    def sync_then_lane(name):
        verdict = yield ask(sync, f"check {name}")
        return (yield ask(lane, f"{name}:{verdict}")), verdict

    units = [(f"u{i}", (lane_then_sync if i % 2 else sync_then_lane)(f"u{i}"))
             for i in range(12)]
    results = {}
    n = drive_units(iter(units), pool_size=4, cap=2, on_done=results.__setitem__)
    assert n == len(units)
    assert results == {
        f"u{i}": (f"lane-m:u{i}" if i % 2 else f"lane-m:u{i}:ok", "ok") for i in range(12)
    }
    assert len(sync_backend.calls) == len(units)


def test_a_full_synchronous_call_does_not_wait_for_a_request_in_flight():
    """The async request cannot end until the synchronous call has been made."""
    script = _Script(["x1"], closed=True)
    lane = _lane_client(script)

    class _OpensTheScript(MockBackend):
        def generate(self, messages_list, **kwargs):
            script.open()
            return super().generate(messages_list, **kwargs)

    sync = make_client(_OpensTheScript("ok"), "lane-sync")
    units = [("x", _unit(lane, "x")), ("y", _unit(sync, "y"))]
    _, done, _ = _drive(units, pool_size=2, cap=1)
    assert done == ["y", "x"]


class _HeldLane:
    """A lane whose replies are held, beside a synchronous client that records.

    A lane reply ends only when the driver starts to wait for one with none
    arrived: the oldest is then released. So whatever the driver decides
    between two replies, it decides with the others still in flight.

    Attributes:
        batches: One ``(size, lane requests in flight)`` per synchronous call.
        labels: The progress label of each synchronous call.
    """

    def __init__(self, monkeypatch):
        self.batches, self.labels = [], []
        self._script = _Script([])
        self._client = _lane_client(self._script)
        self._in_flight = in_flight = deque()
        release = self._script.release

        class _DriverQueue(queue.Queue):
            def get(self, block=True, timeout=None):
                if block and self.empty() and in_flight:
                    release(in_flight.popleft())
                return super().get(block, timeout)

        monkeypatch.setattr(
            drive_module, "queue", SimpleNamespace(Queue=_DriverQueue, Empty=queue.Empty))

    def lane(self, name):
        """A request on the lane."""
        self._in_flight.append(name)
        return LLMRequest.for_client(self._client, [{"role": "user", "content": name}])

    def sync(self, name):
        """A request to the synchronous model, by name."""
        return LLMRequest("sync", [{"role": "user", "content": name}])

    def resolve(self, model):
        """The synchronous client: ``drive_units``' ``resolve``."""
        return self

    def generate(self, messages_list, progress=None, **kwargs):
        self.batches.append((len(messages_list), len(self._in_flight)))
        self.labels.append(progress)
        return ["ok"] * len(messages_list)


def test_only_full_synchronous_calls_go_out_while_requests_are_in_flight(monkeypatch):
    """A group below ``cap`` waits for the lane's replies, which may fill it,
    and goes as it is once the lane is empty."""
    held = _HeldLane(monkeypatch)

    def sync_then_lane(name):
        yield held.sync(name)
        yield held.lane(name)

    def lane_then_sync(name):
        yield held.lane(name)
        yield held.sync(name)

    units = [(f"u{i}", (lane_then_sync if i % 2 else sync_then_lane)(f"u{i}"))
             for i in range(7)]
    n, done, _ = _drive(units, pool_size=7, cap=2, resolve=held.resolve, progress="p")
    # u5's request waits alone through the last four lane replies.
    assert held.batches == [(2, 3), (2, 5), (2, 5), (1, 0)]
    assert held.labels == [f"p round {i} (sync)" for i in (1, 2, 3, 4)]
    assert n == len(units) and done[-1] == "u5"


def test_a_group_that_never_fills_goes_once_nothing_is_in_flight(monkeypatch):
    held = _HeldLane(monkeypatch)

    def one(request):
        yield request

    units = [("s", one(held.sync("s"))), ("x", one(held.lane("x"))),
             ("y", one(held.lane("y")))]
    n, done, _ = _drive(units, pool_size=3, cap=2, resolve=held.resolve)
    assert held.batches == [(1, 0)]
    assert n == len(units) and done == ["x", "y", "s"]


def test_a_wait_that_wakes_with_nothing_arrived_goes_on_waiting(monkeypatch):
    """The driver's wait has a timeout, so that an interrupt gets through;
    waking on it is not the end of the run."""
    timeouts = []

    class _DriverQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            if block and not timeouts:
                timeouts.append(timeout)
                raise queue.Empty        # the timeout passed with no reply
            return super().get(block, timeout)

    monkeypatch.setattr(
        drive_module, "queue", SimpleNamespace(Queue=_DriverQueue, Empty=queue.Empty))
    client = _lane_client(_Script())
    n, done, _ = _drive([("a", _unit(client, "a"))], pool_size=1)
    assert (n, done) == (1, ["a"])
    assert timeouts == [1.0]


def test_a_run_without_async_clients_never_touches_the_event_loop(monkeypatch):
    monkeypatch.setattr(drive_module, "aio", None)
    client = make_client(MockBackend("ok"), "lane-sync")
    n, done, _ = _drive([(k, _unit(client, k, 2)) for k in "abc"], pool_size=2)
    assert n == 3 and done == ["a", "b", "c"]


def test_progress_logs_one_line_every_cap_finished_units(caplog):
    client = _lane_client(_Script(["a1", "b1", "c1", "d1"]))
    with caplog.at_level(logging.INFO, logger="redact.llm_pipeline.drive"):
        _drive([(k, _unit(client, k)) for k in "abcd"], pool_size=4, cap=2, progress="p")
    assert [r.getMessage() for r in caplog.records] == [
        "p: lane-m on the async lane, at most 8 in flight",
        "p: 2 units done, 2 in flight", "p: 4 units done, 0 in flight"]

    caplog.clear()
    client = _lane_client(_Script())
    with caplog.at_level(logging.INFO, logger="redact.llm_pipeline.drive"):
        _drive([(k, _unit(client, k)) for k in "abcd"], pool_size=4, cap=2)
    assert caplog.records == []


def test_progress_announces_each_model_on_the_lane_once_with_its_limit(caplog):
    """A run shorter than ``cap`` units is not silent."""
    script = _Script()
    x = _lane_client(script, "lane-x", workers=2)
    y = _lane_client(script, "lane-y", workers=5)
    units = [("a", _unit(x, "a", 2)), ("b", _unit(y, "b")), ("c", _unit(x, "c"))]
    with caplog.at_level(logging.INFO, logger="redact.llm_pipeline.drive"):
        _drive(units, pool_size=3, cap=8, progress="p")
    assert [r.getMessage() for r in caplog.records] == [
        "p: lane-x on the async lane, at most 2 in flight",
        "p: lane-y on the async lane, at most 5 in flight"]


# --- Failures and teardown -------------------------------------------------------

def test_a_failed_request_propagates_and_cancels_those_in_flight():
    """A failed request is the transport failing: it never reaches on_error,
    and the requests still in flight have stopped when the driver raises."""
    script = _Script(["bad1"], fail={"bad1"})
    client = _lane_client(script)
    done, failed = [], []
    units = [("a", _unit(client, "slow")), ("b", _unit(client, "bad")),
             ("c", _unit(client, "late"))]
    with pytest.raises(RuntimeError, match="bad1 boom"):
        drive_units(iter(units), pool_size=3, on_done=lambda k, v: done.append(k),
                    on_error=lambda k, e: failed.append(k))
    assert done == [] and failed == []
    assert script.log[:4] == [("start", "slow1"), ("start", "bad1"), ("end", "bad1"),
                              ("start", "late1")]
    assert sorted(script.log[4:]) == [("cancel", "late1"), ("cancel", "slow1")]


@pytest.mark.parametrize("error", [KeyError, KeyboardInterrupt])
def test_an_exception_from_on_done_propagates_and_stops_the_lane(error):
    script = _Script(["a1"])
    client = _lane_client(script)

    def on_done(key, value):
        raise error("done boom")

    with pytest.raises(error, match="done boom") as raised:
        drive_units(iter([("a", _unit(client, "a")), ("b", _unit(client, "b"))]),
                    pool_size=2, on_done=on_done, on_error=lambda k, e: None)
    assert raised.value.__context__ is None
    assert script.log == [("start", "a1"), ("end", "a1"), ("start", "b1"), ("cancel", "b1")]


def test_an_exception_from_the_source_stops_the_lane():
    script = _Script([])
    client = _lane_client(script)

    def source():
        yield "a", _unit(client, "a")
        raise OSError("source boom")

    with pytest.raises(OSError, match="source boom"):
        _drive(source(), pool_size=2)
    assert script.log == [("start", "a1"), ("cancel", "a1")]


def test_a_request_that_cannot_be_sent_is_not_the_units_error():
    """A name that does not resolve fails the run, as a failed call does:
    the unit is not handed to on_error, and the request in flight is cancelled."""
    script = _Script([])
    client = _lane_client(script)

    def resolve(name):
        raise KeyError(f"unknown model {name}")

    failed = []
    units = [("a", _unit(client, "a")), ("b", _unit_by_name("ghost", "b"))]
    with pytest.raises(KeyError, match="ghost"):
        drive_units(iter(units), pool_size=2, resolve=resolve,
                    on_done=lambda k, v: None, on_error=lambda k, e: failed.append(k))
    assert failed == []
    assert script.log == [("start", "a1"), ("cancel", "a1")]


def test_cancel_returns_only_when_every_request_has_stopped():
    """A cancelled request needs further turns of the loop to stop; the
    driver's teardown must outlast them."""
    async def turn_of_the_loop():
        stepped = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(stepped.set_result, None)
        await stepped

    async def slow_to_stop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            for _ in range(3):
                await turn_of_the_loop()
            raise

    async def scenario():
        tasks = {asyncio.ensure_future(slow_to_stop()) for _ in range(3)}
        await turn_of_the_loop()
        await drive_module._cancel(tasks)
        return [task.cancelled() for task in tasks]

    assert aio.run(scenario()) == [True, True, True]


def test_callbacks_run_on_the_callers_thread():
    client = _lane_client(_Script())
    threads = []
    units = [("ok", _unit(client, "ok")), ("bad", _fails_after_its_reply(client, "bad", "raises"))]
    drive_units(iter(units), pool_size=2,
                on_done=lambda k, v: threads.append(threading.current_thread()),
                on_error=lambda k, e: threads.append(threading.current_thread()))
    assert threads == [threading.current_thread()] * 2


def _fails_after_its_reply(client, name, how):
    yield LLMRequest.for_client(client, [{"role": "user", "content": f"{name}1"}])
    if how == "raises":
        raise ValueError("bad unit")
    yield "not a request"


@pytest.mark.parametrize("how, error", [("raises", "ValueError"), ("yields-no-request", "TypeError")])
def test_a_unit_that_fails_goes_to_on_error_and_the_others_finish(how, error):
    client = _lane_client(_Script(["bad1", "a1", "b1", "a2"]))
    units = [("a", _unit(client, "a", 2)), ("bad", _fails_after_its_reply(client, "bad", how)),
             ("b", _unit(client, "b"))]
    n, done, failed = _drive(units, pool_size=3)
    assert failed == [("bad", error)] and done == ["b", "a"]
    assert n == 3
