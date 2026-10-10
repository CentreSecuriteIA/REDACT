"""Tests for the request driver (llm_pipeline/drive.py).

Covers drive_generators' batching + error-isolation paths and drive_sync, with
hand-built generators and a fake router (no network).
"""

import pytest

from redact.llm_pipeline import LLMRequest, drive_generators, drive_sync
from tests.conftest import as_resolver


class _FakeRouter:
    def __init__(self, fail=False):
        self.fail = fail
        self.rounds = 0

    def batch_generate(self, model, messages_list, **kw):
        if self.fail:
            raise RuntimeError("batch boom")
        self.rounds += 1
        return [f"{model}:{i}" for i, _ in enumerate(messages_list)]


def _two_round(tag):
    r1 = yield LLMRequest("m", [{"role": "user", "content": f"{tag}-1"}])
    r2 = yield LLMRequest("m", [{"role": "user", "content": f"{tag}-2:{r1}"}])
    return (tag, r1, r2)


def _immediate(tag):
    if False:  # pragma: no cover
        yield
    return (tag, "done")


def _raises():
    yield LLMRequest("m", [{"role": "user", "content": "x"}])
    raise ValueError("bad gen")


def test_drive_generators_multi_round_batched():
    router = _FakeRouter()
    gens = {"a": _two_round("a"), "b": _two_round("b")}
    results = drive_generators(gens, resolve=as_resolver(router), finalize=lambda k, v: v)
    assert results["a"] == ("a", "m:0", "m:0")
    assert results["b"] == ("b", "m:1", "m:1")
    assert router.rounds == 2  # two rounds, each one batched call per model


def test_drive_generators_immediate_completion_at_prime():
    results = drive_generators({"x": _immediate("x")}, resolve=as_resolver(_FakeRouter()), finalize=lambda k, v: v)
    assert results["x"] == ("x", "done")


def test_drive_generators_on_error_isolates():
    results = drive_generators(
        {"ok": _two_round("ok"), "bad": _raises()},
        resolve=as_resolver(_FakeRouter()), finalize=lambda k, v: v,
        on_error=lambda k, exc: ("ERR", type(exc).__name__),
    )
    assert results["ok"][0] == "ok"
    assert results["bad"] == ("ERR", "ValueError")


def test_drive_generators_reraises_without_on_error():
    with pytest.raises(ValueError):
        drive_generators({"bad": _raises()}, resolve=as_resolver(_FakeRouter()), finalize=lambda k, v: v)


def test_drive_generators_batch_failure_propagates_despite_on_error():
    """A dispatch failure is the transport failing, not the unit.

    Routing it through ``on_error`` would hand the caller a result per unit,
    which it then writes *and acks* — an outage would look like a finished run.
    ``on_error`` covers generator errors only.
    """
    with pytest.raises(RuntimeError):
        drive_generators(
            {"a": _two_round("a")}, resolve=as_resolver(_FakeRouter(fail=True)),
            finalize=lambda k, v: v, on_error=lambda k, exc: "FAILED",
        )


def test_drive_generators_finalize_exception_propagates():
    """finalize failing is not the unit failing: on_error never sees it."""
    def finalize(key, value):
        raise KeyError("finalize boom")

    with pytest.raises(KeyError, match="finalize boom") as raised:
        drive_generators(
            {"x": _immediate("x"), "y": _two_round("y")},
            resolve=as_resolver(_FakeRouter()),
            finalize=finalize, on_error=lambda k, exc: "FAILED",
        )
    assert raised.value.__context__ is None  # not chained to StopIteration


def test_drive_generators_batch_failure_reraises():
    with pytest.raises(RuntimeError):
        drive_generators({"a": _two_round("a")}, resolve=as_resolver(_FakeRouter(fail=True)), finalize=lambda k, v: v)


def test_drive_sync_drives_one_generator():
    calls = []

    def call(req: LLMRequest) -> str:
        calls.append(req.messages[-1]["content"])
        return "R:" + req.messages[-1]["content"]

    result = drive_sync(_two_round("t"), call)
    assert result == ("t", "R:t-1", "R:t-2:R:t-1")
    assert calls == ["t-1", "t-2:R:t-1"]


def test_drive_generators_progress_label():
    labels = []

    class _R:
        def batch_generate(self, model, messages_list, progress=None):
            labels.append(progress)
            return ["r"] * len(messages_list)

    drive_generators({"a": _immediate("a")}, resolve=as_resolver(_R()), finalize=lambda k, v: v)  # no rounds
    # a one-round generator without a label passes none; with one, a formatted label
    drive_generators({"b": (lambda: (yield LLMRequest("m", [])))()},
                     resolve=as_resolver(_R()), finalize=lambda k, v: v)
    drive_generators({"c": (lambda: (yield LLMRequest("m", [])))()},
                     resolve=as_resolver(_R()), finalize=lambda k, v: v, progress="lbl")
    assert labels == [None, "lbl round 1 (m)"]


def test_drive_generators_reply_count_mismatch_raises():
    """A model returning fewer replies than requests is a transport failure:
    it raises, and does not leave the unanswered generators out of the result."""
    class _Short:
        def batch_generate(self, model, messages_list, **kw):
            return ["only one"]

    with pytest.raises(RuntimeError, match="returned 1 replies for 2 requests"):
        drive_generators(
            {"a": _two_round("a"), "b": _two_round("b")},
            resolve=as_resolver(_Short()), finalize=lambda k, v: v,
            on_error=lambda k, exc: "FAILED",
        )


def _yields_no_request(first):
    if first is not None:
        yield first
    yield "not a request"


@pytest.mark.parametrize(
    "first", [None, LLMRequest("m", [])], ids=["at-prime", "later"])
def test_drive_generators_isolates_a_generator_that_yields_no_request(first):
    results = drive_generators(
        {"ok": _two_round("ok"), "bad": _yields_no_request(first)},
        resolve=as_resolver(_FakeRouter()), finalize=lambda k, v: v,
        on_error=lambda k, exc: ("ERR", type(exc).__name__),
    )
    assert results["ok"][0] == "ok"
    assert results["bad"] == ("ERR", "TypeError")


def test_run_sync_is_drive_sync():
    from redact.jailbreak.protocol import run_sync

    assert run_sync is drive_sync


class _CapturingRouter:
    """Records the internals_ids kwarg seen per batch_generate call (or its
    absence), so tests can assert exactly what drive_generators forwards."""

    def __init__(self):
        self.internals_ids_seen = []

    def batch_generate(self, model, messages_list, **kw):
        self.internals_ids_seen.append(kw.get("internals_ids"))
        return [f"{model}:{i}" for i, _ in enumerate(messages_list)]


def test_drive_generators_omits_internals_ids_when_none_set():
    router = _CapturingRouter()
    gens = {"a": (lambda: (yield LLMRequest("m", [{"role": "user", "content": "x"}])))()}
    drive_generators(gens, resolve=as_resolver(router), finalize=lambda k, v: v)
    # every request had internals_id=None (the default) -> the kwarg must be
    # omitted entirely, not passed as [None] (which would wrongly trip
    # BatchCaller's guard on a backend that doesn't support internals).
    assert router.internals_ids_seen == [None]


def test_drive_generators_forwards_internals_ids_when_set():
    router = _CapturingRouter()
    gens = {
        "a": (lambda: (yield LLMRequest("m", [{"role": "user", "content": "x"}], internals_id="root/0")))(),
        "b": (lambda: (yield LLMRequest("m", [{"role": "user", "content": "y"}])))(),  # no id
    }
    drive_generators(gens, resolve=as_resolver(router), finalize=lambda k, v: v)
    # mixed batch: one real id, one None, in request order (dict-pooled but
    # order is whatever groups[model] collected them in for this round).
    assert router.internals_ids_seen == [["root/0", None]] or router.internals_ids_seen == [[None, "root/0"]]


# --- Requests that carry their client ----------------------------------------

def _one_round(client, tag):
    reply = yield LLMRequest.for_client(client, [{"role": "user", "content": tag}])
    return reply


def test_two_clients_with_one_name_are_separate_batches():
    from tests.conftest import MockBackend, make_client

    backend_a, backend_b = MockBackend("A"), MockBackend("B")
    gens = {"a": _one_round(make_client(backend_a, "m"), "a"),
            "b": _one_round(make_client(backend_b, "m"), "b")}
    results = drive_generators(gens, resolve=lambda m: pytest.fail("resolved by name"),
                               finalize=lambda k, v: v)
    assert results == {"a": "A", "b": "B"}
    assert len(backend_a.calls) == 1 and len(backend_b.calls) == 1


def test_one_client_shared_by_two_generators_is_one_batch():
    from tests.conftest import make_client
    from tests.llms.test_router import _CountingBackend

    backend = _CountingBackend(["A1", "A2"], native=True)
    client = make_client(backend, "m")
    results = drive_generators({"a": _one_round(client, "a"), "b": _one_round(client, "b")},
                               finalize=lambda k, v: v)
    assert results == {"a": "A1", "b": "A2"}
    assert backend.generate_call_count == 1


def test_a_round_mixing_client_and_name_requests_dispatches_both():
    from tests.conftest import MockBackend, make_client

    backend = MockBackend("A")
    gens = {"c": _one_round(make_client(backend, "m"), "c"), "n": _two_round("n")}
    router = _FakeRouter()
    results = drive_generators(gens, resolve=as_resolver(router), finalize=lambda k, v: v)
    assert results["c"] == "A"
    assert results["n"] == ("n", "m:0", "m:0")
    assert len(backend.calls) == 1 and router.rounds == 2


def test_for_client_sets_model_and_replace_keeps_client():
    import dataclasses

    from tests.conftest import MockBackend, make_client

    client = make_client(MockBackend("A"), "m")
    req = LLMRequest.for_client(client, [], internals_id="x/0")
    assert (req.model, req.client, req.internals_id) == ("m", client, "x/0")
    tagged = dataclasses.replace(req, internals_id="x/1")
    assert tagged.client is client and tagged.internals_id == "x/1"
    assert LLMRequest("m", []).client is None
