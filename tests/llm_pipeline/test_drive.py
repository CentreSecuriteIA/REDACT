"""Tests for the request driver (llm_pipeline/drive.py).

Covers drive_generators' batching + error-isolation paths, drive_sync, and
drive_units' admission, cap and failure paths, with hand-built generators and
a fake router (no network).
"""

import pytest

from redact.llm_pipeline import LLMRequest, drive_generators, drive_sync, drive_units
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
    req = LLMRequest.for_client(client, [], internals_id="x/0", max_tokens=7)
    assert (req.model, req.client, req.internals_id, req.max_tokens) == ("m", client, "x/0", 7)
    tagged = dataclasses.replace(req, internals_id="x/1")
    assert tagged.client is client and tagged.internals_id == "x/1" and tagged.max_tokens == 7
    assert LLMRequest("m", []).client is None and LLMRequest("m", []).max_tokens is None


def test_a_limit_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        LLMRequest("m", [], max_tokens=0)


def _capped(tag, limit, model="m"):
    yield LLMRequest(model, [{"role": "user", "content": f"{tag}1"}], max_tokens=limit)
    return tag


def test_drive_generators_groups_by_limit_and_forwards_it():
    calls = []

    class _R:
        def batch_generate(self, model, messages_list, **kw):
            calls.append((len(messages_list), kw.get("max_tokens")))
            return ["r"] * len(messages_list)

    gens = {"a": _capped("a", 20), "b": _capped("b", 20), "c": _rounds("c", 1)}
    drive_generators(gens, resolve=as_resolver(_R()), finalize=lambda k, v: v)
    assert sorted(calls, key=str) == [(1, None), (2, 20)]


# --- drive_units ---------------------------------------------------------------

class _Recorder:
    """Fake router recording the last user content of every request per call."""

    def __init__(self, fail_on_call=None):
        self.batches: list[list[str]] = []
        self.limits: list = []
        self.fail_on_call = fail_on_call

    def batch_generate(self, model, messages_list, **kw):
        if self.fail_on_call == len(self.batches) + 1:
            raise RuntimeError("batch boom")
        self.batches.append([m[-1]["content"] for m in messages_list])
        self.limits.append(kw.get("max_tokens"))
        return [f"{model}:{m[-1]['content']}" for m in messages_list]


def _rounds(tag, n, model="m"):
    """A unit making ``n`` calls to ``model`` in sequence, then returning ``tag``."""
    for i in range(1, n + 1):
        yield LLMRequest(model, [{"role": "user", "content": f"{tag}{i}"}])
    return tag


def _drive(units, recorder, **kw):
    done, failed = [], []
    n = drive_units(
        units, resolve=as_resolver(recorder),
        on_done=lambda k, v: done.append(k),
        on_error=kw.pop("on_error", lambda k, e: failed.append((k, type(e).__name__))),
        **kw,
    )
    return n, done, failed


def test_drive_units_refills_freed_slots():
    rec = _Recorder()
    units = [("a", _rounds("a", 2)), ("b", _rounds("b", 2)),
             ("c", _rounds("c", 1)), ("d", _rounds("d", 1)), ("e", _rounds("e", 1))]
    n, done, failed = _drive(iter(units), rec, pool_size=3)
    assert rec.batches == [["a1", "b1", "c1"], ["a2", "b2", "d1"], ["e1"]]
    assert done == ["c", "a", "b", "d", "e"] and failed == []
    assert n == 5


def test_drive_units_cap_splits_a_round():
    rec = _Recorder()
    units = [(t, _rounds(t, 1)) for t in "abcd"]
    _drive(iter(units), rec, pool_size=4, cap=2)
    assert rec.batches == [["a1", "b1"], ["c1", "d1"]]


def test_drive_units_deferred_requests_go_before_later_yields():
    rec = _Recorder()
    units = [("a", _rounds("a", 2)), ("b", _rounds("b", 2))] + [(t, _rounds(t, 1)) for t in "cdef"]
    _drive(iter(units), rec, pool_size=6, cap=3)
    assert rec.batches == [["a1", "b1", "c1"], ["d1", "e1", "f1"], ["a2", "b2"]]


def test_drive_units_deferred_unit_keeps_its_slot():
    """A request waiting on the cap still occupies a slot: the source is not
    pulled past ``pool_size`` live units, and the deferred request goes first."""
    rec = _Recorder()
    pulled = []

    def source():
        for t in "abc":
            pulled.append(t)
            yield t, _rounds(t, 1)

    _drive(source(), rec, pool_size=2, cap=1)
    assert rec.batches == [["a1"], ["b1"], ["c1"]]
    assert pulled == ["a", "b", "c"]


def test_drive_units_cap_is_per_group():
    """Each group is capped on its own: two models send two capped calls per
    round rather than sharing one cap."""
    rec = _Recorder()
    units = [(f"a{i}", _rounds(f"a{i}", 1, "ma")) for i in range(4)] + \
            [(f"b{i}", _rounds(f"b{i}", 1, "mb")) for i in range(4)]
    _drive(iter(units), rec, pool_size=8, cap=2)
    assert rec.batches == [["a01", "a11"], ["b01", "b11"], ["a21", "a31"], ["b21", "b31"]]


def test_drive_units_full_groups_go_first_and_a_partial_group_waits():
    rec = _Recorder()
    units = [("b", _rounds("b", 1, "mb")), ("a0", _rounds("a0", 1, "ma")), ("a1", _rounds("a1", 1, "ma"))]
    _drive(iter(units), rec, pool_size=3, cap=2)
    assert rec.batches == [["a01", "a11"], ["b1"]]


def test_drive_units_largest_partial_group_goes_when_none_is_full():
    rec = _Recorder()
    units = [("b", _rounds("b", 1, "mb")), ("a0", _rounds("a0", 1, "ma")), ("a1", _rounds("a1", 1, "ma"))]
    _drive(iter(units), rec, pool_size=3, cap=4)
    assert rec.batches == [["a01", "a11"], ["b1"]]


def test_drive_units_a_tie_between_partial_groups_goes_to_the_oldest_request():
    labels = []

    class _R:
        def batch_generate(self, model, messages_list, progress=None, **kw):
            labels.append(progress)
            return ["r"] * len(messages_list)

    units = [("b", _rounds("b", 1, "mb")), ("a", _rounds("a", 1, "ma"))]
    drive_units(iter(units), resolve=as_resolver(_R()), pool_size=2, cap=2,
                on_done=lambda k, v: None, progress="p")
    assert labels == ["p round 1 (mb)", "p round 2 (ma)"]


def test_drive_units_a_waiting_group_is_sent_once_nothing_is_full():
    """A partial group waits while another keeps filling calls, and the run
    still ends with every unit done."""
    rec = _Recorder()
    units = [("b", _rounds("b", 1, "mb"))] + [(f"a{i}", _rounds(f"a{i}", 1, "ma")) for i in range(6)]
    n, done, _ = _drive(iter(units), rec, pool_size=4, cap=2)
    assert rec.batches == [["a01", "a11"], ["a21", "a31"], ["a41", "a51"], ["b1"]]
    assert n == 7 and done[-1] == "b"


def test_drive_units_two_clients_for_one_model_and_setup_share_a_call():
    """Two client objects for the same model and setup (today's gen and check
    clients) are one group: one call, answered by the first request's client."""
    from tests.conftest import make_client
    from tests.llms.test_router import _CountingBackend

    a, b = _CountingBackend(["A1", "A2"], native=True), _CountingBackend(["B1"], native=True)
    units = [("x", _one_round(make_client(a, "m"), "x")), ("y", _one_round(make_client(b, "m"), "y"))]
    results = {}
    drive_units(iter(units), pool_size=2, on_done=results.__setitem__)
    assert results == {"x": "A1", "y": "A2"}
    assert (a.generate_call_count, b.generate_call_count) == (1, 0)


def test_drive_units_different_models_or_setups_do_not_share_a_call():
    from tests.conftest import make_client
    from tests.llms.test_router import _CountingBackend

    class _OtherSetup(_CountingBackend):
        @property
        def backend_name(self):
            return "other"

    a, b, c = (_CountingBackend(["A"], native=True), _OtherSetup(["B"], native=True),
               _CountingBackend(["C"], native=True))
    units = [("x", _one_round(make_client(a, "m"), "x")),
             ("y", _one_round(make_client(b, "m"), "y")),    # same name, another setup
             ("z", _one_round(make_client(c, "n"), "z"))]    # another model
    results = {}
    drive_units(iter(units), pool_size=3, on_done=results.__setitem__)
    assert results == {"x": "A", "y": "B", "z": "C"}
    assert [a.generate_call_count, b.generate_call_count, c.generate_call_count] == [1, 1, 1]


def test_drive_units_requests_with_different_limits_do_not_share_a_call():
    rec = _Recorder()
    units = [("a", _capped("a", 50)), ("b", _capped("b", 50)), ("c", _rounds("c", 1))]
    _drive(iter(units), rec, pool_size=3, cap=3)
    assert rec.batches == [["a1", "b1"], ["c1"]]      # the larger group goes first
    assert rec.limits == [50, None]                     # forwarded only when set


def test_drive_units_progress_label_names_the_limit():
    labels = []

    class _R:
        def batch_generate(self, model, messages_list, **kw):
            labels.append(kw.get("progress"))
            return ["r"] * len(messages_list)

    units = [("a", _capped("a", 50)), ("b", _rounds("b", 1))]
    drive_units(iter(units), resolve=as_resolver(_R()), pool_size=2, cap=1,
                on_done=lambda k, v: None, progress="p")
    assert labels == ["p round 1 (m, max_tokens=50)", "p round 1 (m)"]   # two groups, one round


def test_drive_units_unit_finishing_at_prime_takes_no_slot():
    rec = _Recorder()
    units = [("x", _immediate("x")), ("y", _immediate("y")),
             ("a", _rounds("a", 1)), ("b", _rounds("b", 1))]
    _, done, _ = _drive(iter(units), rec, pool_size=2)
    assert rec.batches == [["a1", "b1"]]
    assert done == ["x", "y", "a", "b"]


def test_drive_units_all_pure_units_make_no_call():
    """A source of pure transforms finishes inside admission: zero rounds."""
    rec = _Recorder()
    n, done, _ = _drive(iter([(t, _immediate(t)) for t in "xyz"]), rec, pool_size=2)
    assert rec.batches == [] and done == ["x", "y", "z"] and n == 3


def test_drive_units_empty_source_returns_zero():
    rec = _Recorder()
    assert _drive(iter([]), rec, pool_size=4) == (0, [], [])
    assert rec.batches == []


def test_drive_units_consumes_the_source_lazily():
    pulled, snapshot = [], []

    class _Snap(_Recorder):
        def batch_generate(self, model, messages_list, **kw):
            if not snapshot:
                snapshot.extend(pulled)
            return super().batch_generate(model, messages_list, **kw)

    def source():
        for t in "abcde":
            pulled.append(t)
            yield t, _rounds(t, 2)

    _drive(source(), _Snap(), pool_size=2)
    assert snapshot == ["a", "b"]           # no read-ahead past the free slots
    assert pulled == list("abcde")


def test_drive_units_source_exception_propagates():
    """The source failing is not a unit failing: it propagates past ``on_error``,
    and units already finished stay reported."""
    rec = _Recorder()

    def source():
        yield "x", _immediate("x")
        raise OSError("source boom")

    done, failed = [], []
    with pytest.raises(OSError, match="source boom"):
        drive_units(source(), resolve=as_resolver(rec), pool_size=2,
                    on_done=lambda k, v: done.append(k),
                    on_error=lambda k, e: failed.append(k))
    assert done == ["x"] and failed == [] and rec.batches == []


def test_drive_units_dispatch_failure_propagates_and_reports_nothing_in_flight():
    rec = _Recorder(fail_on_call=2)
    units = [("a", _rounds("a", 2)), ("b", _rounds("b", 1)), ("c", _rounds("c", 1))]
    done, failed = [], []
    with pytest.raises(RuntimeError, match="batch boom"):
        drive_units(iter(units), resolve=as_resolver(rec), pool_size=2,
                    on_done=lambda k, v: done.append(k),
                    on_error=lambda k, e: failed.append(k))
    assert done == ["b"] and failed == []


def test_drive_units_reply_count_mismatch_raises():
    class _Short:
        def batch_generate(self, model, messages_list, **kw):
            return ["only one"]

    with pytest.raises(RuntimeError, match="returned 1 replies for 2 requests"):
        _drive(iter([("a", _rounds("a", 1)), ("b", _rounds("b", 1))]), _Short(), pool_size=2)


def test_drive_units_unit_error_goes_to_on_error():
    rec = _Recorder()
    n, done, failed = _drive(iter([("ok", _rounds("ok", 2)), ("bad", _raises())]), rec, pool_size=2)
    assert done == ["ok"] and failed == [("bad", "ValueError")]
    assert n == 2


def test_drive_units_unit_error_at_prime_takes_no_slot():
    rec = _Recorder()
    units = [("bad", _yields_no_request(None)), ("a", _rounds("a", 1)), ("b", _rounds("b", 1))]
    _, done, failed = _drive(iter(units), rec, pool_size=2)
    assert failed == [("bad", "TypeError")] and done == ["a", "b"]
    assert rec.batches == [["a1", "b1"]]


def test_drive_units_unit_error_propagates_without_on_error():
    with pytest.raises(ValueError, match="bad gen"):
        _drive(iter([("ok", _rounds("ok", 2)), ("bad", _raises())]), _Recorder(),
               pool_size=2, on_error=None)


def test_drive_units_on_done_exception_propagates_unchained():
    """on_done failing is not the unit failing: on_error never sees it."""
    def on_done(key, value):
        raise KeyError("done boom")

    with pytest.raises(KeyError, match="done boom") as raised:
        drive_units(iter([("x", _immediate("x"))]), resolve=as_resolver(_Recorder()),
                    pool_size=1, on_done=on_done, on_error=lambda k, e: None)
    assert raised.value.__context__ is None


def test_drive_units_rejects_a_duplicate_live_key():
    with pytest.raises(ValueError, match="already live"):
        _drive(iter([("a", _rounds("a", 1)), ("a", _rounds("a", 1))]), _Recorder(), pool_size=2)


@pytest.mark.parametrize("kw", [{"pool_size": 0}, {"pool_size": 1, "cap": 0}], ids=["pool", "cap"])
def test_drive_units_rejects_pool_size_or_cap_below_one(kw):
    with pytest.raises(ValueError):
        _drive(iter([]), _Recorder(), **kw)


def test_drive_units_progress_label_and_internals_ids():
    labels, ids = [], []

    class _R:
        def batch_generate(self, model, messages_list, **kw):
            labels.append(kw.get("progress"))
            ids.append(kw.get("internals_ids"))
            return ["r"] * len(messages_list)

    units = [("a", (lambda: (yield LLMRequest("m", [], internals_id="root/0")))()),
             ("b", (lambda: (yield LLMRequest("m", [])))())]
    drive_units(iter(units), resolve=as_resolver(_R()), pool_size=2, on_done=lambda k, v: None)
    drive_units(iter([("c", _rounds("c", 1))]), resolve=as_resolver(_R()), pool_size=1,
                on_done=lambda k, v: None, progress="lbl")
    assert labels == [None, "lbl round 1 (m)"]
    assert ids == [["root/0", None], None]   # omitted entirely when no request wants capture
