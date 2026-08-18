"""Tests for the shared model-layer conversation primitives (llms/conversation.py).

Covers drive_generators' batching + error-isolation paths and drive_sync, with
hand-built generators and a fake router (no network).
"""

import pytest

from redact.llms.conversation import LLMRequest, drive_generators, drive_sync


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
    results = drive_generators(gens, router=router, finalize=lambda k, v: v)
    assert results["a"] == ("a", "m:0", "m:0")
    assert results["b"] == ("b", "m:1", "m:1")
    assert router.rounds == 2  # two rounds, each one batched call per model


def test_drive_generators_immediate_completion_at_prime():
    results = drive_generators({"x": _immediate("x")}, router=_FakeRouter(), finalize=lambda k, v: v)
    assert results["x"] == ("x", "done")


def test_drive_generators_on_error_isolates():
    results = drive_generators(
        {"ok": _two_round("ok"), "bad": _raises()},
        router=_FakeRouter(), finalize=lambda k, v: v,
        on_error=lambda k, exc: ("ERR", type(exc).__name__),
    )
    assert results["ok"][0] == "ok"
    assert results["bad"] == ("ERR", "ValueError")


def test_drive_generators_reraises_without_on_error():
    with pytest.raises(ValueError):
        drive_generators({"bad": _raises()}, router=_FakeRouter(), finalize=lambda k, v: v)


def test_drive_generators_batch_failure_isolated():
    results = drive_generators(
        {"a": _two_round("a")}, router=_FakeRouter(fail=True),
        finalize=lambda k, v: v, on_error=lambda k, exc: "FAILED",
    )
    assert results["a"] == "FAILED"


def test_drive_generators_batch_failure_reraises():
    with pytest.raises(RuntimeError):
        drive_generators({"a": _two_round("a")}, router=_FakeRouter(fail=True), finalize=lambda k, v: v)


def test_drive_sync_drives_one_generator():
    calls = []

    def call(req: LLMRequest) -> str:
        calls.append(req.messages[-1]["content"])
        return "R:" + req.messages[-1]["content"]

    result = drive_sync(_two_round("t"), call)
    assert result == ("t", "R:t-1", "R:t-2:R:t-1")
    assert calls == ["t-1", "t-2:R:t-1"]


def test_drive_generators_verbose_progress_label():
    labels = []

    class _R:
        def batch_generate(self, model, messages_list, progress=None):
            labels.append(progress)
            return ["r"] * len(messages_list)

    drive_generators({"a": _immediate("a")}, router=_R(), finalize=lambda k, v: v)  # no rounds
    # a one-round generator with verbose+progress records a formatted label
    drive_generators({"b": (lambda: (yield LLMRequest("m", [])))()},
                     router=_R(), finalize=lambda k, v: v, verbose=True, progress="lbl")
    assert any(p and p.startswith("lbl round 1 (m)") for p in labels)


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
    drive_generators(gens, router=router, finalize=lambda k, v: v)
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
    drive_generators(gens, router=router, finalize=lambda k, v: v)
    # mixed batch: one real id, one None, in request order (dict-pooled but
    # order is whatever groups[model] collected them in for this round).
    assert router.internals_ids_seen == [["root/0", None]] or router.internals_ids_seen == [[None, "root/0"]]
