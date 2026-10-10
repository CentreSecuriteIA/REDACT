"""Tests for the check helpers (llm_pipeline/check.py)."""

import pytest

from redact.llm_pipeline import (
    batch_check_samples,
    check_sample,
    checked,
    drive_sync,
    extracted,
)
from tests.conftest import MockBackend, make_client
from tests.llms.test_router import _CountingBackend


class TestCheckSample:
    def _check(self, backend, response_text):
        b = MockBackend(response_text)
        return check_sample(make_client(b), "sample text",
            build_check_messages=lambda o, s: [{"role": "user", "content": f"Check: {s}"}],
        )

    def test_accepted_yes(self, mock_backend):
        assert self._check(mock_backend, "Yes, this looks good")[0] is True

    def test_accepted_ok(self, mock_backend):
        assert self._check(mock_backend, "ok")[0] is True

    def test_accepted_accept(self, mock_backend):
        assert self._check(mock_backend, "ACCEPT")[0] is True

    def test_accepted_pass(self, mock_backend):
        assert self._check(mock_backend, "Pass - looks fine")[0] is True

    def test_rejected(self, mock_backend):
        accepted, reasoning = self._check(mock_backend, "No, this is off-topic")
        assert accepted is False
        assert "off-topic" in reasoning

    def test_reasoning_empty_on_accept(self, mock_backend):
        _, reasoning = self._check(mock_backend, "yes")
        assert reasoning == ""


class TestBatchCheckSamples:
    def test_check_slices_internals_ids_across_chunks(self):
        backend = _CountingBackend(["yes"] * 5)
        results = batch_check_samples(make_client(backend), [f"s{i}" for i in range(5)],
            build_check_messages=lambda o, s: [{"role": "user", "content": s}],
            batch_size=2,
            internals_ids=[f"id-{i}" for i in range(5)],
        )
        assert [accepted for accepted, _ in results] == [True] * 5
        assert [c["internals_id"] for c in backend.calls] == [
            "id-0", "id-1", "id-2", "id-3", "id-4",
        ]

    @pytest.mark.parametrize("batch_size", [0, -2])
    def test_a_batch_size_below_one_raises(self, batch_size):
        backend = _CountingBackend(["yes", "yes"])
        reason = f"batch_size must be at least 1, got {batch_size}"
        with pytest.raises(ValueError, match=reason):
            batch_check_samples(
                make_client(backend), ["s0", "s1"],
                build_check_messages=lambda o, s: [{"role": "user", "content": s}],
                batch_size=batch_size,
            )
        assert backend.generate_call_count == 0


# --- checked() and extracted(), driven with drive_sync -----------------------

class _Client:
    """Stand-in client: checked()/extracted() read only `.model`."""

    def __init__(self, model):
        self.model = model


GEN, CHECK = _Client("g"), _Client("c")


def _call(replies):
    """A blocking call: pops the next canned reply of the request's client."""
    seen = []

    def call(request):
        seen.append(request)
        return replies[request.client].pop(0)

    call.seen = seen
    return call


def _gen_messages(feedback):
    return [{"role": "user", "content": f"gen|{feedback}"}]


def _check_messages(original, text):
    return [{"role": "user", "content": f"check|{original}|{text}"}]


class TestChecked:
    def test_accepted_first_time(self):
        call = _call({GEN: ["t1"], CHECK: ["Yes"]})
        out = drive_sync(checked(GEN, _gen_messages, CHECK, _check_messages,
                                 original="o", internals_ids=("g", "c")), call)
        assert out == ("t1", True, "")
        assert [(r.client, r.internals_id) for r in call.seen] == [(GEN, "g"), (CHECK, "c")]
        assert call.seen[1].messages[0]["content"] == "check|o|t1"

    def test_rejected_then_accepted_feeds_the_verdict_back(self):
        call = _call({GEN: ["t1", "t2"], CHECK: ["No: too short", "Yes"]})
        out = drive_sync(checked(GEN, _gen_messages, CHECK, _check_messages,
                                 internals_ids=("g", "c"), max_attempts=2), call)
        assert out == ("t2", True, "")
        assert [r.internals_id for r in call.seen] == ["g", "c", "g/attempt_2", "c/attempt_2"]
        assert call.seen[2].messages[0]["content"] == "gen|No: too short"

    def test_all_rejected_returns_the_last_text_and_verdict(self):
        call = _call({GEN: ["t1", "t2"], CHECK: ["No 1", "No 2"]})
        out = drive_sync(checked(GEN, _gen_messages, CHECK, _check_messages,
                                 max_attempts=2), call)
        assert out == ("t2", False, "No 2")
        assert [r.internals_id for r in call.seen] == [None] * 4

    def test_no_checker_is_one_request(self):
        call = _call({GEN: ["t1"]})
        assert drive_sync(checked(GEN, _gen_messages), call) == ("t1", True, "")
        assert len(call.seen) == 1

    @pytest.mark.parametrize("kwargs", [
        {"max_attempts": 0}, {"check": CHECK},
    ], ids=["no-attempts", "check-without-builder"])
    def test_bad_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            drive_sync(checked(GEN, _gen_messages, **kwargs), _call({GEN: ["t"]}))


class TestExtracted:
    def test_parses_first_time(self):
        call = _call({GEN: ["raw"]})
        out = drive_sync(extracted(GEN, lambda: [{"role": "user", "content": "q"}],
                                   lambda raw: [raw.upper()], internals_id="g"), call)
        assert out == (["RAW"], 1)
        assert [r.internals_id for r in call.seen] == ["g"]

    def test_rejected_then_parses(self):
        call = _call({GEN: ["bad", "good"]})
        out = drive_sync(extracted(GEN, lambda: [], lambda raw: [] if raw == "bad" else [raw],
                                   internals_id="g"), call)
        assert out == (["good"], 2)
        assert [r.internals_id for r in call.seen] == ["g", "g/attempt_2"]

    def test_all_rejected(self):
        call = _call({GEN: ["bad"] * 3})
        assert drive_sync(extracted(GEN, lambda: [], lambda raw: []), call) == ([], 3)
        assert len(call.seen) == 3

    def test_zero_attempts_raises(self):
        with pytest.raises(ValueError):
            drive_sync(extracted(GEN, lambda: [], lambda raw: [raw], max_attempts=0),
                       _call({GEN: ["t"]}))
