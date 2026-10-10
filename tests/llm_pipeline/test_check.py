"""Tests for the check helpers (llm_pipeline/check.py)."""

import pytest

from redact.llm_pipeline import batch_check_samples, check_sample
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
