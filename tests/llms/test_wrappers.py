"""Tests for rate limiting, retry wrappers, and batch caller."""

import time
from unittest.mock import MagicMock

import pytest

from redact.llms.wrappers import RateLimiter, with_retries, with_feedback_retries, BatchCaller
from tests.conftest import MockBackend


class TestRateLimiter:
    def test_allows_under_rpm(self):
        limiter = RateLimiter()
        # Default RPM for unknown model is 20 — one call should not block
        start = time.time()
        limiter.wait_if_needed("unknown-model-xyz")
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_creates_per_model_state(self):
        limiter = RateLimiter()
        limiter.wait_if_needed("model-a")
        limiter.wait_if_needed("model-b")
        assert "model-a" in limiter._timestamps
        assert "model-b" in limiter._timestamps

    def test_tracks_timestamps(self):
        limiter = RateLimiter()
        limiter.wait_if_needed("unknown-model-xyz")
        limiter.wait_if_needed("unknown-model-xyz")
        assert len(limiter._timestamps["unknown-model-xyz"]) == 2


class TestWithRetries:
    def test_passes_first_try(self):
        gen = MagicMock(return_value="good")
        check = MagicMock(return_value=(True, ""))
        wrapped = with_retries(gen, check, num_retries=3)
        result, info = wrapped("input")
        assert result == "good"
        assert info == ""
        gen.assert_called_once_with("input")

    def test_retries_on_failure(self):
        gen = MagicMock(side_effect=["bad", "bad", "good"])
        check = MagicMock(side_effect=[(False, "nope"), (False, "nope"), (True, "")])
        wrapped = with_retries(gen, check, num_retries=5)
        result, info = wrapped("input")
        assert result == "good"
        assert info == ""
        assert gen.call_count == 3

    def test_exhausted_returns_discarded(self):
        gen = MagicMock(return_value="bad")
        check = MagicMock(return_value=(False, "nope"))
        wrapped = with_retries(gen, check, num_retries=2)
        result, info = wrapped("input")
        assert info == "DISCARDED"

    def test_returns_last_result_on_exhaustion(self):
        gen = MagicMock(side_effect=["attempt1", "attempt2"])
        check = MagicMock(return_value=(False, "bad"))
        wrapped = with_retries(gen, check, num_retries=2)
        result, info = wrapped("input")
        assert result == "attempt2"


class TestWithFeedbackRetries:
    def test_passes_first_try(self):
        gen = MagicMock(return_value="good")
        check = MagicMock(return_value=(True, ""))
        wrapped = with_feedback_retries(gen, check, num_retries=3)
        result, info = wrapped("input")
        assert result == "good"
        assert info == ""

    def test_passes_feedback_to_generator(self):
        call_args = []

        def gen(text, feedback=""):
            call_args.append(feedback)
            return "attempt"

        check_results = iter([(False, "fix this"), (True, "")])
        def check(orig, gen_text):
            return next(check_results)

        wrapped = with_feedback_retries(gen, check, num_retries=3)
        wrapped("input")
        assert call_args[0] == ""       # first call has no feedback
        assert call_args[1] == "fix this"  # second call gets feedback

    def test_exhausted_includes_feedback(self):
        gen = MagicMock(return_value="bad")
        check = MagicMock(return_value=(False, "still wrong"))
        wrapped = with_feedback_retries(gen, check, num_retries=2)
        result, info = wrapped("input")
        assert "DISCARDED" in info
        assert "still wrong" in info


class TestBatchCaller:
    def test_sequential(self):
        backend = MockBackend(["r0", "r1", "r2"])
        caller = BatchCaller(backend)
        messages_list = [
            [{"role": "user", "content": f"msg{i}"}]
            for i in range(3)
        ]
        results = caller.run(messages_list, "model")
        assert results == ["r0", "r1", "r2"]

    def test_preserves_order(self):
        backend = MockBackend(["a", "b", "c"])
        caller = BatchCaller(backend, max_workers=1)
        messages_list = [
            [{"role": "user", "content": str(i)}]
            for i in range(3)
        ]
        results = caller.run(messages_list, "m")
        assert results == ["a", "b", "c"]

    def test_callback_called(self):
        backend = MockBackend(["x", "y"])
        caller = BatchCaller(backend)
        completed = []
        caller.run(
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            "m",
            on_complete=lambda i, r: completed.append((i, r)),
        )
        assert len(completed) == 2
        assert (0, "x") in completed
        assert (1, "y") in completed

    def test_concurrent(self):
        backend = MockBackend(["a", "b", "c"])
        caller = BatchCaller(backend, max_workers=2)
        messages_list = [
            [{"role": "user", "content": str(i)}]
            for i in range(3)
        ]
        results = caller.run(messages_list, "m")
        assert sorted(results) == ["a", "b", "c"]

    def test_with_rate_limiter(self):
        backend = MockBackend("ok")
        limiter = RateLimiter()
        caller = BatchCaller(backend, rate_limiter=limiter)
        results = caller.run(
            [[{"role": "user", "content": "hi"}]],
            "unknown-test-model",
        )
        assert results == ["ok"]
        assert "unknown-test-model" in limiter._timestamps

    def test_passes_kwargs(self):
        backend = MockBackend("ok")
        caller = BatchCaller(backend)
        caller.run(
            [[{"role": "user", "content": "hi"}]],
            "m",
            max_tokens=50,
        )
        assert backend.calls[0]["max_tokens"] == 50

    def test_progress_label_emits_ticks_without_changing_results(self, capsys):
        backend = MockBackend(["x", "y", "z"])
        caller = BatchCaller(backend)  # series path, max_workers=1
        results = caller.batch_generate(
            [[{"role": "user", "content": str(i)}] for i in range(3)],
            "m",
            progress="gen chunk 1/1",
        )
        assert results == ["x", "y", "z"]
        out = capsys.readouterr().out
        assert "gen chunk 1/1: 0/3 ..." in out   # announce line
        assert "gen chunk 1/1: 3/3 done" in out  # final tick

    def test_progress_chains_with_on_complete(self, capsys):
        backend = MockBackend(["a", "b"])
        caller = BatchCaller(backend)
        seen = []
        results = caller.batch_generate(
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            "m",
            on_complete=lambda i, r: seen.append((i, r)),
            progress="lbl",
        )
        assert results == ["a", "b"]
        assert sorted(seen) == [(0, "a"), (1, "b")]   # user callback still fired
        assert "lbl: 2/2 done" in capsys.readouterr().out
