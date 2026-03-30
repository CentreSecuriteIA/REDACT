"""Tests for generate/check call pairs and retry logic."""

import pytest

from redact.llms.calls import generate_sample, check_sample, generate_with_check
from redact.llms.wrappers import RateLimiter


class TestGenerateSample:
    def test_calls_backend(self, mock_backend):
        messages = [{"role": "user", "content": "hi"}]
        result = generate_sample(mock_backend, "test-model", messages)
        assert result == "mock response"
        assert len(mock_backend.calls) == 1
        assert mock_backend.calls[0]["model"] == "test-model"

    def test_rate_limited(self, mock_backend):
        limiter = RateLimiter()
        messages = [{"role": "user", "content": "hi"}]
        result = generate_sample(mock_backend, "test-model", messages, rate_limiter=limiter)
        assert result == "mock response"

    def test_passes_kwargs(self, mock_backend):
        messages = [{"role": "user", "content": "hi"}]
        generate_sample(mock_backend, "m", messages, max_tokens=100)
        assert mock_backend.calls[0]["max_tokens"] == 100


class TestCheckSample:
    def _check(self, backend, response_text):
        from tests.conftest import MockBackend
        b = MockBackend(response_text)
        return check_sample(
            b, "model", "sample text",
            build_check_messages=lambda s: [{"role": "user", "content": f"Check: {s}"}],
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


class TestGenerateWithCheck:
    def test_success_first_try(self, mock_backend_factory):
        gen = mock_backend_factory("generated text")
        check = mock_backend_factory("Yes")

        result, info = generate_with_check(
            gen, "gen-model", check, "check-model",
            gen_messages=[{"role": "user", "content": "generate"}],
            build_check_messages=lambda s: [{"role": "user", "content": f"check {s}"}],
        )
        assert result == "generated text"
        assert info == ""

    def test_retries_on_rejection(self, mock_backend_factory):
        gen = mock_backend_factory(["bad1", "bad2", "good"])
        check = mock_backend_factory(["No", "No", "Yes"])

        result, info = generate_with_check(
            gen, "m", check, "m",
            gen_messages=[{"role": "user", "content": "gen"}],
            build_check_messages=lambda s: [{"role": "user", "content": s}],
            num_retries=5,
        )
        assert result == "good"
        assert info == ""

    def test_exhausted(self, mock_backend_factory):
        gen = mock_backend_factory("always bad")
        check = mock_backend_factory("No, rejected")

        result, info = generate_with_check(
            gen, "m", check, "m",
            gen_messages=[{"role": "user", "content": "gen"}],
            build_check_messages=lambda s: [{"role": "user", "content": s}],
            num_retries=2,
        )
        assert "DISCARDED" in info

    def test_feedback_mode_requires_callback(self, mock_backend_factory):
        gen = mock_backend_factory("x")
        check = mock_backend_factory("Yes")

        with pytest.raises(ValueError, match="build_gen_messages_with_feedback"):
            generate_with_check(
                gen, "m", check, "m",
                gen_messages=[{"role": "user", "content": "gen"}],
                build_check_messages=lambda s: [{"role": "user", "content": s}],
                use_feedback=True,
            )

    def test_feedback_mode(self, mock_backend_factory):
        gen = mock_backend_factory(["bad", "good"])
        check = mock_backend_factory(["No, try harder", "Yes"])

        result, info = generate_with_check(
            gen, "m", check, "m",
            gen_messages=[{"role": "user", "content": "gen"}],
            build_check_messages=lambda s: [{"role": "user", "content": s}],
            use_feedback=True,
            build_gen_messages_with_feedback=lambda fb: [{"role": "user", "content": f"fix: {fb}"}],
            num_retries=3,
        )
        assert result == "good"
        assert info == ""
