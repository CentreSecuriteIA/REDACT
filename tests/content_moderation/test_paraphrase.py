"""Tests for paraphrase stub functions."""

from tests.conftest import MockBackend
from redact.content_moderation.paraphrase import paraphrase_sample, paraphrase_batch


class TestParaphraseSample:
    def test_returns_unchanged(self):
        backend = MockBackend()
        result = paraphrase_sample(backend, "model", "original text")
        assert result == "original text"


class TestParaphraseBatch:
    def test_returns_all_unchanged(self):
        backend = MockBackend()
        samples = ["text1", "text2", "text3"]
        result = paraphrase_batch(backend, "model", samples)
        assert result == samples

    def test_empty_list(self):
        backend = MockBackend()
        assert paraphrase_batch(backend, "model", []) == []
