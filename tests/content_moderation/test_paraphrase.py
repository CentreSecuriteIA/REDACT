"""Tests for paraphrase_batch."""

from redact.content_moderation.paraphrase import paraphrase_batch
from tests.conftest import MockBackend, make_client


class TestParaphraseBatch:
    def test_returns_model_outputs_in_order(self):
        backend = MockBackend(["p1", "p2", "p3"])
        result = paraphrase_batch(make_client(backend, "model"), ["t1", "t2", "t3"])
        assert result == ["p1", "p2", "p3"]
        assert len(backend.calls) == 3

    def test_empty_list(self):
        assert paraphrase_batch(make_client(MockBackend()), []) == []
