"""Tests for the (real) paraphrase functions."""

from tests.conftest import MockBackend
from redact.content_moderation.paraphrase import paraphrase_sample, paraphrase_batch


class TestParaphraseSample:
    def test_calls_model_and_returns_output(self):
        backend = MockBackend("PARAPHRASED")
        result = paraphrase_sample(backend, "model", "original text")
        assert result == "PARAPHRASED"           # returns the model's output, not the input
        assert len(backend.calls) == 1
        # the sample is passed as the user message (template is "{sample}")
        assert backend.calls[0]["messages"][-1]["content"] == "original text"

    def test_system_prompt_override(self):
        backend = MockBackend("X")
        paraphrase_sample(backend, "model", "s", system_prompt="CUSTOM SYS")
        msgs = backend.calls[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "CUSTOM SYS"}
        assert msgs[1] == {"role": "user", "content": "s"}


class TestParaphraseBatch:
    def test_returns_model_outputs_in_order(self):
        backend = MockBackend(["p1", "p2", "p3"])
        result = paraphrase_batch(backend, "model", ["t1", "t2", "t3"])
        assert result == ["p1", "p2", "p3"]
        assert len(backend.calls) == 3

    def test_empty_list(self):
        assert paraphrase_batch(MockBackend(), "model", []) == []
