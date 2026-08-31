"""Tests for the paraphrase meaning-preservation checker (check→drop)."""

from redact.content_moderation.checker import build_paraphrase_checker
from redact.llms.router import batch_check_samples
from tests.conftest import MockBackend, make_client


def test_message_shape():
    checker = build_paraphrase_checker()
    msgs = checker("the original", "a reworded version")
    assert msgs[0]["role"] == "system" and "paraphrase" in msgs[0]["content"].lower()
    assert msgs[1] == {
        "role": "user",
        "content": "ORIGINAL:\nthe original\n\nPARAPHRASE:\na reworded version",
    }


def test_batched_check_accepts_and_drops():
    # "Yes..." accepts; anything else is a drop with the response as reasoning.
    backend = MockBackend(["Yes, faithful.", "No — it softened the request."])
    checker = build_paraphrase_checker()
    originals = ["orig A", "orig B"]
    paraphrases = ["para A", "para B"]
    results = batch_check_samples(make_client(backend), paraphrases, checker, originals=originals)
    assert results[0][0] is True and results[0][1] == ""
    assert results[1][0] is False and "softened" in results[1][1]
