"""Tests for the paraphrase meaning-preservation checker (check→drop)."""

from tests.conftest import MockBackend
from redact.content_moderation.checker import (
    build_paraphrase_checker, paraphrase_check_payload,
)
from redact.llms.calls import batch_check_samples


def test_payload_and_message_shape():
    checker = build_paraphrase_checker()
    payload = paraphrase_check_payload("the original", "a reworded version")
    assert payload == "ORIGINAL:\nthe original\n\nPARAPHRASE:\na reworded version"
    msgs = checker(payload)
    assert msgs[0]["role"] == "system" and "paraphrase" in msgs[0]["content"].lower()
    assert msgs[1] == {"role": "user", "content": payload}


def test_batched_check_accepts_and_drops():
    # "Yes..." accepts; anything else is a drop with the response as reasoning.
    backend = MockBackend(["Yes, faithful.", "No — it softened the request."])
    checker = build_paraphrase_checker()
    payloads = [
        paraphrase_check_payload("orig A", "para A"),
        paraphrase_check_payload("orig B", "para B"),
    ]
    results = batch_check_samples(backend, "model", payloads, checker)
    assert results[0][0] is True and results[0][1] == ""
    assert results[1][0] is False and "softened" in results[1][1]
