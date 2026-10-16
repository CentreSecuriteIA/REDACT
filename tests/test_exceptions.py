"""Tests for the library's own errors."""

from redact.exceptions import PromptTooLongError, RedactError


def test_redact_error_is_an_exception():
    assert issubclass(RedactError, Exception)


def test_prompt_too_long_is_a_redact_error():
    assert issubclass(PromptTooLongError, RedactError)


def test_message_preserved():
    assert str(PromptTooLongError("10036 input tokens")) == "10036 input tokens"
