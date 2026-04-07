"""Tests for the custom exception hierarchy."""

from redact.exceptions import (
    RedactError,
    ConfigError,
    PromptNotFoundError,
    GenerationError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_redact_error(self):
        for exc_cls in [ConfigError, PromptNotFoundError, GenerationError]:
            assert issubclass(exc_cls, RedactError)

    def test_redact_error_is_exception(self):
        assert issubclass(RedactError, Exception)

    def test_catchable_as_base(self):
        for exc_cls in [ConfigError, PromptNotFoundError, GenerationError]:
            try:
                raise exc_cls("test")
            except RedactError:
                pass  # expected

    def test_message_preserved(self):
        msg = "something went wrong"
        for exc_cls in [RedactError, ConfigError, PromptNotFoundError, GenerationError]:
            exc = exc_cls(msg)
            assert str(exc) == msg
