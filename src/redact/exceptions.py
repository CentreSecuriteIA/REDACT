"""The errors the library defines; elsewhere it raises the built-in ones."""


class RedactError(Exception):
    """Base class of the library's own errors."""


class PromptTooLongError(RedactError):
    """Raised when a prompt is longer than the model's context."""
