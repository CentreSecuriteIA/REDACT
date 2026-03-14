"""Custom exception hierarchy for REDACT.

Users can catch ``RedactError`` to handle all library errors,
or use specific subclasses for fine-grained handling.
"""


class RedactError(Exception):
    """Base exception for the REDACT library."""


class ConfigError(RedactError):
    """Raised when the library is misconfigured (missing env vars, bad settings)."""


class PromptNotFoundError(RedactError):
    """Raised when a prompt template file cannot be found."""


class GenerationError(RedactError):
    """Raised when sample generation fails after exhausting retries."""
