"""The pending model call a generator yields."""

from dataclasses import dataclass


@dataclass
class LLMRequest:
    """A pending LLM call yielded by a generator.

    Attributes:
        model: Model name. Pending requests are grouped by it and sent one
            batch per model.
        messages: Chat messages for the call.
        internals_id: Capture this call's internals under this id. ``None``
            captures nothing.
    """

    model: str
    messages: list[dict]
    internals_id: str | None = None
