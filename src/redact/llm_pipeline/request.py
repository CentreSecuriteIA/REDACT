"""The pending model call a generator yields."""

from __future__ import annotations

from dataclasses import dataclass, field

from redact.llms.client import ModelClient


@dataclass
class LLMRequest:
    """A pending LLM call yielded by a generator.

    Attributes:
        model: Model name: the batch key when no client is attached, and the
            progress label either way.
        messages: Chat messages for the call.
        internals_id: Capture this call's internals under this id. ``None``
            captures nothing.
        client: The client to call. Requests with the same client object go
            out as one batch; ``None`` resolves ``model`` by name.
    """

    model: str
    messages: list[dict]
    internals_id: str | None = None
    client: ModelClient | None = field(default=None, kw_only=True)

    @classmethod
    def for_client(
        cls,
        client: ModelClient,
        messages: list[dict],
        *,
        internals_id: str | None = None,
    ) -> LLMRequest:
        """A request bound to ``client``, with ``model`` set from it."""
        return cls(client.model, messages, internals_id=internals_id, client=client)
