"""Check a model's output with a checker model: the verdict rule and the
single and batched check calls.
"""

from __future__ import annotations

from collections.abc import Callable

from redact.llms.client import ModelClient
from redact.llms.router import batch_generate_samples, generate_sample

#TODO: Decide how strict the accept probe should be (which phrase the checker prompts ask for, word boundary).
_ACCEPT_PREFIXES = ("yes", "ok", "accept", "pass")


def is_accepted(response: str) -> bool:
    """Parse a checker or judge reply into accept/reject.

    The shared acceptance rule. Use it instead of re-implementing the check.

    Args:
        response: Raw reply text.

    Returns:
        True when the reply starts with "yes", "ok", "accept" or "pass"
        (case-insensitive, surrounding whitespace ignored).
    """
    return response.strip().lower().startswith(_ACCEPT_PREFIXES)


def _verdict(response: str) -> tuple[bool, str]:
    """``(accepted, reasoning)``; reasoning is the full reply on rejection."""
    return (True, "") if is_accepted(response) else (False, response)


def check_sample(
    client: ModelClient,
    sample: str,
    build_check_messages: Callable[[str, str], list[dict]],
    original: str = "",
    **kwargs,
) -> tuple[bool, str]:
    """Check one sample with a checker model.

    Args:
        client: The checker model.
        sample: The generated text to validate.
        build_check_messages: ``(original, sample) -> messages``.
        original: Text the sample is compared against, e.g. the input for an
            output check. ``""`` when the checker needs none.
        **kwargs: Forwarded to :meth:`ModelClient.generate`.

    Returns:
        ``(accepted, reasoning)``. ``reasoning`` is empty on acceptance and
        the checker's full reply on rejection.
    """
    messages = build_check_messages(original, sample)
    return _verdict(generate_sample(client, messages, **kwargs))


def batch_check_samples(
    client: ModelClient,
    samples: list[str],
    build_check_messages: Callable[[str, str], list[dict]],
    originals: list[str] | None = None,
    batch_size: int = 32,
    progress: str | None = None,
    internals_ids: list[str | None] | None = None,
    **kwargs,
) -> list[tuple[bool, str]]:
    """Check many samples in chunks, preserving order.

    Args:
        client: The checker model.
        samples: Texts to validate.
        build_check_messages: ``(original, sample) -> messages``, as in
            :func:`check_sample`.
        originals: One comparison text per sample. ``None`` passes ``""`` for
            every item.
        batch_size: Maximum items per chunk, at least 1.
        progress: Label for progress logging. The chunk index is appended
            when there is more than one chunk.
        internals_ids: One capture id (or ``None``) per sample.
        **kwargs: Forwarded to :meth:`ModelClient.generate`.

    Returns:
        ``(accepted, reasoning)`` per sample, in input order.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    if not samples:
        return []
    if originals is None:
        originals = [""] * len(samples)
    if len(originals) != len(samples):
        raise ValueError(
            f"originals must be the same length as samples "
            f"({len(originals)} != {len(samples)})."
        )
    messages_list = [build_check_messages(o, s) for o, s in zip(originals, samples)]
    responses = batch_generate_samples(
        client, messages_list, batch_size=batch_size, progress=progress,
        internals_ids=internals_ids, **kwargs,
    )
    return [_verdict(r) for r in responses]
