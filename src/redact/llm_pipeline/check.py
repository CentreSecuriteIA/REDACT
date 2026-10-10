"""Check a model's output with a checker model: the verdict rule, the single
and batched check calls, and the retry steps a driver runs per unit.
"""

from __future__ import annotations

from collections.abc import Callable, Generator

from redact.llms.client import ModelClient
from redact.llms.router import batch_generate_samples, generate_sample

from .request import LLMRequest

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


def _attempt_id(internals_id: str | None, attempt: int) -> str | None:
    """Attempt 1 keeps the id; later attempts nest under it as ``attempt_n``."""
    if internals_id is None or attempt == 1:
        return internals_id
    return f"{internals_id}/attempt_{attempt}"


def checked(
    gen: ModelClient,
    build_gen_messages: Callable[[str], list[dict]],
    check: ModelClient | None = None,
    build_check_messages: Callable[[str, str], list[dict]] | None = None,
    *,
    original: str = "",
    internals_ids: tuple[str | None, str | None] = (None, None),
    max_attempts: int = 1,
) -> Generator[LLMRequest, str, tuple[str, bool, str]]:
    """Generate, check, and regenerate a rejected reply with the verdict.

    Args:
        gen: The generating model.
        build_gen_messages: ``feedback -> messages``. ``feedback`` is ``""``
            on the first attempt and the checker's reply afterwards.
        check: The checker model. ``None`` accepts the first reply.
        build_check_messages: ``(original, text) -> messages``.
        original: Text the reply is compared against.
        internals_ids: The first attempt's (generation, check) capture ids.
        max_attempts: Generations before giving up, at least 1.

    Returns:
        ``(text, accepted, reasoning)``. ``reasoning`` is the last verdict
        when every attempt was rejected, else ``""``.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}.")
    if check is not None and build_check_messages is None:
        raise ValueError("A check client needs build_check_messages.")
    gen_id, check_id = internals_ids
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        text = yield LLMRequest.for_client(
            gen, build_gen_messages(feedback),
            internals_id=_attempt_id(gen_id, attempt),
        )
        if check is None:
            return text, True, ""
        verdict = yield LLMRequest.for_client(
            check, build_check_messages(original, text),
            internals_id=_attempt_id(check_id, attempt),
        )
        if is_accepted(verdict):
            return text, True, ""
        feedback = verdict
    return text, False, feedback


def extracted(
    gen: ModelClient,
    build_gen_messages: Callable[[], list[dict]],
    extract: Callable[[str], list[str]],
    *,
    internals_id: str | None = None,
    max_attempts: int = 3,
) -> Generator[LLMRequest, str, tuple[list[str], int]]:
    """Generate until ``extract(reply)`` returns samples.

    Args:
        gen: The generating model.
        build_gen_messages: Builds the messages; called once per attempt.
        extract: ``reply -> samples``; ``[]`` for a reply rejected whole.
        internals_id: The first attempt's capture id.
        max_attempts: Generations before giving up, at least 1.

    Returns:
        ``(samples, attempts)``. ``samples`` is ``[]`` when every attempt
        was rejected.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}.")
    for attempt in range(1, max_attempts + 1):
        raw = yield LLMRequest.for_client(
            gen, build_gen_messages(), internals_id=_attempt_id(internals_id, attempt),
        )
        samples = extract(raw)
        if samples:
            return samples, attempt
    return [], max_attempts
