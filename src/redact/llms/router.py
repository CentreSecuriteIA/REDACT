"""Helpers over a :class:`~redact.llms.client.ModelClient`.

Single-sample generation, chunked batches, and the check loop that turns a
checker model's replies into accept/reject verdicts. How a batch is executed
is left to the client.

Usage::

    from redact.llms import ModelClient, generate_sample, check_sample

    gen = ModelClient.create("venice-uncensored")
    check = ModelClient.create("claude-opus-4-6")

    text = generate_sample(gen, messages)
    accepted, reasoning = check_sample(check, text, build_check_messages)
"""

from __future__ import annotations

from collections.abc import Callable

from .client import ModelClient

#TODO: Decide how strict the accept probe should be (which phrase the checker prompts ask for, word boundary).
_ACCEPT_PREFIXES = ("yes", "ok", "accept", "pass")


def _chunk_label(progress: str | None, chunk_i: int, n_chunks: int) -> str | None:
    """Progress label for one chunk, indexed only when there are several."""
    if progress is None:
        return None
    return f"{progress} [{chunk_i}/{n_chunks}]" if n_chunks > 1 else progress


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


def generate_sample(client: ModelClient, messages: list[dict], **kwargs) -> str:
    """Generate one sample (a batch of one, unwrapped).

    Args:
        client: The model to call.
        messages: Chat messages for the item.
        **kwargs: Forwarded to :meth:`ModelClient.generate` (e.g.
            ``max_tokens``, ``temperature``).
    """
    return client.generate([messages], **kwargs)[0]


# TODO(review): ``build_check_messages`` is an ``(original, sample) -> messages``
# callable, and content_moderation/checker.py builds each one as its own closure
# over a PromptTemplate. Define that adapter once on PromptTemplate. The check
# helpers below belong with the pipelines and are due to move out of llms/.


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
    response = client.generate([build_check_messages(original, sample)], **kwargs)[0]
    return (True, "") if is_accepted(response) else (False, response)


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
    if originals is not None and len(originals) != len(samples):
        raise ValueError(
            f"originals must be the same length as samples "
            f"({len(originals)} != {len(samples)})."
        )
    if internals_ids is not None and len(internals_ids) != len(samples):
        raise ValueError(
            f"internals_ids must be the same length as samples "
            f"({len(internals_ids)} != {len(samples)})."
        )

    n_chunks = (len(samples) + batch_size - 1) // batch_size
    results: list[tuple[bool, str]] = []
    for chunk_i, i in enumerate(range(0, len(samples), batch_size), start=1):
        chunk = samples[i : i + batch_size]
        chunk_originals = (
            originals[i : i + batch_size] if originals is not None else [""] * len(chunk)
        )
        messages_list = [build_check_messages(o, s) for o, s in zip(chunk_originals, chunk)]
        responses = client.generate(
            messages_list,
            progress=_chunk_label(progress, chunk_i, n_chunks),
            internals_ids=(
                internals_ids[i : i + batch_size] if internals_ids is not None else None
            ),
            **kwargs,
        )
        for response in responses:
            accepted = is_accepted(response)
            results.append((accepted, "" if accepted else response))
    return results


def batch_generate_samples(
    client: ModelClient,
    messages_list: list[list[dict]],
    batch_size: int = 32,
    progress: str | None = None,
    internals_ids: list[str | None] | None = None,
    **kwargs,
) -> list[str]:
    """Generate many samples in chunks, preserving order.

    Args:
        client: The model to call.
        messages_list: One chat message list per item.
        batch_size: Maximum items per chunk, at least 1.
        progress: Label for progress logging. The chunk index is appended
            when there is more than one chunk.
        internals_ids: One capture id (or ``None``) per item.
        **kwargs: Forwarded to :meth:`ModelClient.generate`.

    Returns:
        Generated text, one per item, in input order.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    if not messages_list:
        return []
    if internals_ids is not None and len(internals_ids) != len(messages_list):
        raise ValueError(
            f"internals_ids must be the same length as messages_list "
            f"({len(internals_ids)} != {len(messages_list)})."
        )

    n_chunks = (len(messages_list) + batch_size - 1) // batch_size
    results: list[str] = []
    for chunk_i, i in enumerate(range(0, len(messages_list), batch_size), start=1):
        results.extend(client.generate(
            messages_list[i : i + batch_size],
            progress=_chunk_label(progress, chunk_i, n_chunks),
            internals_ids=(
                internals_ids[i : i + batch_size] if internals_ids is not None else None
            ),
            **kwargs,
        ))
    return results


def assert_single_sample_per_call(client: ModelClient, samples_per_call: int) -> None:
    """Raise if internals capture is combined with several samples per call.

    When one prompt asks for several samples in one completion, the captured
    internals of that forward pass cannot be attributed to any single sample.
    Pipelines with that shape (e.g. ``InputPipeline.run_from_constitution``)
    call this before dispatch with their own samples-per-call setting.

    Args:
        client: The generation client.
        samples_per_call: Samples one LLM call is asked to produce.

    Raises:
        ValueError: The client's backend supports internals capture and
            ``samples_per_call`` is not 1.
    """
    if client.compute_config.supports_internals and samples_per_call != 1:
        raise ValueError(
            f"{type(client.backend).__name__} supports internals capture, but this call "
            f"requests {samples_per_call} samples per LLM call — one forward pass "
            f"can't be attributed to more than one resulting sample. Set the "
            f"samples-per-call parameter to 1, or don't request internals capture."
        )
