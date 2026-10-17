"""Helpers over a :class:`~redact.llms.client.ModelClient`.

Single-sample generation and chunked batches. How a batch is executed is
left to the client; the check loops are in ``redact.llm_pipeline``.

Usage::

    from redact.llms import ModelClient, generate_sample

    gen = ModelClient.create("venice-uncensored")
    text = generate_sample(gen, messages)
"""

from __future__ import annotations

from .client import ModelClient


def _chunk_label(progress: str | None, chunk_i: int, n_chunks: int) -> str | None:
    """Progress label for one chunk, indexed only when there are several."""
    if progress is None:
        return None
    return f"{progress} [{chunk_i}/{n_chunks}]" if n_chunks > 1 else progress


def generate_sample(client: ModelClient, messages: list[dict], **kwargs) -> str:
    """Generate one sample (a batch of one, unwrapped).

    Args:
        client: The model to call.
        messages: Chat messages for the item.
        **kwargs: Forwarded to :meth:`ModelClient.generate` (e.g.
            ``max_tokens``, ``temperature``).
    """
    return client.generate([messages], **kwargs)[0]


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
