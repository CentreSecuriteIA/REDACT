"""Caller-facing helpers around a :class:`~redact.llms.client.ModelClient`.

The client itself is the model: fully wired at construction, it takes a batch
of messages and returns a batch of replies. This module is the thin layer other
subsystems call on top of that — single-sample convenience, chunking, and the
check loop (build messages with the caller's checker, run them, map the replies
through :func:`is_accepted`).

Nothing here decides *how* a batch executes — rate limiting, fan-out, and the
native-vs-wrapped choice all live inside the client.

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

_ACCEPT_PREFIXES = ("yes", "ok", "accept", "pass")


def _chunk_label(progress: str | None, chunk_i: int, n_chunks: int) -> str | None:
    """Progress label for one chunk — indexed only when there's more than one."""
    if progress is None:
        return None
    return f"{progress} [{chunk_i}/{n_chunks}]" if n_chunks > 1 else progress


def is_accepted(response: str) -> bool:
    """Parse a checker/judge LLM response into accept/reject.

    The single acceptance rule shared by every checker/judge in the library
    (content moderation, output, paraphrase, translation, jailbreak-technique
    checks) — use this rather than reimplementing it at a call site.

    Args:
        response: Raw checker/judge response text.

    Returns:
        True when the response starts with "yes", "ok", "accept", or "pass"
        (case-insensitive).
    """
    return response.strip().lower().startswith(_ACCEPT_PREFIXES)


def generate_sample(client: ModelClient, messages: list[dict], **kwargs) -> str:
    """Generate one sample — a batch of one, unwrapped.

    Args:
        client: The model to call.
        messages: Chat messages for the single item.
        **kwargs: Forwarded to :meth:`ModelClient.generate` (``max_tokens``,
            ``temperature``, ``extra_body``, transport extras).

    Returns:
        Generated text.
    """
    return client.generate([messages], **kwargs)[0]


def check_sample(
    client: ModelClient,
    sample: str,
    build_check_messages: Callable[[str, str], list[dict]],
    original: str = "",
    **kwargs,
) -> tuple[bool, str]:
    """Run one sample past a checker model and interpret the verdict.

    Args:
        client: The checker model.
        sample: The generated text to validate.
        build_check_messages: ``(original, sample) -> messages``. Checkers that
            only need the text to validate ignore ``original``; it exists so
            checkers genuinely comparing two texts (output-vs-input,
            paraphrase-vs-source) get both as real arguments instead of the
            caller concatenating them into ``sample``.
        original: The other half of a two-text comparison; ``""`` when the
            checker doesn't need one.
        **kwargs: Forwarded to :meth:`ModelClient.generate`.

    Returns:
        ``(accepted, reasoning)`` — reasoning is empty on acceptance, and the
        checker's full response on rejection.
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
    """Check many samples, in chunks, preserving order.

    Splits into chunks of ``batch_size`` and hands each to the client, which
    runs it however its transport runs batches. The accept/reject
    interpretation stays here — the client only returns text.

    Args:
        client: The checker model.
        samples: Sample texts to validate.
        build_check_messages: ``(original, sample) -> messages`` — see
            :func:`check_sample` for the two-arg contract.
        originals: One "other half" per sample, for checkers comparing two
            texts. ``None`` means every item gets ``""``.
        batch_size: Max items per chunk.
        progress: Label enabling progress ticks; the chunk index is appended
            when there is more than one chunk.
        internals_ids: One capture id per sample, sliced per chunk.
        **kwargs: Forwarded to :meth:`ModelClient.generate`.

    Returns:
        ``(accepted, reasoning)`` per sample, in input order.
    """
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
    """Generate many samples in chunks — the generation-side counterpart to
    :func:`batch_check_samples`.

    Use this over calling the client directly when the batch is large enough
    to want chunk-level progress, or when a caller needs each reply mapped
    through its own checker afterwards (so a single ``build_check_messages``
    can't describe the whole batch).

    Args:
        client: The model to call.
        messages_list: One chat message list per item.
        batch_size: Max items per chunk.
        progress: Label enabling progress ticks per chunk.
        internals_ids: One capture id per item, sliced per chunk.
        **kwargs: Forwarded to :meth:`ModelClient.generate`.

    Returns:
        Generated text, one per item, in input order.
    """
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
