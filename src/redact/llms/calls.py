"""High-level generate/check call pairs.

These functions compose the low-level backend calls with rate limiting
and retry logic into the generate→check→feedback workflow that is central
to both the content moderation and jailbreak pipelines.

Key feature: gen_backend/gen_model and check_backend/check_model can
differ, supporting patterns like "Venice generates, Claude validates".
"""

from typing import Callable

from .base import LLMBackend
from .wrappers import RateLimiter, with_retries, with_feedback_retries


def generate_sample(
    backend: LLMBackend,
    model: str,
    messages: list[dict],
    rate_limiter: RateLimiter | None = None,
    **kwargs,
) -> str:
    """Generate a single sample with optional rate limiting.

    Args:
        backend: LLM backend to use.
        model: Model identifier.
        messages: Chat messages.
        rate_limiter: Optional shared rate limiter.
        **kwargs: Passed to backend.generate().

    Returns:
        Generated text.
    """
    if rate_limiter:
        rate_limiter.wait_if_needed(model)
    return backend.generate(messages, model, **kwargs)


def check_sample(
    backend: LLMBackend,
    model: str,
    sample: str,
    build_check_messages: Callable[[str], list[dict]],
    rate_limiter: RateLimiter | None = None,
    **kwargs,
) -> tuple[bool, str]:
    """Check a generated sample for acceptance.

    The checker LLM response is parsed for acceptance: if the response
    starts with "yes", "ok", "accept", or "pass" (case-insensitive) the
    sample is accepted. Otherwise it's rejected and the full response
    is returned as feedback/reasoning.

    Args:
        backend: LLM backend for the checker.
        model: Checker model identifier.
        sample: The generated text to validate.
        build_check_messages: Function that takes the sample and returns
                              the message list for the checker.
        rate_limiter: Optional shared rate limiter.
        **kwargs: Passed to backend.generate().

    Returns:
        (accepted, reasoning) — reasoning is empty on acceptance,
        contains the checker's feedback on rejection.
    """
    if rate_limiter:
        rate_limiter.wait_if_needed(model)

    messages = build_check_messages(sample)
    response = backend.generate(messages, model, **kwargs)
    response_stripped = response.strip().lower()

    accepted = any(
        response_stripped.startswith(prefix)
        for prefix in ("yes", "ok", "accept", "pass")
    )

    if accepted:
        return True, ""
    return False, response


def batch_check_samples(
    backend: LLMBackend,
    model: str,
    samples: list[str],
    build_check_messages: Callable[[str], list[dict]],
    batch_size: int = 32,
    **kwargs,
) -> list[tuple[bool, str]]:
    """Check multiple samples in batched engine passes.

    Splits samples into chunks of ``batch_size``, calls
    ``backend.batch_generate()`` once per chunk. Order is preserved.
    Returns a list of ``(accepted, reasoning)`` tuples parallel to ``samples``.

    For vLLM this means one engine pass per chunk (much faster than N
    individual ``generate()`` calls). For API backends the base-class
    ``batch_generate()`` falls back to a sequential loop — behaviour is
    identical to the old per-sample loop.

    Used by both ``InputPipeline.check_samples()`` (content moderation) and
    ``ConstitutionInputPipeline`` — both share the same ``InputPipeline``
    implementation so this function serves both pipelines.

    Args:
        backend: LLM backend for the checker.
        model: Checker model identifier.
        samples: Sample texts to validate.
        build_check_messages: Function(sample_text) -> checker message list.
        batch_size: Max prompts per engine pass (default 32).
        **kwargs: Passed to backend.batch_generate().

    Returns:
        List of (accepted, reasoning) in the same order as ``samples``.
        reasoning is empty string on acceptance, full checker response on
        rejection.
    """
    if not samples:
        return []

    results: list[tuple[bool, str]] = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        messages_list = [build_check_messages(s) for s in chunk]
        responses = backend.batch_generate(messages_list, model, **kwargs)
        for response in responses:
            accepted = any(
                response.strip().lower().startswith(prefix)
                for prefix in ("yes", "ok", "accept", "pass")
            )
            results.append((accepted, "" if accepted else response))
    return results


def generate_with_check(
    gen_backend: LLMBackend,
    gen_model: str,
    check_backend: LLMBackend,
    check_model: str,
    gen_messages: list[dict],
    build_check_messages: Callable[[str], list[dict]],
    rate_limiter: RateLimiter | None = None,
    num_retries: int = 5,
    use_feedback: bool = False,
    build_gen_messages_with_feedback: Callable[[str], list[dict]] | None = None,
    **kwargs,
) -> tuple[str, str]:
    """Generate a sample, check it, retry on rejection.

    Supports:
    - Separate backends/models for generation vs checking
    - Simple retry (use_feedback=False): regenerate from scratch each attempt
    - Feedback retry (use_feedback=True): checker reasoning is passed to
      build_gen_messages_with_feedback to construct a corrected generation prompt

    Args:
        gen_backend: Backend for generation.
        gen_model: Model for generation.
        check_backend: Backend for checking (can be same as gen_backend).
        check_model: Model for checking (can be same as gen_model).
        gen_messages: Base messages for generation.
        build_check_messages: Function(sample) -> checker message list.
        rate_limiter: Optional shared rate limiter.
        num_retries: Maximum attempts.
        use_feedback: If True, use feedback-aware retries.
        build_gen_messages_with_feedback: Required when use_feedback=True.
            Function(feedback) -> updated generation message list.
        **kwargs: Passed to both generate and check calls.

    Returns:
        (result, info) where info="" on success or "DISCARDED; ..." on exhaustion.
    """
    if use_feedback:
        if build_gen_messages_with_feedback is None:
            raise ValueError(
                "build_gen_messages_with_feedback is required when use_feedback=True"
            )

        def gen_fn(text: str, feedback: str = "") -> str:
            msgs = (
                build_gen_messages_with_feedback(feedback)
                if feedback
                else gen_messages
            )
            return generate_sample(gen_backend, gen_model, msgs, rate_limiter, **kwargs)

        def check_fn(original: str, generated: str) -> tuple[bool, str]:
            return check_sample(
                check_backend, check_model, generated,
                build_check_messages, rate_limiter, **kwargs
            )

        wrapped = with_feedback_retries(gen_fn, check_fn, num_retries=num_retries)
        return wrapped("")

    else:
        def gen_fn(text: str) -> str:
            return generate_sample(gen_backend, gen_model, gen_messages, rate_limiter, **kwargs)

        def check_fn(original: str, generated: str) -> tuple[bool, str]:
            return check_sample(
                check_backend, check_model, generated,
                build_check_messages, rate_limiter, **kwargs
            )

        wrapped = with_retries(gen_fn, check_fn, num_retries=num_retries)
        return wrapped("")
