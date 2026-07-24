"""Paraphrasing / fingerprint removal.

Batched paraphrase pass: rephrase generated samples to strip the stylistic
fingerprints of the generating model while preserving meaning. The prompt is
loaded from ``prompts/content_moderation/paraphrase/template.json`` (a 1:1
"rephrase this" instruction).

The real defingerprinting model is trained in a separate repository; here any
capable model (the ``paraphraser`` role) drives the prompt. Whether a paraphrase
actually preserved meaning is a *separate* concern — validated by a dedicated
checker (:func:`redact.content_moderation.checker.build_paraphrase_checker`),
never by the paraphraser itself.

Dispatch goes through a capability-aware :class:`BatchCaller` (one vLLM engine
pass / parallel API thread-pool / series-only sequential), so paraphrasing is
rate-limited and batched like every other stage.
"""

from ..llms.base import LLMBackend
from ..llms.calls import generate_sample
from ..llms.prompts import load_prompt, build_messages
from ..llms.wrappers import RateLimiter, BatchCaller

_PROMPT_DIR = None  # Uses load_prompt() default (package-relative)


def _load_paraphrase_prompt(prompt_dir: str | None = _PROMPT_DIR) -> dict:
    """Load the paraphrase prompt config from JSON."""
    return load_prompt("content_moderation", "paraphrase", prompt_dir=prompt_dir)


def paraphrase_batch(
    backend: LLMBackend,
    model: str,
    samples: list[str],
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | None = _PROMPT_DIR,
    progress: str | None = None,
    **kwargs,
) -> list[str]:
    """Paraphrase a batch of samples in one capability-aware dispatch.

    Args:
        backend: Paraphraser backend.
        model: Paraphraser model identifier.
        samples: Texts to paraphrase.
        rate_limiter: Optional shared rate limiter.
        prompt_dir: Root prompt directory (defaults to the package prompts/).
        progress: Optional progress label for the batch dispatch.
        **kwargs: Passed to ``BatchCaller.batch_generate`` / the backend.

    Returns:
        Paraphrased texts, one per input, in order.
    """
    if not samples:
        return []
    prompt_config = _load_paraphrase_prompt(prompt_dir)
    messages_list = [build_messages(prompt_config, sample=s) for s in samples]
    caller = BatchCaller.from_model(backend, model, rate_limiter=rate_limiter)
    return caller.batch_generate(messages_list, model, progress=progress, **kwargs)


def paraphrase_sample(
    backend: LLMBackend,
    model: str,
    sample: str,
    rate_limiter: RateLimiter | None = None,
    system_prompt: str | None = None,
    prompt_dir: str | None = _PROMPT_DIR,
    **kwargs,
) -> str:
    """Paraphrase a single sample to remove stylistic fingerprints.

    Thin wrapper over :func:`paraphrase_batch`. Pass ``system_prompt`` to override
    the JSON template's system prompt for this call.

    Args:
        backend: Paraphraser backend.
        model: Paraphraser model identifier.
        sample: Text to paraphrase.
        rate_limiter: Optional shared rate limiter.
        system_prompt: Custom system prompt (overrides the JSON template).
        prompt_dir: Root prompt directory.
        **kwargs: Passed to the backend.

    Returns:
        Paraphrased text.
    """
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample},
        ]
        return generate_sample(backend, model, messages, rate_limiter, **kwargs)
    return paraphrase_batch(
        backend, model, [sample], rate_limiter=rate_limiter,
        prompt_dir=prompt_dir, **kwargs,
    )[0]
