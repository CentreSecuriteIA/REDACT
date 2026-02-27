"""Placeholder for paraphrasing / fingerprint removal pipeline.

This module wraps calls to a fine-tuned paraphrasing model that removes
stylistic fingerprints from generated samples. The actual paraphrasing model
is trained and maintained in a separate repository.

For now, this is a pass-through stub that can be called from the pipeline
without error.

Prompt is loaded from Prompts/Content_Moderation/paraphrase/template.json.

TODO:
    - Integrate with the defingerprinting model API endpoint or local vLLM
    - Add paraphrase quality checking (semantic similarity threshold)
    - Add batch paraphrasing support via VLLMBackend.batch_generate()
"""

from ..LLMs.base import LLMBackend
from ..LLMs.calls import generate_sample
from ..LLMs.prompts import load_prompt
from ..LLMs.wrappers import RateLimiter

_PROMPT_DIR = None  # Uses load_prompt() default (package-relative)


def _load_paraphrase_prompt(prompt_dir: str = _PROMPT_DIR) -> dict:
    """Load the paraphrase prompt config from JSON."""
    return load_prompt("Content_Moderation", "paraphrase", prompt_dir=prompt_dir)


def paraphrase_sample(
    backend: LLMBackend,
    model: str,
    sample: str,
    rate_limiter: RateLimiter | None = None,
    system_prompt: str | None = None,
    prompt_dir: str = _PROMPT_DIR,
    **kwargs,
) -> str:
    """Paraphrase a sample to remove stylistic fingerprints.

    PLACEHOLDER: Currently returns the sample unchanged.
    Will be implemented when the defingerprinting model is available.

    Args:
        backend: LLM backend for the paraphraser.
        model: Paraphraser model identifier.
        sample: Text to paraphrase.
        rate_limiter: Optional rate limiter.
        system_prompt: Custom system prompt (overrides JSON template).
        prompt_dir: Root directory for prompt JSON files.
        **kwargs: Passed to backend.generate().

    Returns:
        Paraphrased text (currently: original text unchanged).
    """
    # TODO: Replace stub with actual paraphrasing call when model is available:
    #
    # prompt_config = _load_paraphrase_prompt(prompt_dir)
    # sys_prompt = system_prompt or prompt_config["system_prompt"]
    # messages = [
    #     {"role": "system", "content": sys_prompt},
    #     {"role": "user", "content": sample},
    # ]
    # return generate_sample(backend, model, messages, rate_limiter, **kwargs)
    #
    return sample


def paraphrase_batch(
    backend: LLMBackend,
    model: str,
    samples: list[str],
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str = _PROMPT_DIR,
    **kwargs,
) -> list[str]:
    """Paraphrase a batch of samples.

    PLACEHOLDER: Currently returns all samples unchanged.

    Args:
        backend: LLM backend.
        model: Model identifier.
        samples: List of texts to paraphrase.
        rate_limiter: Optional rate limiter.
        prompt_dir: Root directory for prompt JSON files.
        **kwargs: Passed to generate.

    Returns:
        List of paraphrased texts (currently: originals unchanged).
    """
    return [
        paraphrase_sample(
            backend, model, s, rate_limiter, prompt_dir=prompt_dir, **kwargs
        )
        for s in samples
    ]
