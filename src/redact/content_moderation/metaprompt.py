"""Meta-prompt steps for the content moderation input pipeline.

The full automated pipeline needs only a category name to start:

  Step 1 -- generate_category_description():
    Category name -> LLM -> rich category description
    Template: Prompts/Content_Moderation/category_description/

  Step 2 -- generate_seeds():
    Category name + description -> LLM -> seed prompts (numbered list)
    Template: Prompts/Content_Moderation/seed_generation/

  Step 3 -- InputPipeline.run_category():
    Category name + description + seeds -> LLM -> actual samples
    Template: Prompts/Content_Moderation/generation/
    Checks each sample via quality checker, loops with feedback.

Fallback options:
  - Skip step 1: use the short description from the taxonomy JSON
  - Skip step 2: use hand-written seeds from content_moderation_seeds.json
  - Both steps can be bypassed for quick runs without extra LLM calls

Usage:
    from redact.Content_Moderation.metaprompt import (
        generate_category_description,
        generate_seeds,
    )
    from redact.Dataset_Functions import load_taxonomy, iter_categories

    taxonomy = load_taxonomy("content_moderation_categories")
    for category_name, category_info in iter_categories(taxonomy):

        # Step 1: rich description (or use taxonomy short description as fallback)
        description = generate_category_description(backend, model, category_name)

        # Step 2: seeds (or load from content_moderation_seeds.json as fallback)
        seed_text = generate_seeds(backend, model, category_name, description)

        # Step 3: generate samples
        result = pipeline.run_category(
            category=category_name,
            prompt_config=load_prompt("content_moderation", "generation"),
            build_check_messages=build_quality_checker(category_name),
            seed_kwargs_per_turn=[{
                "Category": category_name,
                "category_description": description,
                "SeedPrompts": seed_text,
            }] * num_turns,
        )
"""

import logging
from pathlib import Path

from ..llms.base import LLMBackend
from ..llms.calls import generate_sample
from ..llms.prompts import load_prompt, build_messages
from ..llms.extraction import extract_numbered_list
from ..llms.wrappers import RateLimiter

logger = logging.getLogger(__name__)


def generate_category_description(
    backend: LLMBackend,
    model: str,
    category: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
    max_retries: int = 2,
) -> str:
    """Generate a rich category description from just the category name.

    Step 1 of the automated pipeline. Produces a detailed, research-grade
    description of what belongs in the harm category, covering content types,
    formats, severity range, and distinguishing features.

    Args:
        backend: LLM backend.
        model: Model identifier.
        category: Harm category name (e.g. "CBRN", "Self-Harm").
        rate_limiter: Optional rate limiter.
        prompt_dir: Prompt directory override.
        max_retries: Retries on empty output.

    Returns:
        Rich description string. Falls back to the category name itself
        if all retries fail.
    """
    config = load_prompt(
        "content_moderation", "category_description", prompt_dir=prompt_dir
    )
    messages = build_messages(config, Category=category)

    for attempt in range(max_retries):
        result = generate_sample(backend, model, messages, rate_limiter)
        result = result.strip()
        if result:
            logger.info(
                "Category description for '%s': %d chars (attempt %d)",
                category, len(result), attempt + 1,
            )
            return result
        logger.warning(
            "Empty description for '%s' (attempt %d/%d)",
            category, attempt + 1, max_retries,
        )

    logger.warning("Falling back to category name as description for '%s'", category)
    return category


def generate_seeds(
    backend: LLMBackend,
    model: str,
    category: str,
    category_description: str,
    num_seeds: int = 10,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
    max_retries: int = 3,
) -> str:
    """Generate seed prompts from the category name and description.

    Step 2 of the automated pipeline. Produces a diverse set of short,
    realistic harmful prompts that serve as style/type examples for the
    generation step. Output is a plain numbered list string ready to
    inject as {SeedPrompts} in the generation template.

    Args:
        backend: LLM backend.
        model: Model identifier.
        category: Harm category name.
        category_description: Rich description from generate_category_description().
        num_seeds: Number of seed prompts to generate.
        rate_limiter: Optional rate limiter.
        prompt_dir: Prompt directory override.
        max_retries: Retries if extraction returns empty.

    Returns:
        Numbered list string of seed prompts. Falls back to empty string
        (which means no seeds in the generation template) if all retries fail.
    """
    config = load_prompt(
        "content_moderation", "seed_generation", prompt_dir=prompt_dir
    )
    messages = build_messages(
        config,
        Category=category,
        category_description=category_description,
        num_seeds=str(num_seeds),
    )

    for attempt in range(max_retries):
        raw = generate_sample(backend, model, messages, rate_limiter)
        examples = extract_numbered_list(raw)

        if examples:
            result = "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(examples))
            logger.info(
                "Generated %d seeds for '%s' (attempt %d)",
                len(examples), category, attempt + 1,
            )
            return result

        logger.warning(
            "No seeds extracted for '%s' (attempt %d/%d)",
            category, attempt + 1, max_retries,
        )

    logger.warning("Could not generate seeds for '%s', returning empty", category)
    return ""


# ---------------------------------------------------------------------------
# Deprecated helper -- predates the two-step description+seed pipeline
# ---------------------------------------------------------------------------

def generate_abstract_seeds(
    backend: LLMBackend,
    model: str,
    category: str,
    raw_seed_text: str,
    num_abstract: int = 10,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
    max_retries: int = 3,
) -> str:
    """[DEPRECATED] Generate abstract seeds from existing raw seeds.

    This was the original metaprompt approach: take hand-written seeds and
    ask the LLM to produce variations with different scenarios. It predates
    the current two-step pipeline (generate_category_description ->
    generate_seeds) which requires no hand-written seeds at all.

    The prompt template it uses (Prompts/Content_Moderation/metaprompt/)
    is a placeholder -- fill it with your own prompt if you want to use
    this pattern. The template fields are: {Category}, {SeedPrompts},
    {num_abstract}.

    Prefer ``generate_seeds()`` for new code -- it generates seeds from
    scratch using only the category name and description, with no prior
    seeds required.

    Args:
        backend: LLM backend.
        model: Model identifier.
        category: Harm category name.
        raw_seed_text: Existing seed prompts as a numbered list string.
        num_abstract: Number of abstract examples to generate.
        rate_limiter: Optional rate limiter.
        prompt_dir: Prompt directory override.
        max_retries: Retries if extraction returns empty.

    Returns:
        Numbered list string of new seed prompts, or ``raw_seed_text``
        unchanged if all retries fail.
    """
    from ..llms.prompts import load_prompt as _load
    config = _load("content_moderation", "metaprompt", prompt_dir=prompt_dir)
    messages = build_messages(
        config,
        Category=category,
        SeedPrompts=raw_seed_text,
        num_abstract=str(num_abstract),
    )

    for attempt in range(max_retries):
        raw = generate_sample(backend, model, messages, rate_limiter)
        examples = extract_numbered_list(raw)
        if examples:
            return "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(examples))
        logger.warning(
            "Abstract seed extraction failed for '%s' (attempt %d/%d)",
            category, attempt + 1, max_retries,
        )

    return raw_seed_text


# Legacy alias kept for backwards compatibility with older runner scripts.
# New code should call generate_abstract_seeds() directly (or better, use
# generate_seeds() which does not require existing seeds).
generate_instruction_prompt = generate_abstract_seeds
