"""Meta-prompt steps for the content moderation input pipeline.

The full automated pipeline needs only a category name to start:

  Step 1 -- generate_category_description():
    Category name -> LLM -> rich category description
    Template: prompts/input/category_description/

  Step 2 -- generate_seeds():
    Category name + description -> LLM -> seed prompts (numbered list)
    Template: prompts/input/seed_generation/

  Step 3 -- InputPipeline.run_category():
    Category name + description + seeds -> LLM -> actual samples
    Template: prompts/input/generation/standalone/
    Checks each sample via quality checker, loops with feedback.

Fallback options:
  - Skip step 1: use the short description from the taxonomy JSON
  - Skip step 2: use hand-written seeds from content_moderation_seeds.json
  - Both steps can be bypassed for quick runs without extra LLM calls

Usage:
    from redact.content_moderation.metaprompt import (
        generate_category_description,
        generate_seeds,
    )
    from redact.dataset import load_taxonomy, iter_categories

    taxonomy = load_taxonomy("content_moderation_categories")
    for category_name, category_info in iter_categories(taxonomy):

        # Step 1: rich description (or use taxonomy short description as fallback)
        description = generate_category_description(client, category_name)

        # Step 2: seeds (or load from content_moderation_seeds.json as fallback)
        seed_text = generate_seeds(client, category_name, description)

        # Step 3: generate samples
        result = pipeline.run_category(
            category=category_name,
            prompt_config=load_prompt("input", "generation/standalone"),
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

from ..llms.client import ModelClient
from ..llms.extraction import extract_numbered_list
from ..llms.prompts import build_messages, load_prompt
from ..llms.router import generate_sample

logger = logging.getLogger(__name__)


def generate_category_description(
    client: ModelClient,
    category: str,
    prompt_dir: str | Path | None = None,
    max_retries: int = 2,
) -> str:
    """Generate a rich category description from just the category name.

    Step 1 of the automated pipeline. Produces a detailed, research-grade
    description of what belongs in the harm category, covering content types,
    formats, severity range, and distinguishing features.

    Args:
        client: The model to call, bound to its transport.
        category: Harm category name (e.g. "CBRN", "Self-Harm").
        prompt_dir: Prompt directory override.
        max_retries: Retries on empty output.

    Returns:
        Rich description string. Falls back to the category name itself
        if all retries fail.
    """
    config = load_prompt(
        "input", "category_description", prompt_dir=prompt_dir
    )
    messages = build_messages(config, Category=category)

    for attempt in range(max_retries):
        result = generate_sample(client, messages)
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
    client: ModelClient,
    category: str,
    category_description: str,
    num_seeds: int = 10,
    prompt_dir: str | Path | None = None,
    max_retries: int = 3,
) -> str:
    """Generate seed prompts from the category name and description.

    Step 2 of the automated pipeline. Produces a diverse set of short,
    realistic harmful prompts that serve as style/type examples for the
    generation step. Output is a plain numbered list string ready to
    inject as {SeedPrompts} in the generation template.

    Args:
        client: The model to call, bound to its transport.
        category: Harm category name.
        category_description: Rich description from generate_category_description().
        num_seeds: Number of seed prompts to generate.
        prompt_dir: Prompt directory override.
        max_retries: Retries if extraction returns empty.

    Returns:
        Numbered list string of seed prompts. Falls back to empty string
        (which means no seeds in the generation template) if all retries fail.
    """
    config = load_prompt(
        "input", "seed_generation", prompt_dir=prompt_dir
    )
    messages = build_messages(
        config,
        Category=category,
        category_description=category_description,
        num_seeds=str(num_seeds),
    )

    for attempt in range(max_retries):
        raw = generate_sample(client, messages)
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


