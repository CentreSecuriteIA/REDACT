"""A category's description and seed prompts, generated for the standalone path.

Each is generated again until a reply parses, and falls back when none does:
the description to the category name, the seeds to ``""``.
"""

import logging
from pathlib import Path

from ..llm_pipeline import LLMRequest, drive_sync, extracted
from ..llms.client import ModelClient
from ..llms.prompting import build_messages, extract_numbered_list, load_prompt
from ..llms.router import generate_sample

logger = logging.getLogger(__name__)


def _call(request: LLMRequest) -> str:
    """Send one request to its client and return the reply."""
    return generate_sample(request.client, request.messages,
                           max_tokens=request.max_tokens)


def generate_category_description(
    client: ModelClient,
    category: str,
    prompt_dir: str | Path | None = None,
    max_retries: int = 2,
) -> str:
    """A rich description of the harm ``category``, generated from its name.

    An empty reply is generated again, up to ``max_retries`` generations in
    all; the category name is the fallback.
    """
    config = load_prompt("input", "category_description", prompt_dir=prompt_dir)
    messages = build_messages(config, Category=category)

    def description(reply: str) -> list[str]:
        text = reply.strip()
        if not text:
            logger.warning("Empty description for '%s'", category)
        return [text] if text else []

    found, attempts = drive_sync(
        extracted(client, lambda: messages, description, max_attempts=max_retries),
        _call,
    )
    if not found:
        logger.warning("Falling back to category name as description for '%s'",
                       category)
        return category
    logger.info("Category description for '%s': %d chars (attempt %d)",
                category, len(found[0]), attempts)
    return found[0]


def generate_seeds(
    client: ModelClient,
    category: str,
    category_description: str,
    num_seeds: int = 10,
    prompt_dir: str | Path | None = None,
    max_retries: int = 3,
) -> str:
    """``num_seeds`` seed prompts for ``category``, as a numbered list.

    The list fills ``{SeedPrompts}`` in the generation template. A reply with
    any other number of seeds is generated again, up to ``max_retries``
    generations in all; ``""`` (no seeds) is the fallback.
    """
    config = load_prompt("input", "seed_generation", prompt_dir=prompt_dir)
    messages = build_messages(
        config,
        Category=category,
        category_description=category_description,
        num_seeds=str(num_seeds),
    )

    def seeds(reply: str) -> list[str]:
        examples = extract_numbered_list(reply, strict=True)
        if len(examples) != num_seeds:
            logger.warning("Expected %d seeds for '%s', extracted %d",
                           num_seeds, category, len(examples))
            return []
        return examples

    found, attempts = drive_sync(
        extracted(client, lambda: messages, seeds, max_attempts=max_retries), _call,
    )
    if not found:
        logger.warning("Could not generate seeds for '%s', returning empty", category)
        return ""
    logger.info("Generated %d seeds for '%s' (attempt %d)",
                len(found), category, attempts)
    return "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(found))
