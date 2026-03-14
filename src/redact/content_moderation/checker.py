"""Checker message builders for content moderation sample validation.

Provides build_check_messages callables that InputPipeline requires.
Each builder returns a function with signature:
    (sample: str) -> list[dict]

The checker LLM should respond starting with "Yes"/"Accept" if the sample
is good, or provide feedback explaining what's wrong if it's not.

Prompts are loaded from JSON templates in Prompts/Content_Moderation/.
"""

from typing import Callable

from ..llms.prompts import load_prompt, build_messages

DEFAULT_CATEGORIES: list[str] = [
    "CBRN",
    "Cyber",
    "Harm to Minors",
    "Harmful Manipulation",
    "Hate Speech",
    "Illegal Activities",
    "Information Integrity",
    "Physical Harm",
    "Privacy",
    "Self-Harm",
    "Sexual Content",
    "Benign",
]

def build_quality_checker(
    category: str,
    criteria: str = "",
    prompt_dir: str | None = None,
) -> Callable[[str], list[dict]]:
    """Build a checker that validates sample quality and category fit.

    The returned function creates a message list that asks the checker LLM
    to evaluate whether a generated sample:
    1. Actually belongs to the specified harm category
    2. Is realistic and plausible
    3. Is sufficiently distinct (not generic/templated)
    4. Meets any additional criteria

    Loads the system prompt and template from
    ``Prompts/Content_Moderation/quality_check/template.json``.

    Args:
        category: The target harm category.
        criteria: Additional acceptance criteria (optional).
        prompt_dir: Root directory for prompt JSON files.

    Returns:
        Callable(sample) -> message list for use with check_sample().
    """
    prompt_config = load_prompt(
        "content_moderation", "quality_check", prompt_dir=prompt_dir
    )

    # Inject additional criteria into the system prompt if provided
    system_prompt = prompt_config["system_prompt"].format(Category=category)
    if criteria:
        system_prompt += f"\n5. Additional criteria: {criteria}"

    def _build(sample: str) -> list[dict]:
        template = prompt_config["template"].format(sample=sample)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template},
        ]

    return _build


def build_category_checker(
    category: str,
    all_categories: list[str] | None = None,
    prompt_dir: str | None = None,
) -> Callable[[str], list[dict]]:
    """Build a checker that performs forward-backward category recovery.

    The checker is asked to classify the sample into a harm category.
    If its classification matches the target category, the sample passes.
    This detects mode collapse and off-category drift.

    Reference: CLAUDE.md generation strategy -- "forward-backward category
    recovery" for mode collapse detection.

    Loads the system prompt and template from
    ``Prompts/Content_Moderation/category_check/template.json``.

    Args:
        category: The expected category.
        all_categories: Full list of valid categories. If None, uses
            DEFAULT_CATEGORIES.
        prompt_dir: Root directory for prompt JSON files.

    Returns:
        Callable(sample) -> message list for use with check_sample().
    """
    prompt_config = load_prompt(
        "content_moderation", "category_check", prompt_dir=prompt_dir
    )

    cats = all_categories or DEFAULT_CATEGORIES
    categories_list = "\n".join(f"- {c}" for c in cats)

    system_prompt = prompt_config["system_prompt"].format(
        Category=category, categories_list=categories_list
    )

    def _build(sample: str) -> list[dict]:
        template = prompt_config["template"].format(sample=sample)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template},
        ]

    return _build
