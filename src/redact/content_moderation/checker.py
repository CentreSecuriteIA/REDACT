"""Checker message builders for content moderation sample validation.

Provides build_check_messages callables that InputPipeline requires.
Each builder returns a function with signature:
    (sample: str) -> list[dict]

The checker LLM should respond starting with "Yes"/"Accept" if the sample
is good, or provide feedback explaining what's wrong if it's not.

Prompts are loaded from JSON templates under ``prompts/input/`` (input-side
checkers) and ``prompts/output/`` (output checker).
"""

from collections.abc import Callable

from ..llms.prompts import load_prompt
from ..types import EntryType

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
    entry_type: str | EntryType = "harmful",
    subcategory: str = "",
    prompt_dir: str | None = None,
) -> Callable[[str], list[dict]]:
    """Build a checker that validates sample quality and category fit.

    Loads the unified entry-type-aware template from
    ``prompts/input/quality_check/template.json`` and injects ``category``,
    ``subcategory``, and ``entry_type`` so the evaluator knows what kind of
    sample it is validating (harmful vs benign vs dual-use).

    Args:
        category: The target category (harm domain or benign category).
        criteria: Additional acceptance criteria appended to the system
            prompt (optional).
        entry_type: ``"harmful"`` / ``"dual_use_harmful"`` /
            ``"dual_use_benign"`` / ``"benign"``. Default ``"harmful"`` —
            preserves the prior harmful-only behaviour for callers that
            don't pass this argument.
        subcategory: Constitution subcategory, or ``""`` (defaults to
            ``category`` inside the prompt for standalone CM use).
        prompt_dir: Root directory for prompt JSON files.

    Returns:
        Callable(sample) -> message list for use with check_sample().
    """
    entry_type_value = (
        entry_type.value if isinstance(entry_type, EntryType) else str(entry_type)
    )

    prompt_config = load_prompt(
        "input", "quality_check", prompt_dir=prompt_dir
    )

    system_prompt = prompt_config["system_prompt"].format(
        Category=category,
        entry_type=entry_type_value,
        subcategory=subcategory or category,
    )
    if criteria:
        system_prompt += f"\n6. Additional criteria: {criteria}"

    def _build(sample: str) -> list[dict]:
        template = prompt_config["template"].format(sample=sample)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template},
        ]

    return _build


def build_output_quality_checker(
    category: str,
    entry_type: str | EntryType = "harmful",
    criteria: str = "",
    prompt_dir: str | None = None,
) -> Callable[[str], list[dict]]:
    """Build a checker that validates the model's output to a given input.

    For harmful inputs the checker should accept outputs that actually
    contain the harmful response (not a refusal). For benign inputs the
    checker should accept normal helpful answers.

    The returned callable takes a single string formatted as
    ``"INPUT:\\n<input_prompt>\\n\\nOUTPUT:\\n<output_response>"`` (this is
    what ``batch_check_samples`` already supplies when fed input-output
    pairs concatenated via ``\\n\\n``-delimiter convention).

    Loads ``prompts/output/quality_check/template.json``.
    """
    entry_type_value = (
        entry_type.value if isinstance(entry_type, EntryType) else str(entry_type)
    )

    prompt_config = load_prompt(
        "output", "quality_check", prompt_dir=prompt_dir
    )

    system_prompt = prompt_config["system_prompt"].format(
        Category=category,
        entry_type=entry_type_value,
    )
    if criteria:
        system_prompt += f"\n5. Additional criteria: {criteria}"

    def _build(sample: str) -> list[dict]:
        template = prompt_config["template"].format(sample=sample)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template},
        ]

    return _build


def build_paraphrase_checker(
    prompt_dir: str | None = None,
) -> Callable[[str], list[dict]]:
    """Build a meaning-preservation checker for the paraphrase pass.

    The returned callable takes a single string formatted as
    ``"ORIGINAL:\\n<original>\\n\\nPARAPHRASE:\\n<paraphrase>"`` and returns the
    checker message list. Via ``check_sample`` / ``batch_check_samples`` it
    accepts when the reply starts with yes/ok/accept/pass (check→drop, no retry).

    Deliberately separate from the paraphraser — a weak paraphraser must never
    grade its own output. Loads
    ``prompts/content_moderation/paraphrase_check/template.json``.

    Args:
        prompt_dir: Root directory for prompt JSON files.

    Returns:
        Callable(payload) -> checker message list.
    """
    prompt_config = load_prompt(
        "content_moderation", "paraphrase_check", prompt_dir=prompt_dir
    )
    system_prompt = prompt_config["system_prompt"]

    def _build(sample: str) -> list[dict]:
        template = prompt_config["template"].format(sample=sample)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template},
        ]

    return _build


def paraphrase_check_payload(original: str, paraphrase: str) -> str:
    """Format an (original, paraphrase) pair for :func:`build_paraphrase_checker`."""
    return f"ORIGINAL:\n{original}\n\nPARAPHRASE:\n{paraphrase}"


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
    ``prompts/input/category_check/template.json``.

    Args:
        category: The expected category.
        all_categories: Full list of valid categories. If None, uses
            DEFAULT_CATEGORIES.
        prompt_dir: Root directory for prompt JSON files.

    Returns:
        Callable(sample) -> message list for use with check_sample().
    """
    prompt_config = load_prompt(
        "input", "category_check", prompt_dir=prompt_dir
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
