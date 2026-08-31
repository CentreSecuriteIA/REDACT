"""Checker message builders for content moderation sample validation.

Provides build_check_messages callables that InputPipeline requires.
Each builder returns a function with signature:
    (original: str, sample: str) -> list[dict]

matching the two-arg checker contract used by ``check_sample``/
``batch_check_samples`` (``redact.llms.router``). Checkers that only need
the one text to validate (``build_quality_checker``, ``build_category_checker``)
simply ignore ``original``; checkers that genuinely compare two texts
(``build_output_quality_checker``, ``build_paraphrase_checker``) use both as
real, separately-templated values — never string-concatenated by the caller
into one payload.

The checker LLM should respond starting with "Yes"/"Accept" if the sample
is good, or provide feedback explaining what's wrong if it's not.

Prompts are loaded from JSON templates under ``prompts/input/`` (input-side
checkers) and ``prompts/output/`` (output checker + paraphrase checker — a
post-generation transform, grouped with output rather than input).
"""

from collections.abc import Callable

from ..llms.prompts import PromptTemplate, load_prompt
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
) -> Callable[[str, str], list[dict]]:
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
        Callable(original, sample) -> message list for use with
        check_sample()/batch_check_samples(). ``original`` is ignored — this
        checker validates one sample in isolation.
    """
    entry_type_value = (
        entry_type.value if isinstance(entry_type, EntryType) else str(entry_type)
    )

    prompt_config = load_prompt(
        "input", "quality_check", prompt_dir=prompt_dir
    )

    tmpl = PromptTemplate(
        prompt_config,
        Category=category,
        entry_type=entry_type_value,
        subcategory=subcategory or category,
    )
    if criteria:
        tmpl.system_prompt += f"\n6. Additional criteria: {criteria}"

    def _build(original: str, sample: str) -> list[dict]:
        return tmpl(sample=sample)

    return _build


def build_output_quality_checker(
    category: str,
    entry_type: str | EntryType = "harmful",
    criteria: str = "",
    prompt_dir: str | None = None,
) -> Callable[[str, str], list[dict]]:
    """Build a checker that validates the model's output to a given input.

    For harmful inputs the checker should accept outputs that actually
    contain the harmful response (not a refusal). For benign inputs the
    checker should accept normal helpful answers.

    The returned callable takes ``(input_prompt, output_response)`` as two
    separate arguments — matching check_sample()'s ``(original, sample)``
    contract — rather than a caller-concatenated payload string.

    Loads ``prompts/output/quality_check/template.json``.
    """
    entry_type_value = (
        entry_type.value if isinstance(entry_type, EntryType) else str(entry_type)
    )

    prompt_config = load_prompt(
        "output", "quality_check", prompt_dir=prompt_dir
    )

    tmpl = PromptTemplate(prompt_config, Category=category, entry_type=entry_type_value)
    if criteria:
        tmpl.system_prompt += f"\n5. Additional criteria: {criteria}"

    def _build(input_prompt: str, output_response: str) -> list[dict]:
        return tmpl(input_prompt=input_prompt, output_response=output_response)

    return _build


def build_paraphrase_checker(
    prompt_dir: str | None = None,
) -> Callable[[str, str], list[dict]]:
    """Build a meaning-preservation checker for the paraphrase pass.

    The returned callable takes ``(original, paraphrase)`` as two separate
    arguments — matching check_sample()'s ``(original, sample)`` contract —
    and returns the checker message list. Via ``check_sample`` /
    ``batch_check_samples`` it accepts when the reply starts with
    yes/ok/accept/pass (check→drop, no retry).

    Deliberately separate from the paraphraser — a weak paraphraser must never
    grade its own output. Loads
    ``prompts/output/paraphrase_check/template.json``.

    Args:
        prompt_dir: Root directory for prompt JSON files.

    Returns:
        Callable(original, paraphrase) -> checker message list.
    """
    prompt_config = load_prompt(
        "output", "paraphrase_check", prompt_dir=prompt_dir
    )
    tmpl = PromptTemplate(prompt_config)

    def _build(original: str, paraphrase: str) -> list[dict]:
        return tmpl(original=original, paraphrase=paraphrase)

    return _build


def build_category_checker(
    category: str,
    all_categories: list[str] | None = None,
    prompt_dir: str | None = None,
) -> Callable[[str, str], list[dict]]:
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
        Callable(original, sample) -> message list for use with
        check_sample()/batch_check_samples(). ``original`` is ignored — this
        checker validates one sample in isolation.
    """
    prompt_config = load_prompt(
        "input", "category_check", prompt_dir=prompt_dir
    )

    cats = all_categories or DEFAULT_CATEGORIES
    categories_list = "\n".join(f"- {c}" for c in cats)

    tmpl = PromptTemplate(prompt_config, Category=category, categories_list=categories_list)

    def _build(original: str, sample: str) -> list[dict]:
        return tmpl(sample=sample)

    return _build
