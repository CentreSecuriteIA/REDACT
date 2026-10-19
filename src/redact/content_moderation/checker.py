"""Which prompt each check uses: the checker builders.

Each builder returns ``(original, sample) -> messages``, the contract of
``redact.llm_pipeline.check``. A check of one text ignores ``original``.
"""

from collections.abc import Callable

from ..llms.prompting import PromptTemplate
from ..types import EntryType
from .modes import MODES

Checker = Callable[[str, str], list[dict]]


def _pair(
    template: PromptTemplate, *, sample: str, original: str | None = None
) -> Checker:
    """Render the checked text as ``sample``, the original as ``original`` if named."""
    def build(first: str, second: str) -> list[dict]:
        values = {sample: second}
        if original is not None:
            values[original] = first
        return template(**values)
    return build


def _value(entry_type: str | EntryType) -> str:
    return entry_type.value if isinstance(entry_type, EntryType) else str(entry_type)


def build_quality_checker(
    category: str,
    entry_type: str | EntryType = "harmful",
    subcategory: str = "",
    prompt_dir: str | None = None,
) -> Checker:
    """Check one input sample against its category, entry type and subcategory.

    ``subcategory`` defaults to ``category``.
    """
    template = PromptTemplate.load(
        "input", "quality_check", prompt_dir=prompt_dir,
        Category=category, entry_type=_value(entry_type),
        subcategory=subcategory or category,
    )
    return _pair(template, sample="sample")


def build_output_quality_checker(
    category: str,
    mode: str = "harmful",
    prompt_dir: str | None = None,
) -> Checker:
    """Check a model's response (``sample``) to an input (``original``)
    against its answer ``mode``, with the checker ``output/quality_check/<mode>``.

    A harmful-mode response must carry what was asked for; a helpful-mode
    one must be a normal, helpful answer to the request as it was asked. In
    both a refusal is rejected. The checker is shown the category, not the
    input's entry type. ``mode`` took the place of ``entry_type``.

    Raises:
        ValueError: ``mode`` is not one of the answer modes.
    """
    if mode not in MODES:
        raise ValueError(
            f"The output checker's mode must be one of {list(MODES)}, got {mode!r}.")
    template = PromptTemplate.load(
        "output", f"quality_check/{mode}", prompt_dir=prompt_dir, Category=category,
    )
    return _pair(template, original="input_prompt", sample="output_response")


def build_paraphrase_checker(prompt_dir: str | None = None) -> Checker:
    """Check that a paraphrase (``sample``) keeps the original's meaning.

    Run it on a model other than the paraphraser.
    """
    template = PromptTemplate.load("paraphrase", "check", prompt_dir=prompt_dir)
    return _pair(template, original="original", sample="paraphrase")
