"""Prompt loading, reply parsing and prompt-tree scaffolding.

- :mod:`prompts`: load a prompt JSON and render it into chat messages, and
  supply the format instruction that asks the model for a reply shape.
- :mod:`extraction`: parse a reply into samples.
- :mod:`scaffold`: write an editable copy of the prompt tree.

``prompts`` imports ``extraction`` for the style names. ``extraction``
imports nothing from this package.
"""

from .extraction import (
    EXTRACTION_STYLES,
    ConstitutionEntry,
    clean_sample,
    extract_and_clean,
    extract_numbered_list,
    extract_structured_qa,
    parse_constitution,
)
from .prompts import (
    PromptTemplate,
    build_messages,
    get_format_instruction,
    load_prompt,
    report_prompt_sources,
    resolve_prompt,
)
from .scaffold import scaffold_prompt_tree

__all__ = [
    "EXTRACTION_STYLES",
    "ConstitutionEntry",
    "PromptTemplate",
    "build_messages",
    "clean_sample",
    "extract_and_clean",
    "extract_numbered_list",
    "extract_structured_qa",
    "get_format_instruction",
    "load_prompt",
    "parse_constitution",
    "report_prompt_sources",
    "resolve_prompt",
    "scaffold_prompt_tree",
]
