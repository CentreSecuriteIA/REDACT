"""Prompt loading, reply parsing and prompt-tree scaffolding.

- :mod:`prompts`: load a prompt JSON and render it into chat messages.
- :mod:`extraction`: parse a reply into samples, and supply the format
  instruction that asks the model for that shape.
- :mod:`scaffold`: write an editable copy of the prompt tree.

``prompts`` and ``extraction`` import each other: ``extraction`` loads the
``format_instructions/`` prompts, and ``prompts`` appends a format instruction
for ``format_style=``. The cycle is broken by a function-local import in
``prompts``.
"""

from .extraction import (
    EXTRACTION_STYLES,
    ConstitutionEntry,
    clean_sample,
    extract_and_clean,
    extract_delimited,
    extract_numbered_list,
    extract_structured_qa,
    get_format_instruction,
    parse_constitution,
)
from .prompts import (
    PromptTemplate,
    build_messages,
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
    "extract_delimited",
    "extract_numbered_list",
    "extract_structured_qa",
    "get_format_instruction",
    "load_prompt",
    "parse_constitution",
    "report_prompt_sources",
    "resolve_prompt",
    "scaffold_prompt_tree",
]
