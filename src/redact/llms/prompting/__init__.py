"""Prompts in, samples out — the round trip between the library and a model.

Three modules, grouped because they are one concern seen from both ends:

- :mod:`prompts` — load a prompt JSON and render it into chat messages.
- :mod:`extraction` — parse the reply back into samples, and supply the
  format instruction that told the model how to shape it.
- :mod:`scaffold` — write an editable copy of the prompt tree (setup tooling;
  the only code here that writes rather than reads).

``prompts`` and ``extraction`` import each other — ``extraction`` needs
``load_prompt`` for the ``format_instructions/`` snippets, and ``prompts``
needs ``get_format_instruction`` for ``format_style=``. That mutual dependency
is why they belong in one package: ``format_style`` is the handshake between
them, naming both the instruction sent and the parser used, so the two cannot
drift apart. The cycle is broken by a function-local import in ``prompts`` (a
package doesn't fix circularity between its own modules — it just makes the
coupling visible and contained).

Named ``prompting`` rather than ``prompts`` to stay distinct from
``redact/prompts/``, the JSON data tree this code reads.
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
