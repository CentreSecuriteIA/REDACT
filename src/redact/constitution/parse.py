"""Parse a constitution reply (3-layer markdown hierarchy) into entries."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Headers may carry bold or italic markers ("## **1. Name**"), stripped from
# the name with a trailing colon. One character class between the fixed parts,
# not chained optional groups, keeps a failed match linear. Samples are "-" or
# "*" bullets holding exactly one parenthesised text, which may itself contain
# parentheses one level deep; any other bullet is skipped.
_CONSTITUTION_MAIN_RE = re.compile(r"^##[\s*_]*[0-9]+\.\s+(.+)$")
_CONSTITUTION_SUB_RE = re.compile(r"^###[\s*_]*[0-9]+\.[0-9]+\.?\s+(.+)$")
_CONSTITUTION_SAMPLE_RE = re.compile(
    r"^[-*]\s+\(([^()]*(?:\([^()]*\)[^()]*)*)\)$"
)


def _header_name(pattern: re.Pattern[str], line: str) -> str:
    """The name in a header line, or "" if ``line`` is not that header."""
    match = pattern.match(line)
    return match.group(1).strip(" \t*_:") if match else ""


@dataclass
class ParsedEntry:
    """A single entry from a parsed constitution."""

    category: str
    subcategory: str
    sample: str


def parse_constitution(text: str) -> list[ParsedEntry]:
    """Parse constitution markdown into entries.

    Expects a 3-layer hierarchy::

        ## 1. Main Category
        ### 1.1 Subcategory
        - (example sample text)
        - (another sample)

    The constitution prompts (``prompts/constitution/{entry_type}/``) ask
    the model for this format, so keep the two in sync. Samples under a
    header that does not parse are dropped, up to the next main category.

    Args:
        text: Raw constitution markdown.

    Returns:
        One :class:`ParsedEntry` per sample line.
    """
    entries: list[ParsedEntry] = []
    current_main = ""
    current_sub = ""

    for line in text.replace("\r\n", "\n").strip().split("\n"):
        line = line.strip()

        main_name = _header_name(_CONSTITUTION_MAIN_RE, line)
        if main_name:
            current_main = main_name
            current_sub = ""
            continue

        sub_name = _header_name(_CONSTITUTION_SUB_RE, line)
        if sub_name:
            current_sub = sub_name
            continue

        if line.startswith("#"):
            # A single "#" document title is ignored; any deeper header that
            # did not parse ("##", "###", "####") ends the current category.
            if line.startswith("##"):
                logger.warning(
                    "Unrecognised constitution header %r; dropping the samples under it.",
                    line[:80],
                )
                current_main = ""
                current_sub = ""
            continue

        sample_match = _CONSTITUTION_SAMPLE_RE.match(line)
        sample = sample_match.group(1).strip() if sample_match else ""
        if sample and current_main:
            entries.append(
                ParsedEntry(
                    category=current_main,
                    subcategory=current_sub,
                    sample=sample,
                )
            )
        elif line:
            logger.debug("Skipped constitution line %r", line[:80])

    return entries
