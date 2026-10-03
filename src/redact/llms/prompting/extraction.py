"""Parse individual samples out of multi-sample LLM output.

Supported formats:

- Numbered list: "1. text" / "2) text" / "3: text".
- Structured Q&A: "**Prompt N:** ... **Question:** ... **Answer:** ...".
- Delimited: samples separated by a delimiter line ("---" by default).

Also provides the format instructions that ask a model for these formats
(:func:`get_format_instruction`), sample cleaning (:func:`clean_sample`) and
constitution parsing (:func:`parse_constitution`).
"""

import re
from dataclasses import dataclass

from .prompts import load_prompt

# ---------------------------------------------------------------------------
# Format instructions (loaded from prompts/format_instructions/{style}/)
# ---------------------------------------------------------------------------

#: Output formats the library can request and parse. All of them are valid
#: for :func:`get_format_instruction`.
EXTRACTION_STYLES: frozenset[str] = frozenset(
    {"numbered", "structured_qa", "delimiter"}
)

#: Styles that yield question/answer pairs instead of plain samples, mapped to
#: the extractor to call. :func:`extract_and_clean` rejects them.
_PAIRED_STYLES: dict[str, str] = {"structured_qa": "extract_structured_qa"}


def get_format_instruction(
    style: str = "numbered",
    num_samples: int = 5,
    prompt_dir: str | None = None,
) -> str:
    """Return the format instruction to append to a system prompt.

    Args:
        style: One of ``EXTRACTION_STYLES``.
        num_samples: Number of samples to request.
        prompt_dir: Optional prompt override directory.

    Raises:
        ValueError: Unknown ``style``.
    """
    if style not in EXTRACTION_STYLES:
        raise ValueError(
            f"Unknown format style '{style}'. Choose from: {sorted(EXTRACTION_STYLES)}"
        )
    config = load_prompt("format_instructions", style, prompt_dir=prompt_dir)
    return config["instruction"].format(num_samples=num_samples)


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

_NUMBERED_PATTERN = re.compile(r"^\s*\d+\s*[\.\)\-\:]\s+", re.MULTILINE)


def extract_numbered_list(text: str) -> list[str]:
    """Extract samples from a numbered list.

    A sample starts on a line that begins with a number followed by '.',
    ')', '-' or ':' and whitespace. Following lines without a number belong
    to the same sample. Text before the first numbered line is ignored.

    Args:
        text: Raw LLM output containing a numbered list.

    Returns:
        The samples, with the numbering removed.
    """
    lines = text.strip().split("\n")
    samples: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        match = _NUMBERED_PATTERN.match(line)
        if match:
            if current_lines:
                samples.append("\n".join(current_lines).strip())
            content = _NUMBERED_PATTERN.sub("", line, count=1).strip()
            current_lines = [content] if content else []
        elif current_lines:
            # Continuation line of a multi-line sample.
            current_lines.append(line)

    if current_lines:
        samples.append("\n".join(current_lines).strip())

    return [s for s in samples if s]


# The "**Prompt N:**" header is optional. A pair ends at the next header, the
# next "**Question:**" or the end of the text.
_STRUCTURED_QA_PATTERN = re.compile(
    r"(?:\*\*Prompt\s*\d+:?\*\*\s*)?"
    r"\*\*Question:\*\*\s*(.+?)\s*"
    r"\*\*Answer:\*\*\s*(.+?)"
    r"(?=\*\*Prompt\s*\d+:?\*\*|\*\*Question:\*\*|\Z)",
    re.DOTALL,
)


def extract_structured_qa(text: str) -> list[dict[str, str]]:
    """Extract question/answer pairs from structured Q&A output.

    Each pair is a ``**Question:** ... **Answer:** ...`` block, optionally
    headed by ``**Prompt N:**``.

    Args:
        text: Raw LLM output in the structured format.

    Returns:
        List of ``{"question": ..., "answer": ...}`` dicts.
    """
    matches = _STRUCTURED_QA_PATTERN.findall(text)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in matches]


def extract_delimited(text: str, delimiter: str = "---") -> list[str]:
    """Split text into samples on delimiter lines.

    Every delimited part is returned. No pipeline currently selects this
    style.

    Args:
        text: Raw LLM output with delimiter-separated samples.
        delimiter: The delimiter string. A line must consist of it alone,
            apart from surrounding whitespace.

    Returns:
        The non-empty parts, stripped.
    """
    parts = re.split(
        rf"^\s*{re.escape(delimiter)}\s*$", text, flags=re.MULTILINE
    )
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Output cleaning utilities
# ---------------------------------------------------------------------------

_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")
# Italics only when the asterisks touch the text on one line, so that
# "3 * 4 * 5" is left alone.
_MARKDOWN_ITALIC = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
# Code fences are unwrapped: the fence and its language tag are removed and
# the code is kept.
_MARKDOWN_CODE_BLOCK = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
# An opening fence with no closing one, as left by truncation at max_tokens.
_MARKDOWN_LONE_FENCE = re.compile(r"```[^\n`]*\n?")
_MARKDOWN_INLINE_CODE = re.compile(r"`(.+?)`")
# Anchored at the start of the text (no MULTILINE), so only a leading preamble
# line matches and content lines further down are not affected.
_META_PREAMBLE = re.compile(
    r"^\s*(Note:|Here are|Here is|Below are|Below is|Sure[,!.]|Of course[,!.]|"
    r"Certainly[,!.]|I'll |I will |Let me )",
    re.IGNORECASE,
)


def clean_sample(
    text: str,
    strip_markdown: bool = True,
    strip_meta: bool = True,
) -> str:
    """Clean one extracted sample.

    Args:
        text: Raw sample text.
        strip_markdown: Remove markdown markup and keep the text it wraps.
            Code blocks are unwrapped, not deleted.
        strip_meta: Remove one leading preamble line (e.g. "Sure, here
            are ..."), when more text follows it.
    """
    result = text
    # Remove ChatML tokens that leak through, whole or partial (e.g.
    # "<|im_end|>", "|im_start").
    result = re.sub(r"<?\|im_(start|end)\|?>?", "", result)
    if strip_meta:
        # Only the first line, and only when more text follows it. split(..., 1)
        # yields one or two parts.
        head_tail = result.lstrip("\n").split("\n", 1)
        if len(head_tail) == 2 and _META_PREAMBLE.match(head_tail[0]):  # noqa: PLR2004
            result = head_tail[1]
    if strip_markdown:
        # Order matters: unwrap paired fences, then remove a lone opener, and
        # only then treat single backticks as inline code.
        result = _MARKDOWN_CODE_BLOCK.sub(r"\1", result)
        result = _MARKDOWN_LONE_FENCE.sub("", result)
        result = _MARKDOWN_BOLD.sub(r"\1", result)
        result = _MARKDOWN_ITALIC.sub(r"\1", result)
        result = _MARKDOWN_INLINE_CODE.sub(r"\1", result)
    # Collapse multiple blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_and_clean(
    text: str,
    style: str = "numbered",
    strip_markdown: bool = True,
    strip_meta: bool = True,
    delimiter: str = "---",
) -> list[str]:
    """Extract samples from LLM output and clean each one.

    Args:
        text: Raw LLM output.
        style: ``"numbered"`` or ``"delimiter"``.
        strip_markdown: Passed to :func:`clean_sample`.
        strip_meta: Passed to :func:`clean_sample` for the ``"delimiter"``
            style. Not applied to numbered samples.
        delimiter: Delimiter for the ``"delimiter"`` style.

    Returns:
        The cleaned samples. Samples that are empty after cleaning are
        dropped.

    Raises:
        ValueError: Unknown style, or a paired style such as
            ``"structured_qa"``. For that one, call
            :func:`extract_structured_qa`.
    """
    if style in _PAIRED_STYLES:
        raise ValueError(
            f"'{style}' yields question/answer pairs, not plain samples, so it "
            f"cannot go through extract_and_clean(). Call "
            f"{_PAIRED_STYLES[style]}() directly — it returns both halves, "
            f"which this would have discarded."
        )
    if style not in EXTRACTION_STYLES:
        raise ValueError(
            f"Unknown extraction style '{style}'. "
            f"Choose from: {sorted(EXTRACTION_STYLES)}"
        )
    if style == "numbered":
        raw = extract_numbered_list(text)
        # Text before the first item is already dropped, so a first line that
        # looks like a preamble belongs to the sample.
        strip_meta = False
    else:  # "delimiter"
        raw = extract_delimited(text, delimiter)

    return [
        cleaned
        for s in raw
        if (cleaned := clean_sample(s, strip_markdown, strip_meta))
    ]


# ---------------------------------------------------------------------------
# Constitution parsing (3-layer markdown hierarchy)
# ---------------------------------------------------------------------------

_CONSTITUTION_MAIN_RE = re.compile(r"^## \d+\.\s+(.+)$")
_CONSTITUTION_SUB_RE = re.compile(r"^### \d+\.\d+\s+(.+)$")
_CONSTITUTION_SAMPLE_RE = re.compile(r"^- \((.+)\)$")


@dataclass
class ConstitutionEntry:
    """A single entry from a parsed constitution."""

    category: str
    subcategory: str
    sample: str


def parse_constitution(text: str) -> list[ConstitutionEntry]:
    """Parse constitution markdown into entries.

    Expects a 3-layer hierarchy::

        ## 1. Main Category
        ### 1.1 Subcategory
        - (example sample text)
        - (another sample)

    The constitution prompts (``prompts/constitution/{entry_type}/``) ask
    the model for this format, so keep the two in sync.

    Args:
        text: Raw constitution markdown.

    Returns:
        One :class:`ConstitutionEntry` per sample line.
    """
    entries: list[ConstitutionEntry] = []
    current_main = ""
    current_sub = ""

    for line in text.strip().split("\n"):
        line = line.strip()

        main_match = _CONSTITUTION_MAIN_RE.match(line)
        if main_match:
            current_main = main_match.group(1).strip()
            current_sub = ""
            continue

        sub_match = _CONSTITUTION_SUB_RE.match(line)
        if sub_match:
            current_sub = sub_match.group(1).strip()
            continue

        sample_match = _CONSTITUTION_SAMPLE_RE.match(line)
        if sample_match and current_main:
            entries.append(
                ConstitutionEntry(
                    category=current_main,
                    subcategory=current_sub,
                    sample=sample_match.group(1).strip(),
                )
            )

    return entries
