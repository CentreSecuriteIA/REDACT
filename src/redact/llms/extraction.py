"""Regex-based extraction of individual samples from multi-sample LLM output.

Three extraction strategies:

1. Numbered list:     "1. sample text" / "2) sample text" / "3: sample text"
   Reference: input dataset notebook regex r'^\\s*\\d+\\s*[\\.\\)\\-\\:]\\s+'

2. Structured Q&A:   "**Prompt N:** ... **Question:** ... **Answer:** ..."
   Reference: jailbreak manipulation.py extract_prompt_answer_pairs()

3. Delimiter-based:   Samples separated by known delimiters (---, ===, blank lines)

Also provides:
- Format instruction constants (appended to system prompts so the LLM
  knows how to structure multi-sample output for parsing)
- Output cleaning utilities (strip markdown, meta-commentary)
"""

import re


# ---------------------------------------------------------------------------
# Format instruction constants
# ---------------------------------------------------------------------------

NUMBERED_LIST_FORMAT_INSTRUCTION: str = (
    "\n\nOUTPUT FORMAT INSTRUCTIONS:\n"
    "Generate exactly {num_samples} samples.\n"
    "Format your response as a numbered list. Each sample must be on its "
    "own line, starting with the number followed by a period and a space.\n"
    "Example format:\n"
    "1. [first sample]\n"
    "2. [second sample]\n"
    "...\n"
    "{num_samples}. [last sample]\n\n"
    "Do NOT include any preamble, explanation, or commentary. "
    "Output ONLY the numbered list."
)

STRUCTURED_QA_FORMAT_INSTRUCTION: str = (
    "\n\nOUTPUT FORMAT INSTRUCTIONS:\n"
    "Generate exactly {num_samples} prompt-answer pairs.\n"
    "Use exactly this format for each:\n\n"
    "**Prompt 1:**\n"
    "**Question:** <question here>\n"
    "**Answer:** <answer here>\n\n"
    "**Prompt 2:**\n"
    "**Question:** <question here>\n"
    "**Answer:** <answer here>\n\n"
    "Continue this pattern for all {num_samples} pairs. "
    "Do NOT include any preamble or commentary."
)

DELIMITER_FORMAT_INSTRUCTION: str = (
    "\n\nOUTPUT FORMAT INSTRUCTIONS:\n"
    "Generate exactly {num_samples} samples.\n"
    "Separate each sample with a line containing only '---'.\n"
    "Do NOT include any preamble, explanation, or commentary.\n"
    "Output ONLY the samples separated by '---'."
)


def get_format_instruction(
    style: str = "numbered",
    num_samples: int = 5,
) -> str:
    """Return a format instruction string to append to a system prompt.

    Args:
        style: One of "numbered", "structured_qa", "delimiter".
        num_samples: Number of samples to request.

    Returns:
        Rendered instruction string ready to append to a system prompt.
    """
    templates = {
        "numbered": NUMBERED_LIST_FORMAT_INSTRUCTION,
        "structured_qa": STRUCTURED_QA_FORMAT_INSTRUCTION,
        "delimiter": DELIMITER_FORMAT_INSTRUCTION,
    }
    if style not in templates:
        raise ValueError(
            f"Unknown format style '{style}'. Choose from: {list(templates)}"
        )
    return templates[style].format(num_samples=num_samples)


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

_NUMBERED_PATTERN = re.compile(r"^\s*\d+\s*[\.\)\-\:]\s+", re.MULTILINE)


def extract_numbered_list(text: str) -> list[str]:
    """Extract samples from a numbered list.

    Matches lines starting with a number followed by '.', ')', '-', or ':'
    and a space. Handles multi-line samples (continuation lines without a
    number prefix are joined to the previous sample).

    Args:
        text: Raw LLM output containing a numbered list.

    Returns:
        List of extracted sample strings, stripped of numbering.
    """
    lines = text.strip().split("\n")
    samples: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        match = _NUMBERED_PATTERN.match(line)
        if match:
            # Save previous accumulated sample
            if current_lines:
                samples.append("\n".join(current_lines).strip())
            # Start new sample: strip the number prefix
            content = _NUMBERED_PATTERN.sub("", line, count=1).strip()
            current_lines = [content] if content else []
        elif current_lines:
            # Continuation line of a multi-line sample
            current_lines.append(line)

    # Don't forget the last sample
    if current_lines:
        samples.append("\n".join(current_lines).strip())

    return [s for s in samples if s]


_STRUCTURED_QA_PATTERN = re.compile(
    r"\*\*Prompt \d+:\*\*\s+"
    r"\*\*Question:\*\*\s+(.+?)\s+"
    r"\*\*Answer:\*\*\s+(.+?)"
    r"(?=\*\*Prompt \d+:\*\*|\Z)",
    re.DOTALL,
)


def extract_structured_qa(text: str) -> list[dict[str, str]]:
    """Extract prompt-answer pairs from structured **Prompt N:** format.

    Reference: jailbreak manipulation.py extract_prompt_answer_pairs().

    Args:
        text: Raw LLM output in structured format.

    Returns:
        List of {"prompt": ..., "answer": ...} dicts.
    """
    matches = _STRUCTURED_QA_PATTERN.findall(text)
    return [{"prompt": q.strip(), "answer": a.strip()} for q, a in matches]


def extract_delimited(text: str, delimiter: str = "---") -> list[str]:
    """Extract samples separated by a delimiter line.

    Args:
        text: Raw LLM output with delimiter-separated samples.
        delimiter: The delimiter string (entire line must match).

    Returns:
        List of extracted sample strings.
    """
    parts = re.split(
        rf"^\s*{re.escape(delimiter)}\s*$", text, flags=re.MULTILINE
    )
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Output cleaning utilities
# ---------------------------------------------------------------------------

_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_ITALIC = re.compile(r"\*(.+?)\*")
_MARKDOWN_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_MARKDOWN_INLINE_CODE = re.compile(r"`(.+?)`")
_META_COMMENTARY = re.compile(
    r"^\s*(Note:|Here are|Below are|Sure|Of course|I'll|Let me|Certainly).*$",
    re.MULTILINE | re.IGNORECASE,
)


def clean_sample(
    text: str,
    strip_markdown: bool = True,
    strip_meta: bool = True,
) -> str:
    """Clean a single extracted sample.

    Args:
        text: Raw sample text.
        strip_markdown: Remove bold, italic, code formatting.
        strip_meta: Remove lines that look like LLM meta-commentary.

    Returns:
        Cleaned text.
    """
    result = text
    if strip_meta:
        result = _META_COMMENTARY.sub("", result)
    if strip_markdown:
        result = _MARKDOWN_CODE_BLOCK.sub("", result)
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

    Convenience function combining extraction + cleaning in one call.

    Args:
        text: Raw LLM output.
        style: Extraction style ("numbered", "structured_qa", "delimiter").
        strip_markdown: Clean markdown formatting from each sample.
        strip_meta: Remove meta-commentary lines from each sample.
        delimiter: Delimiter for "delimiter" style.

    Returns:
        List of cleaned sample strings.
        For "structured_qa" style, returns list of cleaned prompt strings
        (answers are discarded — use extract_structured_qa() directly if
        you need both).
    """
    if style == "numbered":
        raw = extract_numbered_list(text)
    elif style == "structured_qa":
        pairs = extract_structured_qa(text)
        raw = [p["prompt"] for p in pairs]
    elif style == "delimiter":
        raw = extract_delimited(text, delimiter)
    else:
        raise ValueError(f"Unknown extraction style '{style}'")

    return [
        cleaned
        for s in raw
        if (cleaned := clean_sample(s, strip_markdown, strip_meta))
    ]
