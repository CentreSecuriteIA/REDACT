"""Regex-based extraction of individual samples from multi-sample LLM output.

Three extraction strategies:

1. Numbered list:     "1. sample text" / "2) sample text" / "3: sample text"
   Reference: input dataset notebook regex r'^\\s*\\d+\\s*[\\.\\)\\-\\:]\\s+'

2. Structured Q&A:   "**Prompt N:** ... **Question:** ... **Answer:** ..."
   Reference: jailbreak manipulation.py extract_prompt_answer_pairs()

3. Delimiter-based:   Samples separated by known delimiters (---, ===, blank lines)

Also provides:
- Format instructions (loaded from prompts/format_instructions/{style}/,
  appended to system prompts so the LLM knows how to structure multi-sample
  output for parsing — see get_format_instruction()/EXTRACTION_STYLES)
- Output cleaning utilities (strip markdown, meta-commentary)
- Constitution parsing (3-layer hierarchy from markdown)
  Reference: constitutional_classifier constitution_gen.ipynb
"""

import re
from dataclasses import dataclass

from .prompts import load_prompt

# ---------------------------------------------------------------------------
# Format instructions — loaded from prompts/format_instructions/{style}/,
# not hardcoded here, per "prompts are external" (see CLAUDE.md). The set of
# valid styles is exactly this dict's keys, shared with extract_and_clean()'s
# own style dispatch below — the two lived as independently-defined string
# sets before, a coincidental duplication rather than a real coupling
# (both already read the *same* caller-supplied style value, just via two
# separate dicts/if-chains that had no way to notice if they drifted apart).
# EXTRACTION_STYLES is that one shared source of truth.
# ---------------------------------------------------------------------------

EXTRACTION_STYLES: frozenset[str] = frozenset({"numbered", "structured_qa", "delimiter"})


def get_format_instruction(
    style: str = "numbered",
    num_samples: int = 5,
    prompt_dir: str | None = None,
) -> str:
    """Return a format instruction string to append to a system prompt.

    Args:
        style: One of "numbered", "structured_qa", "delimiter"
            (``EXTRACTION_STYLES``).
        num_samples: Number of samples to request.
        prompt_dir: Root directory for prompt JSON files.

    Returns:
        Rendered instruction string ready to append to a system prompt.
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


# The "**Prompt N:**" header is a pure index — the prompts that produce this
# format (format_instructions/structured_qa, jailbreak/benign_generation) put
# nothing after it, the question is on the next line. So it is matched
# *optionally*: requiring a line that carries no information is a failure mode
# for nothing, and a contentless line is exactly what a model drifts on. The
# terminator therefore also accepts the next "**Question:**", so pairs still
# separate correctly when the headers are absent.
_STRUCTURED_QA_PATTERN = re.compile(
    r"(?:\*\*Prompt\s*\d+:?\*\*\s*)?"
    r"\*\*Question:\*\*\s*(.+?)\s*"
    r"\*\*Answer:\*\*\s*(.+?)"
    r"(?=\*\*Prompt\s*\d+:?\*\*|\*\*Question:\*\*|\Z)",
    re.DOTALL,
)


def extract_structured_qa(text: str) -> list[dict[str, str]]:
    """Extract question-answer pairs from structured **Prompt N:** format.

    Reference: jailbreak manipulation.py extract_prompt_answer_pairs().

    Args:
        text: Raw LLM output in structured format.

    Returns:
        List of {"question": ..., "answer": ...} dicts — "question" (not
        "prompt") to match what the format instruction itself labels this
        field (**Question:**), even though the surrounding block is headed
        **Prompt N:**.
    """
    matches = _STRUCTURED_QA_PATTERN.findall(text)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in matches]


def extract_delimited(text: str, delimiter: str = "---") -> list[str]:
    """Extract samples separated by a delimiter line.

    No pipeline currently selects this style — every call site passes
    ``"numbered"`` — but it is kept deliberately. Its original purpose was
    **in-chat chain-of-thought**: let the model reason freely, then emit a
    delimiter and give the final answer after it, so the reasoning can be
    dropped and only the answer parsed out. That makes it the right style for
    any prompt that wants thinking-then-answer from a model without native
    reasoning output, so it stays available rather than being pruned as unused.

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
# Emphasis only when the asterisks actually hug the text, per CommonMark: no
# space after the opener, none before the closer, no line break between. A
# bare "3 * 4 * 5" is arithmetic, not italics, and the permissive r"\*(.+?)\*"
# silently turned it into "3  4  5". Leaving a stray asterisk in a sample is
# far cheaper than deleting the characters between two of them.
_MARKDOWN_ITALIC = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
# Fences are *unwrapped*, not deleted — every other rule here keeps the text
# and drops the markup, and this one used to drop the code with it. Content
# moderation generates samples that legitimately contain scripts, so deleting
# the block deleted the sample's whole point. The optional language tag
# ("```python") goes with the fence.
_MARKDOWN_CODE_BLOCK = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
# An opener with no closer — what truncation at max_tokens looks like. Without
# this the leftover backticks fall through to the inline-code rule, which eats
# two of them and mangles the line ("```python\nimport os" -> "`python...").
_MARKDOWN_LONE_FENCE = re.compile(r"```[^\n`]*\n?")
_MARKDOWN_INLINE_CODE = re.compile(r"`(.+?)`")
# Matches a *leading* preamble line only (anchored, no MULTILINE) so we strip an
# introductory "Sure, here are ...:" line without touching content lines that
# happen to start with the same words further down a sample.
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
    """Clean a single extracted sample.

    Args:
        text: Raw sample text.
        strip_markdown: Remove markdown *markup*, keeping the text it wraps —
            including code, which is unwrapped from its fences rather than
            deleted with them.
        strip_meta: Remove lines that look like LLM meta-commentary.

    Returns:
        Cleaned text.
    """
    result = text
    # Strip ChatML token fragments that may leak through (belt-and-suspenders for vLLM).
    # Regex catches full tokens and partial fragments: <|im_end|, |>, <|im_start, etc.
    result = re.sub(r"<?\|im_(start|end)\|?>?", "", result)
    if strip_meta:
        # Drop only a leading preamble line, and only when real content follows.
        # Avoids gutting content lines that legitimately start with these words.
        head_tail = result.lstrip("\n").split("\n", 1)
        if len(head_tail) == 2 and _META_PREAMBLE.match(head_tail[0]):  # noqa: PLR2004 — split(..., 1) is always length 1 or 2, not a tunable value
            result = head_tail[1]
    if strip_markdown:
        # Order matters: unwrap balanced fences first, then clear any unpaired
        # opener, and only then treat single backticks as inline code.
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

    Convenience function combining extraction + cleaning in one call.

    Args:
        text: Raw LLM output.
        style: Extraction style — one of ``EXTRACTION_STYLES``, the same set
            ``get_format_instruction()`` accepts (they're meant to be used
            together: the format instruction told the model how to
            structure output, this parses that structure back out).
        strip_markdown: Clean markdown formatting from each sample.
        strip_meta: Remove meta-commentary lines from each sample.
        delimiter: Delimiter for "delimiter" style.

    Returns:
        List of cleaned sample strings.
        For "structured_qa" style, returns list of cleaned question strings
        (answers are discarded — use extract_structured_qa() directly if
        you need both).
    """
    if style not in EXTRACTION_STYLES:
        raise ValueError(
            f"Unknown extraction style '{style}'. Choose from: {sorted(EXTRACTION_STYLES)}"
        )
    if style == "numbered":
        raw = extract_numbered_list(text)
    elif style == "structured_qa":
        pairs = extract_structured_qa(text)
        raw = [p["question"] for p in pairs]
    else:  # "delimiter"
        raw = extract_delimited(text, delimiter)

    return [
        cleaned
        for s in raw
        if (cleaned := clean_sample(s, strip_markdown, strip_meta))
    ]


# ---------------------------------------------------------------------------
# Constitution parsing (3-layer markdown hierarchy)
# Reference: constitutional_classifier constitution_gen.ipynb
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
    """Parse a constitution markdown document into structured entries.

    Expects a 3-layer hierarchy::

        ## 1. Main Category
        ### 1.1 Subcategory
        - (example sample text)
        - (another sample)

    This is the counterpart to the constitution generation prompts
    (``prompts/constitution/generation/{entry_type}/template.json``) — each
    one's ``system_prompt`` explicitly instructs the model to produce exactly
    this markdown shape, so the two must be kept in sync if either changes.

    Args:
        text: Raw constitution markdown text.

    Returns:
        List of ConstitutionEntry with category, subcategory, and sample.
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
