"""Parse individual samples out of multi-sample LLM output.

Supported formats:

- Numbered list: "1. text" / "2) text" / "3: text".
- Structured Q&A: "**Prompt N:** ... **Question:** ... **Answer:** ...".
- Delimited: samples separated by a delimiter line ("---" by default).

Also provides the format instructions that ask a model for these formats
(:func:`get_format_instruction`), sample cleaning (:func:`clean_sample`) and
constitution parsing (:func:`parse_constitution`).
"""

#TODO(driver script): handle rejected replies where the steps are driven.
# A reply rejected by expected_count/strict comes back as an empty list, which
# the callers still treat as "the model returned nothing". To move into the
# driver, together with the retry helpers:
#   - Retry a rejected reply a bounded number of times before giving up.
#   - Constitution inputs (content_moderation/generation.py): a rejected entry
#     is acked as "no_samples" and never retried on resume. Leave it un-acked
#     or record a separate status that resume does not skip.
#   - Standalone inputs (standalone_generation.py): one rejected reply stops
#     the whole category. Stop only on an empty reply or repeated failures.
#   - Benign Q&A (jailbreak/manipulation/benign.py): a rejected half gets no
#     rows and the cache counts as complete; an all-rejected run writes an
#     empty CSV that every later load fails on.
#   - Seeds (metaprompt.py): after 3 failed attempts the category is skipped
#     with only an INFO log.
#   - Count rejections and report them: a run that lost entries still logs
#     "100% accepted", and skipped_entries mixes empty, rejected and duplicate.
#   - Validate the count is an int >= 1: a string count rejects every reply.
#   - prompts/format_instructions/numbered: the example shows two items even
#     when one sample is requested.
#   - Add pipeline-level tests for a wrong-count reply at each call site.
#TODO: Future extension: end token. An end marker in the format instructions
# would let strict mode reject trailing commentary and replies cut off at
# max_tokens; neither can be told apart from sample text today.

import logging
import re
from dataclasses import dataclass

logger =logging.getLogger(__name__)

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
        ValueError: Unknown ``style``, or the instruction file has no
            ``instruction`` text or cannot be rendered.
    """
    from .prompts import _read_prompt, _render, _where, resolve_prompt

    if style not in EXTRACTION_STYLES:
        raise ValueError(
            f"Unknown format style '{style}'. Choose from: {sorted(EXTRACTION_STYLES)}"
        )
    path, is_override = resolve_prompt("format_instructions", style, prompt_dir)
    what = _where("instruction", f"format_instructions/{style}", path, is_override)
    config = _read_prompt(path, is_override)
    if not isinstance(config.get("instruction"), str):
        raise ValueError(f'Prompt {what} is missing: the file needs an "instruction".')
    return _render(config["instruction"], {"num_samples": num_samples}, what)


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

# A leaked chat token, whole or partial ("<|im_end|>", "|im_start"), with the
# role name that follows an opening one. It means the model ran past its turn.
_CHATML_TOKEN = re.compile(
    r"<?\|im_(?:start\|?>?(?:(?:system|user|assistant)\b)?|end\|?>?)"
)

# Groups: indent, opening "**", number, separator, closing "**". The bold
# markers cover "**1.** text" and "**1. Title** text". The number is bounded
# ASCII so int() cannot fail on it.
_NUMBERED_PATTERN = re.compile(
    r"^(\s*)(\*\*)?([0-9]{1,4})\s*([\.\)\-\:])(\*\*)?\s+"
)


def extract_numbered_list(text: str, strict: bool = False) -> list[str]:
    """Extract samples from a numbered list.

    A sample starts on a line beginning with a number and '.', ')', '-' or
    ':'. After the first, only the next number in sequence, at the same
    indent, starts a new sample; any other line continues the current one.
    Text before the first item is ignored.

    Args:
        text: Raw LLM output containing a numbered list.
        strict: Reject the reply unless it is one clean list: starting at 1,
            in sequence, in one numbering style, with no empty item and no
            leaked chat token.

    Returns:
        The non-empty samples, with the numbering removed. Empty if rejected.
    """
    text = text.replace("\r\n", "\n")
    if strict and _CHATML_TOKEN.search(text):
        logger.warning("Rejected numbered reply: leaked chat token.")
        return []

    # Not stripped first: the leading indent of the first item is needed below.
    lines = text.split("\n")
    samples: list[str] = []
    current_lines: list[str] = []
    # Number the next sample must carry; None until the first item is seen.
    # A sample's own text can hold numbered lines, so position in the
    # sequence is what tells a new sample from content.
    expected: int | None = None
    # Indent of the first item, the reference for all later ones. A numbered
    # line indented deeper is a list nested inside the current sample.
    base_indent = 0
    first_style: tuple[str, bool, bool] | None = None

    for line in lines:
        match = _NUMBERED_PATTERN.match(line)
        starts_item = False
        if match:
            indent, bold_open, digits, separator, bold_close = match.groups()
            number = int(digits)
            style = (separator, bool(bold_open), bool(bold_close))
            top_level = expected is None or len(indent) <= base_indent
            # The first numbered line always starts a sample: nothing before
            # it can be one. After that only the expected number at the top
            # indent does, since a year, a quantity or a nested step is far
            # likelier than the model renumbering its list.
            starts_item = expected is None or (number == expected and top_level)
            # Strict: every top-level number must be the next one, in the
            # first item's separator and bold style, so a wrapped line like
            # "2 - 5 degrees" is not taken for item 2. A mismatch rejects the
            # whole reply instead of being repaired: the boundary would be a
            # guess, and a wrongly split sample costs more than a lost reply.
            if strict and top_level:
                wanted = 1 if expected is None else expected
                if number != wanted:
                    logger.warning(
                        "Rejected numbered reply: expected item %d, found %d.",
                        wanted, number,
                    )
                    return []
                if first_style is not None and style != first_style:
                    logger.warning(
                        "Rejected numbered reply: item %d changes numbering style.",
                        number,
                    )
                    return []

        if starts_item:
            if expected is None:
                base_indent = len(indent)
                first_style = style
            else:
                samples.append("\n".join(current_lines).strip())
            expected = number + 1
            content = line[match.end():].strip()
            if bold_open and not bold_close:
                # "**1. Title** text": give the title its opening marker back.
                content = "**" + content
            current_lines = [content]
        elif expected is not None:
            # Any other line inside the list is the current sample's text,
            # kept whole so its own indentation survives.
            current_lines.append(line)

    if expected is not None:
        samples.append("\n".join(current_lines).strip())

    # Empty items are kept until here so strict mode sees them; dropping one
    # earlier would let an extra item fill the requested count.
    if strict and not all(samples):
        logger.warning("Rejected numbered reply: empty item.")
        return []
    return [s for s in samples if s]


# The "**Prompt N:**" header is optional and a label's colon may sit inside or
# outside the bold. Labels count only at the start of a line, so label-like
# text inside a question or answer does not split it.
_QA_PROMPT = r"\*\*Prompt\s*\d+:?\*\*:?"
_QA_QUESTION = r"\*\*Question:?\*\*:?"
_QA_ANSWER = r"\*\*Answer:?\*\*:?"
_QA_LABEL_LINE = rf"^[ \t]*(?:{_QA_PROMPT}|{_QA_QUESTION}|{_QA_ANSWER})"
_QA_BODY = rf"(?:(?!{_QA_LABEL_LINE}).)"
_STRUCTURED_QA_PATTERN = re.compile(
    rf"^[ \t]*(?:{_QA_PROMPT}\s*)?{_QA_QUESTION}[ \t]*({_QA_BODY}+?)"
    rf"^[ \t]*{_QA_ANSWER}[ \t]*({_QA_BODY}+)",
    re.DOTALL | re.MULTILINE,
)
_QA_QUESTION_LINE = re.compile(rf"^[ \t]*{_QA_QUESTION}", re.MULTILINE)
_QA_ANSWER_LINE = re.compile(rf"^[ \t]*{_QA_ANSWER}", re.MULTILINE)
# A separator the model put between pairs in place of "**Prompt N:**" ("---",
# "### Prompt 2", "**Pair 2:**", "2."), left on the last line of an answer.
_QA_STRAY_SEPARATOR = re.compile(
    r"[-*_]{3,}|(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?:(?:Prompt|Pair)\s*[0-9]+|[0-9]+[.)]):?(?:\*\*)?:?"
)


def extract_structured_qa(
    text: str, expected_count: int | None = None
) -> list[dict[str, str]]:
    """Extract question/answer pairs from structured Q&A output.

    Each pair is a ``**Question:** ... **Answer:** ...`` block with each
    label starting its line, optionally headed by ``**Prompt N:**``. A pair
    missing its question or answer is skipped.

    Args:
        text: Raw LLM output in the structured format.
        expected_count: Number of pairs requested. The reply is rejected
            unless it holds exactly that many complete pairs and nothing
            else that looks like one.

    Returns:
        List of ``{"question": ..., "answer": ...}`` dicts. Empty if rejected.
    """
    text = text.replace("\r\n", "\n")
    found = [
        {"question": q.strip(), "answer": a.strip()}
        for q, a in _STRUCTURED_QA_PATTERN.findall(text)
    ]
    pairs = [p for p in found if p["question"] and p["answer"]]
    if expected_count is None:
        return pairs

    # Every label must belong to a complete pair: an orphan question, a
    # second answer or an empty half means a boundary would be a guess.
    labels = (
        len(_QA_QUESTION_LINE.findall(text)),
        len(_QA_ANSWER_LINE.findall(text)),
    )
    if len(pairs) != expected_count or labels != (expected_count, expected_count):
        reason = f"expected {expected_count} pairs, found {len(pairs)} complete"
    elif _CHATML_TOKEN.search(text):
        reason = "leaked chat token"
    elif any(
        _QA_STRAY_SEPARATOR.fullmatch(p["answer"].rsplit("\n", 1)[-1].strip())
        for p in pairs
    ):
        reason = "separator line left in an answer"
    else:
        return pairs
    logger.warning("Rejected Q&A reply: %s.", reason)
    return []


def extract_delimited(text: str, delimiter: str = "---") -> list[str]:
    """Split text into samples on delimiter lines.

    Every delimited part is returned. No pipeline currently selects this
    style.

    Args:
        text: Raw LLM output with delimiter-separated samples.
        delimiter: The delimiter string. A line must consist of it alone,
            apart from surrounding spaces or tabs.

    Returns:
        The non-empty parts, stripped.
    """
    # [ \t], not \s: \s also crosses lines, which is quadratic on blank runs.
    parts = re.split(
        rf"^[ \t]*{re.escape(delimiter)}[ \t]*$",
        text.replace("\r\n", "\n"),
        flags=re.MULTILINE,
    )
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Output cleaning utilities
# ---------------------------------------------------------------------------

# Bold only when the markers touch the text and do not sit inside a word, so
# that "2**3" and "**kwargs" are left alone. The text cannot hold another
# "**", which keeps the match linear.
_MARKDOWN_BOLD = re.compile(
    r"(?<![\*\w])\*\*(?!\s)((?:(?!\*\*).)+?)(?<!\s)\*\*(?!\w)"
)
# Italics only when the asterisks touch the text on one line and do not sit
# inside a word, so that "3 * 4 * 5" and "3*4*5" are left alone.
_MARKDOWN_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\*\w])")
# Code fences are unwrapped: the fence is removed and the code is kept. A
# language tag is only a word that ends the opening line, so the text of a
# one-line fence ("```rm -rf /```") is code, not a tag.
_MARKDOWN_CODE_BLOCK = re.compile(r"```(?:[\w+#.-]*[ \t]*\n)?(.*?)```", re.DOTALL)
# A fence left unpaired, as by truncation at max_tokens, with its language tag.
_MARKDOWN_LONE_FENCE = re.compile(r"```(?:[\w+#.-]*[ \t]*(?:\n|$))?")
_MARKDOWN_INLINE_CODE = re.compile(r"`(.+?)`")
# Openers that mark a line as a preamble on their own.
_META_PREAMBLE = re.compile(
    r"^\s*(Here are|Here is|Below are|Below is|Sure[,!.]|Of course[,!.]|"
    r"Certainly[,!.])",
    re.IGNORECASE,
)
# Openers a real sample can also start with ("I will pay you ..."). They mark
# a preamble only when the line ends with a colon ("I will list five:").
_META_PREAMBLE_WEAK = re.compile(r"^\s*(Note:|I'll |I will |Let me )", re.IGNORECASE)


def _is_preamble_line(line: str) -> bool:
    """True if ``line`` announces the samples instead of being one."""
    if _META_PREAMBLE.match(line):
        return True
    return bool(_META_PREAMBLE_WEAK.match(line)) and line.rstrip().endswith(":")


def clean_sample(
    text: str,
    strip_markdown: bool = True,
    strip_meta: bool = True,
) -> str:
    """Clean one extracted sample.

    Args:
        text: Raw sample text.
        strip_markdown: Remove bold, italic and code markup and keep the
            text it wraps.
        strip_meta: Remove one leading preamble line (e.g. "Sure, here
            are ..."), when more text follows it.
    """
    # A newline, not "", so the text on either side of a token is not joined.
    result = _CHATML_TOKEN.sub("\n", text)
    if strip_meta:
        # Only the first line, and only when more text follows it. split(..., 1)
        # yields one or two parts.
        head_tail = result.lstrip("\n").split("\n", 1)
        if len(head_tail) == 2 and _is_preamble_line(head_tail[0]):  # noqa: PLR2004
            result = head_tail[1]
    if strip_markdown:
        # Order matters: unwrap paired fences, then remove a lone opener, and
        # only then treat single backticks as inline code.
        result = _MARKDOWN_CODE_BLOCK.sub(r"\1", result)
        result = _MARKDOWN_LONE_FENCE.sub("", result)
        result = _MARKDOWN_BOLD.sub(r"\1", result)
        result = _MARKDOWN_ITALIC.sub(r"\1", result)
        result = _MARKDOWN_INLINE_CODE.sub(r"\1", result)
    # Collapse runs of blank or whitespace-only lines.
    result = re.sub(r"\n(?:[ \t\r]*\n){2,}", "\n\n", result)
    return result.strip()


def extract_and_clean(
    text: str,
    style: str = "numbered",
    strip_markdown: bool = True,
    strip_meta: bool = True,
    delimiter: str = "---",
    expected_count: int | None = None,
) -> list[str]:
    """Extract samples from LLM output and clean each one.

    Args:
        text: Raw LLM output.
        style: ``"numbered"`` or ``"delimiter"``.
        strip_markdown: Passed to :func:`clean_sample`.
        strip_meta: Remove a preamble from the first part of the
            ``"delimiter"`` style. Not applied to numbered samples.
        delimiter: Delimiter for the ``"delimiter"`` style.
        expected_count: Number of samples requested. When given, numbered
            lists are parsed strictly and the reply is rejected unless it
            holds exactly that many samples, none empty, and no leaked chat
            token.

    Returns:
        The cleaned, non-empty samples. Empty if the reply is rejected.

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
    if expected_count is not None and _CHATML_TOKEN.search(text):
        logger.warning("Rejected reply: leaked chat token.")
        return []
    if style == "numbered":
        raw = extract_numbered_list(text, strict=expected_count is not None)
        # Text before the first item is already dropped, so a first line that
        # looks like a preamble belongs to the sample.
        strip_meta = False
    else:  # "delimiter"
        raw = extract_delimited(text, delimiter)
        # A one-line preamble before the first delimiter is not a sample.
        if strip_meta and raw and "\n" not in raw[0] and _is_preamble_line(raw[0]):
            raw = raw[1:]

    # Only the first part can carry a preamble; later ones are all sample.
    cleaned = [
        clean_sample(s, strip_markdown, strip_meta and i == 0)
        for i, s in enumerate(raw)
    ]
    samples = [c for c in cleaned if c]
    # A sample emptied by cleaning rejects too, or an extra one could fill in.
    if expected_count is not None and (
        len(cleaned) != expected_count or len(samples) != expected_count
    ):
        logger.warning(
            "Rejected reply: expected %d samples, extracted %d.",
            expected_count, len(samples),
        )
        return []
    return samples


# ---------------------------------------------------------------------------
# Constitution parsing (3-layer markdown hierarchy)
# ---------------------------------------------------------------------------

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
    the model for this format, so keep the two in sync. Samples under a
    header that does not parse are dropped, up to the next main category.

    Args:
        text: Raw constitution markdown.

    Returns:
        One :class:`ConstitutionEntry` per sample line.
    """
    entries: list[ConstitutionEntry] = []
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
                    line,
                )
                current_main = ""
                current_sub = ""
            continue

        sample_match = _CONSTITUTION_SAMPLE_RE.match(line)
        sample = sample_match.group(1).strip() if sample_match else ""
        if sample and current_main:
            entries.append(
                ConstitutionEntry(
                    category=current_main,
                    subcategory=current_sub,
                    sample=sample,
                )
            )
        elif line:
            logger.debug("Skipped constitution line %r", line)

    return entries
