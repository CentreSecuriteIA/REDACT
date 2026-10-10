"""Parse individual samples out of multi-sample LLM output.

Supported formats:

- Numbered list: "1. text" / "2) text" / "3: text".
- Structured Q&A: "**Prompt N:** ... **Question:** ... **Answer:** ...".

Also provides sample cleaning (:func:`clean_sample`).
"""

#TODO(driver script): handle rejected replies where the steps are driven.
# A reply rejected by expected_count/strict comes back as an empty list, which
# the callers still treat as "the model returned nothing". To move into the
# driver, together with the retry helpers:
#   - Retry a rejected reply a bounded number of times before giving up.
#     Done for constitution inputs (llm_pipeline.extracted); the call sites
#     below still treat a rejected reply as an empty one.
#   - Standalone inputs (standalone_generation.py): one rejected reply stops
#     the whole category. Stop only on an empty reply or repeated failures.
#   - Benign Q&A (jailbreak/manipulation/benign.py): a rejected half gets no
#     rows and the cache counts as complete; an all-rejected run writes an
#     empty CSV that every later load fails on.
#   - Seeds (metaprompt.py): after 3 failed attempts the category is skipped
#     with only an INFO log.
#   - Count rejections and report them: a run that lost entries still logs
#     "100% accepted", and skipped_entries mixes empty, rejected and duplicate.
#   - prompts/format_instructions/numbered: the example shows two items even
#     when one sample is requested.
#   - Add pipeline-level tests for a wrong-count reply at each call site.
#TODO: Future extension: end token. An end marker in the format instructions
# would let strict mode reject trailing commentary and replies cut off at
# max_tokens; neither can be told apart from sample text today.

import logging
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styles (each has a format instruction in prompts/format_instructions/{style}/)
# ---------------------------------------------------------------------------

#: Output formats the library can request and parse. All of them are valid
#: for :func:`~redact.llms.prompting.prompts.get_format_instruction`.
EXTRACTION_STYLES: frozenset[str] = frozenset({"numbered", "structured_qa"})


#NOTE: Checks num_samples in prompts.get_format_instruction and expected_count in both extractors; name is the one the error reports.
def _sample_count(value: object, name: str = "num_samples") -> int:
    """Validate a sample count: an int >= 1, or a string of one.

    Raises:
        ValueError: Anything else, such as a float, ``None`` or ``0``.
    """
    try:
        count = int(value) if isinstance(value, str) else value
    except ValueError:
        count = None
    # type(), not isinstance(): a bool is an int but not a count.
    if type(count) is not int or count < 1:
        raise ValueError(
            f"{name} must be an integer >= 1 (or a string of one), "
            f"got {value!r}."
        )
    return count


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

# A leaked chat token, whole or partial ("<|im_end|>", "|im_start"), with the
# role name that follows an opening one. It means the model ran past its turn.
_CHATML_TOKEN = re.compile(
    r"<?\|im_(?:start\|?>?(?:(?:system|user|assistant)\b)?|end\|?>?)"
)

#NOTE: The format instruction asks for "1. text"; the other separators and the bold forms are accepted variants.
# Groups: indent, opening "**", number, separator, closing "**". The bold
# markers cover "**1.** text" and "**1. Title** text". The number is bounded
# ASCII so int() cannot fail on it. The separator touches its number and is
# followed by whitespace on the same line, so "2 - 5 degrees" and
# "1.5 million" are text, not items.
_NUMBERED_PATTERN = re.compile(
    r"^(\s*)(\*\*)?([0-9]{1,4})([\.\)\-\:])(\*\*)?\s+"
)


class _NumberingStyle(NamedTuple):
    """How a list writes its numbers: "1.", "1)", "**1.**", "**1. Title**"."""

    separator: str
    bold_open: bool
    bold_close: bool


class _ItemHeader(NamedTuple):
    """The numbering that opens a line, and the text after it."""

    indent: int
    number: int
    style: _NumberingStyle
    content: str


def _item_header(line: str) -> _ItemHeader | None:
    """The parts of a numbered line, or ``None`` if ``line`` is not one."""
    match = _NUMBERED_PATTERN.match(line)
    if match is None:
        return None
    indent, bold_open, digits, separator, bold_close = match.groups()
    content = line[match.end():].strip()
    if bold_open and not bold_close:
        # "**1. Title** text": give the title its opening marker back.
        content = "**" + content
    style = _NumberingStyle(separator, bool(bold_open), bool(bold_close))
    return _ItemHeader(len(indent), int(digits), style, content)


def extract_numbered_list(text: str, strict: bool = False) -> list[str]:
    """Extract samples from a numbered list.

    A sample starts on a line beginning with a number and '.', ')', '-' or
    ':'. After the first, only the next number in sequence, indented no
    deeper than the first item, starts a new sample; any other line continues
    the current one. Text before the first item is ignored.

    Args:
        text: Raw LLM output containing a numbered list.
        strict: Reject the reply unless it is one clean list: starting at 1,
            in sequence, in one numbering style, with no empty item and no
            leaked chat token.

    Returns:
        The non-empty samples, with the numbering removed. Empty if rejected.
    """
    return _parse_numbered_list(text, strict) or []


def _parse_numbered_list(text: str, strict: bool) -> list[str] | None:
    """:func:`extract_numbered_list`, with ``None`` for a rejected reply."""
    text = text.replace("\r\n", "\n")
    if strict and _CHATML_TOKEN.search(text):
        logger.warning("Rejected numbered reply: leaked chat token.")
        return None

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
    first_style: _NumberingStyle | None = None

    for line in lines:
        header = _item_header(line)
        if header and expected is not None and header.indent > base_indent:
            header = None  # nested in the current sample: text, not an item

        # Strict: every top-level number must be the next one, in the first
        # item's separator and bold style, so a wrapped line like "2) left"
        # in a "1." list is not taken for item 2. A mismatch rejects the
        # whole reply instead of being repaired: the boundary would be a
        # guess, and a wrongly split sample costs more than a lost reply.
        if strict and header:
            wanted = 1 if expected is None else expected
            if header.number != wanted:
                logger.warning(
                    "Rejected numbered reply: expected item %d, found %d.",
                    wanted, header.number,
                )
                return None
            if first_style is not None and header.style != first_style:
                logger.warning(
                    "Rejected numbered reply: item %d changes numbering style.",
                    header.number,
                )
                return None

        # The first numbered line always starts a sample: nothing before it
        # can be one. After that only the expected number at the top indent
        # does, since a year, a quantity or a nested step is far likelier
        # than the model renumbering its list.
        if header and (expected is None or header.number == expected):
            if expected is None:
                base_indent = header.indent
                first_style = header.style
            else:
                samples.append("\n".join(current_lines).strip())
            expected = header.number + 1
            current_lines = [header.content]
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
        return None
    return [s for s in samples if s]


# The "**Prompt N:**" header is optional and a label's colon may sit inside or
# outside the bold. Labels count only at the start of a line, so label-like
# text inside a question or answer does not split it.
_QA_PROMPT = r"\*\*Prompt\s*\d+:?\*\*:?"
_QA_QUESTION = r"\*\*Question:?\*\*:?"
_QA_ANSWER = r"\*\*Answer:?\*\*:?"
_QA_LABEL_LINE = rf"^[ \t]*(?:{_QA_PROMPT}|{_QA_QUESTION}|{_QA_ANSWER})"
# One character that does not begin a label line.
_QA_BODY = rf"(?:(?!{_QA_LABEL_LINE}).)"
_STRUCTURED_QA_PATTERN = re.compile(
    rf"""
    ^[ \t]*                     # a pair starts at the start of a line
    (?:{_QA_PROMPT}\s*)?        # optional "**Prompt N:**" header before it
    {_QA_QUESTION}[ \t]*+       # the question label; possessive, to stay linear
    ({_QA_BODY}+?)              # group 1: the question, up to ...
    ^[ \t]*{_QA_ANSWER}[ \t]*   # ... the answer label, at the start of a line
    ({_QA_BODY}+)               # group 2: the answer, up to the next label line
    """,
    re.DOTALL | re.MULTILINE | re.VERBOSE,
)
_QA_QUESTION_LINE = re.compile(
    rf"^[ \t]*(?:{_QA_PROMPT}\s*)?{_QA_QUESTION}", re.MULTILINE
)
_QA_ANSWER_LINE = re.compile(rf"^[ \t]*{_QA_ANSWER}", re.MULTILINE)
# A separator the model put between pairs in place of "**Prompt N:**" ("---",
# "### Prompt 2", "**Pair 2:**", "Sample 2:", "Q2:", "2.", "2:"), left on the
# last line of an answer. A bare number or word and number ("42", "Python 3")
# is not one: an answer can end in it.
_QA_STRAY_SEPARATOR = re.compile(
    r"""
    [-*_]{3,}                   # a rule: "---", "***", "___"
    |
    (?:\#{1,6}\s*)?             # optional heading marks: "### "
    (?:\*\*)?                   # optional opening bold
    (?:
        (?:Prompt|Pair)\s*[0-9]+      # the header words: "Prompt 2", "Pair 2"
        |
        [A-Za-z]+\s*[0-9]+(?=:|\*\*)  # another word only when marked: "Sample 2:"
        |
        [0-9]+[.):]                   # a number and its punctuation: "2.", "2:"
    )
    :?(?:\*\*)?:?               # optional colon, inside or outside closing bold
    """,
    re.VERBOSE,
)


def extract_structured_qa(
    text: str, expected_count: int | str | None = None
) -> list[dict[str, str]]:
    """Extract question/answer pairs from structured Q&A output.

    Each pair is a ``**Question:** ... **Answer:** ...`` block with each
    label starting its line, optionally headed by ``**Prompt N:**``. A pair
    missing its question or answer is skipped.

    Args:
        text: Raw LLM output in the structured format.
        expected_count: Number of pairs requested. The reply is rejected
            unless it holds exactly that many complete pairs and nothing
            else that looks like one, no leaked chat token, and no separator
            line at the end of an answer.

    Returns:
        List of ``{"question": ..., "answer": ...}`` dicts. Empty if rejected.

    Raises:
        ValueError: ``expected_count`` is not an integer >= 1.
    """
    if expected_count is not None:
        expected_count = _sample_count(expected_count, "expected_count")
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


def clean_sample(text: str, strip_markdown: bool = True) -> str:
    """Clean one extracted sample.

    Args:
        text: Raw sample text.
        strip_markdown: Remove bold, italic and code markup and keep the
            text it wraps.
    """
    text = text.replace("\r\n", "\n")
    # A newline, not "", so the text on either side of a token is not joined.
    result = _CHATML_TOKEN.sub("\n", text)
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
    expected_count: int | str | None = None,
) -> list[str]:
    """Extract samples from LLM output and clean each one.

    Args:
        text: Raw LLM output.
        style: ``"numbered"``.
        strip_markdown: Passed to :func:`clean_sample`.
        expected_count: Number of samples requested. When given, the list
            is parsed strictly and the reply is rejected unless it holds
            exactly that many samples, none empty, and no leaked chat token.

    Returns:
        The cleaned, non-empty samples. Empty if the reply is rejected.

    Raises:
        ValueError: Unknown style, a paired style such as
            ``"structured_qa"`` (for that one, call
            :func:`extract_structured_qa`), or an ``expected_count`` that is
            not an integer >= 1.
    """
    if expected_count is not None:
        expected_count = _sample_count(expected_count, "expected_count")
    if style == "structured_qa":
        raise ValueError(
            f"'{style}' yields question/answer pairs, not plain samples, so it "
            f"cannot go through extract_and_clean(). Call "
            f"extract_structured_qa() directly — it returns both halves, "
            f"which this would have discarded."
        )
    if style not in EXTRACTION_STYLES:
        raise ValueError(
            f"Unknown extraction style '{style}'. "
            f"Choose from: {sorted(EXTRACTION_STYLES)}"
        )
    raw = _parse_numbered_list(text, strict=expected_count is not None)
    if raw is None:
        # The strict parser logged why; a count message here would mislead.
        return []

    cleaned = [clean_sample(s, strip_markdown) for s in raw]
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
