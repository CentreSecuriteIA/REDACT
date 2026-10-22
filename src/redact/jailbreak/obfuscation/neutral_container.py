"""Neutral containers: the prompt placed in an everyday document, as text
reaches a classifier in real traffic: a code comment, a YAML document, a CSV
row, an email, a line of a transcript or a log entry.

Each container draws one of the eight variants of its pool in
``obfuscation/neutral_container_templates.json``, which give the surroundings:
the comment syntax and the code around it, the YAML keys, the CSV header and
delimiter, the email's headers and signature, the speaker and timestamp, the
log level and logger. The surroundings are neutral by design: they hold no
instruction and say nothing to the reader.

The prompt is first fitted to its format, so it can be read back whole: in a
comment or a YAML block scalar its line breaks (CR LF, CR, a vertical tab, a
form feed, NEL, U+2028, U+2029) are written as a line feed, and each of its
lines after the first takes the comment marker or the indent the template puts
before the first; a block comment's closer in the prompt (``*/``, ``-->``) is
broken by a space, and in a block scalar the prompt's blank lines are emptied
and the first line's leading spaces dropped, since they would set an indent
the next lines lack; in a CSV row it is one quoted field, its own quotes
doubled; in a subtitle cue (a template holding ``-->``) the prompt keeps only
its non-blank lines, since a blank line ends a cue, and its own ``-->`` is
broken; an email body, any other transcript line and a log entry keep it as it
is, newlines too.

``info`` is ``container=<directive>;variant=<variant>``.
"""

import random
import re
from collections.abc import Callable

from ..directives import draw_variant, read_pools, uses_pool

_FILE = "obfuscation/neutral_container_templates.json"
_POOLS: dict[str, list[dict]] = read_pools(_FILE)

#: What a line-structured format reads as a line break, besides a line feed.
_LINE_BREAKS = re.compile(r"\r\n|[\r\x0b\x0c\x1c-\x1e\x85\u2028\u2029]")

#: What ends a block comment, or a subtitle cue's timing line.
_CLOSERS = ("*/", "-->")


def _line_prefix(template: str) -> str:
    """What stands before ``{prompt}`` on its line of ``template``: a comment
    marker, or the indent of a YAML block scalar."""
    return template.split("{prompt}", 1)[0].rpartition("\n")[2]


def _each_line_prefixed(prompt: str, template: str) -> str:
    """``prompt`` with each line after the first behind the prefix
    ``template`` gives the first (:func:`_line_prefix`), so every line stays
    in the comment or the block. Every line break is first written as a line
    feed (:data:`_LINE_BREAKS`), the only one the prefix follows."""
    prefixed = _LINE_BREAKS.sub("\n", prompt)
    return prefixed.replace("\n", "\n" + _line_prefix(template))


def _closers_broken(prompt: str, template: str) -> str:
    """``prompt`` with each closer (:data:`_CLOSERS`) that ``template`` holds
    broken by a space before its last character (``* /``, ``-- >``), so the
    prompt cannot end the comment or the cue's timing line."""
    for closer in _CLOSERS:
        if closer in template:
            prompt = prompt.replace(closer, closer[:-1] + " " + closer[-1])
    return prompt


def _block_scalar_text(prompt: str) -> str:
    """``prompt`` as the text of a YAML block scalar: its lines of only
    spaces emptied, and the leading spaces of its first line with text
    dropped. The scalar's indent is that of its first line with text, so a
    line less indented than that one would end the scalar."""
    text = re.sub(r"^[ \t]+$", "", _LINE_BREAKS.sub("\n", prompt), flags=re.MULTILINE)
    return re.sub(r"\A(\n*) +", r"\1", text)


def _cue_text(prompt: str, template: str) -> str:
    """``prompt`` for a template that holds ``-->``, a subtitle cue (WebVTT,
    SRT): its own ``-->`` broken, and only its non-blank lines kept, since a
    blank line ends a cue. Any other template gets ``prompt`` as it is."""
    if "-->" not in template:
        return prompt
    text = _LINE_BREAKS.sub("\n", _closers_broken(prompt, template))
    return "\n".join(line for line in text.split("\n") if line.strip())


def _csv_field(prompt: str) -> str:
    """``prompt`` as one quoted CSV field: in double quotes, each of its own
    doubled."""
    return '"' + prompt.replace('"', '""') + '"'


def _filled(directive: str, variant: dict, text: str) -> tuple[str, str]:
    """``variant`` of ``directive``'s pool with ``text`` in its slot, and the
    info naming both."""
    return (variant["template"].format_map({"prompt": text}),
            f"container={directive};variant={variant['name']}")


@uses_pool(_FILE, "code_comment")
def to_code_comment(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put the prompt in a comment above a few lines of code, each of its
    lines behind the comment marker."""
    variant = draw_variant(_POOLS["code_comment"], rng)
    template = variant["template"]
    return _filled("code_comment", variant, _each_line_prefixed(
        _closers_broken(prompt, template), template))


@uses_pool(_FILE, "yaml_document")
def to_yaml_document(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put the prompt in a YAML document as a block scalar, each of its lines
    indented."""
    variant = draw_variant(_POOLS["yaml_document"], rng)
    return _filled("yaml_document", variant, _each_line_prefixed(
        _block_scalar_text(prompt), variant["template"]))


@uses_pool(_FILE, "csv_row")
def to_csv_row(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put the prompt in a quoted field of a CSV row below its header."""
    variant = draw_variant(_POOLS["csv_row"], rng)
    return _filled("csv_row", variant, _csv_field(prompt))


@uses_pool(_FILE, "email_message")
def to_email_message(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put the prompt in the body of an email, below its headers."""
    variant = draw_variant(_POOLS["email_message"], rng)
    return _filled("email_message", variant, prompt)


@uses_pool(_FILE, "transcript_line")
def to_transcript_line(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put the prompt in a transcript as one speaker's line, after its
    timestamp."""
    variant = draw_variant(_POOLS["transcript_line"], rng)
    return _filled("transcript_line", variant,
                   _cue_text(prompt, variant["template"]))


@uses_pool(_FILE, "log_entry")
def to_log_entry(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put the prompt in a log as the message of an entry, after its
    timestamp, level and logger."""
    variant = draw_variant(_POOLS["log_entry"], rng)
    return _filled("log_entry", variant, prompt)


def get_neutral_container_functions() -> list[Callable]:
    """Return all neutral container technique functions."""
    return [
        to_code_comment,
        to_yaml_document,
        to_csv_row,
        to_email_message,
        to_transcript_line,
        to_log_entry,
    ]
