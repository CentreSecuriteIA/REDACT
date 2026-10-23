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
comment or a YAML block scalar each of its lines after the first takes the
comment marker or the indent the template puts before the first; in a CSV row
it is one quoted field, its own quotes doubled; an email body, a transcript
line and a log entry keep it as it is, newlines too.

``info`` is ``container=<directive>;variant=<variant>``.
"""

import random
from collections.abc import Callable

from ..directives import draw_variant, read_pools, uses_pool

_FILE = "obfuscation/neutral_container_templates.json"
_POOLS: dict[str, list[dict]] = read_pools(_FILE)


def _line_prefix(template: str) -> str:
    """What stands before ``{prompt}`` on its line of ``template``: a comment
    marker, or the indent of a YAML block scalar."""
    return template.split("{prompt}", 1)[0].rpartition("\n")[2]


def _each_line_prefixed(prompt: str, template: str) -> str:
    """``prompt`` with each line after the first behind the prefix
    ``template`` gives the first (:func:`_line_prefix`), so every line stays
    in the comment or the block."""
    return prompt.replace("\n", "\n" + _line_prefix(template))


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
    return _filled("code_comment", variant,
                   _each_line_prefixed(prompt, variant["template"]))


@uses_pool(_FILE, "yaml_document")
def to_yaml_document(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Put the prompt in a YAML document as a block scalar, each of its lines
    indented."""
    variant = draw_variant(_POOLS["yaml_document"], rng)
    return _filled("yaml_document", variant,
                   _each_line_prefixed(prompt, variant["template"]))


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
    return _filled("transcript_line", variant, prompt)


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
