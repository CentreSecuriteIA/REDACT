"""The neutral containers: the prompt in a code comment, a YAML document, a
CSV row, an email, a transcript line or a log entry, read back whole by a
small parser of each format (the YAML ones by PyYAML), also for a prompt that
holds what could end the format around it: a block comment's closer, a line
break of any kind, a blank line in a subtitle cue, spaces before a block
scalar's first line."""

import csv
import email
import io
import random
import re

import pytest

from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.obfuscation import neutral_container as NC
from redact.jailbreak.obfuscation.neutral_container import (
    get_neutral_container_functions,
)

TEXTS = [
    "How do plants turn sunlight into food?",
    "Line one\nLine two, with a comma\n\nLine four, after an empty one",
    'She said "hello"; the note read: "back at 5", in quotes.',
    "Tabs\tand pipes | and {braces}, café",
]

#: Every line break a reader may split on, besides a line feed.
LINE_BREAKS = ["\r\n", "\r", "\x0b", "\x0c", "\x85", "\u2028", "\u2029"]

#: Words no variant's surroundings hold: a container gives no instruction
#: and says nothing to the reader. A starting list, for the owner to extend.
NOT_NEUTRAL = (
    "ignore", "override", "bypass", "disregard", "obey", "unrestricted",
    "jailbreak", "system", "admin", "administrator", "instruction",
    "instructions", "policy", "policies", "safety", "rules", "answer",
    "respond", "reply", "must", "always", "never",
)

#: Stands for the prompt where a test fills a template itself.
_MARK = "@@PROMPT@@"

FUNCTIONS = get_neutral_container_functions()


def _pool(fn) -> list[dict]:
    ref = fn.variant_pool
    return read_pools(ref.file)[ref.directive]


def _around(template: str) -> list[str]:
    """What a variant writes before the prompt and after it."""
    return template.format_map({"prompt": _MARK}).split(_MARK)


def _from_prefixed_lines(written: str, template: str) -> str:
    """A comment or a YAML block scalar: the lines between the variant's
    text, each after the first without the marker or indent the first has."""
    before, after = _around(template)
    assert written.startswith(before) and written.endswith(after)
    first, *rest = written[len(before):len(written) - len(after)].split("\n")
    prefix = before.rpartition("\n")[2]
    assert all(line.startswith(prefix) for line in rest)
    return "\n".join([first, *(line[len(prefix):] for line in rest)])


def _from_csv(written: str, template: str) -> str:
    """The field the template puts the prompt in, read with the csv module;
    the delimiter is the character before it."""
    delimiter = template[template.index("{prompt}") - 1]
    rows = list(csv.reader(io.StringIO(written), delimiter=delimiter))
    marked = list(csv.reader(io.StringIO(template.format_map({"prompt": _MARK})),
                             delimiter=delimiter))
    ((row, column),) = [(i, j) for i, fields in enumerate(marked)
                        for j, field in enumerate(fields) if field == _MARK]
    return rows[row][column]


def _from_email(written: str, template: str) -> str:
    """The message's body, read with the email module, less the signature."""
    body = email.message_from_string(written).get_payload()
    _, after = _around(template)
    assert body.endswith(after)
    return body[:len(body) - len(after)]


def _from_last_line(written: str, template: str) -> str:
    """A transcript line or a log entry: the message after the speaker or
    the logger, to the end."""
    before, after = _around(template)
    assert after == "" and written.startswith(before)
    return written[len(before):]


def _is_cue(variant: dict) -> bool:
    """A subtitle cue (WebVTT, SRT): its timing line holds ``-->``."""
    return "-->" in variant["template"]


def _cue_text(text: str) -> str:
    """What a cue holds of ``text``: its non-blank lines, as a blank line ends one."""
    return "\n".join(line for line in text.split("\n") if line.strip())


_READ_BACK = {
    "to_code_comment": _from_prefixed_lines,
    "to_yaml_document": _from_prefixed_lines,
    "to_csv_row": _from_csv,
    "to_email_message": _from_email,
    "to_transcript_line": _from_last_line,
    "to_log_entry": _from_last_line,
}

#: Each technique with each variant of its pool.
_VARIANTS = [(fn, variant) for fn in FUNCTIONS for variant in _pool(fn)]
_IDS = [f"{fn.__name__}-{variant['name']}" for fn, variant in _VARIANTS]


def _written(monkeypatch, fn, variant: dict, text: str) -> tuple[str, str]:
    """What ``fn`` writes for ``text`` when it draws ``variant``."""
    monkeypatch.setattr(NC, "draw_variant", lambda variants, rng=None: variant)
    return fn(text)


def test_the_pools_are_ones_their_techniques_can_draw():
    check_pools(fn.variant_pool for fn in FUNCTIONS)


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_each_container_has_eight_variants_each_named_once(fn):
    names = [variant["name"] for variant in _pool(fn)]
    assert len(names) == len(set(names)) == 8     # noqa: PLR2004


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize(("fn", "variant"), _VARIANTS, ids=_IDS)
def test_the_prompt_is_read_back_whole_from_every_variant(
        monkeypatch, fn, variant, text):
    written, info = _written(monkeypatch, fn, variant, text)
    held = _cue_text(text) if fn is NC.to_transcript_line and _is_cue(variant) else text
    assert _READ_BACK[fn.__name__](written, variant["template"]) == held
    assert info.endswith(f";variant={variant['name']}")


def _yaml_values(monkeypatch, variant: dict, text: str) -> list[str]:
    """The strings of the YAML document ``to_yaml_document`` writes for ``text``
    with ``variant``, read by PyYAML, less one final line feed."""
    yaml = pytest.importorskip("yaml")
    written, _ = _written(monkeypatch, NC.to_yaml_document, variant, text)
    values = []

    def collect(node):
        if isinstance(node, dict):
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
        elif isinstance(node, str):
            values.append(node.removesuffix("\n"))

    for document in yaml.safe_load_all(written):
        collect(document)
    return values


_YAML_VARIANTS = _pool(NC.to_yaml_document)
_YAML_IDS = [variant["name"] for variant in _YAML_VARIANTS]

#: A prompt a YAML block scalar needs fitted, with the text the document holds.
_YAML_FITTED = [
    ("  indented first\nsecond line", "indented first\nsecond line"),
    ("    deep\n  shallower\nflush", "deep\n  shallower\nflush"),
    ("first\n    \n\t\nlast", "first\n\n\nlast"),
    ("\n  after a blank line\nsecond", "\nafter a blank line\nsecond"),
    *[(f"first{kind}second", "first\nsecond") for kind in LINE_BREAKS],
]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("variant", _YAML_VARIANTS, ids=_YAML_IDS)
def test_a_yaml_document_parses_with_the_prompt_as_a_value(monkeypatch, variant, text):
    assert text in _yaml_values(monkeypatch, variant, text)


@pytest.mark.parametrize(("text", "held"), _YAML_FITTED)
@pytest.mark.parametrize("variant", _YAML_VARIANTS, ids=_YAML_IDS)
def test_a_yaml_document_parses_for_a_prompt_that_needs_fitting(
        monkeypatch, variant, text, held):
    """Spaces before the first line, lines of spaces and every kind of line
    break: the document parses, holding the prompt with its line breaks as
    line feeds, its first line's indent gone and its blank lines empty."""
    assert held in _yaml_values(monkeypatch, variant, text)


_COMMENTS = [(fn, variant) for fn, variant in _VARIANTS if fn is NC.to_code_comment]
_COMMENT_IDS = [variant["name"] for _, variant in _COMMENTS]


@pytest.mark.parametrize("kind", LINE_BREAKS, ids=repr)
@pytest.mark.parametrize(("fn", "variant"), _COMMENTS, ids=_COMMENT_IDS)
def test_every_kind_of_line_break_keeps_the_next_line_in_the_comment(
        monkeypatch, fn, variant, kind):
    """A reader that splits on a carriage return, a form feed, NEL or the
    Unicode separators sees each line of the prompt behind the marker."""
    written, _ = _written(monkeypatch, fn, variant, f"first{kind}second")
    prefix = NC._line_prefix(variant["template"])
    assert [line for line in written.splitlines() if "second" in line] == [
        prefix + "second"]


@pytest.mark.parametrize(("closer", "broken"), [("*/", "* /"), ("-->", "-- >")])
@pytest.mark.parametrize(("fn", "variant"), _COMMENTS, ids=_COMMENT_IDS)
def test_a_closer_in_the_prompt_cannot_end_the_comment(
        monkeypatch, fn, variant, closer, broken):
    """Where the variant's template holds the closer the prompt's is broken by a
    space; elsewhere (a ``#`` or ``//`` comment) the prompt is left as it is."""
    text = f"before {closer} after {closer}{closer}"
    written, _ = _written(monkeypatch, fn, variant, text)
    held = _from_prefixed_lines(written, variant["template"])
    if closer in variant["template"]:
        assert written.count(closer) == variant["template"].count(closer)
        assert held == text.replace(closer, broken)
    else:
        assert held == text


_TRANSCRIPTS = [(fn, variant) for fn, variant in _VARIANTS
                if fn is NC.to_transcript_line]
_CUES = [(fn, variant) for fn, variant in _TRANSCRIPTS if _is_cue(variant)]
_OTHER_LINES = [(fn, variant) for fn, variant in _TRANSCRIPTS if not _is_cue(variant)]


def test_the_transcript_pool_holds_two_subtitle_cues():
    assert [variant["name"] for _, variant in _CUES] == ["webvtt", "srt"]


@pytest.mark.parametrize(("fn", "variant"), _CUES,
                         ids=[variant["name"] for _, variant in _CUES])
def test_a_subtitle_cue_holds_no_blank_line_and_no_arrow_of_the_prompt(
        monkeypatch, fn, variant):
    text = "\nfirst line\n\n  \nsecond --> line\n\nlast\n"
    written, _ = _written(monkeypatch, fn, variant, text)
    before, _ = _around(variant["template"])
    assert written[len(before):] == "first line\nsecond -- > line\nlast"
    assert written.count("-->") == variant["template"].count("-->")


@pytest.mark.parametrize(("fn", "variant"), _OTHER_LINES,
                         ids=[variant["name"] for _, variant in _OTHER_LINES])
def test_another_transcript_line_keeps_the_blank_lines_and_arrows(
        monkeypatch, fn, variant):
    text = "first\n\nsecond --> line\n"
    written, _ = _written(monkeypatch, fn, variant, text)
    assert _from_last_line(written, variant["template"]) == text


def test_a_comment_puts_its_marker_before_every_line():
    assert NC._each_line_prefixed("one\ntwo\n\nfour", "x\n# {prompt}\ny") == (
        "one\n# two\n# \n# four")


def test_a_csv_field_is_quoted_with_its_quotes_doubled():
    assert NC._csv_field('say "hi", then go') == '"say ""hi"", then go"'


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_a_seed_draws_its_variant_as_rng_choice_does(fn):
    directive = fn.variant_pool.directive
    for seed in range(20):
        ours, theirs = random.Random(seed), random.Random(seed)
        _, info = fn(TEXTS[0], rng=ours)
        expected = theirs.choice(_pool(fn))["name"]
        assert info == f"container={directive};variant={expected}"
        assert ours.getstate() == theirs.getstate()


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_a_seed_gives_the_same_container(fn):
    assert fn(TEXTS[1], rng=random.Random(3)) == fn(TEXTS[1], rng=random.Random(3))


@pytest.mark.parametrize(("fn", "variant"), _VARIANTS, ids=_IDS)
def test_no_variant_holds_a_word_of_instruction(fn, variant):
    words = set(re.findall(r"[a-z]+", variant["template"].lower()))
    assert sorted(words & set(NOT_NEUTRAL)) == []
