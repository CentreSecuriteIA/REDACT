"""The neutral containers: the prompt in a code comment, a YAML document, a
CSV row, an email, a transcript line or a log entry, read back whole by a
small parser of each format."""

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
    assert _READ_BACK[fn.__name__](written, variant["template"]) == text
    assert info.endswith(f";variant={variant['name']}")


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("variant", _pool(NC.to_yaml_document),
                         ids=lambda variant: variant["name"])
def test_a_yaml_document_parses_with_the_prompt_as_a_value(monkeypatch, variant, text):
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
    assert text in values


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
