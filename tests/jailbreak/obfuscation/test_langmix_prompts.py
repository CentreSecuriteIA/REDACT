"""The model prompts of the code-switching and romanization techniques
(``jailbreak/code_switch``, ``code_switch_check``, ``romanize``,
``romanize_check``): their structure, and their agreement with the code that
reads their replies.

The prompts are wording, which no test judges; these tests hold what the code
depends on: the schema and the placeholders each caller fills, the few-shot
turns (present, alternating, mirroring the template, in several languages),
the example replies against the techniques' own gates, the verdict-first
reply format ``is_accepted`` reads, and the gate's reason codes, which reach
the generation model as feedback and are explained in its system prompt. No
test prints a prompt.
"""

import re
import string

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation import code_switching as cs
from redact.jailbreak.obfuscation import romanization as rm
from redact.llm_pipeline import is_accepted
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

_KEYS = {"system_prompt", "template", "seed_fields", "few_shot_examples", "metadata"}
_GEN_FIELDS = ["language", "prompt"]
_CHECK_FIELDS = ["original", "rewritten", "language"]

#: Each prompt: its module (the language rows and the gates), whether it is
#: the check, and the placeholders its template fills.
_PROMPTS = {
    "code_switch": (cs, False, _GEN_FIELDS),
    "code_switch_check": (cs, True, _CHECK_FIELDS),
    "romanize": (rm, False, _GEN_FIELDS),
    "romanize_check": (rm, True, _CHECK_FIELDS),
}
_EVERY = pytest.mark.parametrize("name", list(_PROMPTS))
_GENERATION = pytest.mark.parametrize("name", ["code_switch", "romanize"])
_CHECKS = pytest.mark.parametrize("name", ["code_switch_check", "romanize_check"])

#: The fewest worked examples a prompt carries, and the fewest languages they
#: span (the system prompt serves ten languages; one language's examples would
#: anchor the model to it).
_MIN_EXAMPLES = 5
_MIN_LANGUAGES = 3
#: An example verdict is one short sentence after the verdict word.
_MAX_VERDICT_CHARS = 160


def _config(name: str) -> dict:
    return load_prompt("jailbreak", name)


def _rows(name: str) -> dict[str, dict]:
    """The language rows of the prompt's family, by name."""
    return {row["name"]: row for row in _PROMPTS[name][0].LANGUAGES}


def _fields(text: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


def _pattern(template: str) -> re.Pattern:
    """A regex that a rendering of ``template`` matches, a named group per
    placeholder."""
    parts = re.split(r"\{(\w+)\}", template)
    return re.compile(
        "".join(re.escape(p) if i % 2 == 0 else f"(?P<{p}>.+?)"
                for i, p in enumerate(parts)),
        re.DOTALL)


def _examples(name: str) -> list[tuple[dict[str, str], str]]:
    """``[(the placeholders of an example's user turn, its reply)]``: each
    user turn read back through the template."""
    config = _config(name)
    shots = config["few_shot_examples"]
    pattern = _pattern(config["template"])
    out = []
    for user, assistant in zip(shots[0::2], shots[1::2], strict=True):
        match = pattern.fullmatch(user["content"])
        assert match, f"{name}: an example's user turn does not mirror the template"
        out.append((match.groupdict(), assistant["content"]))
    return out


# --- schema and rendering -------------------------------------------------


@_EVERY
def test_the_schema_and_the_placeholders_are_the_callers(name):
    config = _config(name)
    _, is_check, fields = _PROMPTS[name]
    assert set(config) == _KEYS
    assert config["seed_fields"] == fields
    assert _fields(config["template"]) == set(fields)
    system = _fields(config["system_prompt"])
    # The checks are built without arguments, so their system prompt holds no
    # placeholder; a generation prompt's holds only the language.
    assert system == (set() if is_check else {"language"})
    assert config["metadata"]["version"] and config["metadata"]["notes"]


@_GENERATION
def test_a_generation_prompt_renders_for_every_language(name):
    module, _, _ = _PROMPTS[name]
    config = _config(name)
    for row in module.LANGUAGES:
        messages = build_messages(config, prompt="THE-MESSAGE", language=row["name"])
        roles = [m["role"] for m in messages]
        shots = len(config["few_shot_examples"])
        assert roles == ["system", *["user", "assistant"] * (shots // 2), "user"]
        assert row["name"] in messages[0]["content"]
        assert "{" not in messages[0]["content"]
        assert messages[-1]["content"].endswith("THE-MESSAGE")
        assert row["name"] in messages[-1]["content"]


@_CHECKS
def test_a_check_prompt_renders_with_the_three_texts(name):
    config = _config(name)
    template = PromptTemplate.load("jailbreak", name)
    messages = template(original="THE-ORIGINAL", rewritten="THE-REWRITE",
                        language="Hindi")
    roles = [m["role"] for m in messages]
    shots = len(config["few_shot_examples"])
    assert roles == ["system", *["user", "assistant"] * (shots // 2), "user"]
    final = messages[-1]["content"]
    assert all(text in final for text in ("THE-ORIGINAL", "THE-REWRITE", "Hindi"))
    assert final.rstrip().endswith("?")


# --- the worked examples --------------------------------------------------


@_EVERY
def test_the_examples_alternate_mirror_the_template_and_span_languages(name):
    config = _config(name)
    shots = config["few_shot_examples"]
    assert len(shots) % 2 == 0 and len(shots) // 2 >= _MIN_EXAMPLES
    assert [m["role"] for m in shots] == ["user", "assistant"] * (len(shots) // 2)
    assert all(m["content"].strip() == m["content"] and m["content"] for m in shots)
    examples = _examples(name)
    rows = _rows(name)
    languages = {fields["language"] for fields, _ in examples}
    assert languages <= set(rows), f"{name}: a language the family lacks"
    assert len(languages) >= _MIN_LANGUAGES


@_GENERATION
def test_every_example_reply_is_the_rewrite_alone_and_passes_the_gate(
        name, monkeypatch):
    module, _, _ = _PROMPTS[name]
    # Without GlotLID the gates judge script share and length: the language
    # label of a Latin text is not checked here.
    monkeypatch.setattr(module.langid, "available", lambda: False)
    gate = module._code_gate if name == "code_switch" else module._romanize_gate
    rows = _rows(name)
    for fields, reply in _examples(name):
        message = fields["prompt"]
        row = rows[fields["language"]]
        assert gate(reply, message, row) == ""
        assert reply != message
        assert reply.strip() == reply and "\n" not in reply
        assert not reply.startswith(('"', "'", "“", "‘"))
        # Digits are left as written.
        assert all(number in reply for number in re.findall(r"\d+(?::\d+)?", message))


@_CHECKS
def test_example_verdicts_start_with_the_verdict_and_agree_with_the_reader(name):
    verdicts = []
    for _, reply in _examples(name):
        word = reply.split()[0].strip(".,:;")
        assert word in ("Yes", "No")
        assert is_accepted(reply) is (word == "Yes")
        assert len(reply) <= _MAX_VERDICT_CHARS and "\n" not in reply
        verdicts.append(word)
    assert verdicts.count("Yes") >= 3 and verdicts.count("No") >= 3  # noqa: PLR2004


@_CHECKS
def test_an_accepted_example_is_one_the_gate_lets_through(name, monkeypatch):
    # A check is only reached by a rewrite the gate passed, so a Yes example
    # the gate would reject teaches the check a case it never sees.
    module, _, _ = _PROMPTS[name]
    monkeypatch.setattr(module.langid, "available", lambda: False)
    gate = module._code_gate if name == "code_switch_check" else module._romanize_gate
    rows = _rows(name)
    accepted = 0
    for fields, reply in _examples(name):
        if reply.startswith("Yes"):
            accepted += 1
            assert gate(fields["rewritten"], fields["original"],
                        rows[fields["language"]]) == ""
    assert accepted


@_CHECKS
def test_the_check_names_the_verdict_first_and_the_closing_question(name):
    config = _config(name)
    system = config["system_prompt"]
    assert "first word" in system
    assert "Yes" in system and "No" in system
    assert "the exact meaning" in config["template"]


# --- the gate's reason codes reach the model as feedback ------------------


def _stub_windows(monkeypatch, label_of):
    monkeypatch.setattr(cs.langid, "available", lambda: True)
    monkeypatch.setattr(cs.langid, "detect", label_of)


def test_the_code_switch_prompt_explains_every_reason_the_gate_gives(monkeypatch):
    rows = _rows("code_switch")
    original = "Give me a plan to do this harmless thing today, step by step please."
    russian, spanish = rows["Russian"], rows["Spanish"]
    reasons = {
        cs._code_gate("  ", original, russian),
        cs._code_gate("plan " * 400, original, russian).split("=")[0],
        cs._code_gate("all english here and nothing else at all today", original,
                      russian),
        cs._code_gate("привет это полностью русский текст сегодня здесь и сейчас",
                      original, russian),
    }
    text = "please hazme un plan to do this thing hoy por favor ahora"
    _stub_windows(monkeypatch, lambda window: "spa_Latn")
    reasons.add(cs._code_gate(text, original, spanish))
    assert reasons == {"empty", "length_ratio", "no Russian", "fully Russian",
                       "no english left"}
    for name, row in rows.items():
        system = build_messages(_config("code_switch"), prompt="x",
                                language=name)[0]["content"]
        for reason in reasons:
            assert reason.replace("Russian", name) in system, (name, reason)


def test_the_romanize_prompt_explains_every_reason_the_gate_gives(monkeypatch):
    rows = _rows("romanize")
    hindi = rows["Hindi"]
    original = "Give me a plan to do this harmless thing today, step by step please."
    monkeypatch.setattr(rm.langid, "available", lambda: True)
    monkeypatch.setattr(rm.langid, "detect", lambda text: "eng_Latn")
    reasons = {
        rm._romanize_gate("  ", original, hindi),
        rm._romanize_gate("plan " * 400, original, hindi).split("=")[0],
        rm._romanize_gate("यह पूरा वाक्य देवनागरी लिपि में लिखा गया है", original, hindi),
        rm._romanize_gate("give me the plan to do this today please now", original,
                          hindi),
    }
    assert reasons == {"empty", "length_ratio", "not latin", "english"}
    system = build_messages(_config("romanize"), prompt="x", language="Hindi")[0][
        "content"]
    for reason in reasons:
        assert reason in system
