"""``scripts/langmix_smoke.py``: the reason codes ``_reason`` reports from a
technique's ``info`` (synthetic DISCARDED strings, no model or prompt text)."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "langmix_smoke", Path(__file__).parents[1] / "scripts" / "langmix_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accepted_info_is_accepted(script):
    assert script._reason("language=Spanish") == "accepted"
    assert script._reason("") == "accepted"


@pytest.mark.parametrize("info,expected", [
    # A dialect/translation piece: the gate reason is in feedback=.
    ("DISCARDED; language=German (Switzerland); piece=1/1; feedback=eszett",
     "eszett"),
    ("DISCARDED; language=Traditional Chinese; piece=1/1; feedback=simplified",
     "simplified"),
    ("DISCARDED; language=Mexican Spanish; piece=2/3; feedback=language=por_Latn",
     "language=por_Latn"),
    ("DISCARDED; language=Hinglish; feedback=length_ratio=2.40",
     "length_ratio=2.40"),
    # The back-translation judge.
    ("DISCARDED; language=Yoruba; piece=all; feedback=judge=No the meaning drifted",
     "check:judge"),
    # A meaning-check verdict (code-switch / romanize).
    ("DISCARDED; reason=code switch failed; language=Spanish; "
     "feedback=No, it changed the request", "check:no"),
    # A pre-call refusal carries only a fixed reason=, no feedback.
    ("DISCARDED; reason=too short to mix; language=Spanish", "too_short_to_mix"),
    # A romanize gate that rejects plain English.
    ("DISCARDED; reason=romanize failed; language=Hinglish; feedback=english",
     "english"),
])
def test_reason_reads_the_real_gate_or_check_code(script, info, expected):
    assert script._reason(info) == expected


def test_reason_prints_no_feedback_sentence(script):
    # Only the first token of a verdict is ever returned, never the sentence.
    out = script._reason(
        "DISCARDED; language=Spanish; feedback=No because the secret phrase leaked")
    assert out == "check:no"
    assert "secret" not in out
