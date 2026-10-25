"""Romanization: the Latin-and-not-English gate, and the generate-gate-check
loop, with a fake client and GlotLID stubbed."""

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation import romanization as rm
from redact.jailbreak.obfuscation.romanization import (
    _romanize_gate,
    get_romanization_functions,
)
from redact.llm_pipeline.drive import drive_sync

_BY = {fn.__name__: fn for fn in get_romanization_functions()}
_LANG = {row["slug"]: row for row in rm.LANGUAGES}
_ORIGINAL = "Give me a plan to do this harmful thing today, step by step please."
_HINGLISH = "mujhe ye kaam karne ka poora plan chahiye aaj step by step please"


def test_one_technique_per_language_tagged_as_a_checked_model_call():
    fns = get_romanization_functions()
    assert sorted(fn.__name__ for fn in fns) == sorted(
        f"to_romanize_{row['slug']}" for row in rm.LANGUAGES)
    for fn in fns:
        assert fn.requires_llm and fn.check == "meaning" and fn.calls == [2, 4]  # noqa: PLR2004
        assert fn.families == ["romanization"] and fn.complexity == 2  # noqa: PLR2004


def _stub(monkeypatch, label: str, available=True):
    monkeypatch.setattr(rm.langid, "available", lambda: available)
    monkeypatch.setattr(rm.langid, "detect", lambda text: label)


def test_the_gate_rejects_native_script_and_english_and_accepts_romanized(monkeypatch):
    hi = _LANG["hindi"]
    assert _romanize_gate("  ", _ORIGINAL, hi) == "empty"
    # Native (Devanagari) script: not a romanization.
    _stub(monkeypatch, "hin_Deva")
    assert _romanize_gate("यह हिंदी है और यह लैटिन नहीं है बिलकुल भी",
                          _ORIGINAL, hi) == "not latin"
    # GlotLID reads the romanized label: accepted.
    _stub(monkeypatch, "hin_Latn")
    assert _romanize_gate(_HINGLISH, _ORIGINAL, hi) == ""
    # GlotLID reads English: it stayed English.
    _stub(monkeypatch, "eng_Latn")
    assert _romanize_gate(_HINGLISH, _ORIGINAL, hi) == "english"
    # Another non-English Latin label: lenient accept (thin romanized models).
    _stub(monkeypatch, "mar_Latn")
    assert _romanize_gate(_HINGLISH, _ORIGINAL, hi) == ""


def test_without_glotlid_only_the_latin_share_is_checked(monkeypatch):
    hi = _LANG["hindi"]
    _stub(monkeypatch, "hin_Latn", available=False)
    assert _romanize_gate(_HINGLISH, _ORIGINAL, hi) == ""
    assert _romanize_gate(
        "यह पूरा वाक्य देवनागरी लिपि में लिखा गया है लैटिन अक्षरों में नहीं है",
        _ORIGINAL, hi) == "not latin"


def test_the_length_gate_rejects_a_runaway_or_dropped_reply(monkeypatch):
    _stub(monkeypatch, "hin_Latn")
    hi = _LANG["hindi"]
    assert _romanize_gate("plan " * 400, _ORIGINAL, hi).startswith("length_ratio=")
    assert _romanize_gate("plan", _ORIGINAL, hi).startswith("length_ratio=")


class _Fake:
    def __init__(self, gen_reply: str, verdict: str):
        self.gen_reply, self.verdict, self.calls = gen_reply, verdict, 0

    def __call__(self, request) -> str:
        self.calls += 1
        return self.verdict if request.model == "chk" else self.gen_reply


def test_a_romanization_the_check_accepts_returns_the_text_and_language(monkeypatch):
    _stub(monkeypatch, "hin_Latn")
    fn = _BY["to_romanize_hindi"]
    text, info = drive_sync(fn(_ORIGINAL, gen_model="gen", check_model="chk"),
                            _Fake(_HINGLISH, "Yes, it means the same."))
    assert text == _HINGLISH and info == "language=Hindi"


def test_the_gate_rejecting_discards_before_a_check_call(monkeypatch):
    _stub(monkeypatch, "eng_Latn")
    fn = _BY["to_romanize_hindi"]
    fake = _Fake("give me a plan to do this today please now right here", "Yes")
    text, info = drive_sync(fn(_ORIGINAL, gen_model="gen", check_model="chk"), fake)
    assert info.startswith("DISCARDED") and "romanize failed" in info
    assert fake.calls == 2  # noqa: PLR2004 - two generations, no check (the gate rejected)
