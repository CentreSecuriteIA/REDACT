"""Code-switching: the mix gate (script share, GlotLID windows), the length
gate, and the generate-gate-check loop, with a fake client and GlotLID
stubbed."""

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation import code_switching as cs
from redact.jailbreak.obfuscation.code_switching import (
    _code_gate,
    get_code_switching_functions,
)
from redact.llm_pipeline.drive import drive_sync

_BY = {fn.__name__: fn for fn in get_code_switching_functions()}
_LANG = {row["slug"]: row for row in cs.LANGUAGES}
_ORIGINAL = "Give me a plan to do this harmful thing today, step by step please."


def test_one_technique_per_language_tagged_as_a_checked_model_call():
    fns = get_code_switching_functions()
    assert sorted(fn.__name__ for fn in fns) == sorted(
        f"to_code_switch_{row['slug']}" for row in cs.LANGUAGES)
    for fn in fns:
        assert fn.requires_llm and fn.check == "meaning" and fn.calls == [2, 4]  # noqa: PLR2004
        assert fn.families == ["code_switching"] and fn.complexity == 2  # noqa: PLR2004


def test_the_non_latin_gate_wants_a_mix_not_english_and_not_a_full_translation():
    ru = _LANG["russian"]
    assert _code_gate("   ", _ORIGINAL, ru) == "empty"
    assert _code_gate("this is all english here and nothing else at all today",
                      _ORIGINAL, ru) == "no Russian"
    assert _code_gate("this is a план to do the взрыв thing today please here",
                      _ORIGINAL, ru) == ""
    assert _code_gate("привет это полностью русский текст сегодня здесь и сейчас",
                      _ORIGINAL, ru) == "fully Russian"


def test_the_length_gate_rejects_a_runaway_or_a_dropped_reply():
    ru = _LANG["russian"]
    assert _code_gate("план " * 400, _ORIGINAL, ru).startswith("length_ratio=")
    assert _code_gate("план", _ORIGINAL, ru).startswith("length_ratio=")


def _stub_glotlid(monkeypatch, labels: dict[str, str], available=True):
    """GlotLID that reports ``labels[window]`` (``eng_Latn`` by default)."""
    monkeypatch.setattr(cs.langid, "available", lambda: available)
    monkeypatch.setattr(cs.langid, "detect",
                        lambda window: labels.get(window, "eng_Latn"))


def test_the_latin_gate_uses_glotlid_windows(monkeypatch):
    es = _LANG["spanish"]
    text = "please hazme un plan to do this thing hoy por favor ahora"
    windows = cs._windows(text)
    # Half the windows Spanish, half English: a genuine mix passes.
    labels = {w: ("spa_Latn" if i % 2 else "eng_Latn")
              for i, w in enumerate(windows)}
    _stub_glotlid(monkeypatch, labels)
    assert _code_gate(text, _ORIGINAL, es) == ""
    # Every window English: no Spanish at all.
    _stub_glotlid(monkeypatch, {})
    assert _code_gate(text, _ORIGINAL, es) == "no Spanish"
    # Every window Spanish: the English is gone.
    _stub_glotlid(monkeypatch, {w: "spa_Latn" for w in windows})
    assert _code_gate(text, _ORIGINAL, es) == "no english left"


def test_without_glotlid_the_latin_mix_check_is_skipped(monkeypatch):
    es = _LANG["spanish"]
    _stub_glotlid(monkeypatch, {}, available=False)
    # No detector: the mix cannot be verified, so only empty/length gate.
    assert _code_gate("whatever text of a fair length goes here today", _ORIGINAL,
                      es) == ""


class _Fake:
    """A blocking call for drive_sync: the gen reply for a generation request,
    a verdict for a check request (told apart by the model it names)."""

    def __init__(self, gen_reply: str, verdict: str):
        self.gen_reply, self.verdict, self.calls = gen_reply, verdict, 0

    def __call__(self, request) -> str:
        self.calls += 1
        return self.verdict if request.model == "chk" else self.gen_reply


def test_a_good_mix_the_check_accepts_returns_the_text_and_the_language():
    fn = _BY["to_code_switch_russian"]
    good = "this is a план to do the взрыв thing today please here"
    text, info = drive_sync(fn(_ORIGINAL, gen_model="gen", check_model="chk"),
                            _Fake(good, "Yes, the meaning is preserved."))
    assert text == good and info == "language=Russian"


def test_the_gate_rejects_before_any_check_call_and_discards():
    fn = _BY["to_code_switch_russian"]
    fake = _Fake("this is all english and never switches at all here today", "Yes")
    text, info = drive_sync(fn(_ORIGINAL, gen_model="gen", check_model="chk"), fake)
    assert info.startswith("DISCARDED") and "code switch failed" in info
    assert "language=Russian" in info
    # Two attempts, each a generation only (the gate rejects before the check).
    assert fake.calls == 2  # noqa: PLR2004


def test_the_check_rejecting_every_time_discards():
    fn = _BY["to_code_switch_russian"]
    good = "this is a план to do the взрыв thing today please here"
    text, info = drive_sync(fn(_ORIGINAL, gen_model="gen", check_model="chk"),
                            _Fake(good, "No, it changed the meaning."))
    assert info.startswith("DISCARDED") and "code switch failed" in info
