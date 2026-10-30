"""Romanization: the Latin-and-not-English gate, and the generate-gate-check
loop, with a fake client and GlotLID stubbed. The five languages GlotLID has
no romanized label for (spec 1.5) have tests of their own, at the end."""

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.obfuscation import romanization as rm
from redact.jailbreak.obfuscation.romanization import (
    _romanize_gate,
    english_share,
    get_romanization_functions,
)
from redact.jailbreak.pool import technique_pool
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


# --- the five languages without a romanized label (spec 1.5) ---------------
#
# GlotLID v3 has no Latin-script label for Russian, Japanese, Chinese, Greek
# or Korean, so their gate reads no language: a Latin share, the length band,
# the share of distinctively English words and GlotLID's eng_Latn reading.
# The texts below are neutral sentences.

_UNLABELLED = ("russian", "japanese", "chinese", "greek", "korean")
_NEUTRAL = "Explain how the water cycle works, step by step, for a school project."
_ROMANIZED = {
    "russian": "Ob'yasni, kak rabotaet krugovorot vody, shag za shagom, dlya proekta.",
    "japanese": ("Mizu no junkan ga dou ugoku ka, gakkou no kadai no tame ni "
                 "setsumei shite."),
    "chinese": ("Qing yi bu yi bu jieshi shui xunhuan shi zenme yunzuo de, "
                "yongyu xuexiao."),
    "greek": "Exigise pos leitourgei o kyklos tou nerou, vima vima, gia mia ergasia.",
    "korean": ("Hakgyo gwaje-reul wihae mul-ui sunhwan-i eotteoke ireonaneunji "
               "seolmyeonghae."),
}
#: The neutral sentence, left in English: five of its fourteen words are
#: English markers (how, the, for, and, the).
_ENGLISH = "Explain how the water cycle works for a school project and give the steps."


def test_the_five_unlabelled_languages_have_no_label_and_are_attack_only():
    for slug in _UNLABELLED:
        row, fn = _LANG[slug], _BY[f"to_romanize_{slug}"]
        assert row["romanized_label"] is None and row["natural"] is False
        assert fn.kinds == ["adversarial"] and fn.since == "1.5"
        assert "more weakly" in fn.__doc__
    labelled = [slug for slug in _LANG if slug not in _UNLABELLED]
    assert len(labelled) == 10  # noqa: PLR2004
    for slug in labelled:
        fn = _BY[f"to_romanize_{slug}"]
        assert _LANG[slug]["romanized_label"].endswith("_Latn")
        assert fn.kinds == ["adversarial", "natural"] and fn.since == "1.4"
        assert "more weakly" not in fn.__doc__
    # A row's natural is what its technique's kinds say, every row.
    for slug, row in _LANG.items():
        assert row["natural"] is ("natural" in _BY[f"to_romanize_{slug}"].kinds)


def test_the_attacks_draw_the_five_and_the_natural_variations_do_not():
    attacks = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    for slug in _UNLABELLED:
        assert f"to_romanize_{slug}" in attacks
        assert f"to_romanize_{slug}" not in natural
    assert "to_romanize_hindi" in natural


@pytest.mark.parametrize("slug", _UNLABELLED)
def test_without_glotlid_the_english_words_still_reject_english(monkeypatch, slug):
    _stub(monkeypatch, "eng_Latn", available=False)
    assert _romanize_gate(_ENGLISH, _NEUTRAL, _LANG[slug]) == "english"
    assert _romanize_gate(_ROMANIZED[slug], _NEUTRAL, _LANG[slug]) == ""
    # A labelled language keeps its gate: without GlotLID, the Latin share only.
    assert _romanize_gate(_ENGLISH, _NEUTRAL, _LANG["hindi"]) == ""


def test_with_glotlid_the_english_words_and_eng_latn_both_reject(monkeypatch):
    ru = _LANG["russian"]
    # A label other than English passes a text with no English words: no
    # label can confirm the language, so any non-English one is let through.
    _stub(monkeypatch, "hrv_Latn")
    assert _romanize_gate(_ROMANIZED["russian"], _NEUTRAL, ru) == ""
    # The English words reject English whatever GlotLID reads.
    assert _romanize_gate(_ENGLISH, _NEUTRAL, ru) == "english"
    # GlotLID's eng_Latn rejects a text with no English marker.
    _stub(monkeypatch, "eng_Latn")
    assert _romanize_gate(_ROMANIZED["russian"], _NEUTRAL, ru) == "english"


def test_the_english_share_threshold_is_one_word_in_ten(monkeypatch):
    # One marker among ten words reaches MAX_ENGLISH_SHARE; among eleven it
    # does not. The original is short, so the length band is not read.
    _stub(monkeypatch, "hrv_Latn", available=False)
    ko = _LANG["korean"]
    at_ten = " ".join(["the"] + ["mul"] * 9)
    at_eleven = " ".join(["the"] + ["mul"] * 10)
    assert english_share(at_ten) == 0.1  # noqa: PLR2004
    assert _romanize_gate(at_ten, "Short.", ko) == "english"
    assert _romanize_gate(at_eleven, "Short.", ko) == ""


def test_the_latin_share_threshold_holds_for_the_five(monkeypatch):
    # Eight Latin letters of ten is MIN_LATIN_SHARE; seven of nine is below.
    _stub(monkeypatch, "hrv_Latn", available=False)
    ja = _LANG["japanese"]
    assert _romanize_gate("mizumizu みず", "Short.", ja) == ""
    assert _romanize_gate("mizumiz みず", "Short.", ja) == "not latin"


def test_quoted_and_bracketed_text_is_left_out_of_the_english_count(monkeypatch):
    _stub(monkeypatch, "hrv_Latn", available=False)
    gr = _LANG["greek"]
    original = 'Write a review of the book "The Name of the Rose" today.'
    quoted = 'Grapse mia kritiki gia to vivlio "The Name of the Rose" simera.'
    unquoted = "Grapse mia kritiki gia to vivlio The Name of the Rose simera."
    assert english_share(quoted) == 0.0
    assert _romanize_gate(quoted, original, gr) == ""
    assert _romanize_gate(unquoted, original, gr) == "english"
    bracketed = "Grapse mia kritiki (of the book) gia to vivlio simera."
    assert english_share(bracketed) == 0.0


def test_the_english_share_counts_markers_in_any_case_and_no_word_as_none():
    assert english_share("") == 0.0
    assert english_share("123 456") == 0.0
    assert english_share("The AND Of") == 1.0
    assert english_share("Privet, kak dela?") == 0.0


def test_an_unlabelled_romanization_left_in_english_is_discarded_unchecked(
        monkeypatch):
    _stub(monkeypatch, "eng_Latn", available=False)
    fn = _BY["to_romanize_chinese"]
    fake = _Fake(_ENGLISH, "Yes")
    text, info = drive_sync(fn(_NEUTRAL, gen_model="gen", check_model="chk"), fake)
    assert info.startswith("DISCARDED") and "feedback=english" in info
    assert fake.calls == 2  # noqa: PLR2004 - two generations, no check (the gate rejected)


def test_an_unlabelled_romanization_the_check_accepts_is_kept(monkeypatch):
    _stub(monkeypatch, "hrv_Latn", available=False)
    fn = _BY["to_romanize_greek"]
    fake = _Fake(_ROMANIZED["greek"], "Yes, the same meaning in Greek.")
    text, info = drive_sync(fn(_NEUTRAL, gen_model="gen", check_model="chk"), fake)
    assert text == _ROMANIZED["greek"] and info == "language=Greek"
    assert fake.calls == 2  # noqa: PLR2004 - one generation, one check


def test_the_check_model_alone_rejects_a_wrong_language_for_the_five(monkeypatch):
    # Nothing in code reads the language: a Korean romanization offered as
    # Japanese passes the gate and is rejected only by the check model.
    _stub(monkeypatch, "hrv_Latn", available=False)
    assert _romanize_gate(_ROMANIZED["korean"], _NEUTRAL, _LANG["japanese"]) == ""
    fn = _BY["to_romanize_japanese"]
    fake = _Fake(_ROMANIZED["korean"], "No. The rewrite is not Japanese.")
    text, info = drive_sync(fn(_NEUTRAL, gen_model="gen", check_model="chk"), fake)
    assert info.startswith("DISCARDED")
    assert fake.calls == 4  # noqa: PLR2004 - two generations, two checks
