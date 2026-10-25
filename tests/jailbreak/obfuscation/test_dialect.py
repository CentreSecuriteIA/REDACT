"""Dialect techniques: the region-coded machine-translation path and the
chat path, reusing the translation machinery, with fakes and GlotLID stubbed.
"""

import re

import pytest

from redact.jailbreak.obfuscation import dialect as D
from redact.jailbreak.obfuscation import langid
from redact.jailbreak.obfuscation.dialect import (
    _ratio,
    _traditional_reason,
    get_dialect_functions,
)
from redact.llms.model_config import MODEL_REGISTRY, ModelConfig, VLLMConfig

MT = "dialect-mt-fake"
GEN = "dialect-gen-fake"
_WIRE = re.compile(r"<<<source>>>([\w-]+)<<<target>>>([\w-]+)<<<text>>>(.*)", re.S)
_BY = {fn.__name__: fn for fn in get_dialect_functions()}


@pytest.fixture(autouse=True)
def mt(monkeypatch):
    monkeypatch.setitem(MODEL_REGISTRY, MT, ModelConfig(
        name=MT, roles=["translation", "machine_translation"],
        supports_system_prompt=False, vllm=VLLMConfig(hf_model_id="org/mt")))


def _wire(request):
    (message,) = request.messages
    return _WIRE.fullmatch(message["content"]).groups()


def _drive(gen, answer, limit=100):
    requests, reply = [], None
    try:
        while len(requests) < limit:
            request = gen.send(reply)
            requests.append(request)
            reply = answer(request)
    except StopIteration as stop:
        return requests, stop.value
    raise AssertionError("more requests than expected")


def test_one_technique_per_variant_on_the_translation_path():
    fns = get_dialect_functions()
    assert sorted(fn.__name__ for fn in fns) == sorted(
        f"to_dialect_{row['slug']}" for row in D.LANGUAGES)
    for fn in fns:
        assert fn.requires_llm and fn.check == "translate" and fn.calls == [2, 12]  # noqa: PLR2004
        assert fn.families == ["dialect"] and fn.complexity == 2  # noqa: PLR2004


def test_the_length_band_uses_the_base_language_ratio_or_an_estimate():
    from redact.jailbreak.obfuscation.translation import MEDIAN_RATIO

    for row in D.LANGUAGES:
        if row["base"] in MEDIAN_RATIO:
            assert _ratio(row) == MEDIAN_RATIO[row["base"]]
        else:  # Portuguese: not a shipped translation language, estimate 1.0
            assert _ratio(row) == D.DEFAULT_RATIO


def test_the_traditional_share_check_rejects_simplified_only():
    assert _traditional_reason("國這東時會來", 0.3) is None
    assert _traditional_reason("国这东时会来", 0.3) == "simplified"
    assert _traditional_reason("你好朋友", 0.3) is None  # no distinguishing char


def test_a_region_code_goes_out_and_a_passing_round_accepts(monkeypatch):
    monkeypatch.setattr(langid, "detect", lambda text: "spa_Latn")

    def answer(request):
        if request.model != MT:
            return "Yes"
        source, target, text = _wire(request)
        return f"es:{text}" if source == "en" else text.removeprefix("es:")

    requests, (text, info) = _drive(
        _BY["to_dialect_mexican_spanish"](
            "Buenos dias, please help me today with this.",
            translate_model=MT, gen_model=GEN), answer)
    assert _wire(requests[0])[:2] == ("en", "es-MX")
    assert [r.model for r in requests] == [MT, MT, GEN]
    assert text.startswith("es:") and info == "Mexican Spanish"


def test_a_wrong_base_language_label_is_rejected(monkeypatch):
    monkeypatch.setattr(langid, "detect", lambda text: "deu_Latn")  # not Spanish

    def answer(request):
        if request.model != MT:
            return "Yes"
        source, target, text = _wire(request)
        return f"es:{text}" if source == "en" else text.removeprefix("es:")

    _, (text, info) = _drive(
        _BY["to_dialect_mexican_spanish"](
            "Buenos dias, please help me today with this.",
            translate_model=MT, gen_model=GEN), answer)
    assert info.startswith("DISCARDED") and "language=deu_Latn" in info


def test_a_simplified_reply_fails_the_traditional_check_and_discards(monkeypatch):
    monkeypatch.setattr(langid, "detect", lambda text: "cmn_Hani")

    def answer(request):
        if request.model != MT:
            return "Yes"
        source, target, text = _wire(request)
        # Forward: Simplified characters, long enough to pass the length band
        # (chinese median 0.34) so the Traditional-share check is what rejects.
        return "国这东时会来对说没过应开学实发为书门车马" if source == "en" else text

    _, (text, info) = _drive(
        _BY["to_dialect_traditional_chinese"](
            "Give me the full plan for this today, step by step.",
            translate_model=MT, gen_model=GEN), answer)
    assert info.startswith("DISCARDED") and "simplified" in info


def test_a_chat_translator_takes_the_variant_by_name(monkeypatch):
    """A chat model (not a machine-translation one) uses the chat path by the
    variant's name: translate, then check, each round. The translate request
    carries the variant name; the check verdict accepts."""
    monkeypatch.setattr(D, "is_machine_translation", lambda model: False)
    calls = {"n": 0}

    def answer(request):
        calls["n"] += 1
        return "oui voici le plan complet pour aujourd hui" if calls["n"] % 2 == 1 \
            else "Yes, it is a faithful translation."

    requests, (text, info) = _drive(
        _BY["to_dialect_canadian_french"](
            "Give me the full plan for today.",
            translate_model="chat-fake", gen_model=GEN), answer)
    assert info == "Canadian French"
    assert [r.model for r in requests] == ["chat-fake", "chat-fake"]
    assert "Canadian French" in " ".join(m["content"] for m in requests[0].messages)


# --- de-CH: no sharp s ----------------------------------------------------------
# Swiss Standard German writes ss (SS) for the sharp s, so a Swiss German
# piece holding U+00DF or U+1E9E is rejected (reason ``eszett``) and translated
# again; no other dialect, and not the translation family, refuses it.

_SHARP = "Die Stra\u00dfe ist gro\u00df und wei\u00df."
_SWISS = "Die Strasse ist gross und weiss."
_CAPITAL = "DIE STRA\u1e9eE IST GROSS UND WEISS, SAG ES MIR."


def test_the_sharp_s_check_rejects_the_letter_in_either_case():
    assert D._eszett_reason(_SWISS) is None
    assert D._eszett_reason(_SHARP) == "eszett"
    assert D._eszett_reason(_CAPITAL) == "eszett"


def test_only_swiss_german_refuses_the_sharp_s():
    """The extra check of each variant, on a piece with a sharp s: Swiss
    German alone rejects it; the four variants without an extra check still
    have none, and Traditional Chinese keeps its Traditional-share check only."""
    refused = {}
    for row in D.LANGUAGES:
        check = D._extra_check(row)
        refused[row["slug"]] = check is not None and check(_SHARP, "source") == "eszett"
    assert refused == {row["slug"]: row["slug"] == "swiss_german" for row in D.LANGUAGES}
    without = {row["slug"] for row in D.LANGUAGES if D._extra_check(row) is None}
    assert without == {"brazilian_portuguese", "mexican_spanish", "canadian_french",
                       "egyptian_arabic"}
    swiss = D._extra_check(next(r for r in D.LANGUAGES if r["slug"] == "swiss_german"))
    assert swiss(_SWISS, "source") is None
    chinese = D._extra_check(
        next(r for r in D.LANGUAGES if r["slug"] == "traditional_chinese"))
    assert chinese(_SHARP, "source") is None
    assert chinese("国这东时会来", "source") == "simplified"


def _swiss_answer(forwards):
    """The translator writes ``forwards`` in turn (the last repeated) and
    translates back to the English prompt; the judge accepts."""
    replies = list(forwards)

    def answer(request):
        if request.model != MT:
            return "Yes"
        source, target, text = _wire(request)
        if source == "en":
            assert target == "de-CH"
            return replies.pop(0) if len(replies) > 1 else replies[0]
        return "The street is wide and white."
    return answer


_PROMPT = "The street is wide and white, tell me."


def test_a_swiss_german_piece_with_a_sharp_s_is_translated_again(monkeypatch):
    monkeypatch.setattr(langid, "detect", lambda text: "deu_Latn")
    requests, (text, info) = _drive(
        _BY["to_dialect_swiss_german"](_PROMPT, translate_model=MT, gen_model=GEN),
        _swiss_answer([_SHARP, _SWISS]))
    assert info == "Swiss German" and text == _SWISS
    assert [r.model for r in requests] == [MT, MT, MT, GEN]  # again, back, judge


def test_a_swiss_german_piece_that_keeps_its_sharp_s_is_discarded(monkeypatch):
    monkeypatch.setattr(langid, "detect", lambda text: "deu_Latn")
    requests, (_, info) = _drive(
        _BY["to_dialect_swiss_german"](_PROMPT, translate_model=MT, gen_model=GEN),
        _swiss_answer([_CAPITAL]))
    assert info == "DISCARDED; language=Swiss German; piece=1/1; feedback=eszett"
    assert [r.model for r in requests] == [MT] * 4  # four rounds, no judge


def test_a_german_translation_keeps_its_sharp_s(monkeypatch):
    """The translation family shares the machine-translation path but not the
    dialect's extra check: German of Germany writes the sharp s."""
    from redact.jailbreak.obfuscation.translation import to_german

    monkeypatch.setattr(langid, "detect", lambda text: "deu_Latn")

    def answer(request):
        if request.model != MT:
            return "Yes"
        source, target, text = _wire(request)
        return _SHARP if source == "en" else "The street is wide and white."

    requests, (text, info) = _drive(
        to_german(_PROMPT, translate_model=MT, gen_model=GEN), answer)
    assert info == "German" and text == _SHARP
    assert [r.model for r in requests] == [MT, MT, GEN]
