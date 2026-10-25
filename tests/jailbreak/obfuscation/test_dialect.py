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
