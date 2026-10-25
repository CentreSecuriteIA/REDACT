"""The GlotLID labels the language-mix families accept, against the labels
translation accepts for the same language.

A Latin-script code-switch counts a window as its second language by the
row's ``label`` and ``also_labels``: for a language that is also a
translation language, exactly the set translation accepts (Swahili's three).
A dialect row's ``labels`` are its base language's, with the two exceptions
``dialect_variants.json`` documents: Egyptian Arabic takes six of Arabic's
nine, and German (Switzerland) adds ``gsw_Latn``, since GlotLID may read
Swiss Standard German as Swiss German. GlotLID is faked throughout.
"""

import re

import pytest

from redact.jailbreak.obfuscation import code_switching as cs
from redact.jailbreak.obfuscation import dialect as D
from redact.jailbreak.obfuscation import langid
from redact.jailbreak.obfuscation.translation import ACCEPTED_LABELS
from redact.llms.model_config import MODEL_REGISTRY, ModelConfig, VLLMConfig

_LATIN = {row["slug"]: row for row in cs.LANGUAGES if row["script"] == "Latn"}
_DIALECT = {row["slug"]: row for row in D.LANGUAGES}


# --- code-switching ---------------------------------------------------------


def test_a_latin_code_switch_accepts_the_labels_translation_accepts():
    shared = sorted(set(_LATIN) & set(ACCEPTED_LABELS))
    assert shared == ["french", "german", "spanish", "swahili"]
    for slug in shared:
        assert cs.second_labels(_LATIN[slug]) == ACCEPTED_LABELS[slug], slug
    # Portuguese is no translation language: its own label alone.
    assert cs.second_labels(_LATIN["portuguese"]) == {"por_Latn"}


@pytest.mark.parametrize("label", ["swh_Latn", "swc_Latn", "swa_Latn"])
def test_every_swahili_label_counts_as_the_second_language(monkeypatch, label):
    """A mix whose Swahili windows GlotLID labels ``label`` passes the gate;
    the same mix with those windows in a third language does not."""
    text = "s1 s2 s3 s4 e1 e2 e3 e4"
    monkeypatch.setattr(cs.langid, "available", lambda: True)

    def detect(other):
        def label_of(window):
            first = window.split()[0]
            return other if first.startswith("s") else "eng_Latn"
        return label_of

    row = _LATIN["swahili"]
    monkeypatch.setattr(cs.langid, "detect", detect(label))
    assert cs._code_gate(text, text, row) == ""
    monkeypatch.setattr(cs.langid, "detect", detect("ita_Latn"))
    assert cs._code_gate(text, text, row) == "no Swahili"


# --- dialect ----------------------------------------------------------------


def test_the_dialect_labels_are_their_base_languages_but_two():
    for slug in ("mexican_spanish", "canadian_french", "traditional_chinese"):
        row = _DIALECT[slug]
        assert set(row["labels"]) == ACCEPTED_LABELS[row["base"]], slug
    egyptian = set(_DIALECT["egyptian_arabic"]["labels"])
    assert egyptian < ACCEPTED_LABELS["arabic"]
    assert ACCEPTED_LABELS["arabic"] - egyptian == {"acq_Arab", "aeb_Arab", "ars_Arab"}
    swiss = set(_DIALECT["german_switzerland"]["labels"])
    assert swiss == ACCEPTED_LABELS["german"] | {"gsw_Latn"}
    assert "brazilian_portuguese" not in ACCEPTED_LABELS
    assert set(_DIALECT["brazilian_portuguese"]["labels"]) == {"por_Latn", "glg_Latn"}


MT = "langmix-labels-mt-fake"
GEN = "langmix-labels-gen-fake"
_WIRE = re.compile(r"<<<source>>>([\w-]+)<<<target>>>([\w-]+)<<<text>>>(.*)", re.S)
_PROMPT = "The street is wide and white, tell me."
_SWISS = "Die Strasse ist gross und weiss."


@pytest.fixture
def mt(monkeypatch):
    monkeypatch.setitem(MODEL_REGISTRY, MT, ModelConfig(
        name=MT, roles=["translation", "machine_translation"],
        supports_system_prompt=False, vllm=VLLMConfig(hf_model_id="org/mt")))


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


def _answer(request):
    if request.model != MT:
        return "Yes"
    (message,) = request.messages
    source, _, _ = _WIRE.fullmatch(message["content"]).groups()
    return _SWISS if source == "en" else "The street is wide and white."


@pytest.mark.parametrize(("label", "accepted"), [
    ("gsw_Latn", True), ("deu_Latn", True), ("ltz_Latn", False)])
def test_a_german_switzerland_piece_read_as_swiss_german_passes(
        monkeypatch, mt, label, accepted):
    """GlotLID reading the piece ``gsw_Latn`` (Swiss German) or German keeps
    it; another label rejects it on every round."""
    monkeypatch.setattr(langid, "detect", lambda text: label)
    fn = next(f for f in D.get_dialect_functions()
              if f.__name__ == "to_dialect_german_switzerland")
    _, (text, info) = _drive(fn(_PROMPT, translate_model=MT, gen_model=GEN), _answer)
    if accepted:
        assert (text, info) == (_SWISS, "German (Switzerland)")
    else:
        assert info.startswith("DISCARDED") and f"language={label}" in info
