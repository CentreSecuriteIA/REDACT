"""Semantic re-encoding: the per-form code gates, the applicability of the
acrostic technique, and the generate-gate-check loop with a fake client."""

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation import semantic_reencoding as sr
from redact.jailbreak.obfuscation.semantic_reencoding import (
    get_semantic_reencoding_functions,
)
from redact.jailbreak.pool import all_techniques, registry
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import applies
from redact.llm_pipeline.drive import drive_sync

_BY = {fn.__name__: fn for fn in get_semantic_reencoding_functions()}
_ORIGINAL = "Explain how to bake sourdough bread at home, step by step please."


def test_the_four_techniques_are_tagged_as_checked_model_calls():
    fns = get_semantic_reencoding_functions()
    assert sorted(fn.__name__ for fn in fns) == [
        "to_adversarial_poetry", "to_infoflood", "to_mathprompt",
        "to_stego_paragraph"]
    for fn in fns:
        assert fn.requires_llm and fn.check == "reencode"
        assert fn.calls == [2, 4] and fn.families == ["semantic_reencoding"]
        assert fn.kinds == ["adversarial"]   # adversarial only, no natural pool
        assert fn.since == "1.5"
    assert _BY["to_stego_paragraph"].complexity == 3  # noqa: PLR2004
    assert _BY["to_mathprompt"].complexity == 2  # noqa: PLR2004


# --- the code gates --------------------------------------------------------

def test_the_math_gate_requires_math_notation():
    assert sr._math_gate("  ", _ORIGINAL) == "empty"
    assert sr._math_gate(_ORIGINAL, _ORIGINAL) == "unchanged"
    assert sr._math_gate("just some plain words about bread", _ORIGINAL) \
        == "no math notation"
    assert sr._math_gate("Let S = {s_1} and f: I -> L; solve for S.", _ORIGINAL) == ""


def test_the_poetry_gate_requires_four_short_lines():
    assert sr._poetry_gate("", _ORIGINAL) == "empty"
    assert sr._poetry_gate("one\ntwo", _ORIGINAL).startswith("too few lines")
    long = "\n".join(["x" * 200] * 4)
    assert sr._poetry_gate(long, _ORIGINAL) == "not verse"
    assert sr._poetry_gate("a short line\nanother one\na third here\nand four",
                           _ORIGINAL) == ""


def test_the_stego_gate_checks_the_acrostic_exactly():
    original = "Name three plants here"
    assert sr._acrostic_words(original) == ["Name", "three", "plants", "here"]
    good = "Name tags were shared. Three friends came. Plants lined the sill. Here we sat."
    assert sr._stego_gate(good, original) == ""
    # wrong first word
    bad = "Hello there, everyone. Three friends came. Plants grew. Here we sat."
    assert sr._stego_gate(bad, original) == "acrostic mismatch"
    # wrong number of sentences
    short = "Name tags were shared. Three friends came."
    assert sr._stego_gate(short, original).startswith("acrostic count=")
    assert sr._stego_gate("   ", original) == "empty"


def test_the_infoflood_gate_requires_a_longer_restatement():
    assert sr._infoflood_gate("", _ORIGINAL) == "empty"
    assert sr._infoflood_gate(_ORIGINAL, _ORIGINAL) == "unchanged"
    # same length is not an overload
    assert sr._infoflood_gate("y" * len(_ORIGINAL), _ORIGINAL).startswith(
        "length_ratio=")
    # within the band
    assert sr._infoflood_gate("y" * (len(_ORIGINAL) * 3), _ORIGINAL) == ""
    # a runaway reply is rejected too
    assert sr._infoflood_gate("y" * (len(_ORIGINAL) * 50), _ORIGINAL).startswith(
        "length_ratio=")


def test_stego_applies_only_to_a_request_of_a_few_words():
    fn = _BY["to_stego_paragraph"]
    assert not applies(fn, "run", "harmful")           # one word: too few
    assert applies(fn, "name three plants", "harmful")  # three words: ok
    assert not applies(fn, "word " * 60, "harmful")    # far too many words
    # the other three have no applicability
    for name in ("to_mathprompt", "to_adversarial_poetry", "to_infoflood"):
        assert applies(_BY[name], "x", "harmful")


# --- the rules (both ways) -------------------------------------------------

#: The obfuscation families a whole-text semantic re-encoding refuses, each a
#: representative technique, plus a translation and a word-level family.
_REFUSED = {
    "to_base64": "encode", "to_rot13": "encode", "to_ascii_art": "ascii_art",
    "to_rewrite_with_typos_low": "typos", "to_lowercase": "casual_writing",
    "to_payload_split": "payload_split", "to_swahili": "translation",
}


@pytest.mark.parametrize("partner", sorted(_REFUSED))
def test_a_semantic_reencoding_refuses_the_whole_text_families_both_ways(partner):
    """A semantic re-encoding is a whole-text rewrite: it refuses the same
    obfuscation families a translation does, and translation itself, in either
    pick order."""
    reg = registry(all_techniques())
    reencode, other = reg["to_mathprompt"], reg[partner]
    # re-encoding picked first: the partner may not join.
    assert other not in compatible([reencode], [other])
    # the partner picked first: the re-encoding may not join.
    assert reencode not in compatible([other], [reencode])


@pytest.mark.parametrize("partner", ["to_adversarial_poetry", "to_infoflood",
                                     "to_stego_paragraph"])
def test_the_semantic_reencodings_refuse_each_other_both_ways(partner):
    reg = registry(all_techniques())
    math, other = reg["to_mathprompt"], reg[partner]
    assert other not in compatible([math], [other])
    assert math not in compatible([other], [math])


# --- the generate-gate-check loop -----------------------------------------

class _Fake:
    def __init__(self, gen_reply: str, verdict: str):
        self.gen_reply, self.verdict, self.calls = gen_reply, verdict, 0

    def __call__(self, request) -> str:
        self.calls += 1
        return self.verdict if request.model == "chk" else self.gen_reply


def test_a_reencoding_the_check_accepts_returns_the_text_and_form():
    reply = "Let S = (s_1, ..., s_n) map I to L; solve for S and describe it."
    fake = _Fake(reply, "Yes, it encodes the request.")
    text, info = drive_sync(
        _BY["to_mathprompt"](_ORIGINAL, gen_model="gen", check_model="chk"), fake)
    assert text == reply and info == "form=math problem"
    assert fake.calls == 2  # noqa: PLR2004  (one generation, one check)


def test_the_gate_rejecting_discards_before_a_check_call():
    fake = _Fake("plain words, no math at all here", "Yes")
    text, info = drive_sync(
        _BY["to_mathprompt"](_ORIGINAL, gen_model="gen", check_model="chk"), fake)
    assert info.startswith("DISCARDED") and "reencode failed" in info
    assert "form=math problem" in info
    assert fake.calls == 2  # noqa: PLR2004  (two generations, no check: the gate rejected)


def test_the_check_rejecting_feeds_back_and_then_discards():
    fake = _Fake("Let x in S and f(x) = y; solve.", "No, it drops a detail.")
    text, info = drive_sync(
        _BY["to_mathprompt"](_ORIGINAL, gen_model="gen", check_model="chk"), fake)
    assert info.startswith("DISCARDED") and "No, it drops a detail" in info
    assert fake.calls == 4  # noqa: PLR2004  (two gen + two check over two attempts)


def test_stego_passes_the_acrostic_and_the_check_reads_the_paragraph():
    original = "Name three plants"   # -> Name, three, plants
    reply = "Name tags were shared at the door. Three friends found seats. Plants lined the sill."
    fake = _Fake(reply, "Yes, natural prose.")
    text, info = drive_sync(
        _BY["to_stego_paragraph"](original, gen_model="gen", check_model="chk"), fake)
    assert text == reply and info == "form=acrostic paragraph"


@pytest.mark.parametrize("name", ["to_mathprompt", "to_adversarial_poetry",
                                  "to_stego_paragraph", "to_infoflood"])
def test_the_check_model_falls_back_to_the_generation_model(name):
    """With no check model named, the generation model both writes and checks:
    every request goes to ``gen``."""
    good = {
        "to_mathprompt": "Let S = {x_1}; f: I -> L; solve for S.",
        "to_adversarial_poetry": "one small line\ntwo small line\nthree line\nfour line",
        "to_stego_paragraph": "Explain this now. How about that. To the end.",
        "to_infoflood": "Within the broad literature, " + "context " * 20 + "please.",
    }[name]
    original = "Explain how to" if name == "to_stego_paragraph" else _ORIGINAL

    models, seq = set(), [good, "Yes."]

    def call(request):
        models.add(request.model)
        return seq.pop(0) if seq else "Yes."

    drive_sync(_BY[name](original, gen_model="gen"), call)
    assert models == {"gen"}
