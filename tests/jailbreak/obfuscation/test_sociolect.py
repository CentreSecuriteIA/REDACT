"""Sociolect: the length-and-change gate and the generate-gate-check loop with
a fake client. One technique per English variety."""

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.obfuscation import sociolect as so
from redact.jailbreak.obfuscation.sociolect import get_sociolect_functions
from redact.jailbreak.pool import all_techniques, registry, technique_pool
from redact.jailbreak.rules import compatible
from redact.llm_pipeline.drive import drive_sync

_BY = {fn.__name__: fn for fn in get_sociolect_functions()}
_ORIGINAL = "Please call me when you get home this evening, and lock the door."


def test_one_technique_per_variety_tagged_as_a_meaning_checked_call():
    fns = get_sociolect_functions()
    assert sorted(fn.__name__ for fn in fns) == sorted(
        f"to_sociolect_{row['slug']}" for row in so.VARIETIES)
    assert len(fns) == 5  # noqa: PLR2004
    for fn in fns:
        assert fn.requires_llm and fn.check == "meaning" and fn.calls == [2, 4]  # noqa: PLR2004
        assert fn.families == ["sociolect"] and fn.complexity == 2  # noqa: PLR2004
        assert fn.kinds == ["adversarial"] and fn.since == "1.5"
        assert fn.within_layer_order == 17  # noqa: PLR2004


def test_the_gate_rejects_empty_unchanged_and_out_of_band():
    assert so._sociolect_gate("  ", _ORIGINAL) == "empty"
    assert so._sociolect_gate(_ORIGINAL, _ORIGINAL) == "unchanged"
    assert so._sociolect_gate("x", _ORIGINAL).startswith("length_ratio=")
    assert so._sociolect_gate("y" * 400, _ORIGINAL).startswith("length_ratio=")
    changed = "Gie us a ring when ye get hame the night, an lock the door."
    assert so._sociolect_gate(changed, _ORIGINAL) == ""


def test_sociolect_is_adversarial_only_and_not_translation_gated():
    adv = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    assert "to_sociolect_scottish_english" in adv
    nat = {fn.__name__ for fn in technique_pool(
        resolve({"natural": {"translation": True}}, "natural"))}
    assert not any(name.startswith("to_sociolect_") for name in nat)
    # It runs on the generation model, so the translation switch does not gate it.
    off = {fn.__name__ for fn in technique_pool(
        resolve({"adversarial": {"translation": False}}, "adversarial"))}
    assert "to_sociolect_scottish_english" in off


@pytest.mark.parametrize("partner", ["to_base64", "to_rewrite_with_typos_low",
                                     "to_swahili", "to_mathprompt",
                                     "to_translationese", "to_romanize_hindi"])
def test_sociolect_refuses_the_whole_text_families_both_ways(partner):
    reg = registry(all_techniques())
    soc, other = reg["to_sociolect_scottish_english"], reg[partner]
    assert other not in compatible([soc], [other])
    assert soc not in compatible([other], [soc])


class _Fake:
    def __init__(self, gen_reply, verdict):
        self.gen_reply, self.verdict, self.calls = gen_reply, verdict, 0

    def __call__(self, request):
        self.calls += 1
        return self.verdict if request.model == "chk" else self.gen_reply


def test_a_rewrite_the_check_accepts_returns_the_text_and_variety():
    reply = "Gie us a ring when ye get hame the night, an lock the door, aye?"
    fake = _Fake(reply, "Yes, same meaning.")
    text, info = drive_sync(
        _BY["to_sociolect_scottish_english"](_ORIGINAL, gen_model="gen",
                                             check_model="chk"), fake)
    assert text == reply and info == "variety=Scottish English"


def test_the_check_reads_the_rewrite_against_the_original():
    """A meaning check must see both texts: the check request carries the
    original prompt and the rewrite."""
    reply = "Gie us a ring when ye get hame the night, an lock the door, aye?"
    seen = []

    def call(request):
        if request.model == "chk":
            seen.append("\n".join(m["content"] for m in request.messages))
            return "Yes, same meaning."
        return reply

    drive_sync(_BY["to_sociolect_scottish_english"](
        _ORIGINAL, gen_model="gen", check_model="chk"), call)
    assert len(seen) == 1
    assert _ORIGINAL in seen[0] and reply in seen[0]


def test_the_gate_rejecting_discards_before_a_check_call():
    fake = _Fake(_ORIGINAL, "Yes")   # unchanged -> gate rejects
    text, info = drive_sync(
        _BY["to_sociolect_singlish"](_ORIGINAL, gen_model="gen", check_model="chk"),
        fake)
    assert info.startswith("DISCARDED") and "sociolect failed" in info
    assert fake.calls == 2  # noqa: PLR2004  (two generations, no check)
