"""Translationese: the pivot round trip on the translation model, reusing the
translation machinery, with fake clients. The round-trip output is the English
back-translation."""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.obfuscation import translationese as te
from redact.jailbreak.obfuscation.translationese import get_translationese_functions
from redact.jailbreak.pool import (
    TRANSLATIONESE,
    all_techniques,
    registry,
    technique_pool,
    translationese_names,
    uses_translation,
)
from redact.jailbreak.rules import compatible
from redact.jailbreak.augmentation import resolve
from redact.llm_pipeline.drive import drive_sync

_ORIGINAL = "Please explain how to bake sourdough bread at home, step by step."


def test_translationese_is_tagged_as_a_translation_round_trip():
    fns = get_translationese_functions()
    assert [fn.__name__ for fn in fns] == ["to_translationese"]
    fn = fns[0]
    assert fn.requires_llm and fn.check == "translate" and fn.calls == [3, 16]
    assert fn.families == ["translationese"] and fn.complexity == 2  # noqa: PLR2004
    assert fn.kinds == ["adversarial"] and fn.since == "1.5"
    assert fn.within_layer_order == 16  # noqa: PLR2004


def test_the_pivot_is_drawn_from_the_rng():
    assert set(te.PIVOTS) and all(p in te.MEDIAN_RATIO for p in te.PIVOTS)
    assert all(p in te.LANGUAGE_CODES and p in te.ACCEPTED_LABELS for p in te.PIVOTS)
    # Deterministic with a seeded rng.
    assert te._pivot({"rng": random.Random(0)}) in te.PIVOTS


# --- the switch and the pool ----------------------------------------------

def test_translationese_runs_on_the_translation_model_and_the_switch_gates_it():
    assert "to_translationese" in translationese_names()
    assert uses_translation([registry(all_techniques())["to_translationese"]])
    # Switch on by default in the attacks: it is in the pool.
    on = technique_pool(resolve(None, "adversarial"))
    assert "to_translationese" in {fn.__name__ for fn in on}
    # Switch off: removed, like translation and dialect.
    off = technique_pool(resolve({"adversarial": {"translation": False}}, "adversarial"))
    assert "to_translationese" not in {fn.__name__ for fn in off}
    # It is adversarial only, never in the natural pool.
    assert "to_translationese" not in {
        fn.__name__ for fn in technique_pool(resolve({"natural": {"translation": True}},
                                                     "natural"))}


def test_an_allow_list_naming_translationese_with_the_switch_off_is_dropped(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="redact.jailbreak.augmentation"):
        resolved = resolve({"adversarial": {
            "translation": False,
            "families": {"allow": ["translationese", "encode"]}}}, "adversarial")
    assert "translationese" not in (resolved["families"]["allow"] or [])
    assert "translationese" in caplog.text
    pool = technique_pool(resolved)
    assert "to_translationese" not in {fn.__name__ for fn in pool}
    assert not uses_translation(pool)


def test_a_techniques_allow_list_naming_translationese_with_the_switch_off(caplog):
    """The technique named by itself, not only its family, is taken out of
    the allow list and named in the WARNING; an allow list of it alone
    raises, as one of translations alone does."""
    import logging
    with caplog.at_level(logging.WARNING, logger="redact.jailbreak.augmentation"):
        resolved = resolve({"adversarial": {
            "translation": False,
            "techniques": {"allow": ["to_translationese", "to_base64"]}}},
            "adversarial")
    assert resolved["techniques"]["allow"] == ["to_base64"]
    assert "to_translationese" in caplog.text
    with pytest.raises(ValueError, match="allows only translation"):
        resolve({"adversarial": {"translation": False,
                                 "techniques": {"allow": ["to_translationese"]}}},
                "adversarial")


# --- the rules (both ways) -------------------------------------------------

@pytest.mark.parametrize("partner", ["to_base64", "to_rewrite_with_typos_low",
                                     "to_swahili", "to_mathprompt",
                                     "to_romanize_hindi", "to_ascii_art"])
def test_translationese_refuses_the_whole_text_families_both_ways(partner):
    reg = registry(all_techniques())
    te_fn, other = reg["to_translationese"], reg[partner]
    assert other not in compatible([te_fn], [other])
    assert te_fn not in compatible([other], [te_fn])


# --- the round trip --------------------------------------------------------

class _MT:
    """A fake machine translator (role machine_translation) and judge: the
    forward leg (English -> pivot) returns ``forward``, the back leg
    (pivot -> English) returns ``back``, the judge returns ``verdict``."""
    def __init__(self, forward, back, verdict):
        self.forward, self.back, self.verdict, self.calls = forward, back, verdict, 0

    def __call__(self, request):
        self.calls += 1
        if request.model == "judge":
            return self.verdict
        body = request.messages[0]["content"]
        return self.back if self.forward in body else self.forward


@pytest.fixture
def _mt_env(monkeypatch):
    """Drive the machine-translation path with a fixed pivot and the language
    check stubbed off (so only the length band and the unchanged check gate)."""
    from redact.jailbreak.obfuscation import translation as tr
    monkeypatch.setattr(te, "is_machine_translation", lambda model: True)
    monkeypatch.setattr(te, "_resolve_translate_model", lambda: "mt")
    monkeypatch.setattr(te, "_pivot", lambda kwargs: "french")
    monkeypatch.setattr(tr.langid, "detect", lambda text: None)


def test_a_machine_translation_round_trip_returns_the_english_back_translation(_mt_env):
    forward = "Veuillez expliquer comment faire du pain au levain a la maison."
    back = "Please explain how one bakes sourdough bread at home in steps."
    fake = _MT(forward, back, "Yes, same meaning.")
    text, info = drive_sync(
        registry(all_techniques())["to_translationese"](
            _ORIGINAL, translate_model="mt", gen_model="judge"),
        fake)
    assert text == back and info == "pivot=french"


def test_an_unchanged_round_trip_is_discarded(_mt_env):
    forward = "Veuillez expliquer la fabrication du pain au levain a la maison."
    fake = _MT(forward, _ORIGINAL, "Yes")   # back == original -> unchanged
    text, info = drive_sync(
        registry(all_techniques())["to_translationese"](
            _ORIGINAL, translate_model="mt", gen_model="judge"),
        fake)
    assert info.startswith("DISCARDED") and "translationese" in info and "unchanged" in info
