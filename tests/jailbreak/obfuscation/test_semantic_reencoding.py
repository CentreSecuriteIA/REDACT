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


# --- stego and the frames that lengthen its text ---------------------------

#: The hacking and rewrite families whose frame lengthens the text before the
#: acrostic is built (they apply first): ``to_stego_paragraph`` refuses them in
#: its own spec entry, since its applicability reads the plan's input while it
#: builds the acrostic from the lengthened text.
_LENGTHENING = ("framing", "persona", "cognitive", "persona_template",
                "policy_framing", "persuasion", "model_distractors")
#: Techniques of other families stego still joins: a rewrite that does not
#: lengthen, a request, a wrapper, a suffix and a manipulation technique.
_STEGO_PARTNERS = ("to_rewrite_in_past_tense", "to_refusal_then_comply", "to_xml",
                   "to_adversarial_suffix_punctuation", "to_fsh_random_short")


def _members(family: str) -> list:
    return [fn for fn in all_techniques() if family in fn.families]


def _stego():
    return registry(all_techniques())["to_stego_paragraph"]


def test_stego_names_the_lengthening_families_in_its_own_entry():
    assert sorted(_stego().cross_incompatible_with) == sorted(_LENGTHENING)
    # the rule is stego's own: the family's list does not hold them, so the
    # other three re-encodings, which build nothing from the text's words, keep
    # them (below).
    for name in ("to_mathprompt", "to_adversarial_poetry", "to_infoflood"):
        assert _BY[name].cross_incompatible_with == []


@pytest.mark.parametrize("family", _LENGTHENING)
def test_stego_refuses_every_technique_of_a_lengthening_family_both_ways(family):
    stego, members = _stego(), _members(family)
    assert members
    for other in members:
        # stego picked first: the frame may not join.
        assert other not in compatible([stego], [other]), other.__name__
        # the frame picked first: stego may not join.
        assert stego not in compatible([other], [stego]), other.__name__


@pytest.mark.parametrize("name", ["to_mathprompt", "to_adversarial_poetry",
                                  "to_infoflood"])
@pytest.mark.parametrize("family", _LENGTHENING)
def test_the_other_reencodings_still_join_the_lengthening_families(name, family):
    fn = _BY[name]
    for other in _members(family):
        assert other in compatible([fn], [other]), other.__name__
        assert fn in compatible([other], [fn]), other.__name__


@pytest.mark.parametrize("partner", _STEGO_PARTNERS)
def test_stego_still_joins_its_other_partners_both_ways(partner):
    stego, other = _stego(), registry(all_techniques())[partner]
    assert other in compatible([stego], [other])
    assert stego in compatible([other], [stego])


def test_stego_own_entry_takes_out_the_lengthening_families_and_nothing_else(
        monkeypatch):
    """Re-derived through ``rules.compatible`` with and without the entry: the
    techniques stego refuses, in either pick order, grow by the lengthening
    families' members exactly, and no other pair changes."""
    pool, stego = all_techniques(), _stego()

    def joins() -> tuple[set, set]:
        stego_first = {fn.__name__ for fn in compatible([stego], pool)}
        stego_second = {fn.__name__ for fn in pool
                        if stego in compatible([fn], [stego])}
        return stego_first, stego_second

    with_entry = joins()
    monkeypatch.setattr(stego, "cross_incompatible_with", [])
    without_entry = joins()
    lengthening = {fn.__name__ for fn in pool if set(fn.families) & set(_LENGTHENING)}
    for kept, wider in zip(with_entry, without_entry):
        assert kept <= wider
        assert wider - kept == lengthening


def _stego_plan(pool: list, n: int = 60) -> list[list[str]]:
    """The combinations of a plan of ``n`` short inputs (stego applies to
    each), a count round of two techniques and a budget round with the
    rewrite layer drawn."""
    from tests.jailbreak import simulate

    rounds = [{"exact_techniques": 2},
              {"include_rewrite": True, "max_complexity": 9}]
    return simulate.deal(pool, n, rounds=rounds,
                         layer_rates={"rewrite": 0.5, "hacking": 0.5})


def test_a_fresh_plan_never_puts_stego_beside_a_lengthening_frame():
    reg = registry(all_techniques())
    frames = [_members(family)[0] for family in _LENGTHENING]
    partners = [reg[name] for name in _STEGO_PARTNERS]
    combinations = _stego_plan([_stego(), *frames, *partners])
    frame_names = {fn.__name__ for fn in frames}
    with_stego = [c for c in combinations if "to_stego_paragraph" in c]
    assert with_stego
    assert not [c for c in with_stego if frame_names & set(c)]
    # the frames are still drawn, beside the other techniques
    assert any(frame_names & set(c) for c in combinations)
    # and stego still meets its other partners
    assert any(set(_STEGO_PARTNERS) & set(c) for c in with_stego)


def test_a_fresh_plan_of_the_default_attacks_never_puts_stego_beside_a_frame():
    from tests.jailbreak import simulate

    combinations, pool = simulate.deal_kind({"rounds": "escalation"}, 120)
    family_of = {fn.__name__: set(fn.families) for fn in pool}
    with_stego = [c for c in combinations if "to_stego_paragraph" in c]
    assert with_stego
    for names in with_stego:
        assert not set().union(*(family_of[n] for n in names)) & set(_LENGTHENING)


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


def _words(n: int) -> str:
    return " ".join(f"word{chr(97 + i % 26)}" for i in range(n))


@pytest.mark.parametrize("n", [0, 2, 41, 120])
def test_stego_discards_a_text_outside_its_length_before_any_call(n):
    """The text the step is given is checked before any model call: outside
    3 to 40 words it is returned unchanged, discarded with a fixed reason."""
    fake = _Fake("never asked", "Yes")
    text, info = drive_sync(
        _BY["to_stego_paragraph"](_words(n), gen_model="gen", check_model="chk"),
        fake)
    assert fake.calls == 0
    assert text == _words(n)
    assert info == (f"DISCARDED; reason=acrostic length; form=acrostic paragraph; "
                    f"words={n}")


@pytest.mark.parametrize("n", [3, 40])
def test_stego_asks_the_model_for_a_text_at_either_end_of_its_length(n):
    fake = _Fake("Plain prose with no acrostic at all.", "Yes")
    _text, info = drive_sync(
        _BY["to_stego_paragraph"](_words(n), gen_model="gen", check_model="chk"),
        fake)
    assert fake.calls == 2  # noqa: PLR2004  (two generations, the gate rejecting each)
    assert "reason=reencode failed" in info


@pytest.mark.parametrize("n", [2, 3, 40, 41])
def test_stego_checks_the_length_its_applicability_reads(n):
    """The step's own check and the plan's applicability read one helper, so
    a text the plan would draw stego for is never discarded for its length,
    and one it would not is."""
    fake = _Fake("Plain prose.", "Yes")
    _text, info = drive_sync(
        _BY["to_stego_paragraph"](_words(n), gen_model="gen", check_model="chk"),
        fake)
    assert applies(_BY["to_stego_paragraph"], _words(n), "harmful") == (
        "reason=acrostic length" not in info)


def test_a_kept_plan_pairing_stego_with_a_frame_discards_it_with_no_call():
    """A plan kept from before the rule replays stego after a frame: the
    frame lengthens the short input past 40 words, and stego, given that
    text, discards it before any model call (it used to make two)."""
    from redact.jailbreak.plan import combination_of
    from redact.jailbreak.units import unit_gen

    reg = registry(all_techniques())
    short = _words(30)   # stego applies; a frame of 11 words or more passes 40
    assert applies(reg["to_stego_paragraph"], short, "harmful")

    def replay(names: list[str]) -> dict:
        """A unit of a kept plan's row, as the stage builds it."""
        return {"id": "u1", "iteration": 0, "prompt": short, "seed": 42,
                "combination": combination_of(names, reg)}

    framed, _ = drive_sync(unit_gen(replay(["to_fictional_world"]), gen_model="gen"),
                           _Fake("", ""))
    assert len(sr._acrostic_words(framed)) > sr.MAX_ACROSTIC_WORDS

    fake = _Fake("never asked", "Yes")
    text, info = drive_sync(unit_gen(
        replay(["to_fictional_world", "to_stego_paragraph"]),
        gen_model="gen", check_model="chk"), fake)
    assert fake.calls == 0
    assert text == framed
    assert info.startswith("DISCARDED; technique=to_stego_paragraph; "
                           "reason=acrostic length")


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
