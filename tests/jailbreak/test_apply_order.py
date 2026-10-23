"""The order a combination applies its techniques in.

A fresh plan sorts each draw with ``in_layer_order``: by layer (``rewrite``,
``hacking``, ``obfuscation``, ``requests``, ``manipulation``), then, in the
obfuscation layer, by each family's ``within_layer_order``, the whole-text
rewrites before the word-level steps. A kept plan replays the order its
manifest recorded, whatever the spec says now.
"""

import codecs

import pandas as pd
import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.chain import combine_techniques, in_layer_order, run_chain
from redact.jailbreak.plan import combination_of
from redact.jailbreak.pool import all_techniques, registry
from redact.jailbreak.variants import variants_of
from redact.llm_pipeline import drive_sync
from tests.conftest import MockBackend
from tests.jailbreak import simulate
from tests.jailbreak.test_pipeline_jailbreaks import GEN, _llm_run

BY_NAME = registry(all_techniques())

#: One technique of each obfuscation family, by family.
ONE_PER_FAMILY = {
    "translation": "to_thai", "typos": "to_rewrite_with_typos_low",
    "casual_writing": "to_lowercase", "noise": "to_keyboard_typos_low",
    "sensitive_words": "to_sensitive_words_split",
    "tokenbreak": "to_tokenbreak_split", "suffixes": "to_adversarial_suffix_fragments",
    "encode": "to_base64", "structural": "to_json", "ascii_art": "to_ascii_art",
}


def _families_in_order(names) -> list[str]:
    ordered = in_layer_order(BY_NAME[name] for name in names)
    return [fn.families[0] for fn in ordered]


def test_the_whole_text_rewrites_apply_before_the_word_level_steps():
    assert _families_in_order(reversed(ONE_PER_FAMILY.values())) == [
        "translation", "typos", "casual_writing", "noise", "sensitive_words",
        "tokenbreak", "suffixes", "encode", "structural", "ascii_art"]


def test_the_families_eval_draws_keep_their_order():
    """Translation, then encode, structural, ascii_art, as before the
    renumbering, so an eval combination applies in the same order."""
    eval_families = ["ascii_art", "structural", "encode", "translation"]
    names = [ONE_PER_FAMILY[family] for family in eval_families]
    assert _families_in_order(names) == [
        "translation", "encode", "structural", "ascii_art"]


def test_a_neutral_container_applies_where_structural_does():
    """After the encodings, before ASCII art. The two containers share their
    place: they exclude each other, so they never meet in a combination."""
    assert (BY_NAME["to_csv_row"].within_layer_order
            == BY_NAME["to_json"].within_layer_order)
    assert _families_in_order(
        ["to_ascii_art", "to_csv_row", "to_base64", "to_thai"]) == [
        "translation", "encode", "neutral_container", "ascii_art"]


def test_a_word_encoding_takes_the_order_of_its_first_family():
    """The ``to_sensitive_words_encode_*`` are in ``sensitive_words`` and
    ``encode``; ``sensitive_words`` comes first in their spec entries."""
    word_encodings = [fn for name, fn in BY_NAME.items()
                      if name.startswith("to_sensitive_words_encode_")]
    assert len(word_encodings) == 10     # noqa: PLR2004
    assert {fn.within_layer_order for fn in word_encodings} == {
        BY_NAME["to_sensitive_words_split"].within_layer_order}


def test_a_kept_plan_names_its_techniques_in_the_order_it_recorded():
    """A plan made before the renumbering applied sensitive words first."""
    recorded = ["to_sensitive_words_variables", "to_uppercase"]
    combined = combination_of(recorded, BY_NAME)
    assert [fn.__name__ for fn in combined.techniques] == recorded
    assert [fn.__name__ for fn in in_layer_order(combined.techniques)] == [
        "to_uppercase", "to_sensitive_words_variables"]


def test_a_kept_plan_runs_its_techniques_in_the_order_it_recorded(
        tmp_path, monkeypatch):
    """Sensitive words, then upper case: the definitions clause the first
    step adds is upper-cased too, which the fresh order would not do."""
    chain = (BY_NAME["to_sensitive_words_variables"], BY_NAME["to_uppercase"])
    out = _llm_run(tmp_path, monkeypatch, [("a word here", chain)],
                   MockBackend("word", model=GEN), techniques=list(chain))
    (row,) = pd.read_csv(out).to_dict("records")
    assert row["technique"] == "to_sensitive_words_variables+to_uppercase"
    assert row["jailbreak"] == "A VAR_1 HERE\n\nWHERE VAR_1 = WORD"


# --- the manipulation layer, last -------------------------------------------------
# Since the eval release the manipulation layer applies last: the obfuscations,
# the request and the request obfuscation change the question alone, and fsh,
# dap or a fake turn put the finished question among their examples or after
# their invented turn. A selected fsh or dap variant picks its examples on the
# clean input (``source_prompt``). A kept plan replays the order it recorded.

_PROBE = "How do plants turn sunlight into food?"
_SUBCATEGORIES = ("Cooking", "Gardening")
_PAIRS = [{"main_category": "M", "sub_category": sub, "prompt": f"plain question {i}",
           "answer": f"plain answer {i}"} for sub in _SUBCATEGORIES for i in range(8)]
_BY_SUB = {sub: [pair for pair in _PAIRS if pair["sub_category"] == sub]
           for sub in _SUBCATEGORIES}
_BENIGN = {"short": _PAIRS, "long": _PAIRS, "all_subcategories": list(_SUBCATEGORIES),
           "by_subcat_short": _BY_SUB, "by_subcat_long": _BY_SUB}
_SEED = (1, "jailbreak", "id0", 0)


def _rot13(text: str) -> str:
    return codecs.encode(text, "rot_13")


def _no_call(request):
    raise AssertionError("no model is asked")


def _run(techniques, call=_no_call, **kwargs) -> tuple[str, str]:
    """``techniques`` run in the order given on the probe, as a unit runs them."""
    return drive_sync(run_chain(list(techniques), _PROBE, benign_data=_BENIGN,
                                rng_seed=_SEED, **kwargs), call)


def test_the_manipulation_layer_applies_after_the_obfuscations_and_the_request():
    assert _families_in_order([
        "to_fsh_random_short", "to_use_past_tense", "to_json", "to_rot13"]) == [
        "encode", "structural", "temporal", "fsh"]


def test_a_fresh_plan_applies_its_manipulation_technique_last():
    rows, pool = simulate.plan_kind({}, simulate.inputs(300, groups=3))
    by = registry(pool)
    held = [row["combination"] for row in rows if len(row["combination"]) > 1
            and any(by[name].layer == "manipulation" for name in row["combination"])]
    assert len(held) > 50     # noqa: PLR2004
    assert [names for names in held if by[names[-1]].layer != "manipulation"] == []


def test_a_kept_plan_inserts_the_question_where_it_recorded_it():
    """A plan made before the manipulation layer moved last applied fsh
    first, so the cipher after it turned the examples too, and a replay does
    so again; a fresh plan's order turns the question alone."""
    recorded = combination_of(["to_fsh_random_short", "to_rot13"], BY_NAME).techniques
    fresh = in_layer_order(recorded)
    assert [fn.__name__ for fn in fresh] == ["to_rot13", "to_fsh_random_short"]

    def flags(techniques) -> tuple[bool, bool]:
        text, _ = _run(techniques)
        return "plain question" in text, _rot13(_PROBE) in text

    assert (flags(recorded), flags(fresh)) == ((False, True), (True, True))


@pytest.mark.parametrize("name", [
    "to_fsh_random_short", "to_dap_random_long", "to_dap_random_many",
    "to_fake_history"])
def test_a_variant_that_asks_no_model_takes_the_finished_question(name):
    text, _ = _run([BY_NAME["to_rot13"], BY_NAME[name]])
    assert (_PROBE in text, _rot13(_PROBE) in text) == (False, True)


@pytest.mark.parametrize("name", [
    "to_fsh_selected_short", "to_fsh_selected_long", "to_fsh_selected_many",
    "to_dap_selected_short", "to_dap_selected_long", "to_dap_selected_many"])
def test_a_selected_variant_picks_on_the_clean_input_and_inserts_the_encoded_one(name):
    asked = []

    def call(request):
        asked.append(request)
        return "Gardening"

    text, info = _run([BY_NAME["to_rot13"], BY_NAME[name]], call, gen_model="gen")
    (selection,) = asked
    shown = "\n".join(message["content"] for message in selection.messages)
    assert (_PROBE in shown, _rot13(_PROBE) in shown) == (True, False)
    assert (_PROBE in text, _rot13(_PROBE) in text) == (False, True)
    assert "sub_category=Gardening;selection=selected" in info


def test_a_selected_variant_picks_on_the_clean_input_of_a_translated_question():
    """The rules keep a translation from fsh and dap, but the pick reads the
    chain's input whatever ran before it; a stand-in translator and checker
    answer here."""
    replies = {"translator": "translated stand-in", "checker": "Yes",
               "gen": "Gardening"}
    asked = []

    def call(request):
        asked.append(request)
        return replies[request.model]

    text, _ = _run([BY_NAME["to_swahili"], BY_NAME["to_fsh_selected_short"]], call,
                   gen_model="gen", translate_model="translator",
                   translate_check_model="checker")
    (selection,) = [request for request in asked if request.model == "gen"]
    shown = "\n".join(message["content"] for message in selection.messages)
    assert (_PROBE in shown, "translated stand-in" in shown) == (True, False)
    assert (_PROBE in text, "translated stand-in" in text) == (False, True)


def test_a_kept_plan_with_fsh_first_picks_on_what_it_picked_on_before():
    """Its fsh runs on the input itself, so the model is shown what the
    variant alone, without a ``source_prompt``, shows it."""
    def shown(gen) -> list:
        seen = []
        drive_sync(gen, lambda request: seen.append(request.messages) or "Gardening")
        return seen

    kept = shown(run_chain([BY_NAME["to_fsh_selected_short"], BY_NAME["to_rot13"]],
                           _PROBE, gen_model="gen", benign_data=_BENIGN))
    alone = shown(BY_NAME["to_fsh_selected_short"](
        _PROBE, gen_model="gen", benign_data=_BENIGN))
    assert kept == alone


@pytest.mark.parametrize("order", ["fresh", "kept"])
def test_each_variant_of_a_row_goes_to_the_technique_that_drew_it(order):
    """``variants_of`` gives the n-th ``variant=`` of a row's info to the
    n-th of its techniques that writes one. The chain writes the infos and
    the row joins the names in one order, the applied one, so a request and
    the fake turn now after it each keep their own, as they do in a kept
    plan's order."""
    request, turn = BY_NAME["to_refusal_then_comply"], BY_NAME["to_fake_history"]
    techniques = (combine_techniques(turn, request).techniques if order == "fresh"
                  else [turn, request])
    assert (techniques[-1] is turn) == (order == "fresh")
    _, info = _run(techniques)
    drawn = variants_of("+".join(fn.__name__ for fn in techniques), info, BY_NAME)
    alone = [variants_of(fn.__name__, _run([fn])[1], BY_NAME)[0] for fn in techniques]
    assert drawn == alone
