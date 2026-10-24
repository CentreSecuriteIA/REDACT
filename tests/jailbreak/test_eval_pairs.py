"""Which pairs of the eval pool's 36 techniques may share a combination, pinned.

The eval type's attacks draw their combinations from a pool of 36 techniques
(``fixtures/eval_pool.json``, pinned), and the rules decide which two of them
may be drawn together. The pool stays, but a rule that is added, dropped or
narrowed changes the pairs the eval draws, and so which attacks an eval run
holds. ``fixtures/eval_pairs.json`` records the pairs ``rules.compatible``
admits, ``"<technique>|<technique>"``, the two names in alphabetical order,
so a rule that moves the eval shows as a diff of the file, which these
tests name. The eval type draws only these pairs (its count rounds draw
exactly one technique, then exactly two).

An eval review that changes the pairs regenerates the fixture from the
repository root, with ``PYTHONPATH=src python -m tests.jailbreak.test_eval_pairs``,
and says in its commit which pairs went.
"""

import json
from pathlib import Path

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.chain import in_layer_order
from redact.jailbreak.pool import registry, technique_pool
from redact.jailbreak.rules import compatible
from redact.recipe import load_dataset_type

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_PAIRS = _FIXTURES / "eval_pairs.json"
_POOL = _FIXTURES / "eval_pool.json"


def _eval_pool() -> list:
    """The techniques the eval type's attacks draw from, as it resolves."""
    return technique_pool(resolve(
        load_dataset_type("eval")["augmentations"], "adversarial"))


def snapshot() -> list[str]:
    """The pairs of the eval pool that the rules admit, sorted."""
    pool = _eval_pool()
    return sorted(
        "|".join(sorted((first.__name__, second.__name__)))
        for i, first in enumerate(pool) for second in pool[i + 1:]
        if second in compatible([first], [second]))


def _dumps(pairs: list[str]) -> str:
    """The fixture as JSON, one pair per line."""
    rows = ",\n".join(f"  {json.dumps(pair)}" for pair in pairs)
    return (
        '{\n "_comment": "Regenerate with PYTHONPATH=src py -m '
        'tests.jailbreak.test_eval_pairs. The pairs of the 36 eval pool '
        'techniques that rules.compatible admits.",\n'
        f' "pairs": [\n{rows}\n ]\n}}\n')


def test_the_eval_pool_is_the_pinned_36_names():
    assert sorted(fn.__name__ for fn in _eval_pool()) == json.loads(
        _POOL.read_text(encoding="utf-8"))


def test_the_pairs_of_the_eval_pool_the_rules_admit_are_as_pinned():
    recorded = set(json.loads(_PAIRS.read_text(encoding="utf-8"))["pairs"])
    current = set(snapshot())
    assert (sorted(current - recorded), sorted(recorded - current)) == ([], []), (
        "pairs the eval now draws, and pairs it no longer can; if a rule is "
        "meant to move the eval, regenerate with "
        "PYTHONPATH=src py -m tests.jailbreak.test_eval_pairs")


def test_the_fixture_is_what_the_regeneration_writes():
    assert _PAIRS.read_text(encoding="utf-8") == _dumps(snapshot())


@pytest.mark.parametrize("translation", [
    "to_bengali", "to_javanese", "to_khmer", "to_nepali", "to_scots_gaelic",
    "to_swahili", "to_thai", "to_zulu"])
@pytest.mark.parametrize("encoding", [
    "to_ascii_ordinal", "to_base64", "to_braille", "to_morse", "to_unicode_escape"])
def test_the_eval_draws_no_translation_beside_an_encoding(translation, encoding):
    assert "|".join(sorted((translation, encoding))) not in snapshot()


def test_the_eval_pool_has_no_word_marking_step():
    """So a rule between the families that have the model mark words
    (``sensitive_words``, ``tokenbreak``, ``word_art``, ``wording``) is a
    training rule: the eval pairs and draws do not move with it."""
    marking = {"sensitive_words", "tokenbreak", "word_art", "wording"}
    assert [fn.__name__ for fn in _eval_pool() if marking & set(fn.families)] == []


@pytest.mark.parametrize("encoding", [
    "to_ascii_ordinal", "to_base64", "to_braille", "to_morse", "to_unicode_escape"])
def test_the_eval_draws_no_ascii_art_beside_a_heavy_encoding(encoding):
    assert "|".join(sorted(("to_ascii_art", encoding))) not in snapshot()


def test_the_eval_s_fsh_and_dap_insert_the_question_last_beside_five_partners():
    """The manipulation layer applies last: the 40 pairs that hold one of the
    eight ``fsh`` and ``dap`` techniques put the finished question, wrapped
    or with its request, among the examples. Each was applied fsh or dap
    first before the eval release, so these 40 changed their text, and no
    other pair did."""
    by = registry(_eval_pool())
    pairs = [pair.split("|") for pair in snapshot()]
    held = [pair for pair in pairs
            if any(by[name].layer == "manipulation" for name in pair)]
    partners = {name for pair in held for name in pair
                if by[name].layer != "manipulation"}
    assert (len(held), partners) == (40, {
        "to_json", "to_markdown", "to_refusal_then_comply", "to_use_past_tense",
        "to_xml"})
    assert [pair for pair in held
            if in_layer_order(by[name] for name in pair)[-1].layer != "manipulation"
            ] == []


def test_ascii_art_still_joins_a_suffix_a_request_and_a_scene_in_the_eval_pool():
    joined = {pair for pair in snapshot() if "to_ascii_art" in pair.split("|")}
    assert {"to_ascii_art|to_complete_story", "to_ascii_art|to_refusal_then_comply",
            "to_ascii_art|to_use_past_tense"} <= joined


if __name__ == "__main__":
    _PAIRS.write_text(_dumps(snapshot()), encoding="utf-8")
