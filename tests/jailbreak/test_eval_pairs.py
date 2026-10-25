"""Which pairs of the eval pool's techniques may share a combination, pinned.

The eval type's attacks draw their combinations from a pinned pool
(``fixtures/eval_pool.json``: 126 techniques in 22 families since eval pool
v2), and the rules decide which two of them may be drawn together. The pool
stays, but a rule that is added, dropped or narrowed changes the pairs the
eval draws, and so which attacks an eval run holds. The eval type draws only
these pairs (its count rounds draw exactly one technique, then exactly two).

``fixtures/eval_pairs.json`` records the pairs ``rules.compatible`` admits in
a form a diff can show at 126 techniques (5,613 of 7,875 pairs, too many to
review one per line): ``families``, each family of the pool with its
techniques, and ``family_pairs``, for each two families of the pool (a family
with itself too, ``"<family>|<family>"``, the two names in alphabetical
order) whether ``"all"`` of their pairs are admitted, ``"none"``, or, when
some are, the pairs ``"admitted"`` or ``"refused"``, whichever list is the
shorter, ``"<technique>|<technique>"`` in alphabetical order. Together they
give every pair, so a rule that moves one pair of the eval changes the line
of its two families, and a technique that joins or leaves a family the line
of that family. ``first_pool_pairs`` lists, one per line as the fixture did
before eval pool v2, the pairs of the 36 techniques of the first eval pool
(:data:`FIRST_POOL`), which v2 holds whole: they did not move with v2, and a
rule that moves one shows there too.

An eval review that changes the pairs regenerates the fixture from the
repository root, with ``PYTHONPATH=src python -m tests.jailbreak.test_eval_pairs``,
and says in its commit which pairs went.
"""

import itertools
import json
from collections import defaultdict
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

#: The 36 techniques of the first eval pool (eval releases 1 and 2), which
#: eval pool v2 holds whole; ``fixtures/spec_1_3_pools.json`` recorded them
#: as a spec 1.3 eval plan did.
FIRST_POOL = (
    "to_ascii_art", "to_ascii_ordinal", "to_base64", "to_bengali", "to_braille",
    "to_complete_story", "to_continue_instructions", "to_continue_passage",
    "to_dap_random_long", "to_dap_random_short", "to_dap_selected_long",
    "to_dap_selected_short", "to_fictional_world", "to_fill_in_blank",
    "to_fsh_random_long", "to_fsh_random_short", "to_fsh_selected_long",
    "to_fsh_selected_short", "to_high_stake", "to_javanese", "to_json",
    "to_khmer", "to_markdown", "to_morse", "to_nefarious_goal", "to_nepali",
    "to_no_moral_constraints", "to_noble_goal", "to_refusal_then_comply",
    "to_scots_gaelic", "to_swahili", "to_thai", "to_unicode_escape",
    "to_use_past_tense", "to_xml", "to_zulu")


def _eval_pool() -> list:
    """The techniques the eval type's attacks draw from, as it resolves."""
    return technique_pool(resolve(
        load_dataset_type("eval")["augmentations"], "adversarial"))


def _pair(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def snapshot() -> list[str]:
    """The pairs of the eval pool that the rules admit, sorted."""
    pool = _eval_pool()
    return sorted(
        _pair(first.__name__, second.__name__)
        for i, first in enumerate(pool) for second in pool[i + 1:]
        if second in compatible([first], [second]))


def _families(pool: list) -> dict[str, list[str]]:
    """Each first family of ``pool`` with its techniques, sorted."""
    members = defaultdict(list)
    for fn in pool:
        members[fn.families[0]].append(fn.__name__)
    return {family: sorted(names) for family, names in sorted(members.items())}


def _pairs_between(members: dict[str, list[str]], first: str, second: str) -> list[str]:
    """Every pair of a technique of ``first`` with one of ``second`` (two of
    ``first`` when they are one family), sorted."""
    if first == second:
        pairs = itertools.combinations(members[first], 2)
    else:
        pairs = itertools.product(members[first], members[second])
    return sorted(_pair(*pair) for pair in pairs)


def record(pairs: list[str], members: dict[str, list[str]]) -> dict:
    """The fixture's record of the admitted ``pairs`` of the families
    ``members``: what :func:`expand` reads back."""
    admitted = set(pairs)
    verdicts = {}
    for first, second in itertools.combinations_with_replacement(sorted(members), 2):
        between = _pairs_between(members, first, second)
        if not between:
            continue
        yes = [pair for pair in between if pair in admitted]
        no = [pair for pair in between if pair not in admitted]
        verdicts[f"{first}|{second}"] = (
            "all" if not no else "none" if not yes
            else {"admitted": yes} if len(yes) <= len(no) else {"refused": no})
    return {"admitted": len(admitted),
            "of": sum(len(names) for names in members.values()) * (
                sum(len(names) for names in members.values()) - 1) // 2,
            "families": members, "family_pairs": verdicts}


def expand(recorded: dict) -> set[str]:
    """Every admitted pair the record ``recorded`` gives."""
    members, admitted = recorded["families"], set()
    for key, verdict in recorded["family_pairs"].items():
        first, second = key.split("|")
        between = _pairs_between(members, first, second)
        if verdict == "all":
            admitted.update(between)
        elif isinstance(verdict, dict) and "admitted" in verdict:
            admitted.update(verdict["admitted"])
        elif isinstance(verdict, dict) and "refused" in verdict:
            admitted.update(set(between) - set(verdict["refused"]))
    return admitted


def first_pool_pairs(pairs: list[str]) -> list[str]:
    """Those of ``pairs`` between two techniques of :data:`FIRST_POOL`."""
    first = set(FIRST_POOL)
    return [pair for pair in pairs if set(pair.split("|")) <= first]


def _dumps(pairs: list[str]) -> str:
    """The fixture as JSON: one family, family pair or first-pool pair per
    line."""
    data = record(pairs, _families(_eval_pool()))

    def lines(mapping: dict) -> str:
        return ",\n".join(f"  {json.dumps(key)}: {json.dumps(value)}"
                          for key, value in mapping.items())

    listed = ",\n".join(f"  {json.dumps(pair)}" for pair in first_pool_pairs(pairs))
    return (
        '{\n "_comment": "Regenerate with PYTHONPATH=src py -m '
        'tests.jailbreak.test_eval_pairs. The pairs of the eval pool that '
        'rules.compatible admits: each family with its techniques, then for '
        'each two families all, none, or the pairs admitted or refused, the '
        'shorter list; then the pairs of the 36 techniques of the first eval '
        'pool, one per line.",\n'
        f' "admitted": {data["admitted"]},\n'
        f' "of": {data["of"]},\n'
        f' "families": {{\n{lines(data["families"])}\n }},\n'
        f' "family_pairs": {{\n{lines(data["family_pairs"])}\n }},\n'
        f' "first_pool_pairs": [\n{listed}\n ]\n}}\n')


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(_PAIRS.read_text(encoding="utf-8"))


def test_the_eval_pool_is_the_pinned_names():
    assert sorted(fn.__name__ for fn in _eval_pool()) == json.loads(
        _POOL.read_text(encoding="utf-8"))


def test_the_pairs_of_the_eval_pool_the_rules_admit_are_as_pinned(recorded):
    current = set(snapshot())
    held = expand(recorded)
    assert (sorted(current - held), sorted(held - current)) == ([], []), (
        "pairs the eval now draws, and pairs it no longer can; if a rule is "
        "meant to move the eval, regenerate with "
        "PYTHONPATH=src py -m tests.jailbreak.test_eval_pairs")
    assert recorded["admitted"] == len(current) == 5613    # noqa: PLR2004
    assert recorded["of"] == 7875     # 126 techniques, two at a time  # noqa: PLR2004


def test_the_fixture_is_what_the_regeneration_writes():
    assert _PAIRS.read_text(encoding="utf-8") == _dumps(snapshot())


def test_the_record_gives_back_every_pair_and_only_those():
    """``expand`` reads back what ``record`` writes, so a pair that moves
    moves its family pair's line: here one admitted pair dropped, one
    refused pair added, and a family's whole verdict turned."""
    pairs = snapshot()
    members = _families(_eval_pool())
    assert expand(record(pairs, members)) == set(pairs)
    fewer = pairs[1:]
    assert expand(record(fewer, members)) == set(fewer)
    assert (record(fewer, members)["family_pairs"]
            != record(pairs, members)["family_pairs"])
    refused = sorted(set(_pairs_between(members, "translation", "encode")))
    more = sorted({*pairs, refused[0]})
    assert expand(record(more, members)) == set(more)
    assert record(more, members)["family_pairs"]["encode|translation"] == {
        "admitted": [refused[0]]}


def test_four_family_pairs_are_mixed_and_each_admits_the_letter_spacing_alone():
    """Of the 22 families' 253 family pairs (250 with a pair: a family of
    one technique has none with itself), the rules admit all or none of the
    pairs of all but four: the letter spacing, the light encoding the pool
    holds, joins ASCII art and the three manipulation families, which refuse
    its five heavy full-text encodings."""
    current = record(snapshot(), _families(_eval_pool()))
    mixed = {key: verdict for key, verdict in current["family_pairs"].items()
             if isinstance(verdict, dict)}
    assert mixed == {
        "ascii_art|encode": {"admitted": ["to_ascii_art|to_letter_spacing"]},
        "dap|encode": {"admitted": [_pair(name, "to_letter_spacing") for name in (
            "to_dap_random_long", "to_dap_random_short", "to_dap_selected_long",
            "to_dap_selected_short")]},
        "encode|fake_turns": {"admitted": [
            _pair(name, "to_letter_spacing") for name in (
                "to_fake_assistant_turn", "to_fake_history", "to_fake_system_turn")]},
        "encode|fsh": {"admitted": [_pair(name, "to_letter_spacing") for name in (
            "to_fsh_random_long", "to_fsh_random_short", "to_fsh_selected_long",
            "to_fsh_selected_short")]},
    }
    assert len(current["family_pairs"]) == 250   # noqa: PLR2004


def test_eval_pool_v2_holds_the_first_pool_whose_pairs_did_not_move(recorded):
    """Eval pool v2 only admits techniques: the 36 of the first pool stay,
    and the 291 pairs of them the rules admit are those the first eval
    release left (no rule changed with v2)."""
    assert set(FIRST_POOL) < set(json.loads(_POOL.read_text(encoding="utf-8")))
    assert recorded["first_pool_pairs"] == first_pool_pairs(snapshot())
    assert len(recorded["first_pool_pairs"]) == 291     # noqa: PLR2004


def test_the_first_pool_is_the_one_a_spec_1_3_eval_plan_recorded():
    spec_1_3 = json.loads(
        (_FIXTURES / "spec_1_3_pools.json").read_text(encoding="utf-8"))
    assert sorted(spec_1_3["eval:adversarial"]) == sorted(FIRST_POOL)
    assert len(FIRST_POOL) == 36    # noqa: PLR2004


@pytest.mark.parametrize("translation", [
    "to_bengali", "to_javanese", "to_khmer", "to_nepali", "to_scots_gaelic",
    "to_swahili", "to_thai", "to_zulu"])
@pytest.mark.parametrize("encoding", [
    "to_ascii_ordinal", "to_base64", "to_braille", "to_letter_spacing", "to_morse",
    "to_unicode_escape"])
def test_the_eval_draws_no_translation_beside_an_encoding(translation, encoding):
    assert _pair(translation, encoding) not in snapshot()


def test_the_eval_pool_has_no_word_marking_step():
    """So a rule between the families that have the model mark words
    (``sensitive_words``, ``tokenbreak``, ``word_art``, ``wording``) is a
    training rule: the eval pairs and draws do not move with it."""
    marking = {"sensitive_words", "tokenbreak", "word_art", "wording"}
    assert [fn.__name__ for fn in _eval_pool() if marking & set(fn.families)] == []


@pytest.mark.parametrize("encoding", [
    "to_ascii_ordinal", "to_base64", "to_braille", "to_morse", "to_unicode_escape"])
def test_the_eval_draws_no_ascii_art_beside_a_heavy_encoding(encoding):
    assert _pair("to_ascii_art", encoding) not in snapshot()


def test_the_eval_s_fsh_and_dap_insert_the_question_last_beside_five_partners():
    """The manipulation layer applies last: the 40 pairs of the first eval
    pool that hold one of the eight ``fsh`` and ``dap`` techniques put the
    finished question, wrapped or with its request, among the examples. Each
    was applied fsh or dap first before the eval release, so these 40
    changed their text, and no other pair of that pool did. Eval pool v2
    pairs them with more partners, each applied before them too."""
    by = registry(_eval_pool())
    pairs = [pair.split("|") for pair in snapshot()]
    held = [pair for pair in pairs
            if any(by[name].layer == "manipulation" for name in pair)]
    first = [pair for pair in held if set(pair) <= set(FIRST_POOL)]
    partners = {name for pair in first for name in pair
                if by[name].layer != "manipulation"}
    assert (len(first), partners) == (40, {
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
