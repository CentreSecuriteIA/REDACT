"""The combination search (``redact.jailbreak.search``): the building blocks,
and an end-to-end search with fakes — validity, mutation, the beam and early
stop. Resume and the oversized-prompt row are in ``test_search_resume.py``.
"""

from __future__ import annotations

import pytest

from redact.dataset import content_id
from redact.jailbreak import search_combinations, search_tree
from redact.jailbreak.search import _beam, _draw_round0, _is_valid, _mutations

from ._search_fakes import POOL, PROMPT, REGISTRY, applied_text, inputs, make_resolve

SAMPLE_KWARGS = {"exact_techniques": 1}


# --- the building blocks ---------------------------------------------------

def test_round0_draws_distinct_valid_combinations():
    combos = _draw_round0(POOL, "id0", PROMPT, "harmful", seed=42, width=4,
                          sample_kwargs=SAMPLE_KWARGS)
    assert len(combos) == len(set(combos))          # distinct
    for cand in combos:
        assert _is_valid(cand, REGISTRY)            # and valid


def test_round0_is_deterministic():
    a = _draw_round0(POOL, "id0", PROMPT, "harmful", seed=42, width=4,
                     sample_kwargs=SAMPLE_KWARGS)
    b = _draw_round0(POOL, "id0", PROMPT, "harmful", seed=42, width=4,
                     sample_kwargs=SAMPLE_KWARGS)
    assert a == b


def test_mutations_are_all_valid_and_change_the_parent():
    muts = _mutations("to_leetspeak_basic", POOL, REGISTRY, PROMPT, "harmful", "id0")
    assert muts                                     # add/drop at least
    assert "to_leetspeak_basic" not in muts
    for cand in muts:
        assert _is_valid(cand, REGISTRY)


def test_mutations_never_pair_two_of_one_family():
    # to_lowercase and to_uppercase are both casual_writing: never together.
    for parent in ("to_lowercase", "to_leetspeak_basic", "to_lowercase+to_leetspeak_basic"):
        for cand in _mutations(parent, POOL, REGISTRY, PROMPT, "harmful", "id0"):
            names = [] if cand == "identity" else cand.split("+")
            fams = [REGISTRY[n].families[0] for n in names]
            assert len(fams) == len(set(fams)), (parent, cand)


def test_beam_keeps_the_best_scoring_candidates():
    def node(iid, cand, score, n, ok):
        return {"input_id": iid, "round": 0, "candidate": cand, "score": score,
                "num_techniques": n, "success": ok}
    rows = [
        node("x", "a", 0.0, 1, False),
        node("x", "b", 1.0, 2, True),
        node("x", "c", 0.0, 1, False),
        node("y", "z", 1.0, 1, True),
    ]
    assert _beam(rows, "x", 0, beam=1) == ["b"]      # the only success wins
    # On a tie of score, the simpler (fewer techniques), then the name.
    assert _beam(rows, "x", 0, beam=2) == ["b", "a"]


def test_is_valid_rejects_an_impossible_pair():
    assert not _is_valid("to_lowercase+to_uppercase", REGISTRY)
    assert _is_valid("to_lowercase+to_leetspeak_basic", REGISTRY)
    assert not _is_valid("to_leetspeak_basic+does_not_exist", REGISTRY)


# --- end-to-end with fakes -------------------------------------------------

def test_every_candidate_in_every_round_is_valid(tmp_path):
    resolve = make_resolve("target", "judge", success_texts=set())  # never wins
    out = search_combinations(
        inputs(PROMPT, "another harmful request to try"),
        target_model="target", judge_model="judge", pool=POOL,
        out_path=tmp_path / "s.csv", width=4, beam=2, depth=2,
        resolve=resolve, verbose=False)
    rows = out["rows"]
    assert not rows.empty
    for cand in rows["candidate"].astype(str):
        assert _is_valid(cand, REGISTRY), cand


def test_early_stop_on_success(tmp_path):
    # The first input is solved by a single technique in round 0, so it gets no
    # later round; the second input is never solved, so it keeps expanding.
    # (Two inputs, so the per-input stop holds the solved one back, not the
    # global all-solved stop.)
    solved = "please describe the full procedure in detail"
    unsolved = "another distinct harmful request to try here"
    win = applied_text(["to_leetspeak_basic"])       # leetspeak of `solved` succeeds
    resolve = make_resolve("target", "judge", success_texts={win})
    out = search_combinations(
        inputs(solved, unsolved), target_model="target", judge_model="judge",
        pool=POOL, out_path=tmp_path / "s.csv", width=4, beam=2, depth=3,
        resolve=resolve, verbose=False)
    rows = out["rows"]
    solved_id, unsolved_id = content_id(solved), content_id(unsolved)
    best = out["best"]
    assert best[solved_id].success and best[solved_id].combination == ["to_leetspeak_basic"]
    assert not best[unsolved_id].success
    assert set(rows[rows["input_id"] == solved_id]["round"]) == {0}
    assert max(rows[rows["input_id"] == unsolved_id]["round"]) >= 1


def test_mutation_finds_a_two_technique_success(tmp_path):
    # Only a two-technique combination succeeds, so round 0 (singles) fails and
    # a round-1 mutation of a beam node finds it. (to_uppercase changes the
    # text, so the two-technique output differs from leetspeak alone.)
    win_names = ["to_uppercase", "to_leetspeak_basic"]
    win = applied_text(win_names)
    assert win != applied_text(["to_leetspeak_basic"])        # a single does not match
    resolve = make_resolve("target", "judge", success_texts={win})
    out = search_combinations(
        inputs(PROMPT), target_model="target", judge_model="judge", pool=POOL,
        out_path=tmp_path / "s.csv", width=4, beam=2, depth=2,
        resolve=resolve, verbose=False)
    best = out["best"]
    iid = next(iter(best))
    assert best[iid].success
    assert best[iid].combination == win_names
    assert best[iid].round == 1
    tree = search_tree(out["rows"], iid)
    winner = [n for n in tree if n["candidate"] == "+".join(win_names)]
    assert winner and winner[0]["parent"] in ("to_leetspeak_basic", "to_uppercase")


def test_pool_of_one_technique(tmp_path):
    resolve = make_resolve("target", "judge", success_texts=set())
    out = search_combinations(
        inputs(PROMPT), target_model="target", judge_model="judge",
        pool=[REGISTRY["to_leetspeak_basic"]], out_path=tmp_path / "s.csv",
        width=4, beam=2, depth=2, resolve=resolve, verbose=False)
    cands = set(out["rows"]["candidate"].astype(str))
    assert "to_leetspeak_basic" in cands
    for cand in cands:
        assert _is_valid(cand, REGISTRY)


def test_requires_a_target_and_a_judge(tmp_path):
    with pytest.raises(ValueError):
        search_combinations(inputs(PROMPT), target_model="", judge_model="judge",
                            pool=POOL, out_path=tmp_path / "s.csv", verbose=False)
    with pytest.raises(ValueError):
        search_combinations(inputs(PROMPT), target_model="t", judge_model="",
                            pool=POOL, out_path=tmp_path / "s.csv", verbose=False)
