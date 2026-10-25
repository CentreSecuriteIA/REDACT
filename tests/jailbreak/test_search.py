"""The combination search (``redact.jailbreak.search``): the building blocks,
and an end-to-end search with fakes — validity, mutation, the beam and early
stop. Resume and the oversized-prompt row are in ``test_search_resume.py``.
"""

from __future__ import annotations

import pytest

from redact.dataset import content_id
from redact.jailbreak import search_combinations, search_tree
from redact.jailbreak.search import (
    _beam,
    _budget_of,
    _draw_round0,
    _is_valid,
    _mutations,
    _mutations_by_kind,
    _plan_round,
)

from ._search_fakes import (
    POOL,
    PROMPT,
    REGISTRY,
    applied_text,
    inputs,
    make_resolve,
    pure_pool,
)

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


def _node(iid, cand, score, n, ok, applied=True):
    return {"input_id": iid, "round": 0, "candidate": cand, "score": score,
            "num_techniques": n, "success": ok, "applied": applied}


def test_beam_keeps_the_best_scoring_candidates():
    rows = [
        _node("x", "a", 0.0, 1, False),
        _node("x", "b", 1.0, 2, True),
        _node("x", "c", 0.0, 2, False),
        _node("y", "z", 1.0, 1, True),
    ]
    assert _beam(rows, "x", 0, beam=1) == ["b"]      # the only success wins
    # Score first, then fewest techniques: b (success), then a (1 < 2).
    assert _beam(rows, "x", 0, beam=2) == ["b", "a"]


def test_beam_breaks_ties_by_a_seeded_draw_not_the_name():
    # Three singles, all failed, same size: the order is a seeded draw, so it is
    # not the alphabetical one for at least one seed, and is deterministic.
    rows = [_node("x", c, 0.0, 1, False) for c in ("a", "b", "c")]
    orders = {tuple(_beam(rows, "x", 0, beam=3, seed=s)) for s in range(12)}
    assert ("a", "b", "c") not in orders or len(orders) > 1   # not fixed alphabetical
    assert _beam(rows, "x", 0, beam=3, seed=5) == _beam(rows, "x", 0, beam=3, seed=5)


def test_beam_excludes_identity_and_non_applied_candidates():
    rows = [
        _node("x", "identity", 0.0, 0, False, applied=True),
        _node("x", "to_bad", 0.0, 1, False, applied=False),   # discarded/error
        _node("x", "to_ok", 0.0, 1, False, applied=True),
    ]
    assert _beam(rows, "x", 0, beam=3) == ["to_ok"]


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


def test_plan_round_reads_only_earlier_rounds(tmp_path):
    # The resume-bug fix: round r's plan reads only rounds < r. A success and
    # other candidates saved IN round r (as a killed run leaves) must not change
    # round r's plan — otherwise a resume would skip the rest of the round.
    iid = content_id(PROMPT)
    input_rows = {iid: {"sample_id": iid, "prompt": PROMPT, "category": "c",
                        "entry_type": "harmful"}}
    budget = _budget_of(SAMPLE_KWARGS)

    def row(r, cand, score=0.0, applied=True):
        return {"input_id": iid, "round": r, "candidate": cand, "score": score,
                "num_techniques": 0 if cand == "identity" else len(cand.split("+")),
                "success": score > 0, "applied": applied}

    r0 = _draw_round0(POOL, iid, PROMPT, "harmful", seed=42, width=4,
                      sample_kwargs=SAMPLE_KWARGS)
    base = [row(0, c) for c in r0]
    clean = _plan_round(1, input_rows, POOL, REGISTRY, base, beam=2, width=4,
                        seed=42, sample_kwargs=SAMPLE_KWARGS, stop_on_success=True,
                        max_candidates=8, budget=budget)
    assert clean                                             # some mutations
    # A killed run that saved a success and others in round 1, plus a round-2 row.
    poisoned = base + [row(1, clean[0][2], score=1.0),
                       row(1, clean[-1][2]), row(2, "identity")]
    resumed = _plan_round(1, input_rows, POOL, REGISTRY, poisoned, beam=2, width=4,
                          seed=42, sample_kwargs=SAMPLE_KWARGS, stop_on_success=True,
                          max_candidates=8, budget=budget)
    assert resumed == clean


def test_max_candidates_caps_the_round(tmp_path):
    pool = pure_pool()
    out = search_combinations(
        inputs(PROMPT), target_model="target", judge_model="judge", pool=pool,
        out_path=tmp_path / "cap.csv", width=4, beam=2, depth=1, max_candidates=5,
        resolve=make_resolve("target", "judge", set()), verbose=False)
    r1 = out["rows"][out["rows"]["round"] == 1]
    # One input, so round 1 holds at most max_candidates rows.
    assert 0 < len(r1) <= 5
    # Uncapped, the same round holds far more (the fan-out F3 describes).
    full = search_combinations(
        inputs(PROMPT), target_model="target", judge_model="judge", pool=pool,
        out_path=tmp_path / "full.csv", width=4, beam=2, depth=1,
        max_candidates=None,
        resolve=make_resolve("target", "judge", set()), verbose=False)
    full1 = full["rows"][full["rows"]["round"] == 1]
    assert len(full1) > 5


def test_mutations_respect_the_round0_budget():
    # A budget round 0 caps complexity and obfuscations; a later round keeps it.
    low = _budget_of({"max_complexity": 1, "max_obfuscations": 1})
    none = None
    parent = "to_lowercase"
    capped = _mutations(parent, pure_pool(), REGISTRY, PROMPT, "harmful", "id0",
                        low)
    uncapped = _mutations(parent, pure_pool(), REGISTRY, PROMPT, "harmful", "id0",
                          none)
    assert set(capped) < set(uncapped)                       # the cap drops some
    for cand in capped:
        if cand == "identity":
            continue
        fns = [REGISTRY[n] for n in cand.split("+")]
        assert sum(getattr(f, "complexity", 0) for f in fns) <= 1
        assert sum(getattr(f, "layer", None) == "obfuscation" for f in fns) <= 1


def test_requires_a_target_and_a_judge(tmp_path):
    with pytest.raises(ValueError):
        search_combinations(inputs(PROMPT), target_model="", judge_model="judge",
                            pool=POOL, out_path=tmp_path / "s.csv", verbose=False)
    with pytest.raises(ValueError):
        search_combinations(inputs(PROMPT), target_model="t", judge_model="",
                            pool=POOL, out_path=tmp_path / "s.csv", verbose=False)
