"""Persistence of the combination search: the oversized-prompt row, resume
after a kill (same rows, no duplicates), and a resume whose settings changed.
"""

from __future__ import annotations

import contextlib

import pytest

from redact.dataset import content_id
from redact.jailbreak import best_combinations, search_combinations

from ._search_fakes import (
    POOL,
    PROMPT,
    applied_for,
    applied_text,
    inputs,
    make_resolve,
    pure_pool,
)


def test_prompt_too_long_gives_a_failed_row_not_a_crash(tmp_path):
    # The target call for the leetspeak text is too long; its unit records a failed
    # row and the search goes on.
    too_long = applied_text(["to_leetspeak_basic"])
    resolve = make_resolve("target", "judge", success_texts=set(),
                           too_long_for={too_long})
    out = search_combinations(
        inputs(PROMPT), target_model="target", judge_model="judge", pool=POOL,
        out_path=tmp_path / "s.csv", width=4, beam=1, depth=0,
        resolve=resolve, verbose=False)
    rows = out["rows"]
    row = rows[rows["candidate"] == "to_leetspeak_basic"].iloc[0]
    assert str(row["success"]).lower() == "false"
    assert "too long" in str(row["judge_reasoning"]).lower()
    # The combination applied; only the target prompt was too long. (The catch
    # records applied=True; an unhandled error would be a DISCARDED error row.)
    assert str(row["applied"]).lower() == "true"


def test_resume_after_a_kill_gives_the_same_rows(tmp_path):
    out_path = tmp_path / "s.csv"
    seeds = inputs(PROMPT, "second harmful request here", "third one as well")
    win = applied_text(["to_uppercase", "to_leetspeak_basic"])

    # A full, clean run for the reference rows.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref = search_combinations(
        seeds, target_model="target", judge_model="judge", pool=POOL,
        out_path=ref_dir / "s.csv", width=4, beam=2, depth=2,
        resolve=make_resolve("target", "judge", {win}), verbose=False)
    ref_ids = sorted(ref["rows"]["sample_id"].astype(str))

    # A run that crashes on the 3rd judge call.
    with pytest.raises(RuntimeError):
        search_combinations(
            seeds, target_model="target", judge_model="judge", pool=POOL,
            out_path=out_path, width=4, beam=2, depth=2,
            resolve=make_resolve("target", "judge", {win}, kill=[3]), verbose=False)

    # Resume to completion with a healthy fake.
    resumed = search_combinations(
        seeds, target_model="target", judge_model="judge", pool=POOL,
        out_path=out_path, width=4, beam=2, depth=2,
        resolve=make_resolve("target", "judge", {win}), verbose=False)
    got_ids = sorted(resumed["rows"]["sample_id"].astype(str))

    assert len(got_ids) == len(set(got_ids))         # no duplicates
    assert got_ids == ref_ids                        # the same rows as a clean run
    assert (best_combinations(resumed["rows"]).keys()
            == best_combinations(ref["rows"]).keys())


def test_resume_after_a_mid_round_kill_matches_a_clean_run(tmp_path):
    # With a full pure pool, one input's round 1 holds well over 32 candidates,
    # so the commit buffer flushes mid-round: a success is acked while later
    # candidates of the same round are still pending. A round's plan must read
    # only the rounds before it, or a resume would skip the rest of the round
    # (the F1 resume bug). The earlier test kills in round 0 and misses this.
    pool = pure_pool()
    seeds = inputs(PROMPT)
    kw = dict(pool=pool, width=4, beam=2, depth=2, max_candidates=None,
              target_model="target", judge_model="judge", verbose=False)

    probe = search_combinations(
        seeds, out_path=tmp_path / "probe.csv",
        resolve=make_resolve("target", "judge", set()), **kw)
    r1 = probe["rows"][probe["rows"]["round"] == 1]["candidate"].astype(str)
    pair = next(c for c in r1 if c.count("+") == 1)
    win = applied_for(PROMPT, pair.split("+"))

    ref = search_combinations(
        seeds, out_path=tmp_path / "ref.csv",
        resolve=make_resolve("target", "judge", {win}), **kw)
    ref_ids = sorted(ref["rows"]["sample_id"].astype(str))
    assert ref["best"][content_id(PROMPT)].success

    for k in (34, 40, 46, 52, 58):
        d = tmp_path / f"k{k}"
        d.mkdir()
        out_path = d / "s.csv"
        with contextlib.suppress(RuntimeError):
            search_combinations(
                seeds, out_path=out_path,
                resolve=make_resolve("target", "judge", {win}, kill=[k]), **kw)
        resumed = search_combinations(
            seeds, out_path=out_path,
            resolve=make_resolve("target", "judge", {win}), **kw)
        got = sorted(resumed["rows"]["sample_id"].astype(str))
        assert len(got) == len(set(got)), f"kill {k}: duplicates"
        assert got == ref_ids, f"kill {k}: {len(got)} rows vs {len(ref_ids)} clean"


def test_resume_with_changed_settings_raises(tmp_path):
    out_path = tmp_path / "s.csv"
    resolve = make_resolve("target", "judge", success_texts=set())
    search_combinations(
        inputs(PROMPT), target_model="target", judge_model="judge", pool=POOL,
        out_path=out_path, width=4, beam=2, depth=1, seed=42,
        resolve=resolve, verbose=False)
    with pytest.raises(ValueError):
        search_combinations(
            inputs(PROMPT), target_model="target", judge_model="judge", pool=POOL,
            out_path=out_path, width=4, beam=2, depth=1, seed=7,   # changed seed
            resolve=resolve, verbose=False)
