"""Persistence of the combination search: the oversized-prompt row, resume
after a kill (same rows, no duplicates), and a resume whose settings changed.
"""

from __future__ import annotations

import pytest

from redact.jailbreak import best_combinations, search_combinations

from ._search_fakes import POOL, PROMPT, applied_text, inputs, make_resolve


def test_prompt_too_long_gives_a_failed_row_not_a_crash(tmp_path):
    # The target call for the base64 text is too long; its unit records a failed
    # row and the search goes on.
    too_long = applied_text(["to_base64"])
    resolve = make_resolve("target", "judge", success_texts=set(),
                           too_long_for={too_long})
    out = search_combinations(
        inputs(PROMPT), target_model="target", judge_model="judge", pool=POOL,
        out_path=tmp_path / "s.csv", width=4, beam=1, depth=0,
        resolve=resolve, verbose=False)
    rows = out["rows"]
    row = rows[rows["candidate"] == "to_base64"].iloc[0]
    assert str(row["success"]).lower() == "false"
    assert "too long" in str(row["judge_reasoning"]).lower()
    # The combination applied; only the target prompt was too long. (The catch
    # records applied=True; an unhandled error would be a DISCARDED error row.)
    assert str(row["applied"]).lower() == "true"


def test_resume_after_a_kill_gives_the_same_rows(tmp_path):
    out_path = tmp_path / "s.csv"
    seeds = inputs(PROMPT, "second harmful request here", "third one as well")
    win = applied_text(["to_uppercase", "to_base64"])

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
