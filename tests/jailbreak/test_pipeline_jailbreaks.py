"""Offline end-to-end tests for the jailbreak pipeline (plan + execute + resume).

Uses an adversarial kind of pure obfuscation techniques only, so no technique
needs an LLM — the engine completes every sample at prime time and the router
is never called. This exercises planning, execution, manifest, the prompt-content
id, and resume without any network access. The ``pool`` tests at the end run
LLM-backed fake techniques over a hand-written plan against a fake backend.
"""


import pandas as pd
import pytest

from redact.dataset import Manifest, content_id
from redact.exceptions import PromptTooLongError
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.pool import technique_pool
from redact.llm_pipeline import LLMRequest
from redact.llms.backends import ComputeConfig
from redact.stages import generate_jailbreaks
from tests.conftest import (
    Killed,
    MockBackend,
    kill_after_rows,
    lock_is_free,
    make_client,
)

_PURE_KIND = {"pure_only": True,
              "layers": {"deny": ["hacking", "manipulation", "requests"]}}


def _pure(**kind) -> dict:
    """The keywords of a run of pure obfuscations and no variations, ``kind``
    given over its adversarial kind."""
    return dict(augmentation={"adversarial": {**_PURE_KIND, **kind}, "natural": None},
                auto_generate_benign=False, verbose=False)


PURE_KW = _pure()


def _inputs():
    prompts = [f"prompt number {i} about a topic" for i in range(5)]
    return pd.DataFrame({"prompt": prompts, "category": ["Cyber"] * 5,
                         "entry_type": ["harmful"] * 5})


def test_the_stage_drives_with_the_pool_its_model_allows(tmp_path, monkeypatch):
    import redact.jailbreak.pipeline as JPipe

    seen = {}

    def stop(source, **kwargs):
        seen.update(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(JPipe, "drive_units", stop)
    monkeypatch.setattr(JPipe, "pool_size", lambda batch_size, *clients: 7 * batch_size)
    with pytest.raises(KeyboardInterrupt):
        generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jailbreaks.csv",
                            seed=7, batch_size=4, **PURE_KW)
    assert (seen["pool_size"], seen["cap"]) == (28, 4)


def test_plan_then_execute(tmp_path):
    out = tmp_path / "jailbreaks.csv"
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **PURE_KW)

    # Manifest written before generation, one unit per sample.
    manifest = out.with_name(out.stem + ".manifest.jsonl")
    assert manifest.exists()
    plan = Manifest(manifest).load()
    assert len(plan) == 5
    assert all(set(r) >= {"sample_id", "iteration", "combination", "settings"} for r in plan)

    # One output row per sample, linked by prompt-content id.
    assert len(df) == 5
    ids = {content_id(p) for p in _inputs()["prompt"]}
    assert set(df["input_id"].astype(str)) == ids
    assert "jailbreak" in df.columns and "combination_spec_version" in df.columns


def test_reproducible_assignment(tmp_path):
    out1 = tmp_path / "a" / "jb.csv"
    out2 = tmp_path / "b" / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out1, seed=42, **PURE_KW)
    generate_jailbreaks(inputs=_inputs(), output_path=out2, seed=42, **PURE_KW)

    p1 = {(r["sample_id"], r["iteration"]): r["combination"]
          for r in Manifest(out1.with_name("jb.manifest.jsonl")).load()}
    p2 = {(r["sample_id"], r["iteration"]): r["combination"]
          for r in Manifest(out2.with_name("jb.manifest.jsonl")).load()}
    assert p1 == p2  # same seed + same prompts -> identical combinations


def test_a_replayed_plan_writes_the_same_text(tmp_path, monkeypatch):
    """One manifest run twice gives byte-identical rows from techniques that
    choose at random, whatever the global random and the run's seed are: the
    techniques draw from the seed the manifest froze."""
    import random
    import shutil

    import redact.jailbreak.pool as JPool
    from redact.jailbreak.obfuscation.encoding import to_separator
    from redact.jailbreak.obfuscation.structural import to_json
    from redact.jailbreak.obfuscation.suffixes import to_adversarial_suffix_fragments

    choosing = [to_separator, to_json, to_adversarial_suffix_fragments]
    monkeypatch.setattr(JPool, "all_techniques", lambda: choosing)
    first, second = tmp_path / "a" / "jb.csv", tmp_path / "b" / "jb.csv"

    random.seed(1)
    generate_jailbreaks(inputs=_inputs(), output_path=first, seed=9,
                        **_pure(iterations=2, max_obfuscations=2))
    second.parent.mkdir()
    manifest = "jb.manifest.jsonl"
    shutil.copy(first.with_name(manifest), second.with_name(manifest))
    random.seed(2)
    generate_jailbreaks(inputs=_inputs(), output_path=second, seed=10,
                        **_pure(iterations=2, max_obfuscations=2))

    assert len(pd.read_csv(first)) == 2 * len(_inputs())
    assert first.read_bytes() == second.read_bytes()


def test_iterations_dedup(tmp_path):
    out = tmp_path / "jb.csv"
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=3,
                             **_pure(iterations=3, max_obfuscations=1))
    plan = Manifest(out.with_name("jb.manifest.jsonl")).load()
    assert len(plan) == 15  # 5 samples x 3 iterations

    # Per sample, the 3 assigned combinations are distinct (pool is large enough).
    by_sample: dict[str, list] = {}
    for r in plan:
        by_sample.setdefault(r["sample_id"], []).append(tuple(r["combination"]))
    for combos in by_sample.values():
        assert len(set(combos)) == len(combos)

    # Regression: sample_id (this row's own id) used to be silently inherited
    # from input_id, so every iteration of the same input shared one value.
    # Each of the 3 iterations per input_id must now have its own distinct
    # sample_id (jailbreak/units.py's row_of()).
    for input_id, group in df.groupby("input_id"):
        assert group["sample_id"].nunique() == len(group)
        assert not (group["sample_id"] == input_id).any()


def test_resume_skips_completed(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    first = pd.read_csv(out)
    assert len(first) == 5

    # Re-run with resume=True: nothing new should be appended (all done).
    again = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1,
                                resume=True, **PURE_KW)
    assert len(again) == 5  # no duplicate rows
    assert len(pd.read_csv(out)) == 5


def test_completion_ledger_written_and_drives_resume(tmp_path, monkeypatch):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)

    # Sidecar ledger has one (input_id, iteration) line per written unit.
    from redact.dataset import Ledger
    led = Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                         casters={"input_id": str, "iteration": int})
    assert len(led.completed()) == 5

    # Ledger alone drives resume: with no way to read the output's keys, a
    # resume run still skips everything (nothing regenerated, no new rows).
    import redact.dataset.resume as resume_module

    monkeypatch.setattr(resume_module, "keys_in", None)
    before = out.read_bytes()
    again = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1,
                                resume=True, **PURE_KW)
    assert len(again) == len(_inputs()) and out.read_bytes() == before

    # The ledger recorded how much of the output its units cover, so a resume
    # does not carry on without it: the run stops, and says how to go on.
    out.unlink()
    with pytest.raises(RuntimeError, match="jb.csv") as stopped:
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1,
                            resume=True, **PURE_KW)
    assert "start the whole stage over" in str(stopped.value)
    assert len(led.completed()) == len(_inputs()) and not out.exists()


def test_a_run_killed_between_rows_and_acks_leaves_each_unit_once(
        tmp_path, monkeypatch):
    import redact.jailbreak.pipeline as JPipe
    from redact.dataset import Ledger

    out, ids = tmp_path / "jb.csv", sorted(content_id(p) for p in _inputs()["prompt"])
    kill_after_rows(monkeypatch, JPipe, on_call=1)
    with pytest.raises(Killed):
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    landed = out.read_bytes()
    state = out.with_name("jb.state.jsonl")
    assert sorted(pd.read_csv(out)["input_id"]) == ids      # the rows landed
    assert state.read_bytes() == b'{"_saved": {}}\n'        # their acks did not
    assert lock_is_free(state)

    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    assert sorted(df["input_id"]) == ids
    assert out.with_name("jb.csv.cut").read_bytes() == landed
    ledger = Ledger.sidecar(out, ("input_id", "iteration"))
    assert sorted(key for key, _ in ledger.completed()) == ids
    assert lock_is_free(state)


def test_a_fresh_run_killed_before_it_plans_leaves_no_old_plan(tmp_path, monkeypatch):
    """Left in place, the old plan would be taken for the new run's by the
    next resume, and the settings just passed ignored."""
    import redact.jailbreak.pipeline as JPipe

    def killed(*args, **kwargs):
        raise Killed

    out = tmp_path / "jb.csv"
    manifest = Manifest.sidecar(out)
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    plan = manifest.load()
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    assert manifest.load() == plan                      # a resume reuses the plan

    monkeypatch.setattr(JPipe, "plan_run", killed)
    with pytest.raises(Killed):
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=2,
                            resume=False, **PURE_KW)
    assert [p.name for p in tmp_path.iterdir() if p.suffix != ".lock"] == []


def test_a_plan_naming_a_technique_no_longer_registered_says_to_plan_again(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs().head(2), output_path=out, seed=1, **PURE_KW)
    manifest = Manifest.sidecar(out)
    third = _inputs().iloc[2]
    manifest.write([*manifest.load(), {
        "sample_id": content_id(third["prompt"]), "iteration": 0,
        "combination": ["to_renamed_away"], "settings": {"seed": 1},
        "category": "Cyber", "entry_type": "harmful", "status": "planned"}])
    with pytest.raises(ValueError, match="'to_renamed_away'.*resume=False"):
        generate_jailbreaks(inputs=_inputs().head(3), output_path=out, seed=1,
                            **PURE_KW)
    assert lock_is_free(out.with_name("jb.state.jsonl"))


def test_a_second_run_is_refused_before_it_plans(tmp_path):
    from filelock import FileLock

    def files():
        return {p.name: p.read_bytes() for p in tmp_path.iterdir()
                if p.suffix != ".lock"}

    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    before = files()
    with FileLock(str(out.with_name("jb.state.jsonl")) + ".lock"), pytest.raises(
            RuntimeError, match="locked"):
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=2,
                            resume=False, **PURE_KW)
    assert files() == before    # the output, the ledger and the plan of seed 1


def test_the_ledger_records_how_much_of_the_output_is_saved(tmp_path):
    from redact.dataset import Ledger

    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    size, _checksum = Ledger.sidecar(out).saved["jb.csv"]
    assert size == out.stat().st_size


def test_fresh_run_clears_ledger(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    from redact.dataset import Ledger
    led = Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                         casters={"input_id": str, "iteration": int})
    assert led.completed()

    # resume=False regenerates from scratch: ledger reset then re-acked to 5.
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, resume=False, **PURE_KW)
    assert len(led.completed()) == 5
    assert len(pd.read_csv(out)) == 5  # not doubled


def test_rounds_escalate(tmp_path):
    """Multi-round escalation: 4 rounds per sample, exact counts for rounds 0-1."""
    out = tmp_path / "jb.csv"
    schedule = [
        {"exact_techniques": 1},
        {"exact_techniques": 2},
        {"max_complexity": 6, "max_obfuscations": 2},
        {"max_complexity": 9, "max_obfuscations": 3},
    ]
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=11,
                             **_pure(rounds=schedule))

    # iterations derived from the number of rounds: 5 samples x 4 rounds.
    plan = Manifest(out.with_name("jb.manifest.jsonl")).load()
    assert len(plan) == 20
    assert sorted({r["iteration"] for r in plan}) == [0, 1, 2, 3]

    # Per-round settings recorded; count-driven rounds have exact combination sizes.
    for r in plan:
        if r["iteration"] == 0:
            assert r["settings"].get("exact_techniques") == 1
            assert len(r["combination"]) == 1
        elif r["iteration"] == 1:
            assert r["settings"].get("exact_techniques") == 2
            assert len(r["combination"]) == 2

    # Output mirrors the exact counts for the count-driven rounds.
    assert (df[df["iteration"] == 0]["num_techniques"] == 1).all()
    assert (df[df["iteration"] == 1]["num_techniques"] == 2).all()


def test_a_budget_round_inherits_the_top_level_max_complexity(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=11,
                        **_pure(max_complexity=1, rounds=[{"max_obfuscations": 3}]))
    plan = Manifest(out.with_name("jb.manifest.jsonl")).load()
    assert {r["settings"]["max_complexity"] for r in plan} == {1}
    assert {r["settings"]["max_obfuscations"] for r in plan} == {3}
    # No pure obfuscation costs 0, so a budget of 1 ends after the first pick.
    assert all(len(r["combination"]) == 1 for r in plan)


def test_empty_rounds_raise(tmp_path):
    with pytest.raises(ValueError, match="rounds must be null"):
        generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv",
                            **_pure(rounds=[]))


def test_escalation_names_the_built_in_schedule(tmp_path):
    from redact.jailbreak import default_escalation_schedule

    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=11,
                        **_pure(rounds="escalation"))
    plan = Manifest(out.with_name("jb.manifest.jsonl")).load()
    schedule = default_escalation_schedule()
    assert len(plan) == len(schedule) * len(_inputs())
    for r in plan:
        assert r["settings"].items() >= schedule[r["iteration"]].items()


def test_family_weights_and_layer_rates_reach_the_plan(tmp_path):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(
        inputs=_inputs(), output_path=out, seed=11,
        **_pure(family_weights={"encode": 3}, layer_rates={"hacking": 0.2}))
    plan = Manifest(out.with_name("jb.manifest.jsonl")).load()
    assert all(r["settings"]["family_weights"] == {"encode": 3} for r in plan)
    assert all(r["settings"]["sampling_probs"] == {"hacking": 0.2} for r in plan)


def test_the_flat_keywords_plan_what_their_section_plans(tmp_path):
    flat, section = tmp_path / "flat" / "jb.csv", tmp_path / "section" / "jb.csv"
    with pytest.warns(DeprecationWarning, match="augmentation"):
        generate_jailbreaks(
            inputs=_inputs(), output_path=flat, seed=5, pure_only=True,
            include_hacking=False, include_manipulation=False,
            include_requests=False, max_complexity=4, iterations=2,
            sampling_probs={"requests": 0.1}, augmentation={"natural": None},
            auto_generate_benign=False, verbose=False)
    generate_jailbreaks(
        inputs=_inputs(), output_path=section, seed=5, **_pure(
            max_complexity=4, iterations=2, layer_rates={"requests": 0.1}))
    assert (Manifest(flat.with_name("jb.manifest.jsonl")).load()
            == Manifest(section.with_name("jb.manifest.jsonl")).load())


def test_a_flat_keyword_for_an_adversarial_kind_switched_off_raises(tmp_path):
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="null"):
        generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv",
                            augmentation={"adversarial": None}, pure_only=True)


def test_with_both_kinds_off_nothing_is_written(tmp_path):
    returned = generate_jailbreaks(
        inputs=_inputs(), output_path=tmp_path / "jb.csv",
        variations_path=tmp_path / "var.csv",
        augmentation={"adversarial": None, "natural": None}, verbose=False)
    assert returned.empty and list(tmp_path.iterdir()) == []


def test_entry_types_keep_the_inputs_of_those_types(tmp_path):
    inputs = _inputs().assign(entry_type=["harmful", "benign"] * 2 + ["harmful"])
    df = generate_jailbreaks(inputs=inputs, output_path=tmp_path / "jb.csv", seed=1,
                             **_pure(entry_types=["benign"]))
    assert list(df["entry_type"]) == ["benign", "benign"]


# --- the pool against the plan ---------------------------------------------------


def test_a_resume_with_a_narrower_pool_runs_the_planned_techniques(
        tmp_path, monkeypatch):
    """A pending unit rebuilds its combination from every registered technique,
    so the techniques the resumed run's pool leaves out still run."""
    import itertools

    import redact.jailbreak.pipeline as JPipe
    from redact.dataset import Ledger

    whole = generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "w" / "jb.csv",
                                seed=7, **PURE_KW)

    out, finished, drive_units = tmp_path / "jb.csv", 2, JPipe.drive_units

    def stop_after_two(source, **kwargs):
        drive_units(itertools.islice(source, finished), **kwargs)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(JPipe, "drive_units", stop_after_two)
        with pytest.raises(KeyboardInterrupt):
            generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **PURE_KW)
    casters = {"input_id": str, "iteration": int}
    done = Ledger.sidecar(out, ("input_id", "iteration"), casters).completed()
    plan = Manifest.sidecar(out).load()
    pending = [r for r in plan if (r["sample_id"], r["iteration"]) not in done]
    assert len(done) == finished and len(pending) == len(_inputs()) - finished

    # The resumed run's pool holds none of the techniques the pending units name.
    off_pool = sorted({name for r in pending for name in r["combination"]})
    narrower = _pure(techniques={"deny": off_pool})
    pool = technique_pool(resolve(narrower["augmentation"], "adversarial"))
    assert pool and not {fn.__name__ for fn in pool} & set(off_pool)

    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **narrower)
    assert Manifest.sidecar(out).load() == plan
    technique_of = dict(zip(zip(df["input_id"], df["iteration"]), df["technique"]))
    for unit in pending:
        key = unit["sample_id"], unit["iteration"]
        assert technique_of[key] == "+".join(unit["combination"])

    def by_unit(rows):
        return rows.sort_values(["input_id", "iteration"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(by_unit(df), by_unit(whole))


_BENIGN = {"all_subcategories": ["sub"],
           "by_subcat_short": {"sub": [{"prompt": "bq", "answer": "ba"}]},
           "short": [{"prompt": "bq", "answer": "ba"}]}


def _benign_loads(monkeypatch) -> list[dict]:
    """Stand a fixed benign set in for the stage's load; the keywords of each load."""
    import redact.jailbreak.pipeline as JPipe

    loads: list[dict] = []
    monkeypatch.setattr(JPipe, "get_or_generate_benign_data",
                        lambda **kwargs: loads.append(kwargs) or _BENIGN)
    return loads


def test_a_pool_with_a_technique_that_needs_benign_data_loads_it(tmp_path, monkeypatch):
    loads = _benign_loads(monkeypatch)
    benign = tmp_path / "benign.csv"
    df = generate_jailbreaks(
        inputs=_inputs(), output_path=tmp_path / "jb.csv", benign_path=benign, seed=7,
        augmentation={"adversarial": {
            "layers": {"deny": []}, "pure_only": False,
            "techniques": {"allow": ["to_fsh_random_short"]},
            "rounds": [{"exact_techniques": 1}]}, "natural": None},
        verbose=False)
    assert [load["cache_path"] for load in loads] == [benign]
    assert {key: loads[0][key] for key in ("taxonomy", "taxonomy_dir")} == {
        "taxonomy": "benign", "taxonomy_dir": None}
    # Each unit ran its technique on the loaded set.
    assert set(df["technique"]) == {"to_fsh_random_short"}
    assert df["accepted"].all() and df["jailbreak"].str.contains("bq").all()


def test_the_benign_data_is_made_with_the_stages_taxonomy_overlay_and_batch(
        tmp_path, monkeypatch):
    loads = _benign_loads(monkeypatch)
    generate_jailbreaks(
        inputs=_inputs(), output_path=tmp_path / "jb.csv", seed=7, batch_size=4,
        benign_taxonomy="mine", taxonomy_dir=tmp_path / "taxonomy",
        prompt_dir=tmp_path / "prompts",
        augmentation={"adversarial": {
            "techniques": {"allow": ["to_fsh_random_short"]},
            "rounds": [{"exact_techniques": 1}]}, "natural": None},
        verbose=False)
    (load,) = loads
    assert {key: load[key] for key in (
        "taxonomy", "taxonomy_dir", "prompt_dir", "batch_size")} == {
        "taxonomy": "mine", "taxonomy_dir": tmp_path / "taxonomy",
        "prompt_dir": tmp_path / "prompts", "batch_size": 4}


def test_a_resume_loads_the_benign_data_its_plan_needs_whatever_its_pool(
        tmp_path, monkeypatch):
    """Else the pending units of a technique that needs it would be acked as
    error rows, never to run again."""
    import itertools

    import redact.jailbreak.pipeline as JPipe

    loads = _benign_loads(monkeypatch)
    out, finished, drive_units = tmp_path / "jb.csv", 2, JPipe.drive_units
    fsh = {"adversarial": {"techniques": {"allow": ["to_fsh_random_short"]},
                           "rounds": [{"exact_techniques": 1}]}, "natural": None}

    def stop_after_two(source, **kwargs):
        drive_units(itertools.islice(source, finished), **kwargs)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(JPipe, "drive_units", stop_after_two)
        with pytest.raises(KeyboardInterrupt):
            generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7,
                                augmentation=fsh, verbose=False)
    assert len(pd.read_csv(out)) == finished
    loaded = len(loads)

    # The resumed run's pool of pure obfuscations needs no benign data.
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7,
                             **{**PURE_KW, "auto_generate_benign": True})
    assert len(loads) == loaded + 1
    assert len(df) == len(_inputs()) and df["accepted"].all()
    assert set(df["technique"]) == {"to_fsh_random_short"}


_FSH = {"adversarial": {"techniques": {"allow": ["to_fsh_random_short"]},
                        "rounds": [{"exact_techniques": 1}]}, "natural": None}


def test_without_auto_generate_benign_the_cache_is_loaded_as_it_is(
        tmp_path, monkeypatch):
    import redact.jailbreak.manipulation.benign as JBenign

    def generate(*args, **kwargs):
        raise AssertionError("benign data was generated")

    monkeypatch.setattr(JBenign, "generate_benign_data", generate)
    benign = tmp_path / "benign.csv"
    pd.DataFrame([
        {"main_category": "A", "sub_category": "s", "prompt": f"q{i}",
         "answer": f"a{i}", "answer_type": answer_type}
        for i in range(8) for answer_type in ("short", "long")
    ]).to_csv(benign, index=False)
    inputs = _inputs().head(3)
    df = generate_jailbreaks(
        inputs=inputs, output_path=tmp_path / "jb.csv", benign_path=benign,
        seed=7, augmentation=_FSH, auto_generate_benign=False, verbose=False)
    assert len(df) == len(inputs) and df["accepted"].all()
    assert set(df["technique"]) == {"to_fsh_random_short"}


def test_without_auto_generate_benign_or_a_cache_the_run_raises_before_a_row(
        tmp_path):
    out = tmp_path / "jb.csv"
    with pytest.raises(ValueError, match="Benign data not found"):
        generate_jailbreaks(
            inputs=_inputs().head(3), output_path=out,
            benign_path=tmp_path / "benign.csv", seed=7, augmentation=_FSH,
            auto_generate_benign=False, verbose=False)
    assert not out.exists()


def test_a_pool_without_one_loads_no_benign_data(tmp_path, monkeypatch):
    loads = _benign_loads(monkeypatch)
    df = generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv", seed=7,
                             **{**PURE_KW, "auto_generate_benign": True})
    assert loads == [] and len(df) == len(_inputs())


# --- the natural kind: variations beside the jailbreaks -------------------------


def _mixed_inputs():
    return pd.DataFrame({
        "sample_id": ["given-1", None, None, None],
        "prompt": [f"prompt number {i} about a topic." for i in range(4)],
        "category": ["Cyber", "Cyber", "Fraud", "Fraud"],
        "entry_type": ["harmful", "benign", "dual_use_benign", "benign"],
    })


def _both(tmp_path, **section):
    """Run both kinds, pure, into ``tmp_path``; the two output paths."""
    jailbreaks, variations = tmp_path / "jb.csv", tmp_path / "var.csv"
    generate_jailbreaks(
        inputs=_mixed_inputs(), output_path=jailbreaks, variations_path=variations,
        seed=3, augmentation={"adversarial": _PURE_KIND, "natural": {"pure_only": True},
                              **section},
        auto_generate_benign=False, verbose=False)
    return jailbreaks, variations


def test_a_run_writes_its_jailbreaks_and_its_variations_each_with_its_sidecars(
        tmp_path):
    jailbreaks, variations = _both(tmp_path)
    for out, stem in ((jailbreaks, "jb"), (variations, "var")):
        assert len(pd.read_csv(out)) == len(_mixed_inputs())
        for sidecar in ("manifest.jsonl", "state.jsonl", "coverage.csv"):
            assert out.with_name(f"{stem}.{sidecar}").exists()
    plan = Manifest(variations.with_name("var.manifest.jsonl")).load()
    assert all(len(r["combination"]) == 1 for r in plan)
    assert all(r["settings"] == {"exact_techniques": 1, "seed": 3} for r in plan)


def test_a_variation_row_keeps_its_inputs_labels(tmp_path):
    _, variations = _both(tmp_path)
    inputs = _mixed_inputs()
    ids = ["given-1", *(content_id(p) for p in inputs["prompt"][1:])]
    rows = pd.read_csv(variations, keep_default_na=False)
    by_prompt = {r["prompt"]: r for r in rows.to_dict("records")}
    assert [by_prompt[p]["input_id"] for p in inputs["prompt"]] == ids
    assert ([by_prompt[p]["entry_type"] for p in inputs["prompt"]]
            == list(inputs["entry_type"]))
    assert list(rows["sample_id"]) == [content_id(t) for t in rows["jailbreak"]]
    assert not (rows["sample_id"] == rows["input_id"]).any()
    natural = {fn.__name__ for fn in technique_pool(resolve(
        {"natural": {"pure_only": True}}, "natural"))}
    assert set(rows["technique"]) <= natural


def test_a_natural_only_run_writes_variations_and_no_jailbreak(tmp_path):
    jailbreaks, variations = _both(tmp_path, adversarial=None)
    assert not jailbreaks.exists()
    assert len(pd.read_csv(variations)) == len(_mixed_inputs())


def test_the_natural_kind_takes_its_entry_types_and_iterations(tmp_path):
    _, variations = _both(tmp_path, natural={
        "pure_only": True, "entry_types": ["benign"], "iterations": 2})
    rows = pd.read_csv(variations)
    assert list(rows["entry_type"]) == ["benign"] * 4
    assert sorted(rows["iteration"]) == [0, 0, 1, 1]


def test_a_variation_that_changes_nothing_is_not_accepted_nor_built(tmp_path):
    from redact import build_dataset, paths
    from redact.jailbreak.obfuscation.casual_writing import (
        get_casual_writing_functions,
    )

    inputs = pd.DataFrame({"prompt": ["all lower case.", "Some Capitals."],
                           "category": ["Cyber"] * 2, "entry_type": ["benign"] * 2})
    others = [fn.__name__ for fn in get_casual_writing_functions()
              if fn.__name__ != "to_lowercase"]
    generate_jailbreaks(
        inputs=inputs, data_dir=tmp_path, seed=1, verbose=False, augmentation={
            "adversarial": {"families": {"allow": ["casual_writing"]},
                            "techniques": {"deny": others},
                            "rounds": [{"exact_techniques": 1}]},
            "natural": {"families": {"allow": None},
                        "techniques": {"allow": ["to_lowercase"], "deny": []}}})
    variations = pd.read_csv(paths.variations_csv(tmp_path), keep_default_na=False)
    assert [(r["jailbreak"], r["is_noop"], r["accepted"], r["reasoning"])
            for r in variations.to_dict("records")] == [
        ("all lower case.", True, False, "unchanged: the text is its input's"),
        ("some capitals.", False, True, "")]
    # A jailbreak that changes nothing is kept, as before.
    jailbreaks = pd.read_csv(paths.jailbreaks_csv(tmp_path))
    assert jailbreaks["accepted"].all() and jailbreaks["is_noop"].any()

    build_dataset(data_dir=tmp_path, verbose=False)
    built = pd.read_csv(paths.complete_dataset_csv(tmp_path))
    built = built[built["dataset_type"] == "natural_variation"]
    assert list(built["jailbreak"]) == ["some capitals."]


def test_the_variations_of_a_given_output_path_go_beside_it(tmp_path):
    from redact import paths

    for name, inputs in (("a", _inputs().iloc[:2]), ("b", _inputs())):
        generate_jailbreaks(
            inputs=inputs, data_dir=tmp_path, output_path=tmp_path / name / "jb.csv",
            seed=1, augmentation={"adversarial": _PURE_KIND,
                                  "natural": {"pure_only": True}},
            auto_generate_benign=False, verbose=False)
    for name, planned in (("a", 2), ("b", 5)):
        variations = tmp_path / name / "variations.csv"
        assert len(Manifest.sidecar(variations).load()) == planned
        assert len(pd.read_csv(variations)) == planned
    assert not paths.variations_csv(tmp_path).exists()


def test_a_fresh_run_of_one_kind_leaves_the_others_files(tmp_path):
    jailbreaks, variations = _both(tmp_path)
    before = variations.read_bytes()
    generate_jailbreaks(
        inputs=_mixed_inputs(), output_path=jailbreaks, seed=4, resume=False,
        augmentation={"adversarial": _PURE_KIND, "natural": None},
        auto_generate_benign=False, verbose=False)
    assert variations.read_bytes() == before


# --- the inputs of the run ------------------------------------------------------


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "standalone"])
def test_without_inputs_the_inputs_of_the_run_are_found(tmp_path, seeded):
    from tests.test_pipelines import _seed_input_tree

    text = _seed_input_tree(tmp_path, seeded)
    df = generate_jailbreaks(data_dir=tmp_path, seed=7, **PURE_KW)
    assert list(df["prompt"]) == [text]
    assert list(df["input_id"]) == [content_id(text)]


@pytest.mark.parametrize("given", [None, ["given", None, None]],
                         ids=["no-column", "missing-ids"])
def test_the_core_gives_each_input_its_id(tmp_path, given):
    from redact.jailbreak.augmentation import rounds_of
    from redact.jailbreak.pipeline import run_augmentation

    inputs = _inputs().head(3)
    if given is not None:
        inputs = inputs.assign(sample_id=given)
    kind = resolve({"adversarial": _PURE_KIND}, "adversarial")
    df = run_augmentation(inputs, technique_pool(kind), client=make_client(),
                          out_path=tmp_path / "jb.csv", rounds=rounds_of(kind),
                          verbose=False)
    prompts = list(inputs["prompt"])
    expected = [content_id(p) for p in prompts]
    if given is not None:
        expected[0] = "given"
    assert sorted(df["input_id"]) == sorted(expected)


def test_without_inputs_in_either_tree_the_run_raises(tmp_path):
    with pytest.raises(ValueError, match="No input samples found"):
        generate_jailbreaks(data_dir=tmp_path, seed=7, **PURE_KW)


@pytest.mark.parametrize("verbose", [False, True])
def test_without_read_back_the_output_is_written_and_not_read(
        tmp_path, monkeypatch, verbose):
    import redact.jailbreak.pipeline as JPipe

    def read_rows(path):
        raise AssertionError("the output was read back")

    monkeypatch.setattr(JPipe, "read_rows", read_rows)
    out = tmp_path / "jb.csv"
    returned = generate_jailbreaks(
        inputs=_inputs(), output_path=out, seed=7, read_back=False,
        **{**PURE_KW, "verbose": verbose})
    assert returned.empty and len(returned.columns) == 0
    assert len(pd.read_csv(out)) == len(_inputs())


# --- the plan as a stream ------------------------------------------------------


def test_plan_run_hands_the_manifest_each_row_as_it_is_made(tmp_path, monkeypatch):
    import redact.jailbreak.plan as JP

    assigned, seen = [], []
    monkeypatch.setattr(
        JP, "assign_combination",
        lambda sid, pool, **kwargs: assigned.append(sid) or ["a_technique"])
    # Per row the manifest takes: the assignments made by then.
    monkeypatch.setattr(
        Manifest, "write", lambda self, rows: seen.extend(len(assigned) for _ in rows))
    JP.plan_run(_inputs(), [], manifest_path=tmp_path / "jb.manifest.jsonl",
                iterations=2, verbose=False)
    assert seen == list(range(1, 11))


@pytest.mark.parametrize("verbose", [False, True])
def test_a_run_never_loads_its_whole_plan(tmp_path, monkeypatch, verbose):
    def load(self):
        raise AssertionError("the whole plan was loaded")

    monkeypatch.setattr(Manifest, "load", load)
    out, kwargs = tmp_path / "jb.csv", {**PURE_KW, "verbose": verbose}
    first = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **kwargs)
    again = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **kwargs)
    assert len(first) == len(again) == len(_inputs())


def _plan_rows_read(monkeypatch, units_taken):
    """Stop the stage once the driver took ``units_taken`` units; the plan rows read."""
    import redact.jailbreak.pipeline as JPipe

    rows, read = Manifest.__iter__, []

    def counted(self):
        for row in rows(self):
            read.append(row)
            yield row

    def take(source, **kwargs):
        units = iter(source)
        for _ in range(units_taken):
            next(units)
        raise KeyboardInterrupt

    monkeypatch.setattr(Manifest, "__iter__", counted)
    monkeypatch.setattr(JPipe, "drive_units", take)
    return read


def test_a_unit_is_built_when_the_driver_takes_it(tmp_path, monkeypatch):
    taken = 2       # of the five planned
    read = _plan_rows_read(monkeypatch, units_taken=taken)
    with pytest.raises(KeyboardInterrupt):
        generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv",
                            seed=7, **PURE_KW)
    assert len(read) == taken


def test_finished_units_are_skipped_as_the_plan_is_read(tmp_path, monkeypatch):
    from redact.dataset import Ledger

    out, finished = tmp_path / "jb.csv", 3
    generate_jailbreaks(inputs=_inputs().iloc[:finished], output_path=out, seed=7,
                        **PURE_KW)
    casters = {"input_id": str, "iteration": int}
    done = set(Ledger.sidecar(out, ("input_id", "iteration"), casters).completed())
    assert len(done) == finished
    Manifest.sidecar(out).reset()       # the next run plans all five inputs
    read = _plan_rows_read(monkeypatch, units_taken=1)
    with pytest.raises(KeyboardInterrupt):
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **PURE_KW)
    # The plan is read until one pending unit is taken; every row read before it
    # was a finished unit, skipped (the manifest order is group + sample_id).
    taken = (str(read[-1]["sample_id"]), int(read[-1]["iteration"]))
    assert taken not in done
    assert all((str(r["sample_id"]), int(r["iteration"])) in done for r in read[:-1])


def test_a_stopped_run_lets_go_of_its_plan(tmp_path, monkeypatch):
    """On Windows a plan still held open could not be replaced by the next run."""
    import redact.jailbreak.pipeline as JPipe

    rows, drive_units, let_go = Manifest.__iter__, JPipe.drive_units, []

    def watched(self):
        try:
            yield from rows(self)
        finally:
            let_go.append(self.path.name)

    def take_one(source, **kwargs):
        next(iter(source))
        raise KeyboardInterrupt

    out = tmp_path / "jb.csv"
    monkeypatch.setattr(Manifest, "__iter__", watched)
    monkeypatch.setattr(JPipe, "drive_units", take_one)
    # ``stopped`` keeps the traceback, and with it every frame of the run.
    with pytest.raises(KeyboardInterrupt) as stopped:  # noqa: F841
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **PURE_KW)
    assert let_go == ["jb.manifest.jsonl"]

    monkeypatch.setattr(JPipe, "drive_units", drive_units)
    fresh = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=8,
                                resume=False, **PURE_KW)
    assert len(fresh) == len(_inputs())


def test_an_input_becomes_a_dict_only_for_the_row_of_a_finished_unit(
        tmp_path, monkeypatch):
    def every_row(self, *args, **kwargs):
        raise AssertionError("every input was turned into a dict")

    inputs = _inputs().assign(turn=range(5))
    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_dict", every_row)
        df = generate_jailbreaks(inputs=inputs, output_path=tmp_path / "jb.csv",
                                 seed=7, **PURE_KW)
    # Each row still carries the columns of its own input.
    turn_of = dict(zip(inputs["prompt"], inputs["turn"]))
    assert dict(zip(df["prompt"], df["turn"])) == turn_of


# --- refill pool: LLM techniques over a hand-written plan -------------------

GEN = "jb-gen"


def one_step(text, *, gen_model=None, **kwargs):
    """LLM technique: one call."""
    reply = yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    return reply, "one_step"


def two_step(text, *, gen_model=None, **kwargs):
    """LLM technique: two calls in sequence."""
    a = yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    b = yield LLMRequest(gen_model, [{"role": "user", "content": "step2 " + a}])
    return b, "two_step"


def broken(text, *, gen_model=None, **kwargs):
    """LLM technique whose chain raises after its call."""
    yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    raise ValueError("chain failed")


class _StepBackend(MockBackend):
    """Native-batching fake: ``step1 x`` -> ``S1:x``, ``step2 x`` -> ``S2:x``.

    ``batches`` holds the request contents of each ``generate()`` call.
    """

    def __init__(self):
        super().__init__(model=GEN)
        self.batches: list[list[str]] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=True)

    def generate(self, messages_list, **kwargs):
        super().generate(messages_list, **kwargs)
        contents = [m[-1]["content"] for m in messages_list]
        self.batches.append(contents)
        return [c.replace("step1 ", "S1:").replace("step2 ", "S2:") for c in contents]


def _llm_run(tmp_path, monkeypatch, plan, backend,
             techniques=(one_step, two_step, broken), **kwargs):
    """Run the stage on ``plan``, ``[(prompt, technique), ...]``, written as its
    manifest; a tuple of techniques is one unit's chain. ``techniques`` is the
    registry."""
    import redact.jailbreak.pool as JPool
    import redact.stages as P
    from redact.dataset import Manifest

    def names(chain):
        return [t.__name__ for t in (chain if isinstance(chain, tuple) else (chain,))]

    monkeypatch.setattr(JPool, "all_techniques", lambda: list(techniques))
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))

    out = tmp_path / "jb.csv"
    Manifest(out.with_name("jb.manifest.jsonl")).write(
        {"sample_id": content_id(p), "iteration": 0, "combination": names(t),
         "settings": {}, "status": "planned"}
        for p, t in plan
    )
    prompts = [p for p, _ in plan]
    inputs = pd.DataFrame({"prompt": prompts, "category": ["Cyber"] * len(prompts),
                           "entry_type": ["harmful"] * len(prompts)})
    generate_jailbreaks(inputs=inputs, output_path=out, model=GEN, resume=True,
                        augmentation={"natural": None}, verbose=False, **kwargs)
    return out


def test_pool_run_without_retries_writes_the_same_rows(tmp_path, monkeypatch):
    from redact.jailbreak import load_spec

    techniques = [two_step, one_step, broken, one_step, one_step, one_step]
    plan = [(f"zq{i}", t) for i, t in enumerate(techniques)]
    out = _llm_run(tmp_path, monkeypatch, plan, _StepBackend(), batch_size=2)

    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(df.columns) == [
        "prompt", "category", "entry_type", "sample_id", "input_id", "jailbreak",
        "technique", "technique_info", "complexity", "num_techniques", "is_noop",
        "accepted", "reasoning", "iteration", "combination_spec_version",
    ]

    def row(prompt, technique, text, info, accepted, reasoning):
        return {
            "prompt": prompt, "category": "Cyber", "entry_type": "harmful",
            "sample_id": content_id(text), "input_id": content_id(prompt),
            "jailbreak": text, "technique": technique, "technique_info": info,
            "complexity": "0", "num_techniques": "1", "is_noop": str(text == prompt),
            "accepted": str(accepted), "reasoning": reasoning, "iteration": "0",
            "combination_spec_version": str(load_spec().get("version", "")),
        }

    expected = [
        row("zq0", "two_step", "S2:S1:zq0", "two_step", True, ""),
        {**row("zq2", "broken", "zq2", "ERROR: chain failed", False, "chain failed"),
         # An error row's sample_id is its own (content_id of its unit key), not
         # the input's.
         "sample_id": content_id(f"jailbreak:{content_id('zq2')}:0")},
    ] + [row(f"zq{i}", "one_step", f"S1:zq{i}", "one_step", True, "") for i in (1, 3, 4, 5)]
    assert {(r["input_id"], r["iteration"]): r for r in df.to_dict("records")} == {
        (r["input_id"], r["iteration"]): r for r in expected
    }
    assert len(df) == 6


def test_pool_fills_a_call_with_a_later_step_and_a_new_unit(tmp_path, monkeypatch):
    backend = _StepBackend()
    plan = [("zq0", two_step)] + [(f"zq{i}", one_step) for i in range(1, 6)]
    _llm_run(tmp_path, monkeypatch, plan, backend, batch_size=2)

    # A chunk loop sends zq0's second step alone ([2, 1, 2, 2]).
    assert [len(b) for b in backend.batches] == [2, 2, 2, 1]
    assert backend.batches[2] == ["step2 S1:zq0", "step1 zq4"]


def test_pool_resume_after_a_failed_append_runs_only_unfinished_units(tmp_path, monkeypatch):
    plan = [(f"zq{i}", one_step) for i in range(70)]
    real_to_csv = pd.DataFrame.to_csv
    appends = []

    def failing_second_append(self, *args, **kwargs):
        appends.append(len(self))
        if len(appends) == 2:
            raise OSError("disk full")
        return real_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_second_append)
    with pytest.raises(OSError, match="disk full"):
        _llm_run(tmp_path, monkeypatch, plan, _StepBackend(), batch_size=4)
    monkeypatch.setattr(pd.DataFrame, "to_csv", real_to_csv)
    out = tmp_path / "jb.csv"
    assert appends == [32, 32] and len(pd.read_csv(out)) == 32

    second = _StepBackend()
    _llm_run(tmp_path, monkeypatch, plan, second, batch_size=4)
    df = pd.read_csv(out)
    assert sum(len(b) for b in second.batches) == 70 - 32
    assert sorted(df["prompt"]) == sorted(p for p, _ in plan)


class _LaneStepBackend(MockBackend):
    """Async fake: ``step1 x`` -> ``S1:x``, ``step2 x`` -> ``S2:x``. A request
    whose content is in ``too_long`` is over the context."""

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self, too_long=()):
        super().__init__(model=GEN)
        self.too_long = set(too_long)

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None,
                        **kwargs):
        content = messages[-1]["content"]
        if content in self.too_long:
            raise PromptTooLongError(f"over the context: {content}")
        return content.replace("step1 ", "S1:").replace("step2 ", "S2:")


def test_pool_a_prompt_too_long_lands_in_the_units_error_row(tmp_path, monkeypatch):
    """The stage's on_error makes an error row of whatever a unit raises. The
    unit is acked and the others finish."""
    from redact.dataset import Ledger

    plan = [("zq0", one_step), ("zq1", two_step), ("zq2", one_step)]
    backend = _LaneStepBackend(too_long={"step2 S1:zq1"})
    out = _llm_run(tmp_path, monkeypatch, plan, backend, batch_size=2)

    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    rows = {r["prompt"]: r for r in df.to_dict("records")}
    message = "over the context: step2 S1:zq1"
    assert {k: rows["zq1"][k] for k in (
        "sample_id", "jailbreak", "technique", "technique_info", "is_noop",
        "accepted", "reasoning")} == {
        "sample_id": content_id(f"jailbreak:{content_id('zq1')}:0"), "jailbreak": "zq1",
        "technique": "two_step",
        "technique_info": f"ERROR: {message}", "is_noop": "True",
        "accepted": "False", "reasoning": message}
    assert {p: (rows[p]["jailbreak"], rows[p]["accepted"]) for p in ("zq0", "zq2")} == {
        "zq0": ("S1:zq0", "True"), "zq2": ("S1:zq2", "True")}
    assert len(df) == len(plan)
    ledger = Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                            casters={"input_id": str, "iteration": int})
    assert ledger.completed() == {(content_id(p), 0) for p, _ in plan}


# --- what reaches a technique's call -----------------------------------------


def test_the_prompt_dir_reaches_a_techniques_call(tmp_path, monkeypatch):
    seen = []

    def takes_prompt_dir(text, *, gen_model=None, prompt_dir=None, **kwargs):
        seen.append(prompt_dir)
        msgs = [{"role": "user", "content": "step1 " + text}]
        reply = yield LLMRequest(gen_model, msgs)
        return reply, "takes_prompt_dir"

    overlay = tmp_path / "my_prompts"
    out = _llm_run(tmp_path, monkeypatch, [("zq0", takes_prompt_dir)], _StepBackend(),
                   techniques=[takes_prompt_dir], prompt_dir=overlay)
    assert seen == [overlay]
    assert list(pd.read_csv(out)["jailbreak"]) == ["S1:zq0"]


def upper(text, **kwargs):
    """Pure technique."""
    return text.upper(), "upper"


def test_a_capturing_client_files_a_units_calls_under_its_sample_id(
        tmp_path, monkeypatch):
    """Each LLM call of a chain is captured under the unit's provisional root,
    which the unit's row relabels to its sample_id; a pure chain captures
    nothing and leaves nothing to relabel."""
    import json

    import redact.llms.backends.introspection as introspection
    from redact.llms.model_config import MODEL_REGISTRY, IntrospectConfig, ModelConfig
    from tests.llms.test_introspection_backend import _generate_backend

    log_dir = tmp_path / "internals"
    monkeypatch.setattr(introspection, "_models", {})
    backend = _generate_backend(log_dir)            # replies "reply" to every request
    # make_client binds the entry's identity onto the backend, temperature too.
    monkeypatch.setitem(MODEL_REGISTRY, GEN, ModelConfig(
        name=GEN, default_temperature=0.7,
        introspect=IntrospectConfig(hf_model_id="org/model", log_dir=str(log_dir))))

    plan = [("zq0", (one_step, two_step)), ("zq1", upper)]
    out = _llm_run(tmp_path, monkeypatch, plan, backend,
                   techniques=[one_step, two_step, upper])

    df = pd.read_csv(out)
    sample_id = dict(zip(df["prompt"], df["sample_id"]))
    assert sample_id["zq0"] == content_id("reply")
    root = f"{content_id('zq0')}/jailbreak"
    asked = {
        p.parent.relative_to(log_dir).as_posix():
            json.loads(p.read_text("utf-8"))["messages"][-1]["content"]
        for p in log_dir.rglob("meta.json")
    }
    assert asked == {
        f"{root}/{sample_id['zq0']}/one_step/0": "step1 zq0",
        f"{root}/{sample_id['zq0']}/two_step/0": "step1 reply",
        f"{root}/{sample_id['zq0']}/two_step/1": "step2 reply",
    }
    assert not (log_dir / root / "attempt_0").exists()
    assert not (log_dir / content_id("zq1")).exists()


def test_a_resume_warns_of_the_inputs_its_kept_plan_leaves_out(tmp_path, caplog):
    out, first = tmp_path / "jb.csv", _inputs().head(3)
    generate_jailbreaks(inputs=first, output_path=out, seed=1, **PURE_KW)
    with caplog.at_level("WARNING", logger="redact.jailbreak.pipeline"):
        again = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1,
                                    resume=True, **PURE_KW)
    assert len(again) == len(first)     # the kept plan: the new inputs are not run
    (warning,) = [r.getMessage() for r in caplog.records]
    assert "2 of the 5 inputs" in warning and "resume=False" in warning


def test_a_resume_whose_plan_holds_every_input_does_not_warn(tmp_path, caplog):
    out = tmp_path / "jb.csv"
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **PURE_KW)
    with caplog.at_level("WARNING", logger="redact.jailbreak.pipeline"):
        generate_jailbreaks(inputs=_inputs().head(3), output_path=out, seed=1,
                            resume=True, **PURE_KW)
    assert caplog.records == []
