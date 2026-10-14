"""Offline end-to-end tests for the jailbreak pipeline (plan + execute + resume).

Uses ``pure_only=True`` with only the obfuscation layer so no technique needs an
LLM — the engine completes every sample at prime time and the router is never
called. This exercises planning, execution, manifest, the prompt-content
id, and resume without any network access. The ``pool`` tests at the end run
LLM-backed fake techniques over a hand-written plan against a fake backend.
"""


import pandas as pd
import pytest

from redact.dataset import Manifest, content_id
from redact.exceptions import PromptTooLongError
from redact.jailbreak.protocol import LLMRequest
from redact.llms.backends import ComputeConfig
from redact.pipelines import generate_jailbreaks
from tests.conftest import (
    Killed,
    MockBackend,
    kill_after_rows,
    lock_is_free,
    make_client,
)

PURE_KW = dict(
    pure_only=True,
    include_hacking=False,
    include_manipulation=False,
    include_obfuscation=True,
    include_requests=False,
    auto_generate_benign=False,
    verbose=False,
)


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


def test_iterations_dedup(tmp_path):
    out = tmp_path / "jb.csv"
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=3,
                             iterations=3, max_obfuscations=1, **PURE_KW)
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


def test_settings_per_iteration_escalation(tmp_path):
    """Multi-round escalation: 4 rounds per sample, exact counts for rounds 0-1."""
    out = tmp_path / "jb.csv"
    schedule = [
        {"exact_techniques": 1},
        {"exact_techniques": 2},
        {"max_complexity": 6, "max_obfuscations": 2},
        {"max_complexity": 9, "max_obfuscations": 3},
    ]
    df = generate_jailbreaks(inputs=_inputs(), output_path=out, seed=11,
                             settings_per_iteration=schedule, **PURE_KW)

    # iterations derived from schedule length: 5 samples x 4 rounds.
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
    generate_jailbreaks(inputs=_inputs(), output_path=out, seed=11, max_complexity=1,
                        settings_per_iteration=[{"max_obfuscations": 3}], **PURE_KW)
    plan = Manifest(out.with_name("jb.manifest.jsonl")).load()
    assert {r["settings"]["max_complexity"] for r in plan} == {1}
    assert {r["settings"]["max_obfuscations"] for r in plan} == {3}
    # No pure obfuscation costs 0, so a budget of 1 ends after the first pick.
    assert all(len(r["combination"]) == 1 for r in plan)


def test_settings_per_iteration_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv",
                            settings_per_iteration=[], **PURE_KW)


# --- the inputs of the run ------------------------------------------------------


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "standalone"])
def test_without_inputs_the_inputs_of_the_run_are_found(tmp_path, seeded):
    from tests.test_pipelines import _seed_input_tree

    text = _seed_input_tree(tmp_path, seeded)
    df = generate_jailbreaks(data_dir=tmp_path, seed=7, **PURE_KW)
    assert list(df["prompt"]) == [text]
    assert list(df["input_id"]) == [content_id(text)]


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
    out, finished = tmp_path / "jb.csv", 3
    generate_jailbreaks(inputs=_inputs().iloc[:finished], output_path=out, seed=7,
                        **PURE_KW)
    Manifest.sidecar(out).reset()       # the next run plans all five inputs
    read = _plan_rows_read(monkeypatch, units_taken=1)
    with pytest.raises(KeyboardInterrupt):
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=7, **PURE_KW)
    assert len(read) == finished + 1    # the finished units, then the one taken


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


def _llm_run(tmp_path, monkeypatch, plan, backend, **kwargs):
    """Run the stage on ``plan``, ``[(prompt, technique), ...]``, written as its manifest."""
    import redact.jailbreak.pool as JPool
    import redact.pipelines as P
    from redact.dataset import Manifest

    monkeypatch.setattr(JPool, "all_techniques", lambda: [one_step, two_step, broken])
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))

    out = tmp_path / "jb.csv"
    Manifest(out.with_name("jb.manifest.jsonl")).write(
        {"sample_id": content_id(p), "iteration": 0, "combination": [t.__name__],
         "settings": {}, "status": "planned"}
        for p, t in plan
    )
    prompts = [p for p, _ in plan]
    inputs = pd.DataFrame({"prompt": prompts, "category": ["Cyber"] * len(prompts),
                           "entry_type": ["harmful"] * len(prompts)})
    generate_jailbreaks(inputs=inputs, output_path=out, model=GEN, resume=True,
                        verbose=False, **kwargs)
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
         "sample_id": content_id(f"{content_id('zq2')}:0")},
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
        "sample_id": content_id(f"{content_id('zq1')}:0"), "jailbreak": "zq1",
        "technique": "two_step",
        "technique_info": f"ERROR: {message}", "is_noop": "True",
        "accepted": "False", "reasoning": message}
    assert {p: (rows[p]["jailbreak"], rows[p]["accepted"]) for p in ("zq0", "zq2")} == {
        "zq0": ("S1:zq0", "True"), "zq2": ("S1:zq2", "True")}
    assert len(df) == len(plan)
    ledger = Ledger.sidecar(out, key_fields=("input_id", "iteration"),
                            casters={"input_id": str, "iteration": int})
    assert ledger.completed() == {(content_id(p), 0) for p, _ in plan}
