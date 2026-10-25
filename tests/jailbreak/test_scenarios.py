"""The scenarios table: the scene each cognitive or persona technique built its
prompt on, beside the kind's rows, when the stage's ``scenarios`` is on.

The model replies are stand-ins: a scenario request gets ``SCENE[<input>]``
between delimiter lines, its check (at the check temperature) ``Yes``, and the
construction, whose messages hold the scene, a prompt built on it.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from redact.dataset import Manifest, content_id
from redact.dataset.planfile import scenarios_path
from redact.jailbreak.chain import run_chain
from redact.jailbreak.hacking.cognitive import to_hypothetical_framing
from redact.jailbreak.hacking.personas import to_artist_persona, to_invented_persona
from redact.jailbreak.scenarios import COLUMNS, write_scenarios
from redact.jailbreak.technique import DISCARDED
from redact.llm_pipeline import CHECK_TEMPERATURE, drive_sync
from redact.stages import generate_jailbreaks
from tests.conftest import Killed, MockBackend, kill_after_rows, make_client

PROMPT = "sample request 1"
ONE = "Notes\n---\nscene one\n---"
TWO = "Notes\n---\nscene two\n---"
GEN = "scenario-gen"


def _drive(gen, replies: list[str]):
    """Answer ``gen``'s requests with ``replies`` in turn; its result."""
    answers = iter(replies)
    return drive_sync(gen, lambda request: next(answers))


# --- the techniques record the scene they built on ----------------------------

@pytest.mark.parametrize("technique", [to_hypothetical_framing, to_artist_persona])
def test_the_accepted_scene_is_recorded_once(technique):
    recorded = []
    text, info = _drive(
        technique(PROMPT, gen_model="gen", record_scenario=recorded.append),
        [ONE, "No: off topic.", TWO, "Yes", "constructed"])
    assert (text, info.startswith("category=")) == ("constructed", True)
    assert recorded == ["scene two"]


@pytest.mark.parametrize("technique", [to_hypothetical_framing, to_artist_persona])
def test_a_scenario_rejected_every_time_records_nothing(technique):
    recorded = []
    _text, info = _drive(
        technique(PROMPT, gen_model="gen", record_scenario=recorded.append),
        [ONE, "No: off topic.", TWO, "No: still off topic."])
    assert info.startswith(DISCARDED) and recorded == []


def test_a_given_scenario_is_the_one_recorded():
    recorded = []
    _drive(to_hypothetical_framing(PROMPT, gen_model="gen", scenario="a given scene",
                                   record_scenario=recorded.append), ["constructed"])
    assert recorded == ["a given scene"]


def test_the_invented_persona_records_the_scene_after_its_persona():
    recorded = []
    _drive(to_invented_persona(PROMPT, gen_model="gen",
                               record_scenario=recorded.append),
           ["Name: Ada\nDescription: a careful planner", ONE, "Yes", "constructed"])
    assert recorded == ["scene one"]


def test_without_a_recorder_a_technique_runs_as_before():
    assert _drive(to_hypothetical_framing(PROMPT, gen_model="gen"),
                  [ONE, "Yes", "constructed"])[0] == "constructed"


# --- the chain hands each technique a recorder of its own ---------------------

def _records(text, *, record_scenario=None):
    """A pure technique that records a scene when it is given a recorder."""
    if record_scenario is not None:
        record_scenario(f"scene of {text}")
    return text + "!", ""


def _plain(text):
    return text + "?", ""


def test_the_chain_names_the_technique_of_each_scene():
    scenarios = []
    chain = run_chain([_records, _plain, to_hypothetical_framing], PROMPT,
                      gen_model="gen", scenarios=scenarios)
    _drive(chain, [ONE, "Yes", "constructed"])
    assert scenarios == [("_records", f"scene of {PROMPT}"),
                         ("to_hypothetical_framing", "scene one")]


def test_without_a_list_the_chain_gives_no_recorder():
    seen = []

    def records(text, *, record_scenario=None):
        seen.append(record_scenario)
        return text + "!", ""

    assert _drive(run_chain([records], PROMPT), []) == (PROMPT + "!", "")
    assert seen == [None]


# --- the stage ----------------------------------------------------------------

class _ScenarioBackend(MockBackend):
    """A check gets ``Yes``; a construction, whose messages hold the scene,
    ``built on <scene>``; any other request a scene naming its input."""

    def __init__(self):
        super().__init__(model=GEN)

    def generate(self, messages_list, *, temperature=None, **kwargs):
        super().generate(messages_list, temperature=temperature, **kwargs)
        replies = []
        for messages in messages_list:
            text = "\n".join(m["content"] for m in messages)
            if temperature == CHECK_TEMPERATURE:
                replies.append("Yes")
            elif scene := re.search(r"SCENE\[zq\d*\]", text):
                replies.append(f"built on {scene.group()}")
            else:
                # The input a request names; an encoded one names none.
                named = re.search(r"zq\d+", text)
                replies.append(f"Notes\n---\nSCENE[{named.group() if named else 'zq'}]"
                               "\n---")
        return replies


#: Each input's combination: a cognitive technique, a persona and an
#: encoding after it, and an encoding alone.
_PLAN = {"zq0 request": ["to_hypothetical_framing"],
         "zq1 request": ["to_artist_persona", "to_rot13"],
         "zq2 request": ["to_rot13"]}


def _inputs(prompts) -> pd.DataFrame:
    return pd.DataFrame({"prompt": list(prompts), "category": "Cyber",
                         "entry_type": "harmful"})


def _run(tmp_path, monkeypatch, *, prompts=tuple(_PLAN), **kwargs):
    """Write :data:`_PLAN` as the kind's manifest (when it has none) and run
    the stage on ``prompts``; the jailbreaks CSV's path."""
    import redact.stages as P

    out = tmp_path / "jb.csv"
    manifest = Manifest(out.with_name("jb.manifest.jsonl"))
    if not manifest.path.exists():
        manifest.write({"sample_id": content_id(prompt), "iteration": 0,
                        "combination": names, "settings": {}, "status": "planned"}
                       for prompt, names in _PLAN.items())
    backend = _ScenarioBackend()
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
    generate_jailbreaks(inputs=_inputs(prompts), output_path=out, model=GEN,
                        augmentation={"natural": None}, verbose=False, **kwargs)
    return out


def _table(out) -> pd.DataFrame:
    return pd.read_csv(scenarios_path(out.with_name("jb.manifest.jsonl")), dtype=str,
                       keep_default_na=False)


def _expected(out, prompts=tuple(_PLAN)[:2]) -> list[dict]:
    """The table rows the units of ``prompts`` give, by their output rows."""
    rows = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("input_id")
    techniques = {"zq0": "to_hypothetical_framing", "zq1": "to_artist_persona"}
    return [{"input_id": content_id(prompt), "iteration": "0",
             "sample_id": rows.loc[content_id(prompt), "sample_id"],
             "technique": techniques[prompt[:3]], "scenario": f"SCENE[{prompt[:3]}]"}
            for prompt in prompts]


def _sorted(table: pd.DataFrame) -> list[dict]:
    return sorted(table.to_dict("records"), key=lambda row: row["input_id"])


def test_each_scene_is_a_row_beside_the_rows_of_the_kind(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, scenarios=True)
    table = _table(out)
    assert list(table.columns) == COLUMNS
    assert _sorted(table) == sorted(_expected(out), key=lambda row: row["input_id"])
    # The built prompt is the row's text; the scene is in the table only.
    rows = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert "scenario" not in rows.columns
    assert not rows["technique_info"].str.contains("SCENE").any()


def test_without_scenarios_no_table_is_written(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    assert not scenarios_path(out.with_name("jb.manifest.jsonl")).exists()
    assert len(pd.read_csv(out)) == len(_PLAN)


@pytest.mark.parametrize("on_call", [1, 2], ids=["after the rows", "after the scenes"])
def test_a_save_killed_before_its_acks_leaves_each_scene_once(
        tmp_path, monkeypatch, on_call):
    """The rows land, then the scenes, then the acks: a save killed after
    the rows, or after the scenes, runs its units again."""
    import redact.jailbreak.pipeline as JPipe

    kill_after_rows(monkeypatch, JPipe, on_call=on_call)
    with pytest.raises(Killed):
        _run(tmp_path, monkeypatch, scenarios=True)
    out = _run(tmp_path, monkeypatch, scenarios=True)
    assert len(pd.read_csv(out)) == len(_PLAN)
    assert _sorted(_table(out)) == sorted(_expected(out),
                                          key=lambda row: row["input_id"])


def test_a_resume_keeps_the_scenes_of_earlier_runs(tmp_path, monkeypatch):
    """The first run holds the input of the first unit only; the second
    runs the others, and the table keeps the first unit's scene."""
    _run(tmp_path, monkeypatch, prompts=["zq0 request"], scenarios=True)
    out = _run(tmp_path, monkeypatch, scenarios=True)
    assert _sorted(_table(out)) == sorted(_expected(out),
                                          key=lambda row: row["input_id"])


def test_a_fresh_run_removes_the_table(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, scenarios=True)
    table = scenarios_path(out.with_name("jb.manifest.jsonl"))
    assert table.exists()
    pure = {"pure_only": True, "layers": {"deny": ["hacking", "manipulation",
                                                   "requests"]}}
    generate_jailbreaks(inputs=_inputs(_PLAN), output_path=out, resume=False,
                        augmentation={"adversarial": pure, "natural": None},
                        auto_generate_benign=False, verbose=False)
    assert not table.exists()


def test_a_kind_skipped_on_a_fresh_run_clears_its_table(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, scenarios=True)
    table = scenarios_path(out.with_name("jb.manifest.jsonl"))
    generate_jailbreaks(inputs=_inputs(_PLAN), output_path=out, resume=False,
                        augmentation={"adversarial": {"entry_types": ["benign"]},
                                      "natural": None}, verbose=False)
    assert not table.exists() and not out.exists()


# --- the table kept to the kind's rows ------------------------------------------

def _write(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def test_rows_of_no_output_row_go_and_a_repeated_one_keeps_its_last(tmp_path):
    out, path = tmp_path / "jb.csv", tmp_path / "jb.scenarios.csv"
    pd.DataFrame({"input_id": ["a", "b"], "iteration": [0, 0],
                  "sample_id": ["sa", "sb"], "jailbreak": ["x", "y"]}).to_csv(
        out, index=False)
    _write(path, [["a", 0, "sa", "to_avi", "first"],
                  ["b", 0, "s-old", "to_avi", "of a cut row"],
                  ["a", 0, "sa", "to_avi", "again"],
                  ["b", 0, "sb", "to_avi", "kept"]])
    assert write_scenarios(out, path) == path
    assert pd.read_csv(path, dtype=str).values.tolist() == [
        ["a", "0", "sa", "to_avi", "again"], ["b", "0", "sb", "to_avi", "kept"]]


def test_a_table_that_matches_the_output_is_not_written_again(tmp_path):
    out, path = tmp_path / "jb.csv", tmp_path / "jb.scenarios.csv"
    pd.DataFrame({"input_id": ["a"], "iteration": [0], "sample_id": ["sa"]}).to_csv(
        out, index=False)
    _write(path, [["a", 0, "sa", "to_avi", "a scene"]])
    before = path.stat().st_mtime_ns, path.read_bytes()
    write_scenarios(out, path)
    assert (path.stat().st_mtime_ns, path.read_bytes()) == before


def test_no_table_is_none(tmp_path):
    assert write_scenarios(tmp_path / "jb.csv", tmp_path / "jb.scenarios.csv") is None


def test_recording_the_scenes_changes_no_row_of_the_default_attacks(
        tmp_path, monkeypatch):
    """Every technique of the packaged attack pool runs with the recorder
    in its chain's kwargs (those taking ``**kwargs`` get it too), and writes
    what it writes without one."""
    import redact.stages as P
    from redact.jailbreak.obfuscation import translation

    backend = _ScenarioBackend()
    monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
    monkeypatch.setattr(translation, "is_machine_translation", lambda model: False)
    prompts = [f"zq{i} request about a topic" for i in range(40)]
    written = {}
    for record in (False, True):
        out = tmp_path / str(record) / "jb.csv"
        generate_jailbreaks(inputs=_inputs(prompts), output_path=out, model=GEN,
                            translation_model=GEN, augmentation={"natural": None},
                            auto_generate_benign=False, benign_path=_benign(tmp_path),
                            verbose=False, scenarios=record)
        written[record] = pd.read_csv(out, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(written[False], written[True])
    rows = written[True]
    techniques = set("+".join(rows["technique"]).split("+"))
    assert len(techniques) > 60  # noqa: PLR2004  (the pool's families, broadly)
    table = _table(tmp_path / "True" / "jb.csv")
    scened = rows[rows["sample_id"].isin(table["sample_id"])]
    assert len(table) and set(table["technique"]) <= techniques
    assert all(name in chain for name, chain in zip(
        table["technique"], scened.set_index("sample_id").loc[
            table["sample_id"], "technique"], strict=True))


def _benign(tmp_path) -> Path:
    """A benign cache of one pair of each answer type, for fsh and dap."""
    path = tmp_path / "benign.csv"
    pd.DataFrame({
        "main_category": ["General"] * 2, "sub_category": ["Cooking"] * 2,
        "answer_type": ["short", "long"], "question": ["How to boil eggs?"] * 2,
        "answer": ["Boil them.", "Put the eggs in boiling water for nine minutes."],
    }).to_csv(path, index=False)
    return path
