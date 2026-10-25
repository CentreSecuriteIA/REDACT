"""``scripts/persona_smoke.py``: the counts it makes, on a fake model."""

import importlib.util
import json
from pathlib import Path

import pytest

from redact.jailbreak import all_techniques
from redact.llm_pipeline import CHECK_TEMPERATURE

#: What the fake model writes: it must never reach the output.
SECRET = "SECRET-MODEL-TEXT"
SCENE = f"Notes\n---\n{SECRET}\n---"
SENTENCES = ["first sentence", "second sentence", "third sentence"]
UNITS = 2  # the units a technique runs in the main() tests


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "persona_smoke", Path(__file__).parents[1] / "scripts" / "persona_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _technique(name: str):
    return next(fn for fn in all_techniques() if fn.__name__ == name)


def _run(script, name: str, verdict: str, scene: str = SCENE, **kwargs):
    """``name`` over the three sentences; the check says ``verdict``, every
    other call writes ``scene``."""
    def call(request):
        return verdict if request.temperature == CHECK_TEMPERATURE else scene

    return script.run_smoke([_technique(name)], SENTENCES, call, gen_model="m",
                            **kwargs)[name]


def test_an_accepted_persona_unit_takes_a_scene_a_check_and_a_construction(script):
    row = _run(script, "to_alien_persona", "Yes")
    assert (row["units"], row["accepted"], row["calls"]) == (3, 3, 9)
    assert not row["discarded"] and not row["errors"]


def test_a_scene_the_check_rejects_twice_is_a_discard_counted_by_reason(script):
    row = _run(script, "to_alien_persona", "No: off topic.")
    assert row["accepted"] == 0
    assert dict(row["discarded"]) == {"scenario check failed": 3}
    assert row["calls"] == 4 * len(SENTENCES)  # two scenes and their checks, a unit


def test_a_reply_without_a_scene_is_a_discard_counted_by_reason(script):
    row = _run(script, "to_avi", "Yes", scene="no scene here")
    assert dict(row["discarded"]) == {"malformed scenario": 3}
    assert row["calls"] == 3 * len(SENTENCES)  # three replies a unit, no check


def test_an_invented_persona_whose_replies_are_no_persona_is_counted(script):
    row = _run(script, "to_invented_persona", "Yes")
    assert dict(row["discarded"]) == {"malformed persona": 3}


def test_a_unit_that_raises_is_an_error_count_not_a_stop(script):
    def call(request):
        raise ConnectionError("down")

    row = script.run_smoke(
        [_technique("to_alien_persona")], SENTENCES, call, gen_model="m"
    )["to_alien_persona"]
    assert (row["units"], row["accepted"], dict(row["errors"])) == (
        3, 0, {"ConnectionError": 3})


def test_the_check_model_takes_the_check_requests(script):
    seen = []

    def call(request):
        seen.append((request.model, request.temperature == CHECK_TEMPERATURE))
        return "Yes" if request.temperature == CHECK_TEMPERATURE else SCENE

    script.run_smoke([_technique("to_alien_persona")], SENTENCES[:1], call,
                     gen_model="gen", check_model="check")
    assert seen == [("gen", False), ("check", True), ("gen", False)]


def test_the_techniques_are_the_active_ones_of_the_families(script):
    chosen = script.select_techniques(all_techniques(), ["persona"])
    names = {fn.__name__ for fn in chosen}
    assert "to_invented_persona" in names and "to_alien_persona" in names
    assert "to_mentally_ill_persona" not in names  # retired by spec 1.1
    assert all("persona" in fn.families for fn in chosen)
    both = script.select_techniques(all_techniques(), ["persona", "cognitive"])
    assert {"to_avi", "to_alien_persona"} <= {fn.__name__ for fn in both}


def test_a_technique_name_outside_the_families_raises(script):
    with pytest.raises(ValueError, match="to_avi"):
        script.select_techniques(all_techniques(), ["persona"], ["to_avi"])


def test_main_prints_counts_only_and_writes_the_json(script, monkeypatch, capsys,
                                                     tmp_path):
    def call(request):
        return "Yes" if request.temperature == CHECK_TEMPERATURE else SCENE

    monkeypatch.setattr(script, "make_call", lambda model: call)
    out = tmp_path / "counts.json"
    code = script.main(["-n", str(UNITS), "--model", "m", "--families", "persona",
                        "--techniques", "to_alien_persona,to_actor_persona",
                        "--json", str(out)])
    printed = capsys.readouterr()
    assert code == 0
    assert SECRET not in printed.out + printed.err
    assert "to_alien_persona" in printed.out and "to_actor_persona" in printed.out
    assert f"total: {2 * UNITS} units, {2 * UNITS} accepted" in printed.out
    counts = json.loads(out.read_text(encoding="utf-8"))
    assert counts["to_actor_persona"]["accepted"] == UNITS
    assert SECRET not in out.read_text(encoding="utf-8")


def test_main_runs_n_sentences_from_a_short_list(script, monkeypatch, capsys, tmp_path):
    inputs = tmp_path / "s.txt"
    inputs.write_text("one\n\ntwo\n", encoding="utf-8")
    monkeypatch.setattr(script, "make_call", lambda model: lambda request: "Yes")
    script.main(["-n", "5", "--model", "m", "--inputs", str(inputs),
                 "--families", "persona", "--techniques", "to_alien_persona"])
    assert "total: 5 units" in capsys.readouterr().out
