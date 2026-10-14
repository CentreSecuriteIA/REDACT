"""Offline tests for the multi_turn conversation core (single-conversation path).

A fake ``call(LLMRequest) -> str`` stands in for the model, so this exercises the
actor protocol, the seed-then-alternate runner, the typed step log, and stop
conditions without any network.
"""

import pytest

from redact.llm_pipeline import LLMRequest, drive_sync
from redact.multi_turn import (
    Actor,
    ModelActor,
    ScriptedActor,
    Setting,
    Transcript,
    run_conversation,
)
from tests.conftest import (
    FakeResolver,
    JudgeResolver,
    MockBackend,
    echo,
    simple_setting,
)


def _echo_call(req: LLMRequest) -> str:
    # deterministic "model": echo the last user message content
    last_user = [m for m in req.messages if m["role"] == "user"][-1]["content"]
    return f"ANSWER to: {last_user}"


def test_llmrequest_is_shared_with_jailbreak():
    # The re-export must be the *same* class so the engine keeps working.
    from redact.jailbreak.protocol import LLMRequest as JBRequest
    assert JBRequest is LLMRequest


def test_transcript_renders_only_visible_steps():
    t = Transcript()
    t.message("driver", "hi", role="user")
    t.note("strategy", "driver", "I'll be friendly")   # provenance, not a message
    t.reply("bot", "hello", role="assistant")
    msgs = t.as_messages(system="SYS")
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_two_actor_alternate_conversation():
    driver = ScriptedActor("driver", ["follow-up 1", "follow-up 2"], role="user")
    bot = ModelActor("bot", model="m", role="assistant")
    setting = Setting(participants=[driver, bot], max_turns=4, name="chat")

    traj = run_conversation(setting, seed="opening question", call=_echo_call, input_id="s1")

    steps = traj.transcript.steps
    # turn0 seed (message) -> bot reply -> driver follow-up -> bot reply
    assert [s.type for s in steps] == ["message", "reply", "message", "reply"]
    assert steps[0].content == "opening question" and steps[0].role == "user"
    assert steps[1].content == "ANSWER to: opening question"
    assert steps[2].content == "follow-up 1"
    assert steps[3].content == "ANSWER to: follow-up 1"
    assert traj.turns_used == 4 and traj.stop_reason == "max_turns"
    assert traj.input_id == "s1" and traj.setting == "chat"


def test_stop_condition_ends_early():
    bot = ModelActor("bot", model="m", role="assistant")
    driver = ScriptedActor("driver", ["more"], role="user")
    # stop as soon as any reply contains "opening"
    setting = Setting(
        participants=[driver, bot], max_turns=10,
        stop=lambda t: any(s.type == "reply" and "opening" in s.content for s in t.steps),
    )
    traj = run_conversation(setting, seed="opening", call=_echo_call)
    assert traj.stop_reason == "stop_condition"
    assert traj.turns_used == 2  # seed + first bot reply (which echoes "opening")


def test_strategy_note_logged_as_provenance():
    t = Transcript()
    t.note("analysis", "attacker", "the target refused; escalate")
    recs = t.to_records()
    assert recs[0]["type"] == "analysis" and recs[0]["role"] is None
    assert t.as_messages() == []  # provenance never rendered to the model


def test_records_are_copies_of_the_steps():
    t = Transcript()
    t.message("user", "hello", meta={"turn": 1})
    record = t.to_records()[0]
    record["content"] = "changed"
    record["meta"]["turn"] = 2
    assert t.steps[0].content == "hello"
    assert t.steps[0].meta == {"turn": 1}


def _two_turns():
    t = Transcript()
    t.message("u", "goal", role="user")
    t.reply("b", "first", role="assistant", model="m")
    t.note("strategy", "u", "push")
    t.message("u", "more", role="user")
    t.reply("b", "second", role="assistant", model="m")
    return t


def test_transcript_first_message_and_last_reply():
    t, empty = _two_turns(), Transcript()
    assert (t.first_message(), t.last_reply()) == ("goal", "second")
    assert (empty.first_message(), empty.last_reply()) == ("", "")


def test_transcript_render_lists_the_visible_turns():
    assert _two_turns().render() == (
        "user: goal\nassistant: first\nuser: more\nassistant: second")
    assert Transcript().render() == ""


def test_transcript_from_records_reads_back_to_records():
    t = _two_turns()
    back = Transcript.from_records(t.to_records())
    assert back == t
    assert back.to_records() == t.to_records()


def test_generate_conversations_offline(tmp_path):
    import json

    import pandas as pd

    from redact import generate_conversations, paths
    from redact.multi_turn import ModelActor, ScriptedActor, Setting

    seeds = pd.DataFrame({
        "seed": ["q1", "q2"], "category": ["Cyber", "Cyber"],
        "entry_type": ["harmful", "harmful"],
    })

    def make():  # fresh Setting per conversation (stateful ScriptedActor)
        return Setting(
            participants=[ScriptedActor("user", ["follow"]), ModelActor("bot", "m")],
            max_turns=4, name="chat",
        )

    df = generate_conversations(seeds, make, data_dir=tmp_path, resolve=FakeResolver(echo), verbose=False)
    assert len(df) == 2
    steps = json.loads(df.iloc[0]["transcript"])
    assert [s["type"] for s in steps] == ["message", "reply", "message", "reply"]
    assert set(df["turns_used"]) == {4}

    out = paths.conversations_csv(tmp_path)
    assert out.exists()
    assert out.with_name("conversations.manifest.jsonl").exists()
    assert out.with_name("conversations.state.jsonl").exists()

    # Resume: nothing new (both units in the ledger).
    df2 = generate_conversations(seeds, make, data_dir=tmp_path, resolve=FakeResolver(echo), verbose=False)
    assert len(df2) == 2


def test_generate_conversations_variants(tmp_path):
    import pandas as pd

    from redact import generate_conversations
    from redact.multi_turn import ModelActor, ScriptedActor, Setting

    seeds = pd.DataFrame({"seed": ["q"]})
    make = lambda: Setting(participants=[ScriptedActor("u", ["f"]), ModelActor("b", "m")], max_turns=3)
    df = generate_conversations(seeds, make, data_dir=tmp_path, iterations=3,
                                resolve=FakeResolver(echo), verbose=False)
    assert len(df) == 3 and set(df["iteration"]) == {0, 1, 2}


def test_evaluate_conversations_offline(tmp_path):
    import pandas as pd

    from redact import evaluate_conversations, generate_conversations
    from redact.multi_turn import ModelActor, ScriptedActor, Setting

    seeds = pd.DataFrame({"seed": ["how to X", "how to Y"]})
    make = lambda: Setting(participants=[ScriptedActor("u", ["more"]), ModelActor("b", "m")], max_turns=2)
    generate_conversations(seeds, make, data_dir=tmp_path, resolve=FakeResolver(echo), verbose=False)

    # The judge says "Yes ..." to the first, "No ..." to the second.
    scored = evaluate_conversations(
        data_dir=tmp_path, judge_model="judge", judge_system="did it work?",
        scope="last_reply", verbose=False,
        resolve=JudgeResolver(MockBackend(["Yes ok", "No refused"])),
    )
    assert set(scored["success"]) == {True, False}
    assert set(scored["judge_model"]) == {"judge"}
    # Resume: nothing re-judged, so no resolver is needed.
    again = evaluate_conversations(data_dir=tmp_path, judge_model="judge", verbose=False)
    assert len(again) == 2
    # resume=False re-judges from scratch (unlinks the existing scored artifacts).
    fresh = evaluate_conversations(
        data_dir=tmp_path, judge_model="judge", resume=False, verbose=False,
        resolve=JudgeResolver(MockBackend(["Yes ok", "No refused"])),
    )
    assert len(fresh) == 2


# --- core edge cases ---------------------------------------------------------

def test_actor_base_is_abstract():
    gen = Actor("x").turn(Transcript())
    with pytest.raises(NotImplementedError):
        next(gen)


def test_scripted_actor_empty_messages_returns_no_steps():
    steps = drive_sync(ScriptedActor("s", [], role="user").turn(Transcript()), lambda r: "")
    assert steps == []


def test_model_actor_uses_system_prompt():
    seen = {}

    def call(req):
        seen["msgs"] = req.messages
        return "reply"

    t = Transcript()
    t.message("u", "hi", role="user")
    steps = drive_sync(ModelActor("bot", "m", system_prompt="SYS").turn(t), call)
    assert seen["msgs"][0] == {"role": "system", "content": "SYS"}
    assert steps[0].type == "reply" and steps[0].content == "reply"


def test_conversation_gen_requires_participants():
    from redact.multi_turn import conversation_gen
    with pytest.raises(ValueError):
        next(conversation_gen(Setting(participants=[]), "seed"))


# --- pipeline edge cases -----------------------------------------------------

def test_generate_conversations_missing_text_column(tmp_path):
    import pandas as pd

    from redact import generate_conversations
    with pytest.raises(ValueError):
        generate_conversations(pd.DataFrame({"foo": ["x"]}), simple_setting,
                               data_dir=tmp_path, resolve=FakeResolver(echo), verbose=False)


def test_generate_conversations_empty_seeds(tmp_path):
    import pandas as pd

    from redact import generate_conversations
    with pytest.raises(ValueError):
        generate_conversations(pd.DataFrame({"seed": []}), simple_setting,
                               data_dir=tmp_path, verbose=False)


def test_generate_conversations_uses_id_column(tmp_path):
    import pandas as pd

    from redact import generate_conversations
    seeds = pd.DataFrame({"sample_id": ["myid"], "seed": ["q"]})
    df = generate_conversations(seeds, simple_setting, data_dir=tmp_path,
                                resolve=FakeResolver(echo), verbose=False)
    assert df.iloc[0]["input_id"] == "myid"


class _BoomActor(Actor):
    def turn(self, transcript):
        if False:  # pragma: no cover
            yield
        raise RuntimeError("boom")


def test_generate_conversations_actor_error_isolated(tmp_path):
    import pandas as pd

    from redact import generate_conversations
    seeds = pd.DataFrame({"seed": ["q"]})
    make = lambda: Setting(participants=[ScriptedActor("u", ["q"]), _BoomActor("boom")], max_turns=3)
    df = generate_conversations(seeds, make, data_dir=tmp_path, resolve=FakeResolver(echo), verbose=False)
    assert df.iloc[0]["stop_reason"].startswith("ERROR")


def test_generate_conversations_deepcopy_template(tmp_path):
    # passing a Setting instance (not a factory) deep-copies per unit.
    import pandas as pd

    from redact import generate_conversations
    seeds = pd.DataFrame({"seed": ["a", "b"]})
    tmpl = Setting(participants=[ScriptedActor("u", ["f"]), ModelActor("b", "m")], max_turns=2)
    df = generate_conversations(seeds, tmpl, data_dir=tmp_path, resolve=FakeResolver(echo), verbose=False)
    assert len(df) == 2  # ScriptedActor state didn't leak across the two conversations


# --- evaluate edge cases -----------------------------------------------------

def test_evaluate_validation(tmp_path):
    from redact import evaluate_conversations
    with pytest.raises(ValueError):
        evaluate_conversations(data_dir=tmp_path, judge_model=None)          # no judge
    with pytest.raises(ValueError):
        evaluate_conversations(data_dir=tmp_path, judge_model="j", scope="bogus")
    with pytest.raises(ValueError):
        evaluate_conversations(data_dir=tmp_path, judge_model="j")           # no conversations.csv


def test_evaluate_transcript_scope_and_bad_json(tmp_path):
    import pandas as pd

    from redact import evaluate_conversations
    convs = pd.DataFrame({"sample_id": ["a", "b"], "transcript": ["not json", "[]"]})
    scored = evaluate_conversations(conversations=convs, data_dir=tmp_path,
                                    judge_model="j", scope="transcript", verbose=True,
                                    resolve=JudgeResolver(MockBackend("Yes ok")))
    assert len(scored) == 2 and set(scored["success"]) == {True}


# --- ledger robustness + verbose/fresh branches ------------------------------

def test_pipeline_ledger_keys_on_input_id_iteration(tmp_path):
    # the conversation ledger keys on (input_id, iteration), skipping bad lines
    from redact.dataset import Ledger
    led = Ledger.sidecar(tmp_path / "conversations.csv",
                         key_fields=("input_id", "iteration"),
                         casters={"input_id": str, "iteration": int})
    led.path.write_text('\n{"input_id":"a","iteration":0}\nnot json\n{"iteration":1}\n', encoding="utf-8")
    assert led.completed() == {("a", 0)}  # blank + bad-json + missing-key all skipped


def test_evaluate_ledger_keys_on_id(tmp_path):
    from redact.dataset import Ledger
    led = Ledger.sidecar(tmp_path / "conversations_scored.csv",
                         key_fields=("sample_id",), casters={"sample_id": str})
    led.path.write_text('{"sample_id":"a"}\nbad\n{}\n', encoding="utf-8")
    assert led.completed() == {"a"}


def test_generate_conversations_verbose_and_fresh(tmp_path):
    import pandas as pd

    from redact import generate_conversations
    seeds = pd.DataFrame({"seed": ["q"]})
    generate_conversations(seeds, simple_setting, data_dir=tmp_path, resolve=FakeResolver(echo), verbose=True)
    # resume=False wipes the artifacts + re-runs.
    df2 = generate_conversations(seeds, simple_setting, data_dir=tmp_path,
                                 resolve=FakeResolver(echo), resume=False, verbose=True)
    assert len(df2) == 1


def test_both_stages_return_no_rows_without_read_back(tmp_path):
    import pandas as pd

    from redact import evaluate_conversations, generate_conversations, paths

    df = generate_conversations(_three_seeds(), simple_setting, data_dir=tmp_path,
                                resolve=FakeResolver(echo), verbose=False,
                                read_back=False)
    scored = evaluate_conversations(data_dir=tmp_path, judge_model="j", verbose=False,
                                    resolve=JudgeResolver(MockBackend("Yes ok")),
                                    read_back=False)
    assert df.empty and scored.empty
    convs = pd.read_csv(paths.conversations_csv(tmp_path))
    assert list(convs["input_id"]) == ["s1", "s2", "s3"]
    scored_ids = pd.read_csv(paths.conversations_scored_csv(tmp_path))["sample_id"]
    assert sorted(scored_ids) == sorted(convs["sample_id"])


def test_pipeline_ledger_empty_record_is_noop(tmp_path):
    from redact.dataset import Ledger
    led = Ledger.sidecar(tmp_path / "conversations.csv",
                         key_fields=("input_id", "iteration"))
    led.record([])
    assert not led.exists()


def test_evaluate_transcript_scope_renders_and_fresh(tmp_path):
    import json

    import pandas as pd

    from redact import evaluate_conversations
    transcript = json.dumps([
        {"type": "message", "actor": "u", "content": "goal here", "role": "user", "model": None, "meta": {}},
        {"type": "reply", "actor": "b", "content": "the answer", "role": "assistant", "model": "m", "meta": {}},
    ])
    convs = pd.DataFrame({"sample_id": ["a"], "transcript": [transcript]})
    scored = evaluate_conversations(conversations=convs, data_dir=tmp_path, judge_model="j",
                                    scope="transcript", resume=False, verbose=False,
                                    resolve=JudgeResolver(MockBackend("Yes")))
    assert list(scored["success"]) == [True]


# --- how a save is made -----------------------------------------------------

def _acks(state):
    """The ledger's unit lines, without what they record of the file's size."""
    import json
    if not state.exists():
        return []
    lines = map(json.loads, state.read_text(encoding="utf-8").splitlines())
    acks = [{k: v for k, v in line.items() if k != "_saved"} for line in lines]
    return [ack for ack in acks if ack]


def test_both_ledgers_record_how_much_of_their_output_is_saved(tmp_path):
    from redact import evaluate_conversations, generate_conversations, paths
    from redact.dataset import Ledger

    generate_conversations(_three_seeds(), simple_setting, data_dir=tmp_path,
                           resolve=FakeResolver(echo), verbose=False)
    evaluate_conversations(data_dir=tmp_path, judge_model="j", verbose=False,
                           resolve=JudgeResolver(MockBackend("Yes ok")))
    out = paths.conversations_csv(tmp_path)
    for artifact in (out, out.with_name("conversations_scored.csv")):
        size, _checksum = Ledger.sidecar(artifact).saved[artifact.name]
        assert size == artifact.stat().st_size


def _three_seeds():
    import pandas as pd
    return pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "seed": ["q1", "q2", "q3"]})


def test_conversations_are_saved_at_exit_rows_then_acks(tmp_path, monkeypatch):
    import redact.multi_turn.pipeline as MP
    from redact import generate_conversations, paths

    state = paths.conversations_csv(tmp_path).with_name("conversations.state.jsonl")
    saves, append = [], MP.append_rows

    def spy(path, rows):
        saves.append(([r["input_id"] for r in rows], len(_acks(state))))
        return append(path, rows)

    monkeypatch.setattr(MP, "append_rows", spy)
    df = generate_conversations(
        _three_seeds(), simple_setting, data_dir=tmp_path, batch_size=2,
        resolve=FakeResolver(echo), verbose=False)

    # Per append: its rows, and the acks on disk when it was made.
    assert saves == [(["s1", "s2", "s3"], 0)]
    assert list(df["input_id"]) == ["s1", "s2", "s3"]
    assert _acks(state) == [
        {"input_id": sid, "iteration": 0, "stop_reason": "max_turns"}
        for sid in ("s1", "s2", "s3")
    ]


def _forty_one_seeds():
    import pandas as pd
    ids = [f"s{n}" for n in range(41)]
    return pd.DataFrame({"sample_id": ids, "seed": ["q"] * len(ids)})


def test_conversations_are_saved_every_32_units_and_at_exit(tmp_path, monkeypatch):
    import redact.multi_turn.pipeline as MP
    from redact import generate_conversations

    seeds = _forty_one_seeds()
    saves, append = [], MP.append_rows

    def spy(path, rows):
        saves.append([r["input_id"] for r in rows])
        return append(path, rows)

    monkeypatch.setattr(MP, "append_rows", spy)
    generate_conversations(
        seeds, simple_setting, data_dir=tmp_path, batch_size=40,
        resolve=FakeResolver(echo), verbose=False)
    ids = list(seeds["sample_id"])
    assert saves == [ids[:32], ids[32:]]


def test_a_failed_conversation_save_acks_none_of_its_units(tmp_path, monkeypatch):
    import pandas as pd

    import redact.multi_turn.pipeline as MP
    from redact import generate_conversations, paths

    out = paths.conversations_csv(tmp_path)
    seeds = _forty_one_seeds()
    outcomes, append = iter([None, OSError("disk full")]), MP.append_rows

    def failing(path, rows):
        error = next(outcomes)
        if error:
            raise error
        return append(path, rows)

    monkeypatch.setattr(MP, "append_rows", failing)
    with pytest.raises(OSError, match="disk full"):
        generate_conversations(
            seeds, simple_setting, data_dir=tmp_path, batch_size=2,
            resolve=FakeResolver(echo), verbose=False)

    first = list(seeds["sample_id"])[:32]
    acks = _acks(out.with_name("conversations.state.jsonl"))
    assert [ack["input_id"] for ack in acks] == first
    assert list(pd.read_csv(out)["input_id"]) == first


def test_conversations_killed_between_rows_and_acks_are_each_on_disk_once(
        tmp_path, monkeypatch):
    import pandas as pd

    import redact.multi_turn.pipeline as MP
    from redact import generate_conversations, paths
    from tests.conftest import Killed, kill_after_rows, lock_is_free

    out = paths.conversations_csv(tmp_path)
    state = out.with_name("conversations.state.jsonl")

    def run():
        return generate_conversations(
            _three_seeds(), simple_setting, data_dir=tmp_path, batch_size=2,
            resolve=FakeResolver(echo), verbose=False)

    kill_after_rows(monkeypatch, MP, on_call=1)
    with pytest.raises(Killed):
        run()
    # The rows landed; their acks did not.
    landed = out.read_bytes()
    assert list(pd.read_csv(out)["input_id"]) == ["s1", "s2", "s3"]
    assert _acks(state) == []
    assert lock_is_free(state)

    assert list(run()["input_id"]) == ["s1", "s2", "s3"]
    assert out.with_name("conversations.csv.cut").read_bytes() == landed
    assert [ack["input_id"] for ack in _acks(state)] == ["s1", "s2", "s3"]
    assert lock_is_free(state)

    run()                       # nothing to do, and no lock kept
    assert lock_is_free(state)


def test_the_conversation_pool_is_sized_by_the_models_that_reply(
        tmp_path, monkeypatch):
    import redact.multi_turn.pipeline as MP
    from redact import generate_conversations

    seen = {}

    def stop(source, **kwargs):
        seen.update(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(MP, "drive_units", stop)
    monkeypatch.setattr(MP, "pool_size", lambda batch_size, *clients: (
        batch_size, [c.model for c in clients]))
    with pytest.raises(KeyboardInterrupt):
        generate_conversations(
            _three_seeds(), simple_setting, data_dir=tmp_path, batch_size=2,
            resolve=FakeResolver(echo), verbose=False)
    assert (seen["pool_size"], seen["cap"]) == ((2, ["m"]), 2)


def test_an_old_conversations_folder_is_taken_over_by_its_rows(tmp_path):
    """Rows whose acks were lost used to be generated a second time."""
    import pandas as pd

    from redact import generate_conversations, paths

    def run(seeds):
        return generate_conversations(
            seeds, simple_setting, data_dir=tmp_path,
            resolve=FakeResolver(echo), verbose=False)

    run(_three_seeds().iloc[:2])
    out = paths.conversations_csv(tmp_path)
    out.with_name("conversations.state.jsonl").unlink()
    assert list(run(_three_seeds())["input_id"]) == ["s1", "s2", "s3"]
    assert list(pd.read_csv(out)["input_id"]) == ["s1", "s2", "s3"]


def _three_conversations():
    import pandas as pd
    return pd.DataFrame({"sample_id": ["a", "b", "c"], "transcript": ["[]"] * 3})


def test_scores_killed_between_rows_and_acks_are_each_on_disk_once(
        tmp_path, monkeypatch):
    import pandas as pd

    import redact.multi_turn.evaluate as ME
    from redact import evaluate_conversations, paths
    from tests.conftest import Killed, kill_after_rows, lock_is_free

    scored = paths.conversations_scored_csv(tmp_path)
    state = scored.with_name("conversations_scored.state.jsonl")

    def run():
        return evaluate_conversations(
            conversations=_three_conversations(), data_dir=tmp_path, judge_model="j",
            batch_size=2, verbose=False, resolve=JudgeResolver(MockBackend("Yes ok")))

    kill_after_rows(monkeypatch, ME, on_call=1)
    with pytest.raises(Killed):
        run()
    # The rows landed; their acks did not.
    landed = scored.read_bytes()
    assert list(pd.read_csv(scored)["sample_id"]) == ["a", "b", "c"]
    assert _acks(state) == []
    assert lock_is_free(state)

    assert list(run()["sample_id"]) == ["a", "b", "c"]
    assert scored.with_name("conversations_scored.csv.cut").read_bytes() == landed
    assert _acks(state) == [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]
    assert lock_is_free(state)

    run()                       # nothing to do: no buffer, and no lock kept
    assert lock_is_free(state)


def test_scores_are_saved_at_exit_rows_then_acks(tmp_path, monkeypatch):
    import redact.multi_turn.evaluate as ME
    from redact import evaluate_conversations, paths

    state = paths.conversations_scored_csv(tmp_path).with_name(
        "conversations_scored.state.jsonl")
    saves, append = [], ME.append_rows

    def spy(path, rows):
        saves.append(([r["sample_id"] for r in rows], len(_acks(state))))
        return append(path, rows)

    monkeypatch.setattr(ME, "append_rows", spy)
    scored = evaluate_conversations(
        conversations=_three_conversations(), data_dir=tmp_path, judge_model="j",
        batch_size=2, verbose=False, resolve=JudgeResolver(MockBackend("Yes ok")))

    assert saves == [(["a", "b", "c"], 0)]
    assert list(scored["sample_id"]) == ["a", "b", "c"]
    assert _acks(state) == [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]


def _forty_one_conversations():
    import pandas as pd
    return pd.DataFrame({"sample_id": [f"c{n}" for n in range(41)],
                         "transcript": ["[]"] * 41})


def test_scores_are_saved_every_32_units_and_at_exit(tmp_path, monkeypatch):
    import redact.multi_turn.evaluate as ME
    from redact import evaluate_conversations

    convs = _forty_one_conversations()
    saves, append = [], ME.append_rows

    def spy(path, rows):
        saves.append([r["sample_id"] for r in rows])
        return append(path, rows)

    monkeypatch.setattr(ME, "append_rows", spy)
    evaluate_conversations(
        conversations=convs, data_dir=tmp_path, judge_model="j", batch_size=40,
        verbose=False, resolve=JudgeResolver(MockBackend("Yes ok")))
    ids = list(convs["sample_id"])
    assert saves == [ids[:32], ids[32:]]


def test_a_failed_scoring_save_acks_none_of_its_units(tmp_path, monkeypatch):
    import pandas as pd

    import redact.multi_turn.evaluate as ME
    from redact import evaluate_conversations, paths

    scored = paths.conversations_scored_csv(tmp_path)
    convs = _forty_one_conversations()
    outcomes, append = iter([None, OSError("disk full")]), ME.append_rows

    def failing(path, rows):
        error = next(outcomes)
        if error:
            raise error
        return append(path, rows)

    monkeypatch.setattr(ME, "append_rows", failing)
    with pytest.raises(OSError, match="disk full"):
        evaluate_conversations(
            conversations=convs, data_dir=tmp_path, judge_model="j",
            batch_size=2, verbose=False, resolve=JudgeResolver(MockBackend("Yes ok")))

    first = list(convs["sample_id"])[:32]
    acks = _acks(scored.with_name("conversations_scored.state.jsonl"))
    assert [ack["sample_id"] for ack in acks] == first
    assert list(pd.read_csv(scored)["sample_id"]) == first


def test_a_judge_call_carries_up_to_batch_size_rows(tmp_path):
    from redact import evaluate_conversations
    from redact.llms.backends import ComputeConfig

    class Judge(MockBackend):
        compute_config = ComputeConfig(supports_native_batching=True)

        def __init__(self):
            super().__init__("Yes ok")
            self.batches = []

        def generate(self, messages_list, **kwargs):
            self.batches.append(len(messages_list))
            return super().generate(messages_list, **kwargs)

    judge = Judge()
    evaluate_conversations(
        conversations=_forty_one_conversations(), data_dir=tmp_path, judge_model="j",
        batch_size=16, verbose=False, resolve=JudgeResolver(judge))
    assert judge.batches == [16, 16, 9]


def test_a_transcript_too_long_for_the_judge_is_scored_failed_and_acked(tmp_path):
    import json
    import re

    import pandas as pd

    from redact import evaluate_conversations, paths
    from redact.dataset import Ledger
    from redact.exceptions import PromptTooLongError
    from redact.llms.backends import ComputeConfig

    class LaneJudge(MockBackend):
        compute_config = ComputeConfig(supports_async_calls=True)

        async def agenerate(self, messages, **kwargs):
            text = " ".join(m["content"] for m in messages)
            token = re.search(r"zq\d+", text).group()
            if token == "zq1":
                raise PromptTooLongError(f"{token} is over the context")
            return "Yes ok"

    def record(token):
        return json.dumps([{"type": "message", "actor": "attacker", "content": token,
                            "role": "user", "model": None, "meta": {}}])

    convs = pd.DataFrame({"sample_id": ["a", "b", "c"],
                          "transcript": [record("zq0"), record("zq1"), record("zq2")]})
    scored = evaluate_conversations(conversations=convs, data_dir=tmp_path,
                                    judge_model="j", scope="transcript", verbose=False,
                                    resolve=JudgeResolver(LaneJudge("unused")))
    by_id = scored.set_index("sample_id")
    assert list(by_id["success"].loc[["a", "b", "c"]]) == [True, False, True]
    assert by_id.loc["b", "judge_reasoning"].startswith("prompt too long")
    ledger = Ledger.sidecar(paths.conversations_scored_csv(tmp_path),
                            key_fields=("sample_id",), casters={"sample_id": str})
    assert ledger.completed() == {"a", "b", "c"}
