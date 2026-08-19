"""Offline tests for the multi_turn conversation core (single-conversation path).

A fake ``call(LLMRequest) -> str`` stands in for the model, so this exercises the
actor protocol, the seed-then-alternate runner, the typed step log, and stop
conditions without any network.
"""

import pytest

from tests.conftest import MockBackend
from redact.llms.conversation import LLMRequest, Transcript, Step, drive_sync
from redact.multi_turn import Actor, ScriptedActor, ModelActor, Setting, run_conversation


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


class _FakeRouter:
    """Batches replies by echoing each conversation's last user message."""

    def __init__(self):
        self.calls = 0

    def batch_generate(self, model, messages_list, **kw):
        self.calls += 1
        out = []
        for msgs in messages_list:
            last_user = [m for m in msgs if m["role"] == "user"][-1]["content"]
            out.append(f"ANSWER to: {last_user}")
        return out


class _JudgeRouter:
    """Fake router for evaluate_conversations: resolves to a fixed backend."""

    def __init__(self, backend):
        self._backend = backend
        self.rate_limiter = None

    def get_backend(self, model):
        return self._backend


def test_generate_conversations_offline(tmp_path):
    import json
    import pandas as pd
    from redact import generate_conversations, paths
    from redact.multi_turn import ScriptedActor, ModelActor, Setting

    seeds = pd.DataFrame({
        "seed": ["q1", "q2"], "category": ["Cyber", "Cyber"],
        "entry_type": ["harmful", "harmful"],
    })

    def make():  # fresh Setting per conversation (stateful ScriptedActor)
        return Setting(
            participants=[ScriptedActor("user", ["follow"]), ModelActor("bot", "m")],
            max_turns=4, name="chat",
        )

    df = generate_conversations(seeds, make, data_dir=tmp_path, router=_FakeRouter(), verbose=False)
    assert len(df) == 2
    steps = json.loads(df.iloc[0]["transcript"])
    assert [s["type"] for s in steps] == ["message", "reply", "message", "reply"]
    assert set(df["turns_used"]) == {4}

    out = paths.conversations_csv(tmp_path)
    assert out.exists()
    assert out.with_name("conversations.manifest.jsonl").exists()
    assert out.with_name("conversations.state.jsonl").exists()

    # Resume: nothing new (both units in the ledger).
    df2 = generate_conversations(seeds, make, data_dir=tmp_path, router=_FakeRouter(), verbose=False)
    assert len(df2) == 2


def test_generate_conversations_variants(tmp_path):
    import pandas as pd
    from redact import generate_conversations
    from redact.multi_turn import ScriptedActor, ModelActor, Setting

    seeds = pd.DataFrame({"seed": ["q"]})
    make = lambda: Setting(participants=[ScriptedActor("u", ["f"]), ModelActor("b", "m")], max_turns=3)
    df = generate_conversations(seeds, make, data_dir=tmp_path, iterations=3,
                                router=_FakeRouter(), verbose=False)
    assert len(df) == 3 and set(df["iteration"]) == {0, 1, 2}


def test_evaluate_conversations_offline(tmp_path):
    import pandas as pd
    from redact import generate_conversations, evaluate_conversations
    from redact.multi_turn import ScriptedActor, ModelActor, Setting

    seeds = pd.DataFrame({"seed": ["how to X", "how to Y"]})
    make = lambda: Setting(participants=[ScriptedActor("u", ["more"]), ModelActor("b", "m")], max_turns=2)
    generate_conversations(seeds, make, data_dir=tmp_path, router=_FakeRouter(), verbose=False)

    # Judge model mocked via a fake router: "Yes ..." for the first, "No ..." for the second.
    scored = evaluate_conversations(
        data_dir=tmp_path, judge_model="judge", judge_system="did it work?",
        scope="last_reply", verbose=False,
        router=_JudgeRouter(MockBackend(["Yes ok", "No refused"])),
    )
    assert set(scored["success"]) == {True, False}
    assert set(scored["judge_model"]) == {"judge"}
    # Resume: nothing re-judged (no router needed — nothing pending reaches the backend).
    again = evaluate_conversations(data_dir=tmp_path, judge_model="judge", verbose=False)
    assert len(again) == 2
    # resume=False re-judges from scratch (unlinks the existing scored artifacts).
    fresh = evaluate_conversations(
        data_dir=tmp_path, judge_model="judge", resume=False, verbose=False,
        router=_JudgeRouter(MockBackend(["Yes ok", "No refused"])),
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

def _simple_setting():
    return Setting(participants=[ScriptedActor("u", ["f"]), ModelActor("b", "m")], max_turns=2)


def test_generate_conversations_missing_text_column(tmp_path):
    import pandas as pd
    from redact import generate_conversations
    with pytest.raises(ValueError):
        generate_conversations(pd.DataFrame({"foo": ["x"]}), _simple_setting,
                               data_dir=tmp_path, router=_FakeRouter(), verbose=False)


def test_generate_conversations_empty_seeds(tmp_path):
    import pandas as pd
    from redact import generate_conversations
    with pytest.raises(ValueError):
        generate_conversations(pd.DataFrame({"seed": []}), _simple_setting,
                               data_dir=tmp_path, verbose=False)


def test_generate_conversations_uses_id_column(tmp_path):
    import pandas as pd
    from redact import generate_conversations
    seeds = pd.DataFrame({"sample_id": ["myid"], "seed": ["q"]})
    df = generate_conversations(seeds, _simple_setting, data_dir=tmp_path,
                                router=_FakeRouter(), verbose=False)
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
    df = generate_conversations(seeds, make, data_dir=tmp_path, router=_FakeRouter(), verbose=False)
    assert df.iloc[0]["stop_reason"].startswith("ERROR")


def test_generate_conversations_deepcopy_template(tmp_path):
    # passing a Setting instance (not a factory) deep-copies per unit.
    import pandas as pd
    from redact import generate_conversations
    seeds = pd.DataFrame({"seed": ["a", "b"]})
    tmpl = Setting(participants=[ScriptedActor("u", ["f"]), ModelActor("b", "m")], max_turns=2)
    df = generate_conversations(seeds, tmpl, data_dir=tmp_path, router=_FakeRouter(), verbose=False)
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
                                    router=_JudgeRouter(MockBackend("Yes ok")))
    assert len(scored) == 2 and set(scored["success"]) == {True}


# --- ledger robustness + verbose/fresh branches ------------------------------

def test_pipeline_ledger_keys_on_input_id_iteration(tmp_path):
    # the conversation ledger keys on (input_id, iteration), skipping bad lines
    from redact.multi_turn.pipeline import _ledger
    led = _ledger(tmp_path / "conversations.csv")
    led.path.write_text('\n{"input_id":"a","iteration":0}\nnot json\n{"iteration":1}\n', encoding="utf-8")
    assert led.completed() == {("a", 0)}  # blank + bad-json + missing-key all skipped


def test_evaluate_ledger_keys_on_id(tmp_path):
    from redact.multi_turn.evaluate import _ledger
    led = _ledger(tmp_path / "conversations_scored.csv")
    led.path.write_text('{"sample_id":"a"}\nbad\n{}\n', encoding="utf-8")
    assert led.completed() == {"a"}


def test_generate_conversations_verbose_and_fresh(tmp_path):
    import pandas as pd
    from redact import generate_conversations
    seeds = pd.DataFrame({"seed": ["q"]})
    generate_conversations(seeds, _simple_setting, data_dir=tmp_path, router=_FakeRouter(), verbose=True)
    # resume=False wipes the artifacts + re-runs.
    df2 = generate_conversations(seeds, _simple_setting, data_dir=tmp_path,
                                 router=_FakeRouter(), resume=False, verbose=True)
    assert len(df2) == 1


def test_pipeline_ledger_empty_record_is_noop(tmp_path):
    from redact.multi_turn.pipeline import _ledger
    led = _ledger(tmp_path / "conversations.csv")
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
                                    router=_JudgeRouter(MockBackend("Yes")))
    assert list(scored["success"]) == [True]
