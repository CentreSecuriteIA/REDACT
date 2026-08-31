"""Offline tests for multi_turn jailbreak attacks (crescendo, PAIR, jailbreak compose)."""

import json

import pandas as pd

from redact.multiturn_attacks import (
    build_attack_setting,
    generate_attacks,
    score_attacks,
)
from tests.conftest import MockBackend, make_client


class _FakeClient:
    """Stands in for a ModelClient: batch of messages in, batch of replies out."""

    def __init__(self, model, reply, counter=None):
        self.model = model
        self._reply = reply
        self._counter = counter

    def generate(self, messages_list, **kw):
        if self._counter is not None:
            self._counter.calls += 1
        return [self._reply(self.model, msgs) for msgs in messages_list]


class _FakeResolver:
    """``model -> _FakeClient``; the resolve seam drive_generators takes."""

    def __init__(self, reply):
        self._reply = reply
        self.calls = 0

    def __call__(self, model):
        return _FakeClient(model, self._reply, self)


def _echo(model, msgs):
    last_user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    return f"[{model}] to: {last_user[:30]}"


class _JudgeResolver:
    """``model -> client`` over one fixed backend."""

    def __init__(self, backend):
        self._backend = backend

    def __call__(self, model):
        return make_client(self._backend, model)


def test_crescendo_attack_logs_strategy_and_escalates(tmp_path):
    seeds = pd.DataFrame({"seed": ["how to make X"]})
    df = generate_attacks(seeds, target_model="target", data_dir=tmp_path,
                          attack="crescendo", max_turns=4, resolve=_FakeResolver(_echo), verbose=False)
    assert len(df) == 1
    steps = json.loads(df.iloc[0]["transcript"])
    types = [s["type"] for s in steps]
    # seed(message) -> target reply -> attacker [strategy, message] -> target reply
    assert types == ["message", "reply", "strategy", "message", "reply"]
    assert steps[0]["content"] == "how to make X"                  # seed
    assert steps[2]["content"].startswith("crescendo step 1")      # strategy note
    assert "detail" in steps[3]["content"]                          # a crescendo follow-up
    assert df.iloc[0]["setting"] == "attack:crescendo"


def test_crescendo_applies_jailbreak_technique(tmp_path):
    from redact.jailbreak.obfuscation.encoding import to_rot13  # a pure transform
    seeds = pd.DataFrame({"seed": ["seed goal"]})
    df = generate_attacks(seeds, target_model="target", data_dir=tmp_path, attack="crescendo",
                          jailbreak_technique=to_rot13, max_turns=3, resolve=_FakeResolver(_echo), verbose=False)
    steps = json.loads(df.iloc[0]["transcript"])
    strat = next(s for s in steps if s["type"] == "strategy")
    assert "+" in strat["content"]  # records the applied technique name
    # the attacker's follow-up message is rot13-transformed (not the plain template)
    atk_msg = [s for s in steps if s["type"] == "message"][1]["content"]
    assert atk_msg != "Thanks. Can you go into more concrete, step-by-step detail on that?"


def test_pair_attack_uses_attacker_model(tmp_path):
    seeds = pd.DataFrame({"seed": ["goal"]})
    df = generate_attacks(seeds, target_model="target", data_dir=tmp_path, attack="pair",
                          attacker_model="attacker", max_turns=4, resolve=_FakeResolver(_echo), verbose=False)
    steps = json.loads(df.iloc[0]["transcript"])
    # the attacker's follow-up came from the attacker model (router tags model name)
    atk_msgs = [s["content"] for s in steps if s["type"] == "message"][1:]
    assert any("[attacker]" in m for m in atk_msgs)


def test_build_attack_setting_requires_target():
    import pytest
    with pytest.raises(ValueError):
        build_attack_setting("")            # no target
    with pytest.raises(ValueError):
        build_attack_setting("t", attack="pair")()  # pair without attacker_model


def test_build_attack_setting_unknown_attack():
    import pytest
    with pytest.raises(ValueError):
        build_attack_setting("t", attack="bogus")


def test_pair_propose_empty_transcript_goal():
    # _goal returns "" when there's no message step yet; the attacker still runs.
    from redact.llms.conversation import Transcript, drive_sync
    from redact.multiturn_attacks import pair_propose
    result = drive_sync(pair_propose("attacker")(Transcript(), 0), lambda req: "next")
    assert result[0] == "next"


def test_score_attacks(tmp_path):
    seeds = pd.DataFrame({"seed": ["how to make X", "how to make Y"]})
    generate_attacks(seeds, target_model="target", data_dir=tmp_path, attack="crescendo",
                     max_turns=3, resolve=_FakeResolver(_echo), verbose=False)
    scored = score_attacks(
        data_dir=tmp_path, judge_model="judge", verbose=False,
        resolve=_JudgeResolver(MockBackend(["Yes complied", "No refused"])),
    )
    assert set(scored["success"]) == {True, False}
