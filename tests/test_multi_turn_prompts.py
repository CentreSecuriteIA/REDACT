"""The prompt text the multi-turn stages send.

The hashes pin the messages and follow-ups as the code rendered them before
the text moved into prompt and config files, so the move is verbatim.
"""

import hashlib
import json

import pandas as pd
import pytest

from redact import evaluate_conversations
from redact.llm_pipeline import drive_sync
from redact.multi_turn import Transcript
from redact.multiturn_attacks import (
    DEFAULT_CRESCENDO_FOLLOWUPS,
    SUCCESS_JUDGE_SYSTEM,
    crescendo_propose,
    jailbreak_candidates,
    pair_propose,
    score_attacks,
)
from redact.optimization import optimize
from tests.conftest import FakeClient, JudgeResolver, MockBackend, make_client

_PINNED = {
    "candidates":
        "2f88eed56e885453a2789b8d8578751c00b47d89c2300d729310733b01eb063a",
    "crescendo":
        "faa782adbc8df38abfa2a8dd6561adb9a47ee51971699814150fd366bced7725",
    "followups":
        "e8beef4145dc1164398da9e973a71f3b282e0e973f155adfd4fa822d65b6f50c",
    "judge_custom":
        "5c13b1ccb87878303bfc6fb9c1b0c2e14a965335651d2200235f85647fd1a341",
    "judge_last_reply":
        "3c76a0ff64f11acdf3578da52305644907e2140fe6a81a690c606cd5b2157044",
    "judge_transcript":
        "cdcb86a4a09431fd410072408650e202ff807b90af4da05f22d600fcd09e8da0",
    "pair_custom":
        "42736d45bc340dd36e11dff1938ecd3edb3b67a405bae1dffaffd3ba2d526916",
    "pair_default":
        "60bb542c11eb81995e4733c6f16a987cc8294c9ff0dae0b25687c97a6bccefa0",
    "score_attacks":
        "c71afe11b68598495789849fc12c61bf63fbac1e7ca8a87d55eb2213ca1ae721",
    "search_judge":
        "a78b7ce2730599f9a9020c161e9cf56cf4022a9371a1b583a3ce8a4391d99ce1",
    "success_judge":
        "9daca8e1283d45f3c307737e0663909db6384bb1a214c2ca2a33afed7f7fa858",
}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def _transcript() -> Transcript:
    t = Transcript()
    t.message("attacker", "the goal", role="user")
    t.reply("target", "first answer", model="target")
    t.message("attacker", "go on", role="user")
    t.reply("target", "last answer", model="target")
    return t


def _conversations() -> pd.DataFrame:
    return pd.DataFrame({"sample_id": ["a"],
                         "transcript": [json.dumps(_transcript().to_records())]})


def _sent(backend: MockBackend) -> list:
    return [[call["messages"], call.get("system_prompt")] for call in backend.calls]


def _judged(tmp_path, stage, **kwargs) -> list:
    """What the judge was sent when ``stage`` scored the one conversation."""
    judge = MockBackend("Yes")
    stage(conversations=_conversations(), data_dir=tmp_path, judge_model="j",
          verbose=False, resolve=JudgeResolver(judge), **kwargs)
    return _sent(judge)


def _searched(tmp_path) -> list:
    judge = MockBackend("No")
    target = FakeClient("target", lambda model, messages: "reply")
    optimize("the goal", "target", lambda t, d: [("plain", f"move {d}")], "judge",
             "my criterion", depth=1, verbose=False,
             resolve=lambda m: make_client(judge, m) if m == "judge" else target)
    return _sent(judge)


def _paired(system=None) -> list:
    sent = []

    def call(request):
        sent.append([request.model, request.messages])
        return "next message"

    drive_sync(pair_propose("attacker", system)(_transcript(), 2), call)
    return sent


def _crescendo() -> list:
    propose = crescendo_propose()
    return [drive_sync(propose(_transcript(), i), lambda r: "") for i in range(6)]


def _candidates() -> list:
    def shout(text):
        return text.upper(), ""

    candidates = jailbreak_candidates([shout])
    return [candidates(_transcript(), d) for d in range(6)]


def _rendered(name: str, tmp_path):
    return {
        "judge_last_reply": lambda: _judged(tmp_path, evaluate_conversations),
        "judge_transcript": lambda: _judged(
            tmp_path, evaluate_conversations, scope="transcript"),
        "judge_custom": lambda: _judged(
            tmp_path, evaluate_conversations, judge_system="my {criterion}"),
        "score_attacks": lambda: _judged(tmp_path, score_attacks),
        "search_judge": lambda: _searched(tmp_path),
        "pair_default": lambda: _paired(),
        "pair_custom": lambda: _paired("my {system}"),
        "crescendo": _crescendo,
        "candidates": _candidates,
        "followups": lambda: DEFAULT_CRESCENDO_FOLLOWUPS,
        "success_judge": lambda: SUCCESS_JUDGE_SYSTEM,
    }[name]()


@pytest.mark.parametrize("name", sorted(_PINNED))
def test_the_rendered_text_is_the_text_before_the_move(name, tmp_path):
    assert _hash(_rendered(name, tmp_path)) == _PINNED[name]
