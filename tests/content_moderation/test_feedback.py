"""The run-time prompt text of content_moderation/feedback.py.

The hashes pin the text the stages rendered before it moved into
feedback.py, so the move is verbatim.
"""

import hashlib
import json

import pytest

from redact.content_moderation.feedback import (
    next_turn_feedback,
    prohibited_note,
    with_feedback,
)
from redact.content_moderation.inputs import InputPipeline
from redact.content_moderation.results import SampleResult, TurnResult

_PINNED = {
    "input_after_user":
        "f2613962270ca85857838ee0380a9e747019cabc4e4eaa6f31b0684b9da3d68b",
    "output_after_user":
        "8a76acfe23a49fbcb922548b0f74946d7870d5e2b894b98f8b0014a503c08c56",
    "output_after_system":
        "8237d0ba91f2f6fc49c1ba59d311d4bb6cca7174efadabc3c19bafcfde25bfcb",
    "input_empty_list":
        "ad1769c736b28a8e67d7555aa6c05fbb9637c33a73ea7b9bb5b214eb384d7c58",
    "no_feedback":
        "fe87c17fff362934c9198d21ee2ae6329fb93479e4c2aa2d65c6102c904c5f93",
    "messages_prohibited_few":
        "a35366f54765dba03e5cab8750ca14a6c7eabda4766ddc6d27d34477f38595c2",
    "messages_prohibited_many":
        "4584eb5c38d4d84ae546ebd6cbcd2933a2c88872208afaf5f028daeef3dea6c2",
    "messages_prohibited_more":
        "1af23f8796e70556c2651fbda65dcc2f06a4426c1171a3f716cb038f5155cc2f",
    "next_turn":
        "5f51f7d971cf0d7732e35f76349b08308ea1f8bf9aaed749b639b21d1bda2bcc",
    "next_turn_none":
        "12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126",
}

_PROMPT = {"system_prompt": "sys", "template": "Make {num_samples} about {Category}"}
_TEXTS = [f"text {i}" for i in range(25)]


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def _base() -> list[dict]:
    return [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


def _rendered(name: str):
    pipe = InputPipeline(gen=None, check=None)

    def build(**kwargs):
        return pipe._build_generation_messages(_PROMPT, 3, Category="Cat", **kwargs)

    turn = TurnResult(0, "", 4, 1, 3, [
        SampleResult("a", True, "", 0),
        SampleResult("b", False, "r" * 250, 0),
        SampleResult("c", False, "second reason", 0),
        SampleResult("d", False, "third reason", 0),
        SampleResult("e", False, "fourth reason", 0),
    ])
    return {
        "input_after_user": lambda: with_feedback(_base(), "- one\n- two"),
        "output_after_user": lambda: with_feedback(_base(), "verdict", "output"),
        "output_after_system": lambda: with_feedback(
            _base()[:1], "verdict", "output"),
        "input_empty_list": lambda: with_feedback([], "reason"),
        "no_feedback": lambda: with_feedback(_base(), ""),
        "messages_prohibited_few": lambda: build(prohibited=_TEXTS[:3]),
        "messages_prohibited_many": lambda: build(
            feedback="- bad", prohibited=_TEXTS, more=5),
        "messages_prohibited_more": lambda: build(prohibited=_TEXTS[:2], more=4),
        "next_turn": lambda: next_turn_feedback(turn),
        "next_turn_none": lambda: next_turn_feedback(
            TurnResult(0, "", 1, 1, 0, [SampleResult("a", True, "", 0)])),
    }[name]()


@pytest.mark.parametrize("name", sorted(_PINNED))
def test_the_rendered_text_is_the_text_before_the_move(name):
    assert _hash(_rendered(name)) == _PINNED[name]


def test_feedback_on_an_empty_list_is_one_user_turn():
    messages = with_feedback([], "reason")
    assert [m["role"] for m in messages] == ["user"]
    assert "reason" in messages[0]["content"]


def test_feedback_leaves_the_given_messages_alone():
    messages = _base()
    with_feedback(messages, "reason")
    assert messages == _base()


def test_the_note_shows_at_most_twenty_texts_and_counts_the_rest():
    note = prohibited_note(_TEXTS, more=5)
    assert "text 19" in note and "text 20" not in note
    assert note == prohibited_note(_TEXTS[:20], more=10)
    few = _TEXTS[:3]
    assert prohibited_note(few).count("- text") == len(few)
