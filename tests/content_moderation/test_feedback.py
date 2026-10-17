"""The run-time prompt text of content_moderation/feedback.py.

The hashes pin the text the stages rendered before it moved into
feedback.py, so the move is verbatim. ``output_empty_list`` and
``messages_prohibited_many`` are pinned as the code before the standalone
inputs were removed rendered them, the second without feedback.
"""

import hashlib
import json

import pytest

from redact.content_moderation.feedback import prohibited_note, with_feedback
from redact.content_moderation.inputs import InputPipeline

_PINNED = {
    "output_after_user":
        "8a76acfe23a49fbcb922548b0f74946d7870d5e2b894b98f8b0014a503c08c56",
    "output_after_system":
        "8237d0ba91f2f6fc49c1ba59d311d4bb6cca7174efadabc3c19bafcfde25bfcb",
    "output_empty_list":
        "c5ba0eec2f835d8693fd57498199df1cf9cd249bfdf54c2a5d07c1e4eca5b821",
    "no_feedback":
        "fe87c17fff362934c9198d21ee2ae6329fb93479e4c2aa2d65c6102c904c5f93",
    "messages_prohibited_few":
        "a35366f54765dba03e5cab8750ca14a6c7eabda4766ddc6d27d34477f38595c2",
    "messages_prohibited_many":
        "03e28f0221650445163e0ed732659a1d0db3c137381990545e23f1892a5d95cb",
    "messages_prohibited_more":
        "1af23f8796e70556c2651fbda65dcc2f06a4426c1171a3f716cb038f5155cc2f",
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

    return {
        "output_after_user": lambda: with_feedback(_base(), "verdict"),
        "output_after_system": lambda: with_feedback(_base()[:1], "verdict"),
        "output_empty_list": lambda: with_feedback([], "reason"),
        "no_feedback": lambda: with_feedback(_base(), ""),
        "messages_prohibited_few": lambda: build(prohibited=_TEXTS[:3]),
        "messages_prohibited_many": lambda: build(prohibited=_TEXTS, more=5),
        "messages_prohibited_more": lambda: build(prohibited=_TEXTS[:2], more=4),
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
