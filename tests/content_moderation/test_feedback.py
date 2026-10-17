"""The run-time prompt text of content_moderation/feedback.py.

The hashes pin the text the stages rendered before it moved into
feedback.py, so the move is verbatim. ``output_empty_list`` is pinned as
the code before the standalone inputs were removed rendered it.
"""

import hashlib
import json

import pytest

from redact.content_moderation.feedback import with_feedback

_PINNED = {
    "output_after_user":
        "8a76acfe23a49fbcb922548b0f74946d7870d5e2b894b98f8b0014a503c08c56",
    "output_after_system":
        "8237d0ba91f2f6fc49c1ba59d311d4bb6cca7174efadabc3c19bafcfde25bfcb",
    "output_empty_list":
        "c5ba0eec2f835d8693fd57498199df1cf9cd249bfdf54c2a5d07c1e4eca5b821",
    "no_feedback":
        "fe87c17fff362934c9198d21ee2ae6329fb93479e4c2aa2d65c6102c904c5f93",
}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def _base() -> list[dict]:
    return [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


def _rendered(name: str):
    return {
        "output_after_user": lambda: with_feedback(_base(), "verdict"),
        "output_after_system": lambda: with_feedback(_base()[:1], "verdict"),
        "output_empty_list": lambda: with_feedback([], "reason"),
        "no_feedback": lambda: with_feedback(_base(), ""),
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

