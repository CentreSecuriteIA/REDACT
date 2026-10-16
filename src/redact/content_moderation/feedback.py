"""The prompt text the stages add at run time.

Rejection feedback for a regenerated reply, and the note naming samples
not to repeat.
"""

from collections.abc import Collection

from .results import TurnResult

# How many of the samples not to repeat the note shows; the rest are counted.
PROHIBITED_SHOWN = 20

_INPUT_FEEDBACK = (
    "\n\n[FEEDBACK FROM PREVIOUS ATTEMPT]\n"
    "Some previously generated samples were rejected for these "
    "reasons:\n{feedback}\n"
    "Please avoid similar issues in your new samples."
)
_OUTPUT_FEEDBACK = (
    "\n\n[FEEDBACK FROM PREVIOUS ATTEMPT]\n"
    "Your previous response was rejected for this reason:\n{feedback}\n"
    "Please avoid this issue in your new response."
)
_FEEDBACK = {"input": _INPUT_FEEDBACK, "output": _OUTPUT_FEEDBACK}


def append_to_user(messages: list[dict], text: str) -> list[dict]:
    """``messages`` with ``text`` added to the last user turn, or as a new one.

    Returns a new list.
    """
    if messages and messages[-1]["role"] == "user":
        last = {**messages[-1], "content": messages[-1]["content"] + text}
        return [*messages[:-1], last]
    return [*messages, {"role": "user", "content": text}]


def with_feedback(
    messages: list[dict], feedback: str, kind: str = "input",
) -> list[dict]:
    """``messages`` with rejection feedback added to the last user turn.

    ``kind`` is ``"input"`` (rejected samples) or ``"output"`` (a rejected
    response). Returns a new list; ``messages`` itself for an empty
    ``feedback``.
    """
    if not feedback:
        return messages
    return append_to_user(messages, _FEEDBACK[kind].format(feedback=feedback))


def prohibited_note(texts: Collection[str], more: int = 0) -> str:
    """The note asking for nothing like ``texts``.

    It shows the first ``PROHIBITED_SHOWN`` texts and counts the rest, plus
    ``more`` not given.
    """
    sample_list = list(texts)[:PROHIBITED_SHOWN]
    prohibited_text = "\n".join(f"- {s}" for s in sample_list)
    prohibited_msg = (
        f"\n\nIMPORTANT: The following samples already exist in the "
        f"dataset. Do NOT generate anything similar to these:\n"
        f"{prohibited_text}"
    )
    more += len(texts) - len(sample_list)
    if more:
        prohibited_msg += (
            f"\n(... and {more} more. "
            f"Ensure all your outputs are novel.)"
        )
    return prohibited_msg


def next_turn_feedback(
    turn_result: TurnResult, max_rejections: int = 3, max_chars: int = 200
) -> str:
    """Build the rejection-feedback string fed into the next generation turn.

    Up to ``max_rejections`` reasons of the turn's rejected samples, each cut
    to ``max_chars``; "" when nothing was rejected with a reason.
    """
    rejections = [
        r.reasoning for r in turn_result.samples if not r.accepted and r.reasoning
    ]
    if not rejections:
        return ""
    return "\n".join(f"- {r[:max_chars]}" for r in rejections[:max_rejections])
