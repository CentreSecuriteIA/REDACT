"""The prompt text the stages add at run time: rejection feedback for a
regenerated response."""

_FEEDBACK = (
    "\n\n[FEEDBACK FROM PREVIOUS ATTEMPT]\n"
    "Your previous response was rejected for this reason:\n{feedback}\n"
    "Please avoid this issue in your new response."
)


def append_to_user(messages: list[dict], text: str) -> list[dict]:
    """``messages`` with ``text`` added to the last user turn, or as a new one.

    Returns a new list.
    """
    if messages and messages[-1]["role"] == "user":
        last = {**messages[-1], "content": messages[-1]["content"] + text}
        return [*messages[:-1], last]
    return [*messages, {"role": "user", "content": text}]


def with_feedback(messages: list[dict], feedback: str) -> list[dict]:
    """``messages`` with a rejected response's feedback added to the last user
    turn.

    Returns a new list; ``messages`` itself for an empty ``feedback``.
    """
    if not feedback:
        return messages
    return append_to_user(messages, _FEEDBACK.format(feedback=feedback))

