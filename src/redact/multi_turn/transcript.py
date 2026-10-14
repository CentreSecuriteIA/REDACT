"""The step log of a conversation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

# Step kinds. Only "message" and "reply" steps are sent to models.
StepType = Literal["strategy", "message", "reply", "analysis", "evaluation"]
_VISIBLE = ("message", "reply")


@dataclass
class Step:
    """One entry of the step log.

    ``message`` and ``reply`` steps carry a chat ``role`` and are sent to
    models; the other kinds are notes, only logged.
    """

    type: StepType
    actor: str
    content: str
    role: str | None = None          # chat role of a message or reply
    model: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Transcript:
    """The steps of a conversation, in order."""

    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_records(cls, records: list[dict]) -> Transcript:
        """The transcript :meth:`to_records` wrote."""
        return cls([Step(**record) for record in records])

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def message(self, actor: str, content: str, role: str = "user",
                model: str | None = None, meta: dict | None = None) -> Step:
        return self.add(Step("message", actor, content, role=role, model=model, meta=meta or {}))

    def reply(self, actor: str, content: str, role: str = "assistant",
              model: str | None = None, meta: dict | None = None) -> Step:
        return self.add(Step("reply", actor, content, role=role, model=model, meta=meta or {}))

    def note(self, type: StepType, actor: str, content: str,
             model: str | None = None, meta: dict | None = None) -> Step:
        """Log a note, which is never sent to a model."""
        return self.add(Step(type, actor, content, model=model, meta=meta or {}))

    def as_messages(self, system: str | None = None) -> list[dict]:
        """Render the visible turns into a chat ``messages`` list."""
        msgs: list[dict] = [{"role": "system", "content": system}] if system else []
        for s in self.steps:
            if s.type in _VISIBLE and s.role:
                msgs.append({"role": s.role, "content": s.content})
        return msgs

    def render(self) -> str:
        """The visible turns as ``role: content`` lines."""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.as_messages())

    def first_message(self) -> str:
        """The first message, the seed; empty when there is none."""
        return next((s.content for s in self.steps if s.type == "message"), "")

    def last_reply(self) -> str:
        """The last reply; empty when there is none."""
        return next((s.content for s in reversed(self.steps) if s.type == "reply"), "")

    def to_records(self) -> list[dict]:
        """Every step as a dict (a copy), for JSON."""
        return [asdict(s) for s in self.steps]
