"""Who speaks: an actor's turn is a generator of model calls."""

from __future__ import annotations

from collections.abc import Generator

from redact.llm_pipeline import LLMRequest

from .transcript import Step, Transcript

# An actor's turn yields LLMRequests, is resumed with replies, and returns the
# Steps it produced (e.g. [strategy?, message] or [reply]).
ActorTurn = Generator[LLMRequest, str, "list[Step]"]


class Actor:
    """Base actor. Subclasses implement :meth:`turn` as a generator."""

    def __init__(self, name: str, role: str = "user"):
        self.name = name
        self.role = role

    def turn(self, transcript: Transcript) -> ActorTurn:  # pragma: no cover
        raise NotImplementedError
        yield  # make this a generator function


class ScriptedActor(Actor):
    """Sends the given messages in order, then repeats the last; calls no model."""

    def __init__(self, name: str, messages: list[str], role: str = "user"):
        super().__init__(name, role)
        self._messages = list(messages)
        self._i = 0

    def turn(self, transcript: Transcript) -> ActorTurn:
        yield from ()  # no LLM call; keeps this a generator
        if not self._messages:
            return []
        content = self._messages[min(self._i, len(self._messages) - 1)]
        self._i += 1
        return [Step("message", self.name, content, role=self.role)]


class ModelActor(Actor):
    """Replies with one call to its model on the visible transcript."""

    def __init__(self, name: str, model: str, role: str = "assistant",
                 system_prompt: str | None = None):
        super().__init__(name, role)
        self.model = model
        self.system_prompt = system_prompt

    def turn(self, transcript: Transcript) -> ActorTurn:
        reply = yield LLMRequest(self.model, transcript.as_messages(self.system_prompt))
        return [Step("reply", self.name, reply, role=self.role, model=self.model)]


class StrategyActor(Actor):
    """Sends the message a ``propose`` generator builds.

    ``propose(transcript, turn_index)`` may yield model calls, and returns the
    message, or ``(message, note)`` to log the note as a ``strategy`` step.
    ``turn_index`` counts this actor's earlier messages in the transcript, so
    the actor holds no state between turns.
    """

    def __init__(self, name: str, propose, role: str = "user"):
        super().__init__(name, role)
        self._propose = propose

    def turn(self, transcript: Transcript) -> ActorTurn:
        turn_index = sum(
            1 for s in transcript.steps if s.type == "message" and s.actor == self.name
        )
        result = yield from self._propose(transcript, turn_index)
        message, note = result if isinstance(result, tuple) else (result, None)
        steps: list[Step] = []
        if note:
            steps.append(Step("strategy", self.name, note))
        steps.append(Step("message", self.name, str(message), role=self.role))
        return steps
