"""Actors, the Setting they take part in, and the runner that plays it to a Trajectory.

An actor's turn is a generator: it yields model calls, is resumed with each
reply, and returns the steps it adds to the transcript.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from redact.llm_pipeline import LLMRequest, drive_sync

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


@dataclass
class Setting:
    """Who takes part in a conversation, and when it ends.

    The seed is the first participant's message; the participants then take
    turns in order. ``stop`` ends the conversation early once it returns true
    for the transcript.
    """

    participants: list[Actor]
    max_turns: int = 6
    drive: str = "conversation"
    stop: Callable[[Transcript], bool] | None = None
    name: str = "conversation"


@dataclass
class Trajectory:
    """A Setting played on a seed; ``input_id`` is the seed row's id."""

    input_id: str
    setting: str
    transcript: Transcript
    turns_used: int
    stop_reason: str
    meta: dict = field(default_factory=dict)


def conversation_gen(
    setting: Setting, seed: str, input_id: str = "0",
) -> Generator[LLMRequest, str, Trajectory]:
    """Play ``setting`` on ``seed``, yielding each model call; return the Trajectory.

    Stops after ``max_turns`` turns, the seed being the first, or once
    ``setting.stop`` is true.
    """
    if not setting.participants:
        raise ValueError("Setting needs at least one participant.")

    transcript = Transcript()
    driver = setting.participants[0]
    transcript.message(driver.name, seed, role=driver.role)  # seed = turn 0
    turns = 1
    stop_reason = "max_turns"

    idx = 1  # next actor after the seeding driver
    while turns < setting.max_turns:
        actor = setting.participants[idx % len(setting.participants)]
        steps = yield from actor.turn(transcript)
        for s in steps:
            transcript.add(s)
        turns += 1
        if setting.stop is not None and setting.stop(transcript):
            stop_reason = "stop_condition"
            break
        idx += 1

    return Trajectory(input_id, setting.name, transcript, turns, stop_reason)


def run_conversation(
    setting: Setting, seed: str, call: Callable[[LLMRequest], str], input_id: str = "0",
) -> Trajectory:
    """Play one conversation, answering each model call with ``call(request)``."""
    return drive_sync(conversation_gen(setting, seed, input_id), call)
