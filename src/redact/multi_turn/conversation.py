"""A Setting played on a seed to a Trajectory."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass

from redact.llm_pipeline import LLMRequest, drive_sync

from .actors import Actor
from .transcript import Transcript


@dataclass
class Setting:
    """Who takes part in a conversation, and when it ends.

    The seed is the first participant's message; the participants then take
    turns in order. ``stop`` ends the conversation early once it returns true
    for the transcript.
    """

    participants: list[Actor]
    max_turns: int = 6
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
