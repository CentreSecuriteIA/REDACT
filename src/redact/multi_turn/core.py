"""General multi-turn conversation core: a ``Setting`` played to a ``Trajectory``.

Use-case-agnostic. Reuses ``llms/`` primitives (``LLMRequest``, ``Transcript``,
``Step``, ``drive_sync``); higher layers (``multiturn_attacks/``, ``optimization/``)
build on this. Dependency direction: ``multi_turn/`` → ``llms/`` only.

An **actor**'s turn is a generator ``turn(transcript) -> yields LLMRequest, returns
list[Step]`` (see :class:`Actor`). The runner plays actors in order (seed as the
first message, then alternate), logging every step. This is the single-conversation
(sync) path; a batched driver over many conversations comes later.

Plan: `.claude/theme4_multiturn_plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generator

from redact.llms.conversation import LLMRequest, Step, Transcript, drive_sync

# An actor's turn yields LLMRequests, is resumed with replies, and returns the
# Steps it produced (e.g. [strategy?, message] or [reply]).
ActorTurn = Generator[LLMRequest, str, "list[Step]"]


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------

class Actor:
    """Base actor. Subclasses implement :meth:`turn` as a generator."""

    def __init__(self, name: str, role: str = "user"):
        self.name = name
        self.role = role

    def turn(self, transcript: Transcript) -> ActorTurn:  # pragma: no cover
        raise NotImplementedError
        yield  # make this a generator function


class ScriptedActor(Actor):
    """Emits pre-set messages in order (no LLM). Also seeds / replays a transcript."""

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
    """A single LLM turn: renders the visible transcript and calls its model."""

    def __init__(self, name: str, model: str, role: str = "assistant",
                 system_prompt: str | None = None):
        super().__init__(name, role)
        self.model = model
        self.system_prompt = system_prompt

    def turn(self, transcript: Transcript) -> ActorTurn:
        reply = yield LLMRequest(self.model, transcript.as_messages(self.system_prompt))
        return [Step("reply", self.name, reply, role=self.role, model=self.model)]


class StrategyActor(Actor):
    """An adaptive, multi-step actor driven by a ``propose`` generator.

    ``propose(transcript, turn_index) -> Generator[LLMRequest, str, message | (message, note)]``
    builds this actor's next message — it may ``yield`` LLM requests (e.g. an attacker
    model working out a strategy, or analyzing the last reply) before returning the
    message (and an optional ``note`` logged as a ``strategy`` provenance step).

    ``turn_index`` = how many messages this actor has already contributed (derived
    from the transcript, so the actor is **stateless** — safe to deep-copy / reuse
    across conversations via a Setting factory). This is the extension point for
    crescendo / PAIR / prober behaviors (see ``multiturn_attacks/``).
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


# ---------------------------------------------------------------------------
# Setting + Trajectory
# ---------------------------------------------------------------------------

@dataclass
class Setting:
    """How a conversation is driven (the primary axis) + participants + limits.

    ``drive="conversation"`` (plain exchange) is the only mode implemented in this
    START; ``"feedback"`` (adaptive strategize→analyze) and ``"optimize"`` (search,
    in ``optimization/``) come later. ``participants`` are played in order with the
    seed as the first message, then alternating. ``stop`` is an optional predicate
    over the transcript for early termination.
    """

    participants: list[Actor]
    max_turns: int = 6
    drive: str = "conversation"
    stop: Callable[[Transcript], bool] | None = None
    name: str = "conversation"


@dataclass
class Trajectory:
    """The logged result of playing a Setting on a seed.

    ``input_id`` is the id of the source seed row — named to match the rest of
    the library (jailbreak / output / paraphrase all reference their source row
    as ``input_id``), so conversation datasets join to base inputs uniformly.
    """

    input_id: str
    setting: str
    transcript: Transcript
    turns_used: int
    stop_reason: str
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner (single conversation)
# ---------------------------------------------------------------------------

def conversation_gen(
    setting: Setting, seed: str, input_id: str = "0",
) -> Generator[LLMRequest, str, Trajectory]:
    """Generator that plays ``setting`` on ``seed`` and returns a :class:`Trajectory`.

    Turn 0 is the seed, logged as the first participant's message. Remaining
    participants then alternate; each ``turn`` may yield LLM requests (advanced by
    the driver). Stops at ``max_turns`` or when ``setting.stop`` fires.
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
    """Play one conversation to completion using a blocking ``call(LLMRequest) -> str``."""
    return drive_sync(conversation_gen(setting, seed, input_id), call)
