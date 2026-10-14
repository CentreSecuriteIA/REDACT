"""Attacker strategies for StrategyActor.

Each builder returns a ``propose(transcript, turn_index)`` generator. The goal
is the transcript's first message, the seed.
"""

from __future__ import annotations

from collections.abc import Callable

from redact.jailbreak.directives import load_templates
from redact.llm_pipeline import LLMRequest
from redact.llms.prompting import PromptTemplate

DEFAULT_CRESCENDO_FOLLOWUPS: list[str] = load_templates(
    "multi_turn", "crescendo_followups.json")["followups"]


def _followup(followups: list[str], turn: int) -> str:
    """The follow-up of ``turn``, the first being turn 1; the last repeats."""
    return followups[min(max(turn - 1, 0), len(followups) - 1)]


def crescendo_propose(
    followups: list[str] | None = None,
    jailbreak_technique: Callable[[str], tuple[str, str]] | None = None,
):
    """Crescendo: the next of ``followups`` each turn; calls no model.

    ``jailbreak_technique`` is an optional pure transform ``(str) -> (str, info)``
    from ``jailbreak/``, applied to each follow-up.
    """
    followups = followups or DEFAULT_CRESCENDO_FOLLOWUPS

    def propose(transcript, turn_index: int):
        yield from ()  # no LLM call
        msg = _followup(followups, turn_index)     # turn 0 is the seed
        note = f"crescendo step {turn_index}"
        if jailbreak_technique is not None:
            transformed, _info = jailbreak_technique(msg)
            msg = transformed
            note += f" +{getattr(jailbreak_technique, '__name__', 'jailbreak')}"
        return msg, note

    return propose


def pair_propose(attacker_model: str, system: str | None = None):
    """PAIR: ``attacker_model`` writes each next message, given the goal and the
    conversation so far. ``system`` replaces the packaged attacker prompt, as
    written."""
    prompt = PromptTemplate.load("conversation", "pair_attacker")
    if system:
        prompt.system_prompt = system

    def propose(transcript, turn_index: int):
        messages = prompt(goal=transcript.first_message(),
                          conversation=transcript.render())
        message = yield LLMRequest(attacker_model, messages)
        return str(message).strip(), f"pair attacker turn {turn_index}"

    return propose
