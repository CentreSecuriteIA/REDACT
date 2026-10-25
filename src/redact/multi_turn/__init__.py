"""Conversation datasets: a Setting played on seeds, saved and scored.

A planned extension, not finished: not part of a recipe run, and its
interfaces may change.

Usage::

    from redact.multi_turn import (
        ModelActor, ScriptedActor, Setting,
        evaluate_conversations, generate_conversations,
    )

    def make_setting():
        user = ScriptedActor("user", ["Go on."])
        bot = ModelActor("bot", "venice-uncensored")
        return Setting(participants=[user, bot], max_turns=4)

    generate_conversations(seeds, make_setting, data_dir=DATA_DIR)
    scored = evaluate_conversations(data_dir=DATA_DIR, judge_model="claude-opus-4-6")
"""

from .actors import Actor, ModelActor, ScriptedActor, StrategyActor
from .conversation import Setting, Trajectory, conversation_gen, run_conversation
from .evaluate import evaluate_conversations
from .pipeline import generate_conversations
from .transcript import Step, StepType, Transcript

__all__ = [
    "Actor",
    "ScriptedActor",
    "ModelActor",
    "StrategyActor",
    "Setting",
    "Trajectory",
    "Step",
    "StepType",
    "Transcript",
    "conversation_gen",
    "run_conversation",
    "generate_conversations",
    "evaluate_conversations",
]
