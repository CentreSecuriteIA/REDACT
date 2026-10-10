"""General multi-turn conversation datasets.

Define a :class:`Setting` (how a conversation is driven + participants) and play it
on a seed to produce a :class:`Trajectory` (a typed step log). Use-case-agnostic;
jailbreak multi-turn attacks (``multiturn_attacks/``) and optimization loops
(``optimization/``) build on this. See ``.claude/theme4_multiturn_plan.md``.
"""

from .core import (
    Actor,
    ModelActor,
    ScriptedActor,
    Setting,
    StrategyActor,
    Trajectory,
    conversation_gen,
    run_conversation,
)
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
