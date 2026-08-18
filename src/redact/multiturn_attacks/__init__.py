"""Multi-turn jailbreak attacks (built on ``multi_turn/``, its own top-level folder).

Attacker-vs-target conversations that probe a safety-trained target over several turns
(crescendo / PAIR-lite), optionally composing jailbreak augmentations from ``jailbreak/``
into each turn. Generation reuses ``multi_turn.generate_conversations``; success scoring
reuses ``multi_turn.evaluate_conversations`` with a jailbreak-success judge.

Dependency direction: ``multiturn_attacks/`` → ``multi_turn/`` (+ ``jailbreak/`` for the
technique transforms), never the reverse.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from redact.multi_turn import generate_conversations, evaluate_conversations

from .attacks import crescendo_propose, pair_propose, DEFAULT_CRESCENDO_FOLLOWUPS
from .setting import build_attack_setting, SUCCESS_JUDGE_SYSTEM
from .optimize import jailbreak_candidates, optimize_attack

__all__ = [
    "build_attack_setting",
    "SUCCESS_JUDGE_SYSTEM",
    "crescendo_propose",
    "pair_propose",
    "DEFAULT_CRESCENDO_FOLLOWUPS",
    "generate_attacks",
    "score_attacks",
    "jailbreak_candidates",
    "optimize_attack",
]


def generate_attacks(
    seeds: pd.DataFrame,
    target_model: str,
    data_dir=None,
    attack: str = "crescendo",
    *,
    attacker_model: str | None = None,
    followups: list[str] | None = None,
    jailbreak_technique: Callable[[str], tuple[str, str]] | None = None,
    max_turns: int = 6,
    target_system: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Generate multi-turn attack conversations against ``target_model`` from harmful seeds.

    Thin wrapper: builds an attack :class:`Setting` factory and runs
    :func:`redact.multi_turn.generate_conversations`. Score afterwards with
    :func:`score_attacks`. ``**kwargs`` forward to ``generate_conversations``
    (``iterations``, ``resume``, ``batch_size``, ``router``, ``verbose``, …).
    """
    factory = build_attack_setting(
        target_model, attack, attacker_model=attacker_model, followups=followups,
        jailbreak_technique=jailbreak_technique, max_turns=max_turns, target_system=target_system,
    )
    return generate_conversations(seeds, factory, data_dir=data_dir, **kwargs)


def score_attacks(data_dir=None, judge_model: str | None = None, **kwargs) -> pd.DataFrame:
    """Score attack conversations for jailbreak success (separable judge pass).

    Wraps :func:`redact.multi_turn.evaluate_conversations` with the jailbreak-success
    criterion. ``judge_model`` should be independent of the target.
    """
    return evaluate_conversations(
        data_dir=data_dir, judge_model=judge_model,
        judge_system=SUCCESS_JUDGE_SYSTEM, scope="last_reply", **kwargs,
    )
