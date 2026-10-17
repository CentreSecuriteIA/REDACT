"""The attack stage: generate and score."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from redact.multi_turn import evaluate_conversations, generate_conversations

from .setting import build_attack_setting, success_judge_system


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
    prompt_dir=None,
    **kwargs,
) -> pd.DataFrame:
    """Play an attack against ``target_model`` on every seed and save the conversations.

    ``prompt_dir`` is the prompt overlay of the pair attacker. ``**kwargs`` go to :func:`redact.multi_turn.generate_conversations`
    (``iterations``, ``resume``, ``batch_size``, ``resolve``, ``verbose``, ...).
    Score the result with :func:`score_attacks`.
    """
    factory = build_attack_setting(
        target_model, attack, attacker_model=attacker_model, followups=followups,
        jailbreak_technique=jailbreak_technique, max_turns=max_turns, target_system=target_system,
        prompt_dir=prompt_dir,
    )
    return generate_conversations(seeds, factory, data_dir=data_dir, **kwargs)


def score_attacks(
    data_dir=None, judge_model: str | None = None, prompt_dir=None, **kwargs,
) -> pd.DataFrame:
    """Score saved attack conversations for jailbreak success.

    ``judge_model`` should not be the target. The judge's prompts are read
    through the overlay ``prompt_dir``. ``**kwargs`` go to
    :func:`redact.multi_turn.evaluate_conversations`.
    """
    return evaluate_conversations(
        data_dir=data_dir, judge_model=judge_model,
        judge_system=success_judge_system(prompt_dir), scope="last_reply",
        prompt_dir=prompt_dir, **kwargs,
    )
