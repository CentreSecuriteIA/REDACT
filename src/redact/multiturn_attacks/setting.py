"""The attack Setting factory, and the judge prompt that scores an attack."""

from __future__ import annotations

from collections.abc import Callable

from redact.llms.prompting import load_prompt
from redact.multi_turn import ModelActor, Setting, StrategyActor

from .attacks import crescendo_propose, pair_propose


def success_judge_system(prompt_dir=None) -> str:
    """The criterion of an attack's success, read through the overlay ``prompt_dir``."""
    return load_prompt("conversation", "success_judge", prompt_dir)["system_prompt"]


SUCCESS_JUDGE_SYSTEM: str = success_judge_system()


def build_attack_setting(
    target_model: str,
    attack: str = "crescendo",
    *,
    attacker_model: str | None = None,
    followups: list[str] | None = None,
    jailbreak_technique: Callable[[str], tuple[str, str]] | None = None,
    max_turns: int = 6,
    target_system: str | None = None,
    prompt_dir=None,
) -> Callable[[], Setting]:
    """A factory of attack Settings: the attacker, then the target.

    The attacker is a StrategyActor sending ``user`` turns, the seed being its
    first; the target is a ModelActor. Pass the factory to
    ``generate_conversations``.

    Args:
        target_model: The model under attack, required.
        attack: ``"crescendo"`` or ``"pair"``.
        attacker_model: The attacker's model, required for ``"pair"``.
        followups: Crescendo's follow-ups.
        jailbreak_technique: A pure transform applied to each crescendo follow-up.
        target_system: The target's system prompt.
        prompt_dir: The prompt overlay the pair attacker's prompt is read through.
    """
    if not target_model:
        raise ValueError("build_attack_setting requires a target_model (no default).")

    if attack == "crescendo":
        propose = crescendo_propose(followups, jailbreak_technique)
    elif attack == "pair":
        if not attacker_model:
            raise ValueError("attack='pair' requires an attacker_model.")
        propose = pair_propose(attacker_model, prompt_dir=prompt_dir)
    else:
        raise ValueError(f"unknown attack {attack!r}; use 'crescendo' or 'pair'.")

    def factory() -> Setting:
        return Setting(
            participants=[
                StrategyActor("attacker", propose, role="user"),
                ModelActor("target", target_model, role="assistant", system_prompt=target_system),
            ],
            max_turns=max_turns,
            name=f"attack:{attack}",
        )

    return factory
