"""Multi-turn jailbreak attacks: a Setting against a safety-trained target.

An attacker (crescendo or PAIR) talks to the target over several turns, and a
separate judge pass scores whether the target complied.
"""

from .attacks import DEFAULT_CRESCENDO_FOLLOWUPS, crescendo_propose, pair_propose
from .optimize import jailbreak_candidates, optimize_attack
from .pipeline import generate_attacks, score_attacks
from .setting import SUCCESS_JUDGE_SYSTEM, build_attack_setting

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
