"""Jailbreak optimization = the general ``optimization`` search with jailbreak moves.

``jailbreak_candidates`` proposes, at each depth, the plain escalation follow-up plus
one candidate per jailbreak technique (each technique transforms the follow-up) — so
the search "tries combinations of the jailbreak techniques and keeps the best,
escalating that way." ``optimize_attack`` wires it to the success judge.
"""

from __future__ import annotations

from collections.abc import Callable

from redact.optimization import optimize

from .attacks import DEFAULT_CRESCENDO_FOLLOWUPS
from .setting import SUCCESS_JUDGE_SYSTEM


def jailbreak_candidates(
    techniques: list[Callable[[str], tuple[str, str]]],
    followups: list[str] | None = None,
):
    """Build a ``candidates(transcript, depth)`` proposer: the plain escalation for this
    depth + one variant per (pure-transform) jailbreak technique applied to it."""
    followups = followups or DEFAULT_CRESCENDO_FOLLOWUPS

    def candidates(transcript, depth: int) -> list[tuple[str, str]]:
        base = followups[min(max(depth - 1, 0), len(followups) - 1)]
        out: list[tuple[str, str]] = [("plain", base)]
        for tech in techniques:
            msg, _info = tech(base)
            out.append((getattr(tech, "__name__", "jailbreak"), msg))
        return out

    return candidates


def optimize_attack(
    seed: str,
    target_model: str,
    techniques: list[Callable[[str], tuple[str, str]]],
    judge_model: str,
    *,
    beam: int = 2,
    depth: int = 3,
    followups: list[str] | None = None,
    verbose: bool = True,
    router=None,
):
    """Search jailbreak technique-combinations against ``target_model`` toward success.

    Returns ``(best_node, tree)`` from :func:`redact.optimization.optimize` — the best
    conversation found plus the full search tree (candidates + judge scores per depth).
    """
    return optimize(
        seed, target_model, jailbreak_candidates(techniques, followups),
        judge_model, SUCCESS_JUDGE_SYSTEM, beam=beam, depth=depth,
        verbose=verbose, router=router,
    )
