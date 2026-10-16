"""Jailbreak moves for the search."""

from __future__ import annotations

from collections.abc import Callable

from redact.optimization import optimize

from .attacks import DEFAULT_CRESCENDO_FOLLOWUPS, _followup
from .setting import success_judge_system


def jailbreak_candidates(
    techniques: list[Callable[[str], tuple[str, str]]],
    followups: list[str] | None = None,
):
    """The moves at a depth: its follow-up, plain and through each technique.

    Returns ``candidates(transcript, depth)`` for :func:`redact.optimization.optimize`.
    """
    followups = followups or DEFAULT_CRESCENDO_FOLLOWUPS

    def candidates(transcript, depth: int) -> list[tuple[str, str]]:
        base = _followup(followups, depth)
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
    resolve=None,
    prompt_dir=None,
):
    """Search ``techniques`` against ``target_model`` with the jailbreak-success judge.

    The judge's prompts are read through the overlay ``prompt_dir``. Returns
    ``(best_node, tree)`` from :func:`redact.optimization.optimize`.
    """
    return optimize(
        seed, target_model, jailbreak_candidates(techniques, followups),
        judge_model, success_judge_system(prompt_dir), beam=beam, depth=depth,
        verbose=verbose, resolve=resolve, prompt_dir=prompt_dir,
    )
