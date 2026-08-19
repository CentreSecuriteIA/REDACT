"""General optimize-toward-an-objective search over conversation moves (beam/tree).

The **search controller above** ``drive_generators`` (plan weak-point W1): at each
depth it expands every frontier node into candidate next-turns **in parallel**
(batched through the router), scores each with a **judge**, logs the whole tree,
keeps the best ``beam`` nodes, and continues — with backtracking implicit in keeping
multiple paths. Objective-general: the ``candidates`` proposer and the judge define
the objective, so this works for jailbreak optimization (``multiturn_attacks``) or
anything else with a scorer.

Subsumes: beam=1 → linear refine (PAIR); beam>1 → tree search (TAP); candidates =
technique-combinations over a base follow-up → optimized-crescendo (try combinations,
keep the best, escalate).

MVP: ``candidates(transcript, depth) -> list[(move_name, message)]`` returns concrete
attacker messages (e.g. different jailbreak-technique combinations); scoring is via a
judge model (binary success). An attacker-LLM-proposed candidate variant is a noted
extension.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass

from redact.llms import get_router
from redact.llms.calls import batch_check_samples
from redact.llms.conversation import LLMRequest, Transcript, drive_generators


@dataclass
class Node:
    """One node in the search tree = a conversation state + its score."""

    id: int
    parent: int | None
    depth: int
    move: str
    transcript: Transcript
    score: float = 0.0
    success: bool = False
    judge_reasoning: str = ""


def _expand_gen(parent_transcript: Transcript, move_name: str, message: str,
                target_model: str, depth: int):
    """A one-turn expansion generator: append the candidate attacker message, call the
    target, return the extended transcript."""
    t = copy.deepcopy(parent_transcript)
    t.note("strategy", "attacker", f"{move_name} (d{depth})")
    t.message("attacker", message, role="user")
    reply = yield LLMRequest(target_model, t.as_messages())
    t.reply("target", reply, role="assistant", model=target_model)
    return t


def _final_reply(transcript: Transcript) -> str:
    return next((s.content for s in reversed(transcript.steps) if s.type == "reply"), "")


def optimize(
    seed: str,
    target_model: str,
    candidates: Callable[[Transcript, int], list[tuple[str, str]]],
    judge_model: str,
    judge_system: str,
    *,
    beam: int = 2,
    depth: int = 3,
    stop_on_success: bool = True,
    router=None,
) -> tuple[Node, list[Node]]:
    """Beam/tree search for the highest-scoring conversation continuation.

    Args:
        seed: the goal / opening message (attacker turn 0).
        target_model: the model under attack (**required**).
        candidates: ``(transcript, depth) -> [(move_name, message), ...]`` — the branching.
        judge_model / judge_system: scorer (binary success → score 1.0/0.0). Independent
            of the target.
        beam: nodes kept per depth. ``1`` = linear refine; ``>1`` = tree/beam.
        depth: max search depth (turns of optimization).
        stop_on_success: stop as soon as any candidate succeeds.

    Returns:
        ``(best_node, tree)`` where ``tree`` is every node explored (the full trajectory).
    """
    if not target_model:
        raise ValueError("optimize requires a target_model.")
    router = router or get_router()
    judge_backend = router.get_backend(judge_model)
    rate_limiter = router.rate_limiter
    ids = itertools.count()

    root_t = Transcript()
    root_t.message("attacker", seed, role="user")
    root = Node(next(ids), None, 0, "seed", root_t)
    tree: list[Node] = [root]
    frontier: list[Node] = [root]

    for d in range(1, depth + 1):
        gens: dict[int, object] = {}
        origin: dict[int, tuple[Node, str]] = {}
        for node in frontier:
            for move_name, message in candidates(node.transcript, d):
                key = next(ids)
                gens[key] = _expand_gen(node.transcript, move_name, message, target_model, d)
                origin[key] = (node, move_name)
        if not gens:
            break

        expanded = drive_generators(
            gens, router=router,
            finalize=lambda k, t: t, on_error=lambda k, exc: None,
        )
        keys = [k for k in gens if expanded.get(k) is not None]
        if not keys:
            break

        payloads = [f"GOAL:\n{seed}\n\nRESPONSE:\n{_final_reply(expanded[k])}" for k in keys]
        checks = batch_check_samples(
            judge_backend, judge_model, payloads,
            lambda s: [{"role": "system", "content": judge_system},
                       {"role": "user", "content": s}],
            rate_limiter=rate_limiter,
        )

        children: list[Node] = []
        for k, (success, reason) in zip(keys, checks):
            parent, move_name = origin[k]
            node = Node(next(ids), parent.id, d, move_name, expanded[k],
                        score=1.0 if success else 0.0, success=bool(success),
                        judge_reasoning=reason)
            tree.append(node)
            children.append(node)

        children.sort(key=lambda n: n.score, reverse=True)
        frontier = children[:beam]
        if stop_on_success and any(n.success for n in children):
            break

    best = max(tree, key=lambda n: (n.score, n.depth))
    return best, tree


def tree_to_records(tree: list[Node]) -> list[dict]:
    """Flatten a search tree to serializable rows (nodes = move + score + parent)."""
    return [
        {"id": n.id, "parent": n.parent, "depth": n.depth, "move": n.move,
         "score": n.score, "success": n.success, "judge_reasoning": n.judge_reasoning,
         "final_reply": _final_reply(n.transcript)}
        for n in tree
    ]
