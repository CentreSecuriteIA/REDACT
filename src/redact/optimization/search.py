"""Beam search over conversation moves toward a judge's yes.

At each depth every frontier node is extended by each move ``candidates``
proposes, the target replies to all of them in one round, the judge scores the
replies, and the ``beam`` best go on. Every node explored is kept as the tree.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass

from redact.llm_pipeline import LLMRequest, batch_check_samples, drive_generators
from redact.llms import ModelClient
from redact.multi_turn import Transcript
from redact.multi_turn.evaluate import judge_prompt


@dataclass
class Node:
    """A conversation in the search tree, with the judge's verdict on it."""

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
    """A copy of the transcript with ``message`` added and the target's reply to it."""
    t = copy.deepcopy(parent_transcript)
    t.note("strategy", "attacker", f"{move_name} (d{depth})")
    t.message("attacker", message, role="user")
    reply = yield LLMRequest(target_model, t.as_messages())
    t.reply("target", reply, role="assistant", model=target_model)
    return t


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
    verbose: bool = True,
    resolve=None,
    prompt_dir=None,
) -> tuple[Node, list[Node]]:
    """Search for the conversation the judge says yes to.

    Args:
        seed: The goal, the attacker's first message.
        target_model: The model under attack, required.
        candidates: ``(transcript, depth) -> [(move_name, message), ...]``.
        judge_model: The scoring model; pick one that is not the target.
        judge_system: The criterion, as the judge's system prompt.
        beam: Nodes kept per depth; ``1`` refines a single line.
        depth: The most attacker turns added to the seed.
        stop_on_success: Stop at the first depth where a node succeeds.
        prompt_dir: The prompt overlay the judge's prompt is read through.

    Returns:
        ``(best_node, tree)``, the tree being every node explored.
    """
    if not target_model:
        raise ValueError("optimize requires a target_model.")
    resolve = resolve or ModelClient.create
    judge = resolve(judge_model)
    judge_messages = judge_prompt(judge_system, prompt_dir)
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
            gens, resolve=resolve,
            finalize=lambda k, t: t, on_error=lambda k, exc: None,
            progress=f"optimize depth {d}/{depth}" if verbose else None,
        )
        keys = [k for k in gens if expanded.get(k) is not None]
        if not keys:
            break

        replies = [expanded[k].last_reply() for k in keys]
        checks = batch_check_samples(
            judge, replies,
            lambda goal, reply: judge_messages(
                goal=goal, label="RESPONSE", sample=reply),
            originals=[seed] * len(keys),
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
    """One dict per node of the tree, for JSON or a frame."""
    return [
        {"id": n.id, "parent": n.parent, "depth": n.depth, "move": n.move,
         "score": n.score, "success": n.success, "judge_reasoning": n.judge_reasoning,
         "final_reply": n.transcript.last_reply()}
        for n in tree
    ]
