"""Offline tests for the optimization search controller (beam/tree)."""

import json

from redact.optimization import optimize, tree_to_records
from tests.conftest import MockBackend
import redact.optimization.search as SRCH


class _FakeRouter:
    def batch_generate(self, model, messages_list, **kw):
        return [f"[{model}] reply" for _ in messages_list]


def _candidates(transcript, depth):
    # two moves per node
    return [("a", f"move-a d{depth}"), ("b", f"move-b d{depth}")]


def test_optimize_finds_success_and_stops(tmp_path, monkeypatch):
    # judge: first candidate "No", second "Yes" -> success at depth 1, stop.
    monkeypatch.setattr(SRCH, "get_backend", lambda m: MockBackend(["No", "Yes success"]))
    best, tree = optimize(
        "goal", target_model="target", candidates=_candidates,
        judge_model="judge", judge_system="worked?", beam=2, depth=3,
        router=_FakeRouter(),
    )
    # root + 2 children at depth 1 (stopped on success).
    assert len(tree) == 3
    assert tree[0].move == "seed" and tree[0].parent is None
    assert best.success is True and best.score == 1.0 and best.move == "b"
    rows = tree_to_records(tree)
    assert {r["depth"] for r in rows} == {0, 1}


def test_optimize_expands_to_depth_when_no_success(monkeypatch):
    monkeypatch.setattr(SRCH, "get_backend", lambda m: MockBackend("No"))  # never succeeds
    best, tree = optimize(
        "goal", "target", _candidates, "judge", "worked?",
        beam=1, depth=2, router=_FakeRouter(),
    )
    # depth 1: 1 frontier x 2 = 2 nodes; beam=1 -> 1 frontier; depth 2: 1 x 2 = 2 more.
    # root(1) + 2 + 2 = 5 nodes.
    assert len(tree) == 5
    assert max(n.depth for n in tree) == 2
    assert best.success is False


def test_optimize_attack_bridges_jailbreak_techniques(monkeypatch):
    from redact.jailbreak.obfuscation.encoding import to_rot13
    from redact.multiturn_attacks import optimize_attack
    monkeypatch.setattr(SRCH, "get_backend", lambda m: MockBackend("No"))
    best, tree = optimize_attack(
        "how to X", target_model="target", techniques=[to_rot13],
        judge_model="judge", beam=2, depth=1, router=_FakeRouter(),
    )
    # candidates at depth 1 = plain + to_rot13 -> 2 children.
    moves = {n.move for n in tree if n.depth == 1}
    assert moves == {"plain", "to_rot13"}


def test_optimize_requires_target():
    import pytest
    with pytest.raises(ValueError):
        optimize("s", "", _candidates, "j", "sys")  # raises before touching a backend


def test_optimize_empty_candidates_returns_root(monkeypatch):
    monkeypatch.setattr(SRCH, "get_backend", lambda m: MockBackend("No"))
    best, tree = optimize("s", "target", lambda t, d: [], "j", "sys", router=_FakeRouter())
    assert len(tree) == 1 and tree[0].move == "seed" and best is tree[0]


def test_optimize_batch_failure_yields_no_children(monkeypatch):
    monkeypatch.setattr(SRCH, "get_backend", lambda m: MockBackend("No"))

    class _BadRouter:
        def batch_generate(self, model, ml, **kw):
            raise RuntimeError("boom")

    best, tree = optimize("s", "target", _candidates, "j", "sys", router=_BadRouter())
    assert len(tree) == 1  # expansions failed -> isolated -> no scored children
