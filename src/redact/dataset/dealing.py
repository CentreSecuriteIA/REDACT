"""Balanced dealing: the running counts of one group, so a planner gives each
name its share and each optional part its rate, evenly over the group."""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Sequence


def stable_seed(*parts) -> int:
    """A seed from ``parts`` that is the same in every process.

    SHA-256, not the builtin ``hash``, which is salted per process.
    """
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8"))
    return int(digest.hexdigest()[:16], 16)


class Dealer:
    """Running counts of one group's names and optional parts.

    ``weights`` sets a name's share (1 by default); :meth:`pick` takes the
    name with the lowest count over weight. The planner records what each
    unit was given in ``counts`` (names), ``members`` (the members of a name,
    for :meth:`pick_member`), ``included`` (parts) and ``spent`` (the cost
    the units spent, for :meth:`bank`), and sets ``position`` to the units
    dealt so far, including the current. ``key`` names the group; with the
    part and the position it seeds where in the group each part's picks
    fall. ``targets`` holds each part's picks due by the last unit asked.
    """

    def __init__(self, weights: dict | None = None, key: tuple = ()) -> None:
        self.weights = dict(weights or {})
        self.key = key
        self.position = 0
        self.counts: Counter = Counter()
        self.members: Counter = Counter()
        self.included: Counter = Counter()
        self.spent = 0
        self.targets: dict[str, int] = {}

    def score(self, name) -> float:
        return self.counts[name] / self.weights.get(name, 1.0)

    def pick(self, rng: random.Random, names: Sequence):
        """The name of ``names`` with the lowest score, ties broken by ``rng``."""
        best = min(self.score(name) for name in names)
        return rng.choice([name for name in names if self.score(name) == best])

    def pick_member(self, rng: random.Random, names: Sequence):
        """The name of ``names`` with the lowest count in ``members``, ties
        broken by ``rng``. The planner passes the members of the name
        :meth:`pick` chose, so they come out even; ``members`` is kept apart
        from ``counts``, so :meth:`pick` deals the same with or without it."""
        least = min(self.members[name] for name in names)
        return rng.choice([name for name in names if self.members[name] == least])

    def include(self, part: str, rate: float) -> bool:
        """Whether this unit takes ``part``: the picks so far are below
        ``rate * position`` shifted by a jitter in [0, 1) seeded by the
        group, the part and the position, so a group gets ``rate * size``
        picks rounded down or up, every prefix of it within one pick of
        ``rate * position``, and a group of one takes the part with
        probability ``rate``. The jitter is drawn afresh for each unit: one
        offset per part repeated a part's picks every ``1 / rate`` units,
        so two parts of one rate fell on the same units, or never on the
        same, for a whole group."""
        jitter = random.Random(stable_seed(*self.key, part, self.position)).random()
        self.targets[part] = math.floor(rate * self.position + jitter)
        return self.included[part] < self.targets[part]

    def bank(self, per_unit: int) -> int:
        """What the units dealt before this one left unspent of ``per_unit``
        each: ``per_unit`` times their number, less ``spent``, never below 0.
        Units each given at most ``per_unit`` plus the bank keep ``spent`` at
        or below ``per_unit`` times ``position``: a cheap unit leaves what it
        did not spend to the units after it."""
        return max(0, per_unit * (self.position - 1) - self.spent)
