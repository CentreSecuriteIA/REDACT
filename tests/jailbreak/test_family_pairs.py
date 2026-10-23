"""The refused family pairs, pinned.

``fixtures/family_pairs.json`` records, for every two families the rules ever
keep apart, how many of their techniques' pairs ``rules.compatible`` refuses
and out of how many: ``"<family>|<family>": [refused, total]``, the two names
in alphabetical order. A technique counts under its first family; each pair
of techniques counts once in either order, so a rule kept in one direction
only (a family's ``cross_incompatible_with`` that its partner's list does not
repeat) changes a count. A pair of families with no refused pair is not
listed; one family's techniques never meet each other (one technique per
family) and are not counted.

A rule that is added, dropped or narrowed, or a technique that is refused
where its family's others are not, changes a line, which this test names; the
digests of a plan's draws change for any of them, but only say that something
did. A change meant to alter the matrix regenerates the fixture, from the
repository root, with ``PYTHONPATH=src python -m tests.jailbreak.test_family_pairs``.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.rules import compatible

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "family_pairs.json"


def snapshot() -> dict[str, list[int]]:
    """``[refused, total]`` for each pair of families with a refused pair of
    techniques, as the rules give them now."""
    techniques = all_techniques()
    refused: Counter = Counter()
    total: Counter = Counter()
    for first in techniques:
        for second in techniques:
            a, b = first.families[0], second.families[0]
            if a == b:
                continue
            pair = "|".join(sorted((a, b)))
            total[pair] += 1
            if second not in compatible([first], [second]):
                refused[pair] += 1
    return {pair: [refused[pair], total[pair]] for pair in sorted(refused)}


def _dumps(pairs: dict[str, list[int]]) -> str:
    """The fixture as JSON, one pair per line."""
    rows = ",\n".join(f"  {json.dumps(pair)}: {json.dumps(counts)}"
                      for pair, counts in pairs.items())
    return (
        '{\n "_comment": "Regenerate with PYTHONPATH=src py -m '
        'tests.jailbreak.test_family_pairs. [refused, total] ordered pairs of '
        'techniques, by their first families.",\n'
        f' "pairs": {{\n{rows}\n }}\n}}\n')


@pytest.fixture(scope="module")
def recorded() -> dict[str, list[int]]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["pairs"]


def test_the_refused_family_pairs_are_as_pinned(recorded):
    current = snapshot()
    kept_apart = sorted(pair for pair in current if pair not in recorded)
    let_through = sorted(pair for pair in recorded if pair not in current)
    changed = sorted(f"{pair}: {recorded[pair]} -> {current[pair]}"
                     for pair in current if pair in recorded
                     and current[pair] != recorded[pair])
    assert (kept_apart, let_through, changed) == ([], [], []), (
        "families newly kept apart, families let through, and counts that "
        "changed; if a rule is meant to change, regenerate with "
        "PYTHONPATH=src py -m tests.jailbreak.test_family_pairs")


def test_the_fixture_is_what_the_regeneration_writes():
    assert _FIXTURE.read_text(encoding="utf-8") == _dumps(snapshot())


if __name__ == "__main__":
    _FIXTURE.write_text(_dumps(snapshot()), encoding="utf-8")
