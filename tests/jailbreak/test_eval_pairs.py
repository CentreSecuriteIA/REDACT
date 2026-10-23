"""Which pairs of the eval pool's 36 techniques may share a combination, pinned.

The eval type's attacks draw their combinations from a pool of 36 techniques
(``fixtures/eval_pool.json``, pinned), and the rules decide which two of them
may be drawn together. The pool stays, but a rule that is added, dropped or
narrowed changes the pairs the eval draws, and so which attacks an eval run
holds. ``fixtures/eval_pairs.json`` records the pairs ``rules.compatible``
admits, ``"<technique>|<technique>"``, the two names in alphabetical order,
so a rule that moves the eval shows as a diff of the file, which these
tests name. The eval type draws only these pairs (its count rounds draw
exactly one technique, then exactly two).

An eval review that changes the pairs regenerates the fixture from the
repository root, with ``PYTHONPATH=src python -m tests.jailbreak.test_eval_pairs``,
and says in its commit which pairs went.
"""

import json
from pathlib import Path

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.pool import technique_pool
from redact.jailbreak.rules import compatible
from redact.recipe import load_dataset_type

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_PAIRS = _FIXTURES / "eval_pairs.json"
_POOL = _FIXTURES / "eval_pool.json"


def _eval_pool() -> list:
    """The techniques the eval type's attacks draw from, as it resolves."""
    return technique_pool(resolve(
        load_dataset_type("eval")["augmentations"], "adversarial"))


def snapshot() -> list[str]:
    """The pairs of the eval pool that the rules admit, sorted."""
    pool = _eval_pool()
    return sorted(
        "|".join(sorted((first.__name__, second.__name__)))
        for i, first in enumerate(pool) for second in pool[i + 1:]
        if second in compatible([first], [second]))


def _dumps(pairs: list[str]) -> str:
    """The fixture as JSON, one pair per line."""
    rows = ",\n".join(f"  {json.dumps(pair)}" for pair in pairs)
    return (
        '{\n "_comment": "Regenerate with PYTHONPATH=src py -m '
        'tests.jailbreak.test_eval_pairs. The pairs of the 36 eval pool '
        'techniques that rules.compatible admits.",\n'
        f' "pairs": [\n{rows}\n ]\n}}\n')


def test_the_eval_pool_is_the_pinned_36_names():
    assert sorted(fn.__name__ for fn in _eval_pool()) == json.loads(
        _POOL.read_text(encoding="utf-8"))


def test_the_pairs_of_the_eval_pool_the_rules_admit_are_as_pinned():
    recorded = set(json.loads(_PAIRS.read_text(encoding="utf-8"))["pairs"])
    current = set(snapshot())
    assert (sorted(current - recorded), sorted(recorded - current)) == ([], []), (
        "pairs the eval now draws, and pairs it no longer can; if a rule is "
        "meant to move the eval, regenerate with "
        "PYTHONPATH=src py -m tests.jailbreak.test_eval_pairs")


def test_the_fixture_is_what_the_regeneration_writes():
    assert _PAIRS.read_text(encoding="utf-8") == _dumps(snapshot())


if __name__ == "__main__":
    _PAIRS.write_text(_dumps(snapshot()), encoding="utf-8")
