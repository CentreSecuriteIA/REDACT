"""Deal a plan of synthetic inputs and read its shares: a test utility.

:func:`deal` plans ``n`` inputs through :func:`redact.jailbreak.plan.plan_run`,
in a temporary folder removed afterwards, and returns every unit's
combination (:func:`plan_rows` its manifest row). :func:`shares` gives each
technique's share of the units and its share within its family, and
:func:`spend` what the units spent in complexity. They are for tests and for
tuning a kind; the run's coverage table does not hold them. From the
repository root, ``py -m tests.jailbreak.simulate [N]`` prints the shares for
the default adversarial kind, and ``py -m tests.jailbreak.simulate [N]
budget`` what the unit budget and the group budget spend
(:func:`compare_budgets`).
"""

import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.dataset import Manifest
from redact.jailbreak.augmentation import resolve, rounds_of
from redact.jailbreak.plan import plan_run
from redact.jailbreak.pool import technique_pool

#: The complexity from which a technique is heavy: the budget has room for
#: one only when the unit's other picks leave it.
HEAVY = 3


def inputs(n: int, groups: int = 1) -> pd.DataFrame:
    """``n`` harmful inputs spread over ``groups`` categories, so the plan
    deals ``groups`` groups."""
    return pd.DataFrame({
        "prompt": [f"simulated input {i} about a topic" for i in range(n)],
        "category": [f"category {i % groups}" for i in range(n)],
        "entry_type": ["harmful"] * n,
    })


def plan_rows(
    frame: pd.DataFrame,
    pool: list[Callable],
    *,
    seed: int = 42,
    rounds: list[dict] | None = None,
    family_weights: dict | None = None,
    layer_rates: dict | None = None,
) -> list[dict]:
    """The manifest rows of a plan of the inputs in ``frame``, round after
    round; ``rounds`` is one sampler-settings dict per round (one budget
    round of the defaults when ``None``)."""
    rounds = rounds or [{}]
    with tempfile.TemporaryDirectory() as folder:
        manifest = Path(folder) / "simulated.manifest.jsonl"
        plan_run(frame, pool, manifest_path=manifest, seed=seed,
                 iterations=len(rounds), settings_per_iteration=rounds,
                 family_weights=family_weights, layer_rates=layer_rates,
                 verbose=False)
        return Manifest(manifest).load()


def deal(pool: list[Callable], n: int, *, groups: int = 1, **plan) -> list[list[str]]:
    """The combination of every unit of a plan of ``n`` inputs in ``groups``
    groups, round after round; ``plan`` is :func:`plan_rows`' keywords."""
    return [row["combination"] for row in plan_rows(inputs(n, groups), pool, **plan)]


def plan_kind(section: dict, frame: pd.DataFrame, *, kind: str = "adversarial",
              seed: int = 42, base: dict | None = None) -> tuple[list[dict], list]:
    """:func:`plan_rows` for an augmentation ``kind`` resolved from
    ``section``, as the stage plans it; the rows and the kind's pool. ``base``
    is a dataset type's section under ``section`` (the eval type's, say)."""
    resolved = resolve({kind: section}, kind, base=base)
    pool = technique_pool(resolved)
    rows = plan_rows(
        frame, pool, seed=seed, rounds=rounds_of(resolved),
        family_weights=resolved["family_weights"] or None,
        layer_rates=resolved.get("layer_rates") or None)
    return rows, pool


def deal_kind(section: dict, n: int, *, kind: str = "adversarial",
              groups: int = 1, seed: int = 42,
              base: dict | None = None) -> tuple[list[list[str]], list]:
    """:func:`deal` for an augmentation ``kind``: the combinations of a plan
    of ``n`` inputs in ``groups`` groups (:func:`plan_kind`) and the pool."""
    rows, pool = plan_kind(section, inputs(n, groups), kind=kind, seed=seed,
                           base=base)
    return [row["combination"] for row in rows], pool


def shares(combinations: list[list[str]], pool: list[Callable]) -> dict[str, dict]:
    """Per pool technique: its first ``family``, its ``count``, its ``share``
    of the units and its ``share_in_family``, its count over its family's
    (0 for a family never drawn)."""
    family_of = {fn.__name__: (getattr(fn, "families", None) or [""])[0] for fn in pool}
    counts = Counter(name for names in combinations for name in names)
    per_family = Counter()
    for name, count in counts.items():
        per_family[family_of.get(name, "")] += count
    units = len(combinations) or 1
    return {
        name: {
            "family": family,
            "count": counts[name],
            "share": counts[name] / units,
            "share_in_family": (counts[name] / per_family[family]
                                if per_family[family] else 0.0),
        }
        for name, family in family_of.items()
    }


def top_over_even(table: dict, family: str) -> float:
    """The largest share of ``family`` one technique has, over an even share."""
    in_family = [row["share_in_family"] for row in table.values()
                 if row["family"] == family]
    return max(in_family) * len(in_family)


def spend(combinations: list[list[str]], pool: list[Callable], *,
          limit: int = 6) -> dict:
    """What the units spent: the ``mean`` complexity of a unit, the
    ``highest`` of one unit, the units ``above`` ``limit``, the mean number of
    ``obfuscations`` a unit draws, and the ``heavy`` draws, techniques of
    complexity :data:`HEAVY` or more."""
    by_name = {fn.__name__: fn for fn in pool}
    units = [[by_name[name] for name in names] for names in combinations]
    costs = [sum(fn.complexity for fn in fns) for fns in units]
    return {
        "mean": sum(costs) / len(costs),
        "highest": max(costs),
        "above": sum(cost > limit for cost in costs),
        "obfuscations": sum(fn.layer == "obfuscation" for fns in units
                            for fn in fns) / len(units),
        "heavy": sum(fn.complexity >= HEAVY for fns in units for fn in fns),
    }


def compare_budgets(n: int = 3000, *, groups: int = 3, seed: int = 42,
                    limit: int = 6, ceiling: int = 9) -> str:
    """What ``n`` units of the default adversarial kind spend under the unit
    budget and under the group budget with ``ceiling``, as text: the numbers
    of :func:`spend`, then for the ``encode`` and ``sensitive_words``
    families the top technique's share over an even share and the heavy
    techniques drawn."""
    runs = {"unit": {"max_complexity": limit},
            "group": {"max_complexity": limit, "budget": "group",
                      "max_unit_complexity": ceiling}}
    lines = [f"{'budget':<8} {'mean':>5} {'highest':>8} {f'above {limit}':>8} "
             f"{'obfusc.':>8} {'heavy':>6}  families"]
    for budget, section in runs.items():
        drawn, pool = deal_kind(section, n, groups=groups, seed=seed)
        table = shares(drawn, pool)
        stats = spend(drawn, pool, limit=limit)
        complexity = {fn.__name__: fn.complexity for fn in pool}
        lines.append(
            f"{budget:<8} {stats['mean']:>5.2f} {stats['highest']:>8} "
            f"{stats['above']:>8} {stats['obfuscations']:>8.2f} {stats['heavy']:>6}")
        for family in ("encode", "sensitive_words"):
            heavy = [name for name, row in table.items()
                     if row["family"] == family and complexity[name] >= HEAVY]
            drawn_heavy = sorted(name for name in heavy if table[name]["count"])
            lines.append(f"{'':<8}   {family}: top {top_over_even(table, family):.2f}x "
                         f"an even share; heavy drawn: {len(drawn_heavy)} of "
                         f"{len(heavy)} {drawn_heavy or ''}")
    return "\n".join(lines)


def report(table: dict[str, dict]) -> str:
    """``table`` (:func:`shares`) as text, one technique a line, by family."""
    lines = [f"{'family':<20} {'technique':<52} {'count':>6} {'share':>7} "
             f"{'in family':>9}"]
    for name, row in sorted(table.items(), key=lambda kv: (kv[1]["family"], kv[0])):
        lines.append(f"{row['family']:<20} {name:<52} {row['count']:>6} "
                     f"{row['share']:>7.3f} {row['share_in_family']:>9.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    if "budget" in sys.argv:
        print(compare_budgets(size))
    else:
        drawn, kind_pool = deal_kind({}, size, groups=3)
        print(report(shares(drawn, kind_pool)))
