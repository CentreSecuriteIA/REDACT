"""Deal a plan of synthetic inputs and read its shares: a test utility.

:func:`deal` plans ``n`` inputs through :func:`redact.jailbreak.plan.plan_run`,
in a temporary folder removed afterwards, and returns every unit's
combination; :func:`shares` gives each technique's share of the units and its
share within its family. The shares are for tests and for tuning a kind; the
run's coverage table does not hold them. From the repository root,
``py -m tests.jailbreak.simulate [N]`` prints them for the default
adversarial kind.
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


def inputs(n: int, groups: int = 1) -> pd.DataFrame:
    """``n`` harmful inputs spread over ``groups`` categories, so the plan
    deals ``groups`` groups."""
    return pd.DataFrame({
        "prompt": [f"simulated input {i} about a topic" for i in range(n)],
        "category": [f"category {i % groups}" for i in range(n)],
        "entry_type": ["harmful"] * n,
    })


def deal(
    pool: list[Callable],
    n: int,
    *,
    groups: int = 1,
    seed: int = 42,
    rounds: list[dict] | None = None,
    family_weights: dict | None = None,
    layer_rates: dict | None = None,
) -> list[list[str]]:
    """The combination of every unit of a plan of ``n`` inputs, round after
    round; ``rounds`` is one sampler-settings dict per round (one budget
    round of the defaults when ``None``)."""
    rounds = rounds or [{}]
    with tempfile.TemporaryDirectory() as folder:
        manifest = Path(folder) / "simulated.manifest.jsonl"
        plan_run(inputs(n, groups), pool, manifest_path=manifest, seed=seed,
                 iterations=len(rounds), settings_per_iteration=rounds,
                 family_weights=family_weights, layer_rates=layer_rates,
                 verbose=False)
        return [row["combination"] for row in Manifest(manifest).load()]


def deal_kind(section: dict, n: int, *, kind: str = "adversarial",
              groups: int = 1, seed: int = 42,
              base: dict | None = None) -> tuple[list[list[str]], list]:
    """:func:`deal` for an augmentation ``kind`` resolved from ``section``, as
    the stage plans it; the combinations and the kind's pool. ``base`` is a
    dataset type's section under ``section`` (the eval type's, say)."""
    resolved = resolve({kind: section}, kind, base=base)
    pool = technique_pool(resolved)
    combinations = deal(
        pool, n, groups=groups, seed=seed, rounds=rounds_of(resolved),
        family_weights=resolved["family_weights"] or None,
        layer_rates=resolved.get("layer_rates") or None)
    return combinations, pool


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
    drawn, kind_pool = deal_kind({}, size, groups=3)
    print(report(shares(drawn, kind_pool)))
