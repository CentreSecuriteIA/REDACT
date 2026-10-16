"""The plan of a run: one combination per input and iteration, frozen in the
manifest, and read back.

A manifest row is ``{sample_id, iteration, combination, settings, category,
entry_type, status}``: ``combination`` names the techniques in the order they
apply, ``settings`` what the sampler was given, with the run's ``seed``.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from redact.dataset import Manifest, row_id

from .chain import _stable_seed, combine_techniques
from .sampling import sample_combination

logger = logging.getLogger(__name__)


class _Dealer:
    """Running family and layer counts of one ``(category, entry_type,
    iteration)`` group, so the planner deals families and layers evenly.

    ``weights`` sets a family's share within its layer (1 by default). The
    planner sets ``position`` to the units dealt so far, including the current,
    and calls ``observe`` with the techniques each unit was given. ``key``
    names the group; it seeds where in the group each layer's picks fall.
    """

    def __init__(self, weights: dict | None = None, key: tuple = ()) -> None:
        self.weights = dict(weights or {})
        self.key = key
        self.position = 0
        self.family_counts: Counter = Counter()
        self.layer_counts: Counter = Counter()
        self._offsets: dict[str, float] = {}

    def score(self, family: str) -> float:
        return self.family_counts[family] / self.weights.get(family, 1.0)

    def include(self, layer: str, rate: float) -> bool:
        """Whether this unit takes ``layer``: the picks so far are below
        ``rate * position`` shifted by the group's seeded offset in [0, 1), so
        a group gets ``rate * size`` picks rounded down or up, and a group of
        one takes the layer with probability ``rate``."""
        if layer not in self._offsets:
            seeded = random.Random(_stable_seed(*self.key, layer))
            self._offsets[layer] = seeded.random()
        return self.layer_counts[layer] < math.floor(
            rate * self.position + self._offsets[layer])

    def observe(self, techniques) -> None:
        for fn in techniques:
            if fn is None:
                continue
            families = getattr(fn, "families", None) or []
            if families:
                self.family_counts[families[0]] += 1
            layer = getattr(fn, "layer", None)
            if layer is not None:
                self.layer_counts[layer] += 1


def assign_combination(
    sample_id: str,
    pool: list[Callable],
    *,
    seed: int = 42,
    iteration: int = 0,
    used: tuple | list | set = (),
    max_resamples: int = 8,
    dealer=None,
    **sample_kwargs,
) -> list[str]:
    """The names of the combination one unit is assigned, in applied order.

    The draw is seeded by ``(seed, sample_id, iteration)``, so it does not
    depend on the order or the chunking of the inputs. It is drawn again, up
    to ``max_resamples`` times, while it equals a combination in ``used`` (the
    sample's earlier iterations); the last draw is kept if none differs. The
    first draw is dealt from ``dealer``; a resample draws uniformly, so it can
    escape a collision the deal would otherwise force on a small pool.
    ``sample_kwargs`` go to :func:`sample_combination`. An empty list is the
    identity.
    """
    used_set = {tuple(c) for c in used}
    base = _stable_seed(seed, sample_id, iteration)
    chosen: list[str] = []
    for attempt in range(max_resamples):
        rng = random.Random(base + attempt)
        fn = sample_combination(
            rng, pool, dealer=(dealer if attempt == 0 else None), **sample_kwargs)
        names = [getattr(t, "__name__", "") for t in getattr(fn, "techniques", [])]
        chosen = names
        if tuple(names) not in used_set:
            break
    return chosen


def combination_of(names: list[str], registry: dict[str, Callable]) -> Callable:
    """The combined technique a manifest row names, in the order it names them.

    An unknown name raises ``KeyError``, so a stale manifest fails loudly.
    """
    fns = [registry[n] for n in names]
    return combine_techniques(*fns, sort_by_hierarchy=False)


def plan_run(
    inputs: pd.DataFrame,
    pool: list,
    *,
    manifest_path,
    seed: int = 42,
    iterations: int = 1,
    settings_per_iteration: list[dict] | None = None,
    sample_kwargs: dict | None = None,
    family_weights: dict | None = None,
    layer_rates: dict | None = None,
    text_col: str = "prompt",
    id_col: str = "sample_id",
    verbose: bool = True,
) -> Path:
    """Write the plan of every input and iteration to ``manifest_path``.

    A unit's ``sample_id`` is the input's ``id_col`` when it has one, else the
    content id of ``text_col``. Each iteration's sampler settings are
    ``sample_kwargs`` with ``settings_per_iteration[iteration]`` merged over
    them, key by key.

    The units of one ``(category, entry_type, iteration)`` group are dealt in
    ``sample_id`` order from a shared :class:`_Dealer`, so each layer appears at
    its rate (``layer_rates`` over the spec's) and each family at its weight
    (``family_weights``, equal by default), evenly across the group and
    independent of the input order; a row's ``settings`` record the weights
    when given. A coverage table is written beside the manifest. Returns
    ``manifest_path``.
    """
    base_kwargs = dict(sample_kwargs or {})
    if layer_rates is not None:
        base_kwargs = {"sampling_probs": dict(layer_rates), **base_kwargs}
    manifest_path = Path(manifest_path)
    by_name = {fn.__name__: fn for fn in pool}
    planned = 0

    # Each (category, entry_type) subgroup's input rows, dealt in sample_id
    # order so the plan does not depend on the order the inputs arrived in; a
    # row keeps its own dedup history across its iterations.
    subgroups: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for idx, (_, row) in enumerate(inputs.iterrows()):
        sid = row_id(row, text_col, id_col)
        key = (str(row.get("category", "")), str(row.get("entry_type", "")))
        subgroups[key].append((sid, idx))
    for members in subgroups.values():
        members.sort()
    used_by_row: dict[int, list[tuple]] = defaultdict(list)
    recorded = {"family_weights": dict(family_weights)} if family_weights else {}

    # Yielded to the manifest one at a time: the plan is never held whole.
    def units():
        nonlocal planned
        for it in range(iterations):
            settings = (
                {**base_kwargs, **settings_per_iteration[it]}
                if settings_per_iteration is not None
                else base_kwargs
            )
            for key in sorted(subgroups):
                category, entry_type = key
                dealer = _Dealer(family_weights, key=(seed, category, entry_type, it))
                for sid, idx in subgroups[key]:
                    dealer.position += 1
                    names = assign_combination(
                        sid, pool, seed=seed, iteration=it,
                        used=used_by_row[idx], dealer=dealer, **settings,
                    )
                    used_by_row[idx].append(tuple(names))
                    dealer.observe(by_name.get(n) for n in names)
                    planned += 1
                    yield {
                        "sample_id": sid,
                        "iteration": it,
                        "combination": names,
                        "settings": {**settings, "seed": seed, **recorded},
                        "category": category,
                        "entry_type": entry_type,
                        "status": "planned",
                    }

    Manifest(manifest_path).write(units())
    coverage(manifest_path, pool)

    if verbose:
        logger.info("Planned %d units (%d samples x %d) -> %s",
                    planned, len(inputs), iterations, manifest_path)
    return manifest_path


def coverage_path(manifest_path) -> Path:
    """The coverage CSV beside the manifest at ``manifest_path``."""
    manifest_path = Path(manifest_path)
    suffix = ".manifest.jsonl"
    stem = (manifest_path.name[:-len(suffix)] if manifest_path.name.endswith(suffix)
            else manifest_path.stem)
    return manifest_path.with_name(f"{stem}.coverage.csv")


def _manifest_rows(manifest_path: Path):
    """The planned rows read straight off the file (bad lines skipped)."""
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def coverage(manifest_path, pool: list[Callable]) -> Path | None:
    """Write ``<out>.coverage.csv`` beside the manifest and log a summary.

    One row per layer, family and technique drawn, with its count per category,
    entry type and iteration, plus a zero row for every pool technique never
    drawn, so the file records the whole pool. Returns the coverage path, or
    ``None`` when the manifest does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    out = coverage_path(manifest_path)

    by_name = {fn.__name__: fn for fn in pool}
    counts: dict[str, Counter] = {
        "layer": Counter(), "family": Counter(), "technique": Counter()}
    drawn: set[str] = set()
    for row in _manifest_rows(manifest_path):
        category = str(row.get("category", ""))
        entry_type = str(row.get("entry_type", ""))
        iteration = row.get("iteration", "")
        for name in row.get("combination", []) or []:
            drawn.add(name)
            fn = by_name.get(name)
            counts["technique"][(name, category, entry_type, iteration)] += 1
            families = getattr(fn, "families", None) or []
            if families:
                counts["family"][(families[0], category, entry_type, iteration)] += 1
            layer = getattr(fn, "layer", None)
            if layer is not None:
                counts["layer"][(layer, category, entry_type, iteration)] += 1

    rows = []
    for level in ("layer", "family", "technique"):
        for key in sorted(counts[level], key=lambda k: tuple(map(str, k))):
            name, category, entry_type, iteration = key
            rows.append({"level": level, "name": name, "category": category,
                         "entry_type": entry_type, "iteration": iteration,
                         "count": counts[level][key]})
    for fn in pool:
        if fn.__name__ not in drawn:
            rows.append({"level": "technique", "name": fn.__name__, "category": "",
                         "entry_type": "", "iteration": "", "count": 0})

    pd.DataFrame(rows, columns=[
        "level", "name", "category", "entry_type", "iteration", "count"
    ]).to_csv(out, index=False)
    logger.info("Coverage: %d of %d pool techniques drawn -> %s",
                len(drawn), len(pool), out)
    return out


def default_escalation_schedule() -> list[dict]:
    """The built-in four rounds, each one round's sampler settings.

    Rounds one and two draw exactly one and two techniques; rounds three and
    four draw by budget, the fourth with the larger one. It is what an
    adversarial kind's ``"rounds": "escalation"`` names; each round is merged
    over the kind's budget settings.
    """
    return [
        {"exact_techniques": 1},
        {"exact_techniques": 2},
        {
            "max_complexity": 6,
            "max_obfuscations": 2,
            "sampling_probs": {
                "hacking": 0.7, "manipulation": 0.5,
                "requests": 0.8, "request_obfuscation": 0.5,
            },
            "allow_request_obfuscation": True,
        },
        {
            "max_complexity": 9,
            "max_obfuscations": 3,
            "sampling_probs": {
                "hacking": 0.9, "manipulation": 0.7,
                "requests": 0.9, "request_obfuscation": 0.7,
            },
            "allow_request_obfuscation": True,
        },
    ]
