"""The plan of a run: one combination per input and iteration, frozen in the
manifest, and read back.

A manifest row is ``{sample_id, iteration, combination, settings, category,
entry_type, status}``: ``combination`` names the techniques in the order they
apply, ``settings`` what the sampler was given, with the run's ``seed``.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from redact.dataset import Manifest, row_id

from .chain import combine_techniques
from .sampling import sample_combination

logger = logging.getLogger(__name__)


def _stable_seed(*parts) -> int:
    """A seed from ``parts`` that is the same in every process.

    SHA-256, not the builtin ``hash``, which is salted per process.
    """
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8"))
    return int(digest.hexdigest()[:16], 16)


def build_function_registry(pool: list[Callable]) -> dict[str, Callable]:
    """Each technique of ``pool`` by its ``__name__``."""
    return {getattr(fn, "__name__", ""): fn for fn in pool}


def assign_combination(
    sample_id: str,
    pool: list[Callable],
    *,
    seed: int = 42,
    iteration: int = 0,
    used: tuple | list | set = (),
    max_resamples: int = 8,
    **sample_kwargs,
) -> list[str]:
    """The names of the combination one unit is assigned, in applied order.

    The draw is seeded by ``(seed, sample_id, iteration)``, so it does not
    depend on the order or the chunking of the inputs. It is drawn again, up
    to ``max_resamples`` times, while it equals a combination in ``used`` (the
    sample's earlier iterations); the last draw is kept if none differs.
    ``sample_kwargs`` go to :func:`sample_combination`. An empty list is the
    identity.
    """
    used_set = {tuple(c) for c in used}
    base = _stable_seed(seed, sample_id, iteration)
    chosen: list[str] = []
    for attempt in range(max_resamples):
        rng = random.Random(base + attempt)
        fn = sample_combination(rng, pool, **sample_kwargs)
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
    text_col: str = "prompt",
    id_col: str = "sample_id",
    verbose: bool = True,
) -> Path:
    """Write the plan of every input and iteration to ``manifest_path``.

    A unit's ``sample_id`` is the input's ``id_col`` when it has one, else the
    content id of ``text_col``. Each iteration's sampler settings are
    ``sample_kwargs`` with ``settings_per_iteration[iteration]`` merged over
    them, key by key. Returns ``manifest_path``.
    """
    sample_kwargs = sample_kwargs or {}
    manifest_path = Path(manifest_path)
    planned = 0

    # Yielded to the manifest one at a time: the plan is never held whole.
    def units():
        nonlocal planned
        for _, row in inputs.iterrows():
            sid = row_id(row, text_col, id_col)

            used: list[tuple] = []
            for it in range(iterations):
                settings = (
                    {**sample_kwargs, **settings_per_iteration[it]}
                    if settings_per_iteration is not None
                    else sample_kwargs
                )
                names = assign_combination(
                    sid, pool, seed=seed, iteration=it, used=used, **settings
                )
                used.append(tuple(names))
                planned += 1
                yield {
                    "sample_id": sid,
                    "iteration": it,
                    "combination": names,
                    "settings": {**settings, "seed": seed},
                    "category": str(row.get("category", "")),
                    "entry_type": str(row.get("entry_type", "")),
                    "status": "planned",
                }

    Manifest(manifest_path).write(units())

    if verbose:
        logger.info("Planned %d units (%d samples x %d) -> %s",
                    planned, len(inputs), iterations, manifest_path)
    return manifest_path


def default_escalation_schedule() -> list[dict]:
    """The built-in four rounds, each one round's sampler settings.

    Rounds one and two draw exactly one and two techniques; rounds three and
    four draw by budget, the fourth with the larger one. Pass it, or a list
    like it, as ``settings_per_iteration``; each round is merged over the
    run's base settings.
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
