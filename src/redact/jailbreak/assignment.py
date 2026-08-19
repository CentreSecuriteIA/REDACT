"""Deterministic combination assignment (manifest planning) + reconstruction.

Split out of jailbreak/utils.py (see that module's docstring for the full
7-concern breakdown this split resolved). This one owns the manifest-facing
half: assigning a technique combination to a sample reproducibly
(assign_combination, on top of sample_combination), and reconstructing a
combined callable from the technique names a manifest recorded
(build_combination / build_function_registry) — used by jailbreak/manifest.py.
"""

import hashlib
import random
from collections.abc import Callable

from .chain import combine_techniques
from .sampling import sample_combination


def _stable_seed(*parts) -> int:
    """Deterministic, process-independent seed from arbitrary parts.

    Uses SHA-256 (not builtin ``hash``, which is salted per process) so the
    same ``(seed, sample_id, iteration)`` always yields the same RNG stream —
    a requirement for reproducible, chunk-independent combination assignment.
    """
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8"))
    return int(digest.hexdigest()[:16], 16)


def build_function_registry(pool: list[Callable]) -> dict[str, Callable]:
    """Map technique ``__name__`` -> function for reconstruction from a manifest."""
    return {getattr(fn, "__name__", ""): fn for fn in pool}


def assign_combination(
    sample_id: str,
    pool: list[Callable],
    *,
    seed: int = 42,
    iteration: int = 0,
    used: "tuple | list | set" = (),
    max_resamples: int = 8,
    **sample_kwargs,
) -> list[str]:
    """Deterministically assign a technique combination to one sample.

    Seeds an RNG from ``hash(seed, sample_id, iteration)`` so assignment is
    reproducible and independent of input order / chunking. Resamples (bounded
    by ``max_resamples``) until the combination's technique-name tuple is not
    already in ``used`` — so repeated iterations of the same sample don't draw
    duplicates. Falls back to the last sampled combination if a fresh one can't
    be found within the budget.

    Args:
        sample_id: Stable per-sample key (the prompt-content MD5).
        pool: Tagged technique functions to sample from.
        seed: Global run seed.
        iteration: Round index (0 for single-round runs).
        used: Already-assigned combinations for this sample, each a sequence of
            technique names.
        max_resamples: Max resample attempts to avoid a duplicate.
        **sample_kwargs: Forwarded to :func:`sample_combination`
            (``max_complexity``, ``include_*``, ``sampling_probs``, ...).

    Returns:
        Ordered list of technique names (empty list == identity / no-op).
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


def build_combination(names: list[str], registry: dict[str, Callable]) -> Callable:
    """Reconstruct a combined technique callable from recorded technique names.

    ``sort_by_hierarchy=False`` preserves the exact order recorded at planning
    time (which was already hierarchy-sorted by :func:`sample_combination`).
    Unknown names raise ``KeyError`` so a stale manifest fails loudly.
    """
    fns = [registry[n] for n in names]
    return combine_techniques(*fns, sort_by_hierarchy=False)
