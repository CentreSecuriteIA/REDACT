"""Draw one combination for a unit: by budget, or by exact count."""

import random
from collections.abc import Callable

from .chain import combine_techniques, in_layer_order
from .rules import compatible, request_obfuscation_candidates
from .technique import load_spec


def _pick_by_family(
    rng: random.Random, candidates: list[Callable], dealer=None,
) -> Callable:
    """Pick a family, then a technique within it.

    Without a ``dealer`` the family is uniform. With one, it is the eligible
    family with the lowest count-over-weight so far in the group (ties broken by
    ``rng``), so a group's families come out even. The technique within the
    family is uniform either way.
    """
    family_map: dict[str, list[Callable]] = {}
    for fn in candidates:
        fam = (getattr(fn, "families", None) or [""])[0]
        family_map.setdefault(fam, []).append(fn)
    families = list(family_map)
    if dealer is None:
        chosen_family = rng.choice(families)
    else:
        best = min(dealer.score(fam) for fam in families)
        chosen_family = rng.choice([f for f in families if dealer.score(f) == best])
    return rng.choice(family_map[chosen_family])


def sample_combination(
    rng: random.Random,
    pool: list[Callable],
    max_complexity: int = 6,
    max_obfuscations: int = 2,
    include_hacking: bool = True,
    include_manipulation: bool = True,
    include_obfuscation: bool = True,
    include_requests: bool = True,
    allow_request_obfuscation: bool = False,
    sampling_probs: dict | None = None,
    exact_techniques: int | None = None,
    dealer=None,
) -> Callable:
    """Sample a random valid technique combination from pool.

    Two modes:

    - **Budget-driven** (default): probabilistic per-layer picking governed by
        ``max_complexity``, ``max_obfuscations``, and ``sampling_probs``. The
        hacking, manipulation and requests layers are picked first, at their
        rates; the obfuscations then fill what the budget leaves, up to
        ``max_obfuscations``.
    - **Count-driven** (``exact_techniques`` set): delegates to
        :func:`sample_exact_combination`, which picks exactly that many compatible
        techniques and ignores the complexity / obfuscation budget entirely. The
        budget knobs are meaningless in this mode and are not consulted.

    Args:
        rng: Random instance for reproducibility.
        pool: All available tagged technique functions.
        max_complexity: Maximum total complexity score across all selected techniques.
        max_obfuscations: Maximum number of distinct obfuscation families to pick.
        include_hacking: Whether to consider hacking-layer techniques.
        include_manipulation: Whether to consider manipulation-layer techniques.
        include_obfuscation: Whether to consider obfuscation-layer techniques.
        include_requests: Whether to consider request-layer techniques.
        allow_request_obfuscation: If True, optionally add a light obfuscation after
            the request layer (encode_light or translation only).
        sampling_probs: Per-layer inclusion probabilities. Overrides the
            ``sampling_probs`` block in combination_spec.json. Recognised keys:
            ``hacking``, ``manipulation``, ``requests``, ``request_obfuscation``.
        exact_techniques: If set, switch to count-driven mode and pick exactly this
            many techniques (best-effort if the pool runs out). The budget knobs are
            ignored in this mode.

    Returns:
        A combined technique function, or an identity function if nothing was selected.
    """
    if exact_techniques is not None:
        return sample_exact_combination(
            rng, pool, exact_techniques,
            include_hacking=include_hacking,
            include_manipulation=include_manipulation,
            include_obfuscation=include_obfuscation,
            include_requests=include_requests,
            dealer=dealer,
        )

    spec = load_spec()
    probs = dict(spec.get("layer_rates", {}))
    if sampling_probs:
        probs.update(sampling_probs)
    p_hacking = probs.get("hacking", 0.5)
    p_manipulation = probs.get("manipulation", 0.33)
    p_requests = probs.get("requests", 0.7)
    p_request_obfuscation = probs.get("request_obfuscation", 0.5)
    selected: list[Callable] = []
    remaining_complexity = max_complexity
    current_pool = list(pool)

    def _update_pool() -> None:
        nonlocal current_pool
        current_pool = compatible(selected, current_pool, remaining_complexity)

    def _pick_from_layer(layer: str) -> Callable | None:
        candidates = [f for f in current_pool if getattr(f, "layer", None) == layer]
        if not candidates:
            return None
        return _pick_by_family(rng, candidates, dealer)

    def _commit(fn: Callable) -> None:
        selected.append(fn)
        nonlocal remaining_complexity
        remaining_complexity -= getattr(fn, "complexity", 0)
        _update_pool()

    def _include(layer: str, prob: float) -> bool:
        """Whether this unit takes a pick in ``layer``.

        A dealer includes the layer while its share of the group is below
        ``prob``; without one it is a per-unit Bernoulli draw.
        """
        if dealer is None:
            return rng.random() < prob
        return dealer.include(layer, prob)

    # Phase 1: Hacking
    if include_hacking and _include("hacking", p_hacking):
        pick = _pick_from_layer("hacking")
        if pick is not None:
            _commit(pick)

    # Phase 2: Manipulation
    if include_manipulation and _include("manipulation", p_manipulation):
        pick = _pick_from_layer("manipulation")
        if pick is not None:
            _commit(pick)

    # Phase 3: Requests, before the obfuscations, so the rated layers take
    # their picks from the budget first; the apply order is the layers'.
    if include_requests and _include("requests", p_requests):
        pick = _pick_from_layer("requests")
        if pick is not None:
            _commit(pick)

    # Phase 4: Obfuscation (up to max_obfuscations families), what the budget leaves
    if include_obfuscation:
        obfusc_count = 0
        while obfusc_count < max_obfuscations:
            candidates = [f for f in current_pool if getattr(f, "layer", None) == "obfuscation"]
            if not candidates:
                break
            pick = _pick_by_family(rng, candidates, dealer)
            _commit(pick)
            obfusc_count += 1

    # Phase 5: Request-layer obfuscation (optional, if enabled). Kept out of
    # `selected` because it is the one pick whose position is *not* its layer's:
    # it obfuscates the finished request, so it runs after the request template.
    request_obfuscation: Callable | None = None
    if allow_request_obfuscation and rng.random() < p_request_obfuscation:
        req_obfusc_candidates = request_obfuscation_candidates(current_pool)
        if req_obfusc_candidates:
            request_obfuscation = _pick_by_family(rng, req_obfusc_candidates, dealer)

    if not selected and request_obfuscation is None:
        # combine_techniques() with no techniques returns a no-op combined
        # callable named "identity" with .techniques == [].
        return combine_techniques()

    # Order the regular picks here and chain in that exact order: passing the
    # request-layer obfuscation to combine_techniques() for sorting would move
    # it back into the obfuscation slot, i.e. before the request template.
    ordered = in_layer_order(selected)
    if request_obfuscation is not None:
        ordered.append(request_obfuscation)
    return combine_techniques(*ordered, sort_by_hierarchy=False)


def sample_exact_combination(
    rng: random.Random,
    pool: list[Callable],
    n: int,
    *,
    include_hacking: bool = True,
    include_manipulation: bool = True,
    include_obfuscation: bool = True,
    include_requests: bool = True,
    dealer=None,
) -> Callable:
    """Pick exactly ``n`` compatible techniques (count-driven, not budget-driven).

    Unlike :func:`sample_combination`, the **count is the only knob** — there is no
    complexity or obfuscation budget. Each pick is still validated through
    :func:`redact.jailbreak.rules.compatible` (with ``remaining_complexity=None``, so complexity
    never filters anything) to enforce layer caps (1 hacking / 1 manipulation /
    1 request) and cross-incompatibilities. The obfuscation layer has no per-layer
    pick cap, so e.g. ``n=2`` may yield two obfuscation-family techniques — intended
    diversity.

    Best-effort: if the compatible pool is exhausted before ``n`` picks (only possible
    via caps / incompatibilities, never via complexity), fewer techniques are returned.
    With the full pool, ``n`` of 1 or 2 is always reachable.

    Args:
        rng: Random instance for reproducibility.
        pool: All available tagged technique functions.
        n: Exact number of techniques to select.
        include_hacking: Layer toggle applied to the initial candidate pool.
        include_manipulation: Layer toggle, as ``include_hacking``.
        include_obfuscation: Layer toggle, as ``include_hacking``.
        include_requests: Layer toggle, as ``include_hacking``.

    Returns:
        A combined technique function (``"identity"`` if ``n <= 0`` or nothing picked).
    """
    layer_included = {
        "hacking": include_hacking,
        "manipulation": include_manipulation,
        "obfuscation": include_obfuscation,
        "requests": include_requests,
    }
    current_pool = [
        f for f in pool if layer_included.get(getattr(f, "layer", None), True)
    ]

    selected: list[Callable] = []
    for _ in range(max(0, n)):
        if not current_pool:
            break
        pick = _pick_by_family(rng, current_pool, dealer)
        selected.append(pick)
        # remaining_complexity=None → no complexity filtering; only caps + incompat.
        current_pool = compatible(selected, current_pool, None)

    if not selected:
        return combine_techniques()
    return combine_techniques(*selected)
