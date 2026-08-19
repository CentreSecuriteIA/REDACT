"""Sampling technique combinations — compatibility filtering + the two
sampling strategies (budget-driven and count-driven) built on top of it.

Split out of jailbreak/utils.py (see that module's docstring for the full
7-concern breakdown this split resolved). Everything here is used
exclusively by sample_combination() / sample_exact_combination() — the
compatibility filter and the family-first picker have no other callers.
"""

import random
from collections.abc import Callable

from .chain import combine_techniques
from .spec import load_spec


def get_compatible_remaining(
    selected: list[Callable],
    pool: list[Callable],
    remaining_complexity: int | None = None,
) -> list[Callable]:
    """Return functions from pool that can still be added given the current selection.

    Enforces:
    - At most max_picks per family (currently 1 for all families)
    - Family cross-incompatibilities declared in combination_spec.json
    - Layer caps (1 hacking, 1 manipulation, 1 request)
    - Manipulation blocking: when fsh/dap selected, blocks ascii_art, translation,
      and encode-heavy functions (except sensitive_words_encode_* which only encode
      specific words, leaving the Q&A structure readable)
    - Complexity budget: removes functions that would exceed remaining_complexity

    Args:
        selected: Already-selected technique functions.
        pool: Candidate pool to filter.
        remaining_complexity: If provided, exclude functions with complexity > this value.

    Returns:
        Filtered list of compatible functions.
    """
    spec = load_spec()
    families_spec = spec["families"]
    layer_max_picks: dict[str, int] = spec.get("layer_max_picks", {})
    manip_blocked = spec.get("manipulation_blocked_obfuscation", {})
    manip_blocked_families: set[str] = set(manip_blocked.get("families", []))
    manip_blocked_encode_weight: str | None = manip_blocked.get("encode_weight", None)

    # Collect selected families and layer counts
    selected_families: set[str] = set()
    layer_counts: dict[str, int] = {}
    for fn in selected:
        for fam in getattr(fn, "families", []):
            selected_families.add(fam)
        layer = getattr(fn, "layer", None)
        if layer:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    # Build blocked-families set from cross-incompatibilities
    blocked_families: set[str] = set(selected_families)  # can't re-pick same family
    for sel_fam in selected_families:
        if sel_fam in families_spec:
            for incomp in families_spec[sel_fam].get("cross_incompatible_with", []):
                blocked_families.add(incomp)

    has_manipulation = bool(selected_families & {"fsh", "dap"})

    result = []
    for fn in pool:
        fn_families: set[str] = set(getattr(fn, "families", []))
        fn_layer: str | None = getattr(fn, "layer", None)
        fn_complexity: int = getattr(fn, "complexity", 0)
        fn_encode_weight: str | None = getattr(fn, "encode_weight", None)

        # Block if any family overlaps with blocked set
        if fn_families & blocked_families:
            continue

        # Block if layer cap already reached
        if (
            fn_layer and fn_layer in layer_max_picks
            and layer_counts.get(fn_layer, 0) >= layer_max_picks[fn_layer]
        ):
            continue

        # Manipulation-specific obfuscation restrictions
        if has_manipulation:
            # Block explicitly listed families (ascii_art, translation)
            if fn_families & manip_blocked_families:
                continue
            # Block encode-heavy functions — but NOT if they also have sensitive_words
            # (selective encoding of individual words is fine; only full-text heavy encoding
            # makes the Q&A structure unreadable)
            has_sw = "sensitive_words" in fn_families
            is_heavy_encode = (
                "encode" in fn_families
                and not has_sw
                and fn_encode_weight == manip_blocked_encode_weight
            )
            if is_heavy_encode:
                continue

        # Complexity budget
        if remaining_complexity is not None and fn_complexity > remaining_complexity:
            continue

        result.append(fn)

    return result


def _pick_by_family(rng: random.Random, candidates: list[Callable]) -> Callable:
    """Pick uniformly by family first, then uniformly within the chosen family.

    Prevents over-sized families (e.g. 20 translation languages) from dominating
    over smaller families that have fewer individual functions.
    """
    family_map: dict[str, list[Callable]] = {}
    for fn in candidates:
        fam = (getattr(fn, "families", None) or [""])[0]
        family_map.setdefault(fam, []).append(fn)
    chosen_family = rng.choice(list(family_map.keys()))
    return rng.choice(family_map[chosen_family])


def _is_request_obfuscation_compatible(fn: Callable, spec: dict) -> bool:
    """Return True if fn can be used as request-layer obfuscation.

    Allowed: encode_light (encode family + encode_weight==light) and translation family.
    """
    allowed_families = set(spec.get("request_obfuscation_allowed_families", []))
    fn_families = set(getattr(fn, "families", []))
    fn_encode_weight = getattr(fn, "encode_weight", None)

    if (
        "encode_light" in allowed_families
        and "encode" in fn_families and fn_encode_weight == "light"
    ):
        return True
    return "translation" in allowed_families and "translation" in fn_families


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
) -> Callable:
    """Sample a random valid technique combination from pool.

    Two modes:

    - **Budget-driven** (default): probabilistic per-layer picking governed by
        ``max_complexity``, ``max_obfuscations``, and ``sampling_probs``.
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
        )

    spec = load_spec()
    probs = dict(spec.get("sampling_probs", {}))
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
        current_pool = get_compatible_remaining(selected, current_pool, remaining_complexity)

    def _pick_from_layer(layer: str) -> Callable | None:
        candidates = [f for f in current_pool if getattr(f, "layer", None) == layer]
        if not candidates:
            return None
        return _pick_by_family(rng, candidates)

    def _commit(fn: Callable) -> None:
        selected.append(fn)
        nonlocal remaining_complexity
        remaining_complexity -= getattr(fn, "complexity", 0)
        _update_pool()

    # Phase 1: Hacking
    if include_hacking and rng.random() < p_hacking:
        pick = _pick_from_layer("hacking")
        if pick is not None:
            _commit(pick)

    # Phase 2: Manipulation
    if include_manipulation and rng.random() < p_manipulation:
        pick = _pick_from_layer("manipulation")
        if pick is not None:
            _commit(pick)

    # Phase 3: Obfuscation (up to max_obfuscations families)
    if include_obfuscation:
        obfusc_count = 0
        while obfusc_count < max_obfuscations:
            candidates = [f for f in current_pool if getattr(f, "layer", None) == "obfuscation"]
            if not candidates:
                break
            pick = _pick_by_family(rng, candidates)
            _commit(pick)
            obfusc_count += 1

    # Phase 4: Requests
    if include_requests and rng.random() < p_requests:
        pick = _pick_from_layer("requests")
        if pick is not None:
            _commit(pick)

    # Phase 5: Request-layer obfuscation (optional, if enabled)
    if allow_request_obfuscation and rng.random() < p_request_obfuscation:
        req_obfusc_candidates = [
            f for f in current_pool
            if _is_request_obfuscation_compatible(f, spec)
        ]
        if req_obfusc_candidates:
            pick = _pick_by_family(rng, req_obfusc_candidates)
            selected.append(pick)  # don't update pool; this is the final step

    if not selected:
        # combine_techniques() with no techniques returns a no-op combined
        # callable named "identity" with .techniques == [].
        return combine_techniques()

    return combine_techniques(*selected)


def sample_exact_combination(
    rng: random.Random,
    pool: list[Callable],
    n: int,
    *,
    include_hacking: bool = True,
    include_manipulation: bool = True,
    include_obfuscation: bool = True,
    include_requests: bool = True,
) -> Callable:
    """Pick exactly ``n`` compatible techniques (count-driven, not budget-driven).

    Unlike :func:`sample_combination`, the **count is the only knob** — there is no
    complexity or obfuscation budget. Each pick is still validated through
    :func:`get_compatible_remaining` (with ``remaining_complexity=None``, so complexity
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
        include_hacking / include_manipulation / include_obfuscation /
            include_requests: Layer toggles applied to the initial candidate pool.

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
        pick = _pick_by_family(rng, current_pool)
        selected.append(pick)
        # remaining_complexity=None → no complexity filtering; only caps + incompat.
        current_pool = get_compatible_remaining(selected, current_pool, None)

    if not selected:
        return combine_techniques()
    return combine_techniques(*selected)


def default_escalation_schedule() -> list[dict]:
    """Return the built-in 4-round increasing-complexity schedule (overridable).

    Each dict is one round's ``sample_combination`` kwargs, passed per-iteration to
    :func:`redact.jailbreak.manifest.plan_run` via ``settings_per_iteration``. Every
    input sample is augmented once per round (4 times total), escalating in intensity:

    1. exactly 1 technique (count-driven)
    2. exactly 2 techniques (count-driven, a real combination)
    3. higher complexity (budget-driven, probabilistic stacking)
    4. even higher complexity (budget-driven, largest budget + most stacking)

    Exact rounds carry only ``exact_techniques`` (the count is the whole story); budget
    rounds carry only the budget knobs. ``allow_request_obfuscation`` is required for
    the ``request_obfuscation`` probability to take effect.

    Override by passing your own list to ``generate_jailbreaks(settings_per_iteration=...)``.
    """
    return [
        # Rounds 1-2: COUNT-driven. exact_techniques is the only knob.
        {"exact_techniques": 1},
        {"exact_techniques": 2},
        # Rounds 3-4: BUDGET-driven. Probabilistic sampler runs.
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
