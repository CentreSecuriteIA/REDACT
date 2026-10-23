"""Draw one combination for a unit: by budget, or by exact count."""

import random
from collections.abc import Callable

from .chain import combine_techniques, in_layer_order
from .rules import compatible, request_obfuscation_candidates
from .technique import load_spec

#: The complexity a budget draw may spend when it is given none.
DEFAULT_MAX_COMPLEXITY = 6

#: The prefix of the request obfuscation's family names in a dealer's
#: counts: its picks are counted apart from the obfuscation layer's, so they
#: do not starve the layer's ``encode`` and ``sensitive_words``.
REQUEST_SCOPE = "request_obfuscation:"


def _pick_by_family(
    rng: random.Random, candidates: list[Callable], dealer=None, scope: str = "",
) -> Callable:
    """Pick a family, then a technique within it.

    Without a ``dealer`` both picks are uniform. With one, the family is the
    eligible family with the lowest count-over-weight so far in the group, and
    the technique the eligible member of that family with the lowest count so
    far (:meth:`~redact.dataset.dealing.Dealer.pick_member`), ties broken by
    ``rng``, so a group's families and each family's members come out even.
    The dealer counts the families under ``scope`` + their names
    (:data:`REQUEST_SCOPE` for the request obfuscation); the members' counts
    are shared.
    """
    family_map: dict[str, list[Callable]] = {}
    for fn in candidates:
        fam = (getattr(fn, "families", None) or [""])[0]
        family_map.setdefault(fam, []).append(fn)
    families = list(family_map)
    if dealer is None:
        return rng.choice(family_map[rng.choice(families)])
    fam = dealer.pick(rng, [scope + f for f in families])[len(scope):]
    members = {fn.__name__: fn for fn in family_map[fam]}
    return members[dealer.pick_member(rng, list(members))]


def sample_combination(
    rng: random.Random,
    pool: list[Callable],
    max_complexity: int = DEFAULT_MAX_COMPLEXITY,
    max_obfuscations: int = 2,
    include_rewrite: bool = False,
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
        rewrite layer is picked first, when included and at its rate; then
        the hacking, manipulation and requests layers, at their rates; then
        the request obfuscation, when allowed, the unit holds a request and
        at its rate; the obfuscations then fill what the budget leaves, up
        to ``max_obfuscations`` with the request obfuscation counted.
    - **Count-driven** (``exact_techniques`` set): delegates to
        :func:`sample_exact_combination`, which picks exactly that many compatible
        techniques and ignores the complexity / obfuscation budget entirely. The
        budget knobs are meaningless in this mode and are not consulted, nor
        are the rates and ``include_rewrite``: a rewrite technique in the pool
        may be drawn.

    Args:
        rng: Random instance for reproducibility.
        pool: All available tagged technique functions.
        max_complexity: Maximum total complexity score across all selected techniques.
        max_obfuscations: Maximum number of distinct obfuscation families to
            pick, the request obfuscation included.
        include_rewrite: Whether to consider rewrite-layer techniques; off
            unless a round says so (the layer is off by default). Phase 0
            is decided (a dealer's ``include``, else an ``rng`` draw) only
            when it is on, the ``rewrite`` rate is above 0 and the pool holds
            a rewrite technique, so a draw without it consumes what it did
            before the layer existed.
        include_hacking: Whether to consider hacking-layer techniques.
        include_manipulation: Whether to consider manipulation-layer techniques.
        include_obfuscation: Whether to consider obfuscation-layer techniques.
        include_requests: Whether to consider request-layer techniques.
        allow_request_obfuscation: If True, optionally add a light obfuscation after
            the request technique (encode_light or translation only). It is
            decided (a dealer's ``include``, else an ``rng`` draw) only once a
            request technique is picked, so a unit without one takes none,
            and drawn before the
            other obfuscations, taking one ``max_obfuscations`` slot and its
            complexity from what they may use; with ``max_obfuscations`` 0 it
            has no slot.
        sampling_probs: Per-layer inclusion probabilities. Overrides the
            ``layer_rates`` block in combination_spec.json. Recognised keys:
            ``rewrite``, ``hacking``, ``manipulation``, ``requests``,
            ``request_obfuscation``.
        exact_techniques: If set, switch to count-driven mode and pick exactly this
            many techniques (best-effort if the pool runs out). The budget knobs are
            ignored in this mode.
        dealer: The plan's :class:`~redact.dataset.dealing.Dealer` for the unit's
            group, which deals the layers, families and techniques; ``None``
            draws each from ``rng``.

    Returns:
        A combined technique function, or an identity function if nothing was
        selected. In budget mode its ``request_obfuscation`` is the request
        obfuscation it holds (applied last), or ``None``.
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
    p_rewrite = probs.get("rewrite", 0)
    p_hacking = probs.get("hacking", 0.5)
    p_manipulation = probs.get("manipulation", 0.33)
    p_requests = probs.get("requests", 0.7)
    p_request_obfuscation = probs.get("request_obfuscation", 0.5)
    selected: list[Callable] = []
    remaining_complexity = max_complexity
    # Each pick, the first and the request obfuscation included, is drawn
    # from what fits the budget the earlier picks leave.
    current_pool = compatible(selected, pool, remaining_complexity)

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

    # Phase 0: Rewrite, first, as it applies first. Decided only when the
    # round includes it, its rate is above 0 and a rewrite technique fits,
    # so a draw without it makes no rng draw and no dealer call: the draws
    # are those made before the layer existed.
    rewrites = [f for f in current_pool if getattr(f, "layer", None) == "rewrite"]
    if (include_rewrite and p_rewrite > 0 and rewrites
            and _include("rewrite", p_rewrite)):
        _commit(_pick_by_family(rng, rewrites, dealer))

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
    request: Callable | None = None
    if include_requests and _include("requests", p_requests):
        request = _pick_from_layer("requests")
        if request is not None:
            _commit(request)

    # Phase 5, decided and drawn before phase 4: a request-layer obfuscation
    # (optional, if enabled), only for a unit that holds a request, since it
    # obfuscates the finished request. It holds one of the max_obfuscations
    # slots and its complexity, so the obfuscations fill what it leaves and
    # the combination never holds more than max_obfuscations of them. Its
    # families are dealt under REQUEST_SCOPE, apart from phase 4's.
    request_obfuscation: Callable | None = None
    obfuscation_slots = max_obfuscations
    if (request is not None and allow_request_obfuscation and max_obfuscations > 0
            and _include("request_obfuscation", p_request_obfuscation)):
        req_obfusc_candidates = request_obfuscation_candidates(current_pool)
        if req_obfusc_candidates:
            request_obfuscation = _pick_by_family(
                rng, req_obfusc_candidates, dealer, scope=REQUEST_SCOPE)
            _commit(request_obfuscation)
            obfuscation_slots -= 1

    # Phase 4: Obfuscation (up to the slots left), what the budget leaves
    if include_obfuscation:
        obfusc_count = 0
        while obfusc_count < obfuscation_slots:
            candidates = [f for f in current_pool if getattr(f, "layer", None) == "obfuscation"]
            if not candidates:
                break
            pick = _pick_by_family(rng, candidates, dealer)
            _commit(pick)
            obfusc_count += 1

    if not selected:
        # combine_techniques() with no techniques returns a no-op combined
        # callable named "identity" with .techniques == [].
        return combine_techniques()

    # Order the regular picks here and chain in that exact order. The request
    # obfuscation is the one pick whose position is *not* its layer's: it
    # obfuscates the finished request, so it runs after the request template;
    # sorting it with the others would move it back into the obfuscation slot.
    ordered = in_layer_order(fn for fn in selected if fn is not request_obfuscation)
    if request_obfuscation is not None:
        ordered.append(request_obfuscation)
    combined = combine_techniques(*ordered, sort_by_hierarchy=False)
    combined.request_obfuscation = request_obfuscation
    return combined


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
    never filters anything) to enforce layer caps (1 rewrite / 1 hacking /
    1 manipulation / 1 request) and cross-incompatibilities. The obfuscation
    layer has no per-layer pick cap, so e.g. ``n=2`` may yield two
    obfuscation-family techniques — intended diversity. No rate is read: a
    rewrite technique in the pool is dealt as any family's, so the pool (a
    kind's ``layers.deny``), not a rate, keeps the rewrite layer out of a
    count round.

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
