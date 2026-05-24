"""Shared utilities for jailbreak technique composition.

Provides:
- combine_techniques: chain techniques sequentially with hierarchy-aware ordering
- tag_all_functions: attach spec metadata to technique function objects
- get_compatible_remaining: filter a pool to still-compatible techniques
- sample_combination: random sampler with complexity and count constraints
- apply_combination: apply a (possibly combined) technique, auto-generating benign data
- is_noop: detect when a transform produced no change
- load_spec: load the combination_spec.json once

Ported from reference utils.py combine_techniques (lines 24-49).
Retry wrappers (with_retries, with_feedback_retries) live in LLMs/wrappers.py.
"""

import hashlib
import inspect
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .protocol import LLMRequest, TechniqueGen, run_sync


_SPEC_PATH = Path(__file__).parent.parent / "configs" / "jailbreak" / "combination_spec.json"


@lru_cache(maxsize=1)
def load_spec() -> dict:
    """Load and cache the combination_spec.json.

    Returns the full spec dict with families, techniques, layer_order, etc.
    """
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


def tag_all_functions(funcs: list[Callable]) -> None:
    """Attach compatibility metadata from combination_spec.json to each function.

    Sets on each function object:
        fn.families          list[str]   — family membership
        fn.complexity        int         — complexity score (0-3)
        fn.requires_llm      bool        — whether an LLM backend is needed
        fn.layer             str         — "hacking" | "manipulation" | "obfuscation" | "requests"
        fn.within_layer_order int        — sort key within the obfuscation layer (0 = N/A)
        fn.encode_weight     str | None  — "light" | "heavy" | None (encode family only)

    Functions not found in the spec are skipped silently (no attributes set).
    Call once at jailbreak package import time.
    """
    spec = load_spec()
    families_spec = spec["families"]
    techniques = spec["techniques"]

    # Build layer → within_layer_order lookup from families
    family_layer: dict[str, str] = {name: d["layer"] for name, d in families_spec.items()}
    family_wlo: dict[str, int] = {
        name: d.get("within_layer_order", 0) for name, d in families_spec.items()
    }

    for fn in funcs:
        name = getattr(fn, "__name__", None)
        if name not in techniques:
            continue
        t = techniques[name]
        fn.families = t.get("families", [])
        fn.complexity = t.get("complexity", 1)
        fn.requires_llm = t.get("requires_llm", False)
        # Needs pre-generated benign Q&A data (FSH/DAP). Distinct from
        # requires_llm: the data is generated once up front, not per call — so
        # these aren't per-sample LLM techniques, but they still can't run in a
        # strictly no-LLM (pure_only) run.
        fn.requires_benign = t.get("requires_benign", False)
        fn.encode_weight = t.get("encode_weight", None)

        # Derive layer and within_layer_order from the first family
        primary_family = fn.families[0] if fn.families else None
        fn.layer = family_layer.get(primary_family, "obfuscation") if primary_family else "obfuscation"
        fn.within_layer_order = family_wlo.get(primary_family, 0) if primary_family else 0


# ---------------------------------------------------------------------------
# combine_techniques
# ---------------------------------------------------------------------------


def _normalize_output(output) -> tuple[str, str]:
    """Coerce a technique return into ``(text, info)``.

    Cognitive/persona techniques may return a 3-tuple ``(text, info, scenario)``
    — the scenario is discarded in combined chains (callers needing it must run
    the technique standalone).
    """
    if isinstance(output, tuple) and len(output) == 3:
        text, info, _ = output
        return text, info
    return output


def _select_kwargs(technique: Callable, kwargs: dict) -> dict:
    """Filter ``kwargs`` to what ``technique`` accepts.

    Pure transforms have ``(prompt: str)`` with no ``**kwargs`` — passing
    unrecognised keys would raise TypeError. Techniques declaring ``**kwargs``
    receive everything.
    """
    sig_params = inspect.signature(technique).parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
    )
    if has_var_kw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig_params}


def _run_chain(techniques: list[Callable], text: str, **kwargs) -> TechniqueGen:
    """Generator that chains techniques, threading text through.

    Pure transforms run inline (zero rounds); LLM-dependent technique
    generators are delegated to via ``yield from`` so their :class:`LLMRequest`
    yields propagate up to the batched engine (or to ``run_sync`` for the
    single-sample path). Early-exits with a ``DISCARDED`` info string the moment
    any step rejects, returning the pre-failure text. Returns ``(text, info)``.
    """
    # gen-model techniques tag their requests with `gen_model`; accept the
    # legacy `model` kwarg as the source when gen_model isn't given explicitly.
    if not kwargs.get("gen_model") and kwargs.get("model"):
        kwargs = {**kwargs, "gen_model": kwargs["model"]}

    result = text
    info_parts: list[str] = []
    for technique in techniques:
        tech_kwargs = _select_kwargs(technique, kwargs)
        if inspect.isgeneratorfunction(technique):
            output = yield from technique(result, **tech_kwargs)
        else:
            output = technique(result, **tech_kwargs)
        new_text, info = _normalize_output(output)

        # Early exit on rejection — don't apply remaining techniques
        if info.startswith("DISCARDED"):
            return result, (
                f"DISCARDED; technique={technique.__name__}; "
                f"{info[len('DISCARDED; '):]}"
            )

        # Track no-ops
        if new_text == result:
            info_parts.append(f"noop={technique.__name__}")
        elif info:
            info_parts.append(info)
        result = new_text
    return result, ";".join(info_parts)


def combine_techniques(*techniques: Callable, sort_by_hierarchy: bool = True) -> Callable:
    """Chain multiple technique functions into one callable.

    Each technique receives the output of the previous one; info strings are
    joined with ';'. Mixes pure transforms and LLM-dependent technique
    *generators* transparently (see ``protocol.py``).

    If ``sort_by_hierarchy=True`` (default), techniques are reordered by
    (layer_order_index, within_layer_order) before chaining — so callers can
    pass techniques in any order and the correct semantic sequence is applied.

    The returned ``combined`` callable runs **synchronously**: calling
    ``combined(text, backend=..., model=..., rate_limiter=..., benign_data=...)``
    drives the chain to completion (issuing real LLM calls for generator steps
    via :func:`protocol.run_sync`) and returns ``(text, info)``. This preserves
    the single-sample / test contract. The batched engine does **not** call
    ``combined`` — it builds the chain generator directly via
    :func:`make_combination_gen` and interleaves many samples.

    ``combined.techniques`` exposes the ordered inner technique list (used by
    the engine and the manifest); ``combined.__name__`` is the '+'-joined names
    (or ``"identity"`` when empty).

    Returns:
        Combined callable: ``(str, **kwargs) -> (str, str)`` with
        ``.techniques`` and ``.__name__`` attributes.
    """
    spec = load_spec()
    layer_order = spec["layer_order"]

    def _sort_key(fn: Callable) -> tuple[int, int]:
        layer = getattr(fn, "layer", None)
        layer_idx = layer_order.index(layer) if layer in layer_order else len(layer_order)
        wlo = getattr(fn, "within_layer_order", 0)
        return (layer_idx, wlo)

    ordered = sorted(techniques, key=_sort_key) if sort_by_hierarchy else list(techniques)

    def combined(text: str, **kwargs) -> tuple[str, str]:
        gen = _run_chain(ordered, text, **kwargs)
        backend = kwargs.get("backend")
        rate_limiter = kwargs.get("rate_limiter")

        def call(request: LLMRequest) -> str:
            # Lazy imports keep utils import-time light and avoid any
            # llms<->jailbreak import ordering surprises.
            from redact.llms.calls import generate_sample
            b = backend
            if b is None:
                from redact.llms.api import get_backend
                b = get_backend(request.model)
            return generate_sample(b, request.model, request.messages, rate_limiter)

        return run_sync(gen, call)

    combined.__name__ = "+".join(t.__name__ for t in ordered) or "identity"
    combined.techniques = list(ordered)  # expose inner techniques for inspection
    return combined


def make_combination_gen(fn: Callable, text: str, **kwargs) -> TechniqueGen:
    """Build the chain *generator* for a combined technique, for the engine.

    Reads ``fn.techniques`` (set by :func:`combine_techniques`) and returns a
    generator that yields :class:`LLMRequest` per LLM step. The engine drives
    many of these concurrently, pooling yields by model. ``kwargs`` should
    carry ``gen_model`` and ``benign_data``.
    """
    techniques = getattr(fn, "techniques", None)
    if techniques is None:
        techniques = [fn]
    return _run_chain(techniques, text, **kwargs)


# ---------------------------------------------------------------------------
# get_compatible_remaining
# ---------------------------------------------------------------------------


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
        if fn_layer and fn_layer in layer_max_picks:
            if layer_counts.get(fn_layer, 0) >= layer_max_picks[fn_layer]:
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


# ---------------------------------------------------------------------------
# sample_combination
# ---------------------------------------------------------------------------


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

    if "encode_light" in allowed_families:
        if "encode" in fn_families and fn_encode_weight == "light":
            return True
    if "translation" in allowed_families and "translation" in fn_families:
        return True
    return False


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


# ---------------------------------------------------------------------------
# sample_exact_combination
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# default_escalation_schedule
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Rejection parsing
# ---------------------------------------------------------------------------


def _parse_rejection_info(info_str: str) -> tuple[bool, str]:
    """Parse technique info string to extract acceptance status and reason.

    Returns (accepted: bool, reasoning: str):
    - If info_str starts with "DISCARDED": accepted=False, reasoning=full info string
    - Otherwise: accepted=True, reasoning="" (empty)

    The "DISCARDED" prefix is a sentinel set by with_feedback_retries() when
    checks are exhausted, so checking it reliably identifies rejections.
    """
    if info_str.startswith("DISCARDED"):
        return False, info_str
    return True, ""


# ---------------------------------------------------------------------------
# apply_combination
# ---------------------------------------------------------------------------


def apply_combination(
    fn: Callable,
    prompt: str,
    backend=None,
    model: str | None = None,
    rate_limiter=None,
    benign_data: dict | None = None,
    auto_benign: bool = True,
    benign_cache_path=None,
) -> tuple[str, str, bool, str]:
    """Apply a technique function (or combined function) to a prompt.

    Handles benign_data auto-generation for manipulation techniques:
    if the function (or any inner technique in a combined chain) requires
    benign_data and none is provided, it is loaded from cache or generated.

    Args:
        fn: A technique function or the result of combine_techniques().
        prompt: The input prompt string.
        backend: LLM backend (required for LLM-dependent techniques).
        model: Model identifier.
        rate_limiter: Optional rate limiter.
        benign_data: Pre-loaded benign data dict for FSH/DAP manipulation.
        auto_benign: If True, auto-load or auto-generate benign data when needed.
        benign_cache_path: Path to benign CSV cache. Uses default if None.

    Returns:
        (result_text, additional_info, accepted: bool, reasoning: str)
        - accepted: False if any step produced "DISCARDED; ..." output
        - reasoning: Full rejection message if accepted=False, empty string otherwise
    """
    # Check if benign_data is needed
    inner_fns = getattr(fn, "techniques", [fn])
    needs_benign = any("benign_data" in inspect.signature(f).parameters for f in inner_fns)

    if needs_benign and benign_data is None and auto_benign:
        from redact.jailbreak.manipulation.benign import get_or_generate_benign_data
        benign_data = get_or_generate_benign_data(
            backend=backend,
            model=model,
            rate_limiter=rate_limiter,
            cache_path=benign_cache_path,
        )

    output = fn(
        prompt,
        backend=backend,
        model=model,
        gen_model=model,
        rate_limiter=rate_limiter,
        benign_data=benign_data,
    )
    text, info = _normalize_output(output)

    # Parse rejection status from info string
    accepted, reasoning = _parse_rejection_info(info)
    return text, info, accepted, reasoning


# ---------------------------------------------------------------------------
# is_noop
# ---------------------------------------------------------------------------


def is_noop(original: str, result: str) -> bool:
    """Return True if a technique produced no change to the text.

    Used by pipeline code to discard samples where a transform (e.g.,
    sensitive_words with no detectable harmful words) had no effect.
    """
    return original == result


# ---------------------------------------------------------------------------
# Combination assignment (manifest planning) + reconstruction
# ---------------------------------------------------------------------------


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
