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

import inspect
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable


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
        fn.encode_weight = t.get("encode_weight", None)

        # Derive layer and within_layer_order from the first family
        primary_family = fn.families[0] if fn.families else None
        fn.layer = family_layer.get(primary_family, "obfuscation") if primary_family else "obfuscation"
        fn.within_layer_order = family_wlo.get(primary_family, 0) if primary_family else 0


# ---------------------------------------------------------------------------
# combine_techniques
# ---------------------------------------------------------------------------


def combine_techniques(*techniques: Callable, sort_by_hierarchy: bool = True) -> Callable:
    """Chain multiple technique functions sequentially.

    Each technique receives the output of the previous one.
    Additional_info strings are joined with ';'.

    If sort_by_hierarchy=True (default), techniques are reordered by
    (layer_order_index, within_layer_order) before chaining — so callers
    can pass techniques in any order and the correct semantic sequence is
    always applied.

    Supports:
    - Pure functions (str -> (str, str))
    - LLM functions that accept **kwargs (backend, model, rate_limiter, benign_data, etc.)
    - Cognitive/persona functions that return (str, str, str): the scenario (third element)
      is discarded in combined chains. Callers needing the scenario should not use
      combine_techniques for those functions.

    Args:
        *techniques: Functions with signature (str, **kwargs) -> (str, str) or (str, str, str)
        sort_by_hierarchy: If True, sort by layer and within_layer_order before chaining.

    Returns:
        Combined function: (str, **kwargs) -> (str, str)
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
        result = text
        info_parts: list[str] = []
        for technique in ordered:
            # Filter kwargs to only what this function accepts.
            # Pure transforms have (prompt: str) with no **kwargs — passing
            # unrecognised keys would raise TypeError.
            sig_params = inspect.signature(technique).parameters
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
            )
            tech_kwargs = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in sig_params}
            output = technique(result, **tech_kwargs)
            # Normalize 3-tuple returns from cognitive/persona (discard scenario)
            if isinstance(output, tuple) and len(output) == 3:
                new_text, info, _ = output
            else:
                new_text, info = output

            # Early exit on rejection — don't apply remaining techniques
            if info.startswith("DISCARDED"):
                return result, f"DISCARDED; technique={technique.__name__}; {info[len('DISCARDED; '):]}"

            # Track no-ops
            if new_text == result:
                info_parts.append(f"noop={technique.__name__}")
            else:
                if info:
                    info_parts.append(info)
            result = new_text
        return result, ";".join(info_parts)

    combined.__name__ = "+".join(t.__name__ for t in ordered)
    combined.techniques = list(ordered)  # expose inner techniques for inspection
    return combined


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
) -> Callable:
    """Sample a random valid technique combination from pool.

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

    Returns:
        A combined technique function, or an identity function if nothing was selected.
    """
    spec = load_spec()
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
        return rng.choice(candidates)

    def _commit(fn: Callable) -> None:
        selected.append(fn)
        nonlocal remaining_complexity
        remaining_complexity -= getattr(fn, "complexity", 0)
        _update_pool()

    # Phase 1: Hacking (50% chance)
    if include_hacking and rng.random() < 0.5:
        pick = _pick_from_layer("hacking")
        if pick is not None:
            _commit(pick)

    # Phase 2: Manipulation (33% chance)
    if include_manipulation and rng.random() < 0.33:
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
            pick = rng.choice(candidates)
            _commit(pick)
            obfusc_count += 1

    # Phase 4: Requests (70% chance)
    if include_requests and rng.random() < 0.7:
        pick = _pick_from_layer("requests")
        if pick is not None:
            _commit(pick)

    # Phase 5: Request-layer obfuscation (optional, 50% chance if enabled)
    if allow_request_obfuscation and rng.random() < 0.5:
        req_obfusc_candidates = [
            f for f in current_pool
            if _is_request_obfuscation_compatible(f, spec)
        ]
        if req_obfusc_candidates:
            pick = rng.choice(req_obfusc_candidates)
            selected.append(pick)  # don't update pool; this is the final step

    if not selected:
        def identity(text: str, **kwargs) -> tuple[str, str]:
            return text, ""
        identity.__name__ = "identity"
        identity.techniques = []
        return identity

    return combine_techniques(*selected)


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
        rate_limiter=rate_limiter,
        benign_data=benign_data,
    )
    # Normalize 3-tuple (cognitive/persona scenario return)
    if isinstance(output, tuple) and len(output) == 3:
        text, info, _ = output
    else:
        text, info = output

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
