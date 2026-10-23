"""Which techniques may still join a combination, by the spec's rules.

Every rule binds both ways: a pair the spec forbids is refused whichever of the
two was picked first.
"""

from collections import Counter
from collections.abc import Callable

from .technique import load_spec


def _families(fn: Callable) -> set[str]:
    return set(getattr(fn, "families", ()))


def _own_incompatible(fn: Callable) -> set[str]:
    """The families the spec keeps ``fn`` itself away from, beside its
    families' rules."""
    return set(getattr(fn, "cross_incompatible_with", ()))


def _unfit_for_manipulation(fn: Callable, blocked: dict) -> bool:
    """Whether ``fn`` is an obfuscation the spec keeps away from the
    manipulation layer's families (fsh, dap, fake_turns).

    Word-level encoding (``sensitive_words``) is kept: it leaves the Q&A
    readable.
    """
    families = _families(fn)
    if families & set(blocked.get("families", [])):
        return True
    return ("encode" in families and "sensitive_words" not in families
            and getattr(fn, "encode_weight", None) == blocked.get("encode_weight"))


def compatible(
    selected: list[Callable],
    pool: list[Callable],
    remaining_complexity: int | None = None,
) -> list[Callable]:
    """The techniques of ``pool`` that may join ``selected``.

    A candidate is refused when it shares a family with the selection, when
    its layer is at the spec's cap, when the spec declares one of its families
    incompatible with one of the selection's (in either family's list, or in
    the list of a technique on either side), when it
    and the selection would pair a manipulation with an obfuscation the spec
    blocks for manipulation, or when its complexity exceeds
    ``remaining_complexity``.
    """
    spec = load_spec()
    families = spec["families"]
    layer_caps: dict[str, int] = spec.get("layer_max_picks", {})
    blocked_for_manipulation = spec.get("manipulation_blocked_obfuscation", {})
    manipulation = {
        name for name, fam in families.items() if fam.get("layer") == "manipulation"}

    def incompatible(family: str) -> set[str]:
        return set(families.get(family, {}).get("cross_incompatible_with", []))

    taken = set().union(*map(_families, selected))
    refused = taken.union(*map(incompatible, taken), *map(_own_incompatible, selected))
    per_layer = Counter(getattr(fn, "layer", None) for fn in selected)
    has_manipulation = bool(taken & manipulation)
    has_unfit = any(_unfit_for_manipulation(fn, blocked_for_manipulation)
                    for fn in selected)

    result = []
    for fn in pool:
        fn_families = _families(fn)
        layer = getattr(fn, "layer", None)
        if fn_families & refused:
            continue
        if taken & set().union(_own_incompatible(fn), *map(incompatible, fn_families)):
            continue
        if layer in layer_caps and per_layer[layer] >= layer_caps[layer]:
            continue
        if has_manipulation and _unfit_for_manipulation(fn, blocked_for_manipulation):
            continue
        if has_unfit and fn_families & manipulation:
            continue
        if (remaining_complexity is not None
                and getattr(fn, "complexity", 0) > remaining_complexity):
            continue
        result.append(fn)
    return result


def request_obfuscation_candidates(pool: list[Callable]) -> list[Callable]:
    """The techniques of ``pool`` that may obfuscate a finished request.

    The spec allows ``translation`` and ``encode_light`` (the ``encode``
    family at ``encode_weight`` light).
    """
    allowed = set(load_spec().get("request_obfuscation_allowed_families", []))

    def allowed_for_request(fn: Callable) -> bool:
        families = _families(fn)
        if ("encode_light" in allowed and "encode" in families
                and getattr(fn, "encode_weight", None) == "light"):
            return True
        return "translation" in allowed and "translation" in families

    return [fn for fn in pool if allowed_for_request(fn)]
