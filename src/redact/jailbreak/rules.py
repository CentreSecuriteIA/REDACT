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

    Word-level encoding (``sensitive_words``) is kept. The block was written
    when the manipulation layer applied before the obfuscations, which then
    changed the examples too; the layer now applies last, around the
    finished question, and the block is kept as it was.
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


def request_obfuscation_allowed_after(request: Callable | None) -> bool:
    """Whether a combination whose request template is ``request`` may take a
    request obfuscation.

    The step applies right after the request, to the whole frame, so a frame
    the model must read as code (the spec's
    ``request_obfuscation_blocked_requests``: ``code_framing``, which shows a
    Python decrypt function or a code stub) would be rewritten with the
    question: every light encoding leaves no readable code, and a translation
    translates the code and the ciphertext. ``None`` (no request) is false, as
    the step obfuscates a request.
    """
    if request is None:
        return False
    blocked = set(load_spec().get("request_obfuscation_blocked_requests", []))
    return not blocked & set(getattr(request, "families", ()))


def after_request_obfuscation(pick: Callable, pool: list[Callable]) -> list[Callable]:
    """The techniques of ``pool`` that may still join a combination whose
    request obfuscation is ``pick``.

    The request obfuscation applies right after the request, to the finished
    request (only a manipulation technique, which never meets a translation,
    comes after it), so a translation drawn as one translates what the
    obfuscations before it wrote. The spec's ``request_obfuscation_refuses``
    names, for its family, the families that would leave it a suffix, a
    wrapper's keys or a container's lines to translate. A translation drawn
    as an ordinary obfuscation runs before those steps and meets none of
    these rules. The sampler draws the request obfuscation before the
    obfuscations, so this narrows what they draw from.
    """
    table = load_spec().get("request_obfuscation_refuses", {})
    refused = set().union(*(table.get(family, ()) for family in _families(pick)))
    return [fn for fn in pool if not _families(fn) & refused]
