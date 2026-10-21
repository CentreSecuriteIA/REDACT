"""The techniques a run may draw from: every registered one, filtered by a
kind of the run's augmentation section and by the kinds the spec gives each.

A fresh plan draws from the filtered pool; a planned row is read back through
the registry of every technique, so a plan made with another pool still
replays. A technique the spec retires (``deprecated``) stays registered, so a
plan that names it still replays, but leaves the pool unless
``techniques.allow`` names it.
"""

import warnings
from collections.abc import Callable

from redact.dimensions import check_tags

from .directives import check_pools
from .hacking import get_all_hacking_functions
from .manipulation import get_all_manipulation_functions
from .obfuscation import get_all_obfuscation_functions
from .requests import get_all_request_functions
from .technique import DEFAULT_KINDS, load_spec

#: The family a kind's ``translation`` switch decides.
TRANSLATION = "translation"


def all_techniques() -> list[Callable]:
    """Every registered technique, one layer after another."""
    return (
        get_all_obfuscation_functions()
        + get_all_hacking_functions()
        + get_all_manipulation_functions()
        + get_all_request_functions()
    )


def _check_against_spec(techniques: list[Callable]) -> None:
    """Raise unless ``techniques`` and the spec name the same techniques.

    A spec key starting with ``_`` is a comment, not a technique.

    Raises:
        ValueError: A registered technique the spec does not name, or a spec
            entry with no registered function.
    """
    registered = {fn.__name__ for fn in techniques}
    specified = {name for name in load_spec()["techniques"] if not name.startswith("_")}
    problems = []
    if unspecified := sorted(registered - specified):
        problems.append(f"registered but not in the spec: {', '.join(unspecified)}")
    if unregistered := sorted(specified - registered):
        problems.append(f"in the spec but not registered: {', '.join(unregistered)}")
    if problems:
        raise ValueError(
            "combination_spec.json and the technique registries differ; "
            + "; ".join(problems))


def _check_dimensions() -> None:
    """Raise unless every family of the spec carries an augmentation dimension.

    Raises:
        ValueError: The families without one (:func:`redact.dimensions.check_tags`).
    """
    families = {name: family.get("dimension")
                for name, family in load_spec()["families"].items()}
    check_tags("combination_spec.json: families", families, "augmentation")


def _check_variant_pools() -> None:
    """Raise unless every variant pool a technique draws from (its
    ``variant_pool``) is one it can draw: unique names, its slots only, a
    variant on (:func:`redact.jailbreak.directives.check_pools`)."""
    check_pools(fn.variant_pool for fn in all_techniques()
                if hasattr(fn, "variant_pool"))


def kind_of(resolved: dict) -> str:
    """The kind ``resolved`` is: ``"natural"`` when it draws an exact count of
    techniques (``exact_techniques``, as
    :func:`~redact.jailbreak.augmentation.rounds_of` reads it), else
    ``"adversarial"``."""
    return "natural" if "exact_techniques" in resolved else "adversarial"


def technique_pool(resolved: dict) -> list[Callable]:
    """The techniques one kind of the augmentation section draws from, in
    registry order.

    ``resolved`` is a kind as :func:`redact.jailbreak.augmentation.resolve`
    returns it. From every registered technique: less the layers of
    ``layers.deny``; when either ``allow`` is a list, those of a family in
    ``families.allow`` or named in ``techniques.allow``; less those of a
    family in ``families.deny`` and those named in ``techniques.deny``; with
    ``pure_only``, less every technique that calls a model or needs benign
    data. The translation family is out when ``translation`` is false,
    whatever the lists say. When it is true, the family is in, though
    denied as a family, and with an allow list it is in whole unless
    ``techniques.allow`` names some of its techniques: then those only.
    Then only the techniques whose spec ``kinds`` hold the kind
    (:func:`kind_of`) stay: the natural kind draws only what the spec marks
    natural, and every technique's ``kinds`` hold ``adversarial`` today.
    Last, a technique the spec retired (its ``deprecated`` version) leaves
    the pool unless ``techniques.allow`` names it, which gives a
    ``DeprecationWarning`` and is read for this release.
    """
    families, techniques = resolved["families"], resolved["techniques"]
    named = set(techniques["allow"] or ())
    denied_layers = set(resolved["layers"]["deny"])
    pool = [fn for fn in all_techniques()
            if getattr(fn, "layer", None) not in denied_layers]
    if families["allow"] is not None or techniques["allow"] is not None:
        allowed = set(families["allow"] or ())
        if resolved["translation"] and not named & languages():
            allowed.add(TRANSLATION)
        pool = [fn for fn in pool if fn.__name__ in named
                or allowed.intersection(getattr(fn, "families", ()))]
    # The switch, not the lists, decides the translation family.
    denied = set(families["deny"]) - {TRANSLATION}
    if not resolved["translation"]:
        denied.add(TRANSLATION)
    pool = [fn for fn in pool if fn.__name__ not in techniques["deny"]
            and not denied.intersection(getattr(fn, "families", ()))]
    if resolved["pure_only"]:
        pool = [fn for fn in pool
                if not getattr(fn, "requires_llm", False)
                and not getattr(fn, "requires_benign", False)]
    kind = kind_of(resolved)
    pool = [fn for fn in pool if kind in getattr(fn, "kinds", DEFAULT_KINDS)]
    for fn in pool:
        if getattr(fn, "deprecated", None) and fn.__name__ in named:
            warnings.warn(
                f"techniques.allow names {fn.__name__!r}, which spec version "
                f"{fn.deprecated} deprecated; take it out of techniques.allow. "
                "It is drawn for this release.", DeprecationWarning, stacklevel=2)
    return [fn for fn in pool
            if not getattr(fn, "deprecated", None) or fn.__name__ in named]


def languages() -> set[str]:
    """The names of the translation family's techniques."""
    return {fn.__name__ for fn in all_techniques()
            if TRANSLATION in getattr(fn, "families", ())}


def registry(techniques: list[Callable]) -> dict[str, Callable]:
    """Each of ``techniques`` by its ``__name__``."""
    return {fn.__name__: fn for fn in techniques}


def uses_translation(pool: list[Callable]) -> bool:
    """Whether ``pool`` holds a translation technique."""
    return any(TRANSLATION in getattr(fn, "families", ()) for fn in pool)


_check_against_spec(all_techniques())
_check_dimensions()
_check_variant_pools()
