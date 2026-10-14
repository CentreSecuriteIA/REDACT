"""The techniques a run may draw from: every registered one, filtered by the
run's flags.

A fresh plan draws from the filtered pool; a planned row is read back through
the registry of every technique, so a plan made with another pool still
replays.
"""

from collections.abc import Callable

from .hacking import get_all_hacking_functions
from .manipulation import get_all_manipulation_functions
from .obfuscation import get_all_obfuscation_functions
from .requests import get_all_request_functions
from .technique import load_spec


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


def technique_pool(
    *,
    include_hacking: bool = True,
    include_manipulation: bool = True,
    include_obfuscation: bool = True,
    include_requests: bool = True,
    include_translation: bool = True,
    pure_only: bool = False,
) -> list[Callable]:
    """The techniques a fresh plan may draw from, in registry order.

    A layer whose flag is off is left out, ``include_translation=False`` leaves
    out the translation family, and ``pure_only`` every technique that calls a
    model or needs benign data.
    """
    layer_on = {
        "hacking": include_hacking,
        "manipulation": include_manipulation,
        "obfuscation": include_obfuscation,
        "requests": include_requests,
    }
    pool = [fn for fn in all_techniques()
            if layer_on.get(getattr(fn, "layer", None), True)]
    if pure_only:
        pool = [fn for fn in pool
                if not getattr(fn, "requires_llm", False)
                and not getattr(fn, "requires_benign", False)]
    if not include_translation:
        pool = [fn for fn in pool if "translation" not in getattr(fn, "families", ())]
    return pool


def registry(techniques: list[Callable]) -> dict[str, Callable]:
    """Each of ``techniques`` by its ``__name__``."""
    return {fn.__name__: fn for fn in techniques}


def uses_translation(pool: list[Callable]) -> bool:
    """Whether ``pool`` holds a translation technique."""
    return any("translation" in getattr(fn, "families", ()) for fn in pool)


_check_against_spec(all_techniques())
