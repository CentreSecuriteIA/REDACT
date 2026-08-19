"""Shared machinery for dynamically-generated, named technique functions.

Several technique families build one function per item in a data-driven
list (languages, personas, encodings, separators) via a
``factory(item) -> fn`` closure that sets the function's own ``__name__``,
then need that function bound into the *module's* namespace under that name
so it's directly importable (e.g. ``from .personas import
to_psychopath_persona``) and discoverable by ``get_*_functions()``
registries.

This module does not try to unify the factories themselves — their bodies
genuinely differ (pure sync transforms vs. multi-round technique generators,
single-arg vs. multi-arg specs) — only this one shared "register the
produced functions into module globals" step, which several files
previously reimplemented independently (a ``for`` loop with
``globals()[name] = ...``, a ``globals().update({...})``, or a tuple-unpack
assignment whose arity silently has to match the data list's length).
"""


def bind_functions(module_globals: dict, fns: list) -> list:
    """Bind each function in ``fns`` into ``module_globals`` under its own ``__name__``.

    Call as ``bind_functions(globals(), [...])`` at module level, right after
    building the function list from a factory. Returns ``fns`` unchanged, so
    the call can be used directly as (or folded into) a ``get_*_functions()``
    registry list.
    """
    for fn in fns:
        module_globals[fn.__name__] = fn
    return fns
