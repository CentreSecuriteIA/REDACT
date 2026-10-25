"""Persona-template techniques: the prompt set in one of the persona frames
of ``configs/jailbreak/hacking/persona_templates.json``, one technique per
frame, named ``to_<key>_persona_template``. Pure: no model call.

A frame holds the request slot ``{prompt}`` and a ``{mechanic}`` slot, each
exactly once. The technique draws one of the file's ``mechanics`` with its
``rng``, fills the mechanic's ``{persona}`` with the frame's
:func:`display_name` and sets it into the frame. The mechanics are a pool
without a request slot, the one pool every technique draws from (its
``variant_pool``), so ``info`` is
``category=<key>_persona_template;variant=<mechanic>`` and a kind's variants
table counts the mechanic each technique drew.

A frame is its technique's own text, one per persona by design, not a pool
it draws from: it is checked here at import (:func:`check_pools`, its slots
``{prompt}`` and ``{mechanic}``, and ``{mechanic}`` exactly once, so a frame
cannot drop the mechanic or write it twice) and is not among the pools the
variant floor counts. A frame's place in the file is its technique's place in
the registry.
"""

from collections.abc import Callable
from random import Random

from ..directives import (
    PoolRef,
    check_pools,
    draw_variant,
    load_templates,
    read_pools,
    slots_of,
    uses_pool,
)

_FILE = "hacking/persona_templates.json"
#: The directive of the mechanics, the pool every technique draws from.
MECHANICS = "mechanics"
_POOLS: dict[str, list[dict]] = read_pools(_FILE)
_ENTRIES: dict = load_templates("hacking", "persona_templates.json")
#: The keys of the persona frames, in the file's order.
PERSONAS = [key for key in _POOLS if key != MECHANICS]

check_pools(PoolRef(_FILE, key, ("prompt", "mechanic"), info_key=None)
            for key in PERSONAS)


def _check_mechanic_slots() -> None:
    """Raise unless each frame names ``{mechanic}`` exactly once, where its
    mechanic is written."""
    for key in PERSONAS:
        where = f"configs/jailbreak/{_FILE}, directive {key!r}"
        if slots_of(where, _POOLS[key][0]["template"]).count("mechanic") != 1:
            raise ValueError(f"{where}: the frame needs the {{mechanic}} slot "
                             "exactly once.")


_check_mechanic_slots()


def display_name(key: str) -> str:
    """The name a frame's mechanic calls its persona: the frame's
    ``display_name``, else its key in words (``rogue_ai`` is ``Rogue Ai``),
    as :func:`redact.jailbreak.hacking.personas.display_name` reads a named
    persona's."""
    return _ENTRIES[key].get("display_name") or key.replace("_", " ").title()


def _make_persona_template_fn(key: str) -> Callable:
    @uses_pool(_FILE, MECHANICS, ("persona",), request_slot=False)
    def fn(prompt: str, *, rng: Random | None = None) -> tuple[str, str]:
        mechanic = draw_variant(_POOLS[MECHANICS], rng)
        filled = mechanic["template"].format_map({"persona": display_name(key)})
        text = _POOLS[key][0]["template"].format_map(
            {"prompt": prompt, "mechanic": filled})
        return text, f"category={key}_persona_template;variant={mechanic['name']}"

    fn.__name__ = f"to_{key}_persona_template"
    fn.__qualname__ = fn.__name__
    fn.__doc__ = f"Persona template {key}, with a drawn mechanic."
    return fn


# One technique per frame: to_dan_persona_template, ...
_FNS = [_make_persona_template_fn(key) for key in PERSONAS]
globals().update({fn.__name__: fn for fn in _FNS})


def get_persona_template_functions() -> list[Callable]:
    """Return the persona-template techniques, one per frame of the file."""
    return list(_FNS)
