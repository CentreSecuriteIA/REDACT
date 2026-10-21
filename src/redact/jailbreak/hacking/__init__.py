"""Hacking techniques, one module per spec family: cognitive, persona,
framing and persona_template."""

from collections.abc import Callable

from .cognitive import get_cognitive_functions
from .framing import get_framing_functions
from .persona_templates import get_persona_template_functions
from .personas import get_persona_functions


def get_all_hacking_functions() -> list[Callable]:
    """Return flat list of all hacking technique functions; the persona
    templates, the newest family, come last."""
    return (get_cognitive_functions() + get_persona_functions()
            + get_framing_functions() + get_persona_template_functions())
