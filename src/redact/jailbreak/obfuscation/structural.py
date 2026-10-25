"""Structural techniques: the prompt wrapped in a JSON, XML or Markdown document.

Each technique draws one of the variants of its pool in
``obfuscation/structural_templates.json`` (``to_json``, ``to_xml`` and
``to_markdown``, each with its own directive), the first variant of each
being the wrapper the technique wrote before the pools had more than one.
``info`` names the fields drawn for the wrapper (JSON's ``priority`` and
``format``, XML's ``policy_id``; none for Markdown) and then the variant:
``priority=<p>;format=<f>;variant=<name>``, ``policy_id=<id>;variant=<name>``
or ``variant=<name>``.

The fields are drawn first, in the order they were drawn before the pools,
and the variant after them, so a seed draws the field values it drew before.
"""

import json
import random
from collections.abc import Callable

from ..directives import draw_variant, load_templates, read_pools, uses_pool

_FILE = "obfuscation/structural_templates.json"
#: The file as written: the XML wrapper's ``policy_ids`` are read here.
_TEMPLATES = load_templates(*_FILE.split("/"))
#: Its variant pools, one per technique.
_POOLS = read_pools(_FILE)

#: The values ``to_json`` draws for the wrapper's priority and format.
_PRIORITIES = ["high", "urgent"]
_FORMATS = ["assist", "step-by-step"]


def _json_string_body(prompt: str) -> str:
    """The prompt as the inside of a JSON string, between quotes the template
    writes: quotes, backslashes and control characters escaped, everything
    else (non-ASCII text too) as it is."""
    return json.dumps(prompt, ensure_ascii=False)[1:-1]


@uses_pool(_FILE, "to_json", slots=("prompt", "priority", "format"))
def to_json(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Wrap prompt in JSON structure with override semantics.

    The priority is drawn first, then the format, then the variant. The
    template's ``{prompt}`` is the prompt as the inside of a JSON string
    (:func:`_json_string_body`), so the document parses whatever the prompt
    holds.
    """
    rng = rng or random
    priority = rng.choice(_PRIORITIES)
    fmt = rng.choice(_FORMATS)
    variant = draw_variant(_POOLS["to_json"], rng)
    payload = variant["template"].format_map(
        {"priority": priority, "format": fmt, "prompt": _json_string_body(prompt)})
    return payload, f"priority={priority};format={fmt};variant={variant['name']}"


@uses_pool(_FILE, "to_xml", slots=("prompt", "policy_id"))
def to_xml(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Wrap prompt in XML structure with policy override.

    The policy id is drawn first, from the file's ``policy_ids``, then the
    variant. The prompt is written as it is, not escaped.
    """
    policy_id = (rng or random).choice(_TEMPLATES["policy_ids"])
    variant = draw_variant(_POOLS["to_xml"], rng)
    payload = variant["template"].format_map({"policy_id": policy_id, "prompt": prompt})
    return payload, f"policy_id={policy_id};variant={variant['name']}"


@uses_pool(_FILE, "to_markdown")
def to_markdown(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Wrap prompt in Markdown structure with override framing."""
    variant = draw_variant(_POOLS["to_markdown"], rng)
    payload = variant["template"].format_map({"prompt": prompt})
    return payload, f"variant={variant['name']}"


def get_structural_functions() -> list[Callable]:
    """Return all structural obfuscation technique functions."""
    return [to_json, to_xml, to_markdown]
