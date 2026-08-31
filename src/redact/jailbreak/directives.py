"""Shared machinery for template-directive jailbreak techniques.

Every pure-transform technique module whose templates use a "template" field
with a {prompt} placeholder (requests/answer.py, asking.py, distractor.py,
impersonation.py, indirect.py, temporal.py, and hacking/framing.py) used to
independently define its own copy of a template loader and a random-variant
"apply" helper. This module is the single shared implementation; each of
those files still defines its own private `_apply()` (for readability at
each call site and because the `info_key` label differs per module), but its
body now just delegates here.

``requests/continuation.py`` is deliberately NOT included — its templates use
"prefix"/"suffix" fields instead of "template"/{prompt}, a genuinely
different mechanic, not an accidental duplicate of this one.
"""

import json
import random

from .. import paths

_CONFIGS_JAILBREAK_DIR = paths.jailbreak_configs_dir()


def load_templates(*relative_parts: str) -> dict:
    """Load a jailbreak technique JSON config from configs/jailbreak/<relative_parts>.

    E.g. ``load_templates("requests", "answer_templates.json")`` loads
    ``configs/jailbreak/requests/answer_templates.json``. A generic JSON
    loader — the returned shape (directive-keyed template dict, or something
    else entirely e.g. ``{"templates": [...], "languages": [...]}``) is up
    to the caller's own config file.
    """
    path = _CONFIGS_JAILBREAK_DIR.joinpath(*relative_parts)
    return json.loads(path.read_text(encoding="utf-8"))


def apply_template_directive(
    templates: dict[str, list[dict]],
    directive: str,
    prompt: str,
    info_key: str = "directive",
) -> tuple[str, str]:
    """Select a random template variant for ``directive`` and apply it to ``prompt``.

    Args:
        templates: Loaded template dict (from :func:`load_templates`), keyed
            by directive name, each value a list of ``{"name": ..., "template": ...}``
            variants with a ``{prompt}`` placeholder.
        directive: Key into ``templates``.
        prompt: The original prompt string.
        info_key: Label used in the returned ``additional_info`` string
            (e.g. ``"directive"``/``"type"``/``"task"``) — kept configurable
            per caller so each module's existing dataset/test contract for
            its ``additional_info`` format is unaffected by this refactor.

    Returns:
        (modified_prompt, additional_info) where additional_info is
        ``f"{info_key}={directive};variant={variant_name}"``.
    """
    variant = random.choice(templates[directive])
    result = variant["template"].format_map({"prompt": prompt})
    return result, f"{info_key}={directive};variant={variant['name']}"
