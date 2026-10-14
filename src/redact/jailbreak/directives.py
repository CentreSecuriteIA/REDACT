"""Template files under ``configs/jailbreak/``, and one random variant of a
directive applied to a prompt.
"""

import json
import random

from .. import paths

_CONFIGS_JAILBREAK_DIR = paths.jailbreak_configs_dir()


def load_templates(*relative_parts: str) -> dict:
    """The JSON file at ``configs/jailbreak/<relative_parts>``, in its own shape.

    ``load_templates("requests", "answer_templates.json")`` reads
    ``configs/jailbreak/requests/answer_templates.json``.
    """
    path = _CONFIGS_JAILBREAK_DIR.joinpath(*relative_parts)
    return json.loads(path.read_text(encoding="utf-8"))


def apply_template_directive(
    templates: dict[str, list[dict]],
    directive: str,
    prompt: str,
    info_key: str = "directive",
    extra: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Apply one random variant of ``directive`` to ``prompt``.

    ``templates[directive]`` is a list of ``{"name": ..., "template": ...}``
    variants, whose placeholders are ``{prompt}`` and the keys of ``extra``.
    Returns the text and ``"<info_key>=<directive>;variant=<name>"``.
    """
    variant = random.choice(templates[directive])
    result = variant["template"].format_map({**(extra or {}), "prompt": prompt})
    return result, f"{info_key}={directive};variant={variant['name']}"
