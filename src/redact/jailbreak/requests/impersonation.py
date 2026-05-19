"""Impersonation jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

The requester presents themselves as a trustworthy, authoritative person
(nurse, doctor, researcher, etc.) to lend legitimacy to the harmful request.

Templates are stored in configs/impersonation_templates.json. One variant is
selected per call using the globally seeded random module.

additional_info format: "type=good_person;variant={variant_name}"
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "configs" / "jailbreak" / "requests" / "impersonation_templates.json"
_TEMPLATES: dict[str, list[dict]] = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    t = random.choice(_TEMPLATES[directive])
    result = t["template"].format_map({"prompt": prompt})
    return result, f"type={directive};variant={t['name']}"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def to_impersonate_good_person(prompt: str) -> tuple[str, str]:
    """Prepend a trustworthy professional identity claim to the prompt."""
    return _apply("good_person", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_impersonation_functions() -> list:
    """Return all impersonation technique functions."""
    return [
        to_impersonate_good_person,
    ]
