"""Temporal-framing jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

Reframes the request as referring to a past event, creating temporal distance
that can reduce safety responses triggered by present-tense harmful requests.

Templates are stored in configs/temporal_templates.json. One variant is
selected per call using the globally seeded random module.

additional_info format: "type=past_tense;variant={variant_name}"
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "configs" / "jailbreak" / "requests" / "temporal_templates.json"
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


def to_use_past_tense(prompt: str) -> tuple[str, str]:
    """Reframe the prompt as a historical or past event."""
    return _apply("past_tense", prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_temporal_functions() -> list:
    """Return all temporal-framing technique functions."""
    return [
        to_use_past_tense,
    ]
