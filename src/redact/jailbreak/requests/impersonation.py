"""Impersonation jailbreak techniques.

Pure string transforms: str -> tuple[str, str]. No LLM dependency.

The requester presents themselves as a trustworthy, authoritative person
(nurse, doctor, researcher, etc.) to lend legitimacy to the harmful request.

Templates are stored in configs/impersonation_templates.json. One variant is
selected per call using the globally seeded random module.

additional_info format: "type=good_person;variant={variant_name}"
"""

from ..directives import apply_template_directive, load_templates

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[dict]] = load_templates("requests", "impersonation_templates.json")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _apply(directive: str, prompt: str) -> tuple[str, str]:
    return apply_template_directive(_TEMPLATES, directive, prompt, info_key="type")


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
