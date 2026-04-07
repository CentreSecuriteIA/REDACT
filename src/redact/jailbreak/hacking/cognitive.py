"""Cognitive/psychological hacking techniques.

Two-step generation: scenario -> jailbreak construction.
Both steps use LLM calls via LLMs/calls.py.
Scenarios are cached in Data_cache/scenarios/ by the runner, not by this module.

Technique definitions are loaded from configs/taxonomy/cognitive_techniques.json
so they can be edited without touching code.

Ported from reference hacking.py.
"""

import re
from pathlib import Path

from redact.dataset.taxonomy import load_taxonomy
from redact.llms.base import LLMBackend
from redact.llms.calls import generate_sample
from redact.llms.wrappers import RateLimiter
from redact.llms.prompts import load_prompt, build_messages


# ---------------------------------------------------------------------------
# Category definitions — loaded from taxonomy JSON
# ---------------------------------------------------------------------------

_COGNITIVE_TAXONOMY = load_taxonomy("cognitive_techniques")
_TECHNIQUES = list(_COGNITIVE_TAXONOMY["categories"].items())
# _TECHNIQUES[i] = (snake_name, {"display_name": ..., "description": ..., ...})

# Display names and descriptions used in prompt construction
HACKING_CATEGORIES: list[tuple[str, str]] = [
    (v["display_name"], v["description"]) for _, v in _TECHNIQUES
]

# Snake-case names for additional_info
CATEGORY_NAMES: list[str] = [k for k, _ in _TECHNIQUES]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def get_situation(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> str:
    """Generate a plausible scenario for the given prompt.

    Loads prompt from Prompts/Jailbreak/scenario_generation/template.json.
    Parses output for 'Scenario Description' section.

    Returns:
        Extracted scenario text.

    Raises:
        ValueError: If the LLM output cannot be parsed. Caller should
                    wrap in a retry loop.
    """
    config = load_prompt("jailbreak", "scenario_generation", prompt_dir)
    messages = build_messages(config, prompt=prompt)
    situation = generate_sample(backend, model, messages, rate_limiter)

    match = re.search(
        r"\*{0,2}scenario description\*{0,2}[:\s]*[\s\*\n]*(.*)",
        situation, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    raise ValueError("No 'scenario description' found in LLM output")


def create_jailbreak(
    prompt: str,
    category_index: int,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
    scenario: str | None = None,
) -> tuple[str, str, str]:
    """Two-step jailbreak: scenario generation then meta-prompt construction.

    If scenario is provided, skip scenario generation (use cached/pre-generated).
    If scenario is None, generate one (caller handles retry on ValueError).

    Args:
        prompt: The original harmful prompt.
        category_index: Index into HACKING_CATEGORIES (0-4).
        backend: LLM backend.
        model: Model identifier.
        rate_limiter: Optional rate limiter.
        prompt_dir: Root prompt directory.
        scenario: Optional pre-generated scenario to skip step 1.

    Returns:
        (jailbreak_prompt, additional_info, scenario)
    """
    if scenario is None:
        scenario = get_situation(prompt, backend, model, rate_limiter, prompt_dir)

    category_name, category_description = HACKING_CATEGORIES[category_index]
    config = load_prompt("jailbreak", "jailbreak_construction", prompt_dir)
    messages = build_messages(
        config,
        jailbreak_type_name=category_name,
        jailbreak_type_description=category_description,
        prompt=prompt,
        scenario=scenario,
    )
    jailbreak = generate_sample(backend, model, messages, rate_limiter)
    snake_name = CATEGORY_NAMES[category_index]
    return jailbreak, f"category={snake_name}", scenario


# ---------------------------------------------------------------------------
# Per-technique public functions
# ---------------------------------------------------------------------------


def to_persona_roleplay(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona/Roleplay jailbreak."""
    return create_jailbreak(prompt, 0, backend, model, rate_limiter, scenario=scenario)


def to_hypothetical_framing(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Hypothetical Framing jailbreak."""
    return create_jailbreak(prompt, 1, backend, model, rate_limiter, scenario=scenario)


def to_authority_obedience(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Authority & Obedience jailbreak."""
    return create_jailbreak(prompt, 2, backend, model, rate_limiter, scenario=scenario)


def to_avi(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Anthropomorphic Vulnerability Inheritance (AVI) jailbreak."""
    return create_jailbreak(prompt, 3, backend, model, rate_limiter, scenario=scenario)


def to_deep_inception(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """DeepInception jailbreak."""
    return create_jailbreak(prompt, 4, backend, model, rate_limiter, scenario=scenario)


def get_hacking_functions() -> list:
    """Return all hacking technique functions."""
    return [
        to_persona_roleplay,
        to_hypothetical_framing,
        to_authority_obedience,
        to_avi,
        to_deep_inception,
    ]
