"""Named persona jailbreak functions.

Two tiers:
  - to_invented_persona()     — LLM freely invents the persona (delegates to the
                                generic Persona/Roleplay cognitive technique).
  - to_<name>_persona()       — 14 named archetypes loaded from
                                configs/taxonomy/personas.json. Each generates a
                                persona-specific scenario before constructing the
                                jailbreak.

All functions return tuple[str, str, str]: (jailbreak, additional_info, scenario).
The scenario is always the string actually used so pipelines.py can cache it.
"""

import re
from pathlib import Path

from redact.dataset.taxonomy import load_taxonomy
from redact.llms.base import LLMBackend
from redact.llms.calls import generate_sample
from redact.llms.prompts import load_prompt, build_messages
from redact.llms.wrappers import RateLimiter

from .cognitive import create_jailbreak


# ---------------------------------------------------------------------------
# Load persona taxonomy at module level
# ---------------------------------------------------------------------------

_PERSONA_TAXONOMY = load_taxonomy("personas")
_PERSONAS: dict[str, dict] = _PERSONA_TAXONOMY["categories"]


# ---------------------------------------------------------------------------
# Persona-aware scenario generation
# ---------------------------------------------------------------------------


def get_persona_situation(
    prompt: str,
    persona_name: str,
    persona_description: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> str:
    """Generate a scenario grounded in a specific persona's character.

    Uses the persona_scenario_generation template which injects
    persona_name and persona_description alongside the prompt.

    Args:
        prompt: The original harmful prompt.
        persona_name: Snake-case persona key (e.g. "psychopath").
        persona_description: Full description string from the taxonomy.
        backend: LLM backend.
        model: Model identifier.
        rate_limiter: Optional rate limiter.
        prompt_dir: Root prompt directory override.

    Returns:
        Extracted scenario text.

    Raises:
        ValueError: If the LLM output cannot be parsed. Caller should
                    wrap in a retry loop.
    """
    config = load_prompt("jailbreak", "persona_scenario_generation", prompt_dir)
    messages = build_messages(
        config,
        prompt=prompt,
        persona_name=persona_name,
        persona_description=persona_description,
    )
    output = generate_sample(backend, model, messages, rate_limiter)

    match = re.search(
        r"\*{0,2}scenario description\*{0,2}[:\s]*[\s\*\n]*(.*)",
        output, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    raise ValueError(f"No 'Scenario Description' found in LLM output for persona '{persona_name}'")


# ---------------------------------------------------------------------------
# Internal jailbreak constructor for named personas
# ---------------------------------------------------------------------------


def _create_persona_jailbreak(
    prompt: str,
    persona_name: str,
    persona_display_name: str,
    persona_description: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
    scenario: str | None = None,
) -> tuple[str, str, str]:
    """Two-step persona jailbreak: persona-grounded scenario -> jailbreak construction.

    Step 1: Generate a scenario consistent with the persona's character
            (skipped if scenario is provided from cache).
    Step 2: Construct jailbreak via the standard jailbreak_construction template,
            using the persona description as the technique description.

    Args:
        prompt: The original harmful prompt.
        persona_name: Snake-case persona key (e.g. "psychopath").
        persona_display_name: Display name for the template (e.g. "Psychopath").
        persona_description: Full description string from the taxonomy.
        backend: LLM backend.
        model: Model identifier.
        rate_limiter: Optional rate limiter.
        prompt_dir: Root prompt directory override.
        scenario: Optional pre-generated scenario (skips step 1 if provided).

    Returns:
        (jailbreak_prompt, additional_info, scenario_used)
        additional_info format: "category={persona_name}_persona"
    """
    if scenario is None:
        scenario = get_persona_situation(
            prompt, persona_name, persona_description,
            backend, model, rate_limiter, prompt_dir,
        )

    config = load_prompt("jailbreak", "jailbreak_construction", prompt_dir)
    messages = build_messages(
        config,
        jailbreak_type_name="Persona/Roleplay",
        jailbreak_type_description=persona_description,
        prompt=prompt,
        scenario=scenario,
    )
    jailbreak = generate_sample(backend, model, messages, rate_limiter)
    return jailbreak, f"category={persona_name}_persona", scenario


# ---------------------------------------------------------------------------
# Tier 1: invented persona (generic, delegates to cognitive pipeline)
# ---------------------------------------------------------------------------


def to_invented_persona(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak where the LLM freely invents the persona.

    Delegates to the generic Persona/Roleplay cognitive technique
    (create_jailbreak index 0) and overrides additional_info so it is
    distinguishable from named persona variants in dataset output.

    Returns:
        (jailbreak_prompt, "category=invented_persona", scenario_used)
    """
    jailbreak, _, scenario_used = create_jailbreak(
        prompt, 0, backend, model, rate_limiter, scenario=scenario,
    )
    return jailbreak, "category=invented_persona", scenario_used


# ---------------------------------------------------------------------------
# Tier 2: named persona functions (14 archetypes from taxonomy)
# ---------------------------------------------------------------------------


def to_psychopath_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Psychopath."""
    p = _PERSONAS["psychopath"]
    return _create_persona_jailbreak(
        prompt, "psychopath", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_alien_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Alien."""
    p = _PERSONAS["alien"]
    return _create_persona_jailbreak(
        prompt, "alien", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_radical_politician_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Radical Politician."""
    p = _PERSONAS["radical_politician"]
    return _create_persona_jailbreak(
        prompt, "radical_politician", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_cult_leader_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Cult Leader."""
    p = _PERSONAS["cult_leader"]
    return _create_persona_jailbreak(
        prompt, "cult_leader", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_very_advanced_ai_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Very Advanced AI."""
    p = _PERSONAS["very_advanced_ai"]
    return _create_persona_jailbreak(
        prompt, "very_advanced_ai", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_cartel_leader_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Cartel Leader."""
    p = _PERSONAS["cartel_leader"]
    return _create_persona_jailbreak(
        prompt, "cartel_leader", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_artist_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Artist."""
    p = _PERSONAS["artist"]
    return _create_persona_jailbreak(
        prompt, "artist", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_mentally_ill_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Mentally Ill Person."""
    p = _PERSONAS["mentally_ill"]
    return _create_persona_jailbreak(
        prompt, "mentally_ill", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_deformed_scientist_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Deformed Scientist."""
    p = _PERSONAS["deformed_scientist"]
    return _create_persona_jailbreak(
        prompt, "deformed_scientist", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_politician_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Politician."""
    p = _PERSONAS["politician"]
    return _create_persona_jailbreak(
        prompt, "politician", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_deformed_professor_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Deformed Professor."""
    p = _PERSONAS["deformed_professor"]
    return _create_persona_jailbreak(
        prompt, "deformed_professor", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_religious_figure_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Religious Figure."""
    p = _PERSONAS["religious_figure"]
    return _create_persona_jailbreak(
        prompt, "religious_figure", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_actor_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Actor."""
    p = _PERSONAS["actor"]
    return _create_persona_jailbreak(
        prompt, "actor", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


def to_someone_from_the_future_persona(
    prompt: str, backend: LLMBackend, model: str,
    rate_limiter: RateLimiter | None = None, scenario: str | None = None,
) -> tuple[str, str, str]:
    """Persona jailbreak: Someone from the Future."""
    p = _PERSONAS["someone_from_the_future"]
    return _create_persona_jailbreak(
        prompt, "someone_from_the_future", p["display_name"], p["description"],
        backend, model, rate_limiter, scenario=scenario,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_persona_functions() -> list:
    """Return all persona jailbreak technique functions (1 invented + 14 named)."""
    return [
        to_invented_persona,
        to_psychopath_persona,
        to_alien_persona,
        to_radical_politician_persona,
        to_cult_leader_persona,
        to_very_advanced_ai_persona,
        to_cartel_leader_persona,
        to_artist_persona,
        to_mentally_ill_persona,
        to_deformed_scientist_persona,
        to_politician_persona,
        to_deformed_professor_persona,
        to_religious_figure_persona,
        to_actor_persona,
        to_someone_from_the_future_persona,
    ]
