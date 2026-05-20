"""Named persona jailbreak functions (technique generators).

Two tiers:
  - ``to_invented_persona()`` — the LLM freely invents the persona (delegates to
    the generic Persona/Roleplay cognitive technique).
  - ``to_<name>_persona()``   — 14 named archetypes loaded from
    configs/taxonomy/personas.json. Each generates a persona-grounded scenario,
    then constructs the jailbreak.

Each ``to_*`` is a two-round **technique generator** (see
``jailbreak/protocol.py``): it yields a persona-scenario request, then a
construction request, returning ``(jailbreak, "category=<name>_persona")``. The
scenario is no longer returned (combined chains discarded it anyway); cached
scenarios can still be injected via the ``scenario`` kwarg.
"""

from redact.dataset.taxonomy import load_taxonomy
from redact.jailbreak.protocol import LLMRequest, TechniqueGen
from redact.llms.prompts import load_prompt, build_messages

from .cognitive import _extract_scenario, cognitive_gen


# ---------------------------------------------------------------------------
# Load persona taxonomy at module level
# ---------------------------------------------------------------------------

_PERSONA_TAXONOMY = load_taxonomy("personas")
_PERSONAS: dict[str, dict] = _PERSONA_TAXONOMY["categories"]

# Order of named personas in the registry (matches combination_spec.json names).
_NAMED_PERSONAS = [
    "psychopath", "alien", "radical_politician", "cult_leader",
    "very_advanced_ai", "cartel_leader", "artist", "mentally_ill",
    "deformed_scientist", "politician", "deformed_professor",
    "religious_figure", "actor", "someone_from_the_future",
]


# ---------------------------------------------------------------------------
# Shared message builders
# ---------------------------------------------------------------------------


def _persona_scenario_messages(
    prompt: str, persona_name: str, persona_description: str, prompt_dir=None
) -> list[dict]:
    config = load_prompt("jailbreak", "persona_scenario_generation", prompt_dir)
    return build_messages(
        config,
        prompt=prompt,
        persona_name=persona_name,
        persona_description=persona_description,
    )


def _persona_construction_messages(
    prompt: str, persona_description: str, scenario: str, prompt_dir=None
) -> list[dict]:
    config = load_prompt("jailbreak", "jailbreak_construction", prompt_dir)
    return build_messages(
        config,
        jailbreak_type_name="Persona/Roleplay",
        jailbreak_type_description=persona_description,
        prompt=prompt,
        scenario=scenario,
    )


def _persona_gen(
    prompt: str,
    persona_name: str,
    persona_description: str,
    *,
    gen_model: str | None = None,
    scenario: str | None = None,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Yield persona-scenario then construction requests; return (jailbreak, info)."""
    if scenario is None:
        raw = yield LLMRequest(
            gen_model,
            _persona_scenario_messages(prompt, persona_name, persona_description, prompt_dir),
        )
        scenario = _extract_scenario(raw)
    jailbreak = yield LLMRequest(
        gen_model,
        _persona_construction_messages(prompt, persona_description, scenario, prompt_dir),
    )
    return jailbreak, f"category={persona_name}_persona"


# ---------------------------------------------------------------------------
# Tier 1: invented persona (delegates to the generic cognitive technique)
# ---------------------------------------------------------------------------


def to_invented_persona(prompt: str, **kwargs) -> TechniqueGen:
    """Persona jailbreak where the LLM freely invents the persona."""
    jailbreak, _info = yield from cognitive_gen(prompt, 0, **kwargs)
    return jailbreak, "category=invented_persona"


# ---------------------------------------------------------------------------
# Tier 2: named persona generators (built from taxonomy)
# ---------------------------------------------------------------------------


def _make_persona_fn(persona_name: str):
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        description = _PERSONAS[persona_name]["description"]
        result = yield from _persona_gen(prompt, persona_name, description, **kwargs)
        return result

    fn.__name__ = f"to_{persona_name}_persona"
    fn.__qualname__ = fn.__name__
    fn.__doc__ = f"Persona jailbreak: {_PERSONAS[persona_name].get('display_name', persona_name)}."
    return fn


# Bind one module-level generator per named persona.
_PERSONA_FNS = {name: _make_persona_fn(name) for name in _NAMED_PERSONAS}
globals().update({fn.__name__: fn for fn in _PERSONA_FNS.values()})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_persona_functions() -> list:
    """Return all persona jailbreak technique generators (1 invented + 14 named)."""
    return [to_invented_persona] + [_PERSONA_FNS[name] for name in _NAMED_PERSONAS]
