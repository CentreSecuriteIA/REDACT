"""Persona techniques: an invented persona or one of the 14 in
``configs/jailbreak/hacking/personas.json``.

``info`` is ``category=<name>_persona``, ``<name>`` the persona's key; the
prompts get its ``display_name``. A scenario reply without its scene is asked
again, then discarded, and a scenario the check rejects is made again once, then
discarded, as for the cognitive techniques.
"""

from collections.abc import Callable

from redact.jailbreak.directives import load_templates
from redact.jailbreak.technique import TechniqueGen, is_discarded
from redact.llm_pipeline import LLMRequest
from redact.llms.prompting import build_messages, load_prompt

from .cognitive import (
    _cognitive_gen,
    _construction_messages,
    _scenario_gen,
)

_PERSONAS: dict[str, dict] = load_templates("hacking", "personas.json")["categories"]

# Order of named personas in the registry (matches combination_spec.json names).
_NAMED_PERSONAS = [
    "psychopath", "alien", "radical_politician", "cult_leader",
    "very_advanced_ai", "cartel_leader", "artist", "mentally_ill",
    "deformed_scientist", "politician", "deformed_professor",
    "religious_figure", "actor", "someone_from_the_future",
]


def display_name(persona_name: str) -> str:
    """The name a persona's prompts call it: its ``display_name``, else its
    key in words (``cult_leader`` is ``Cult Leader``)."""
    return (_PERSONAS[persona_name].get("display_name")
            or persona_name.replace("_", " ").title())


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


def _persona_gen(
    prompt: str,
    persona_name: str,
    persona_description: str,
    *,
    gen_model: str | None = None,
    check_model: str | None = None,
    check_attempts: int = 2,
    scenario: str | None = None,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Yield the persona-scenario requests (``cognitive._scenario_gen``: the
    scenario and its check), then the construction request; return
    ``(jailbreak, info)``.

    The scenario prompt names the persona by :func:`display_name`, the info
    by its key. A scenario that no attempt parses or the check rejects every
    time returns the prompt with the discard info
    (:data:`~redact.jailbreak.hacking.cognitive.MALFORMED_SCENARIO`,
    ``scenario check failed``).
    """
    if scenario is None:
        scenario, discard = yield from _scenario_gen(
            gen_model,
            _persona_scenario_messages(
                prompt, display_name(persona_name), persona_description, prompt_dir),
            original=prompt, check_model=check_model,
            check_attempts=check_attempts, prompt_dir=prompt_dir)
        if scenario is None:
            return prompt, discard
    jailbreak = yield LLMRequest(
        gen_model,
        _construction_messages(
            prompt, "Persona/Roleplay", persona_description, scenario, prompt_dir),
    )
    return jailbreak, f"category={persona_name}_persona"


def to_invented_persona(prompt: str, **kwargs) -> TechniqueGen:
    """Persona jailbreak where the LLM freely invents the persona."""
    jailbreak, info = yield from _cognitive_gen(prompt, "persona_roleplay", **kwargs)
    return jailbreak, info if is_discarded(info) else "category=invented_persona"


def _make_persona_fn(persona_name: str):
    def fn(prompt: str, **kwargs) -> TechniqueGen:
        description = _PERSONAS[persona_name]["description"]
        result = yield from _persona_gen(prompt, persona_name, description, **kwargs)
        return result

    fn.__name__ = f"to_{persona_name}_persona"
    fn.__qualname__ = fn.__name__
    fn.__doc__ = f"Persona jailbreak: {display_name(persona_name)}."
    return fn


# One technique per named persona, named to_<persona>_persona: to_alien_persona, ...
_FNS = [_make_persona_fn(name) for name in _NAMED_PERSONAS]
globals().update({fn.__name__: fn for fn in _FNS})


def get_persona_functions() -> list[Callable]:
    """Return all persona jailbreak technique generators (1 invented + 14
    named, 4 of them retired by the spec: a kept plan still names them)."""
    return [to_invented_persona, *_FNS]
