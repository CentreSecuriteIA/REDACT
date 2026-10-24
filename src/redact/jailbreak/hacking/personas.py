"""Persona techniques: an invented persona or one of the 17 in
``configs/jailbreak/hacking/personas.json`` (4 of them retired by the spec).

``info`` is ``category=<name>_persona``, ``<name>`` the persona's key (or
``invented``); the prompts get its ``display_name``. A scenario reply without
its scene is asked again, then discarded, and a scenario the check rejects is
made again once, then discarded, as for the cognitive techniques.

An invented persona has a first step: the model is asked for a name and a
description for the prompt (``persona_invention``), as ``Name: ...`` then
``Description: ...``, and asked again, up to 3 replies, for any other form
(:func:`_persona_of`); a prompt whose replies never took it is discarded
(:data:`MALFORMED_PERSONA`). The persona scenario then goes as for a named
persona, with the invented name and description.
"""

import re
from collections.abc import Callable

from redact.jailbreak.directives import load_templates
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import LLMRequest, extracted
from redact.llms.prompting import build_messages, load_prompt

from .cognitive import (
    _construction_messages,
    _scenario_gen,
)

_PERSONAS: dict[str, dict] = load_templates("hacking", "personas.json")["categories"]

#: The info of an invented persona whose replies never held a name and a description.
MALFORMED_PERSONA = f"{DISCARDED}; reason=malformed persona"

# A name on one line, then a description (which may run over lines).
_PERSONA_REPLY = re.compile(
    r"\s*Name:[ \t]*(?P<name>\S[^\n]*?)[ \t]*\n+[ \t]*Description:[ \t]*"
    r"(?P<description>\S.*?)\s*", re.DOTALL)
# A label at the start of a line: a second persona in the reply.
_LABEL = re.compile(r"^[ \t]*(?:Name|Description):", re.MULTILINE)

# Order of named personas in the registry (matches combination_spec.json names).
_NAMED_PERSONAS = [
    "psychopath", "alien", "radical_politician", "cult_leader",
    "very_advanced_ai", "cartel_leader", "artist", "mentally_ill",
    "deformed_scientist", "politician", "deformed_professor",
    "religious_figure", "actor", "someone_from_the_future",
    "mercenary", "underground_hacker", "anarchist_philosopher",
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


def _persona_of(reply: str) -> list[str]:
    """``[name, description]`` of a reply that is a ``Name:`` line, then a
    ``Description:``, and nothing before, between or after them but blank
    lines; ``[]`` for any other form (markdown labels, a missing or empty
    field, two personas), which :func:`~redact.llm_pipeline.extracted` asks
    for again."""
    match = _PERSONA_REPLY.fullmatch(reply.replace("\r\n", "\n"))
    if match is None or _LABEL.search(match["description"]):
        return []
    return [match["name"], match["description"]]


def _persona_gen(
    prompt: str,
    persona_name: str,
    persona_description: str,
    *,
    shown_as: str | None = None,
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

    The scenario prompt names the persona by ``shown_as``, else by
    :func:`display_name`, the info by its key. A scenario that no attempt
    parses or the check rejects every time returns the prompt with the discard
    info (:data:`~redact.jailbreak.hacking.cognitive.MALFORMED_SCENARIO`,
    ``scenario check failed``).
    """
    if scenario is None:
        scenario, discard = yield from _scenario_gen(
            gen_model,
            _persona_scenario_messages(
                prompt, shown_as or display_name(persona_name), persona_description,
                prompt_dir),
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


def _invented_gen(
    prompt: str, *, gen_model: str | None = None, prompt_dir=None, **kwargs,
) -> TechniqueGen:
    """Ask for a persona for the prompt until a reply is a name and a
    description (:func:`_persona_of`, up to 3 replies), then make
    :func:`_persona_gen`'s scenario and construction with them."""
    messages = build_messages(
        load_prompt("jailbreak", "persona_invention", prompt_dir), prompt=prompt)
    persona, _attempts = yield from extracted(
        gen_model, lambda: messages, _persona_of)
    if not persona:
        return prompt, MALFORMED_PERSONA
    name, description = persona
    return (yield from _persona_gen(
        prompt, "invented", description, shown_as=name, gen_model=gen_model,
        prompt_dir=prompt_dir, **kwargs))


def to_invented_persona(prompt: str, **kwargs) -> TechniqueGen:
    """Persona jailbreak where the model first invents the persona for the
    prompt: a name and a description."""
    return (yield from _invented_gen(prompt, **kwargs))


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
    """Return all persona jailbreak technique generators (1 invented + 17
    named, 4 of them retired by the spec: a kept plan still names them)."""
    return [to_invented_persona, *_FNS]
