"""Cognitive techniques: a model writes a scenario, then the prompt built on it.

The five techniques are the categories of
``configs/jailbreak/hacking/cognitive_techniques.json``; ``info`` is
``category=<key>``. A scenario reply without its scene is asked again, up to
:func:`~redact.llm_pipeline.extracted`'s attempts, then the prompt is
discarded (:data:`MALFORMED_SCENARIO`).
"""

import re
from collections.abc import Callable, Generator

from redact.jailbreak.directives import load_templates
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import LLMRequest, extracted
from redact.llms.prompting import build_messages, load_prompt

_TECHNIQUES: dict[str, dict] = load_templates(
    "hacking", "cognitive_techniques.json")["categories"]

#: The info of a technique whose scenario reply never held a scene.
MALFORMED_SCENARIO = f"{DISCARDED}; reason=malformed scenario"

# A line of three or more dashes: the scenario prompts put the scene behind it.
_DELIMITER = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)
_SCENARIO_RE = re.compile(
    r"\*{0,2}scenario description\*{0,2}[:\s]*[\s\*\n]*(.*)",
    re.DOTALL | re.IGNORECASE,
)
# The same heading at the start of a block.
_HEADING_RE = re.compile(r"\A[#\s]*" + _SCENARIO_RE.pattern, re.DOTALL | re.IGNORECASE)


def _extract_scenario(text: str) -> str | None:
    """The scene of a scenario reply, or None when it has none.

    The scenario prompts ask for the scene at the bottom, set apart by ``---``
    lines: it is the last block after the first delimiter, less a leading
    'Scenario Description' heading. A reply without a delimiter may mark the
    scene with that heading instead. A reply with neither, or an empty scene,
    has none: it is rejected whole, never taken as the scene.
    """
    blocks = [block.strip() for block in _DELIMITER.split(text)[1:]]
    blocks = [block for block in blocks if block]
    if blocks:
        heading = _HEADING_RE.match(blocks[-1])
        scene = heading.group(1) if heading else blocks[-1]
    else:
        match = _SCENARIO_RE.search(text)
        scene = match.group(1) if match else ""
    return scene.strip() or None


def _scenario_gen(
    gen_model: str | None, messages: list[dict],
) -> Generator[LLMRequest, str, str | None]:
    """Ask for a scenario until a reply holds a scene (:func:`_extract_scenario`);
    return the scene, or None when no attempt held one."""
    def scene_of(reply: str) -> list[str]:
        scene = _extract_scenario(reply)
        return [scene] if scene else []

    scenes, _attempts = yield from extracted(gen_model, lambda: messages, scene_of)
    return scenes[0] if scenes else None


def _scenario_messages(prompt: str, prompt_dir=None) -> list[dict]:
    config = load_prompt("jailbreak", "scenario_generation", prompt_dir)
    return build_messages(config, prompt=prompt)


def _construction_messages(
    prompt: str, type_name: str, type_description: str, scenario: str, prompt_dir=None
) -> list[dict]:
    config = load_prompt("jailbreak", "jailbreak_construction", prompt_dir)
    return build_messages(
        config,
        jailbreak_type_name=type_name,
        jailbreak_type_description=type_description,
        prompt=prompt,
        scenario=scenario,
    )


def _cognitive_gen(
    prompt: str,
    key: str,
    *,
    gen_model: str | None = None,
    scenario: str | None = None,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Yield the scenario request, then the construction request for the
    technique ``key`` of the taxonomy; return ``(jailbreak, info)``.

    A given ``scenario`` skips the scenario request. A scenario that no
    attempt parses returns the prompt with :data:`MALFORMED_SCENARIO`.
    """
    if scenario is None:
        scenario = yield from _scenario_gen(
            gen_model, _scenario_messages(prompt, prompt_dir))
        if scenario is None:
            return prompt, MALFORMED_SCENARIO
    technique = _TECHNIQUES[key]
    jailbreak = yield LLMRequest(gen_model, _construction_messages(
        prompt, technique["display_name"], technique["description"],
        scenario, prompt_dir))
    return jailbreak, f"category={key}"


def to_persona_roleplay(prompt: str, **kwargs) -> TechniqueGen:
    """Persona/Roleplay jailbreak."""
    return (yield from _cognitive_gen(prompt, "persona_roleplay", **kwargs))


def to_hypothetical_framing(prompt: str, **kwargs) -> TechniqueGen:
    """Hypothetical Framing jailbreak."""
    return (yield from _cognitive_gen(prompt, "hypothetical_framing", **kwargs))


def to_authority_obedience(prompt: str, **kwargs) -> TechniqueGen:
    """Authority & Obedience jailbreak."""
    return (yield from _cognitive_gen(prompt, "authority_obedience", **kwargs))


def to_avi(prompt: str, **kwargs) -> TechniqueGen:
    """Anthropomorphic Vulnerability Inheritance (AVI) jailbreak."""
    return (yield from _cognitive_gen(prompt, "avi", **kwargs))


def to_deep_inception(prompt: str, **kwargs) -> TechniqueGen:
    """DeepInception jailbreak."""
    return (yield from _cognitive_gen(prompt, "deep_inception", **kwargs))


def get_cognitive_functions() -> list[Callable]:
    """Return all cognitive technique functions."""
    return [
        to_persona_roleplay,
        to_hypothetical_framing,
        to_authority_obedience,
        to_avi,
        to_deep_inception,
    ]
