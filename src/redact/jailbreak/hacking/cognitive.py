"""Cognitive techniques: a model writes a scenario, then the prompt built on it.

The five techniques are the categories of
``configs/jailbreak/hacking/cognitive_techniques.json``; ``info`` is
``category=<key>``.
"""

import re
from collections.abc import Callable

from redact.jailbreak.directives import load_templates
from redact.jailbreak.technique import TechniqueGen
from redact.llm_pipeline import LLMRequest
from redact.llms.prompting import build_messages, load_prompt

_TECHNIQUES: dict[str, dict] = load_templates(
    "hacking", "cognitive_techniques.json")["categories"]


_SCENARIO_RE = re.compile(
    r"\*{0,2}scenario description\*{0,2}[:\s]*[\s\*\n]*(.*)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_scenario(text: str) -> str:
    """Pull the 'Scenario Description' section from an LLM reply, else the whole text."""
    match = _SCENARIO_RE.search(text)
    return match.group(1).strip() if match else text.strip()


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

    A given ``scenario`` skips the scenario request.
    """
    if scenario is None:
        raw = yield LLMRequest(gen_model, _scenario_messages(prompt, prompt_dir))
        scenario = _extract_scenario(raw)
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
