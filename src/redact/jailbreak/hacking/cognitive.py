"""Cognitive techniques: a model writes a scenario, a check model reads it
against the prompt, then the prompt is built on it.

The five techniques are the categories of
``configs/jailbreak/hacking/cognitive_techniques.json``; ``info`` is
``category=<key>``. A scenario reply without its scene is asked again, up to
:func:`~redact.llm_pipeline.extracted`'s attempts, then the prompt is
discarded (:data:`MALFORMED_SCENARIO`); a scenario the check rejects is made
again once with the verdict, then discarded (``reason=scenario check
failed``).
"""

import re
from collections.abc import Callable, Generator
from functools import partial

from redact.content_moderation.feedback import with_feedback
from redact.jailbreak.directives import load_templates
from redact.jailbreak.technique import DISCARDED, TechniqueGen
from redact.llm_pipeline import CHECK_TEMPERATURE, LLMRequest, extracted, is_accepted
from redact.llms.prompting import PromptTemplate, build_messages, load_prompt

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
    gen_model: str | None,
    messages: list[dict],
    *,
    original: str,
    check_model: str | None = None,
    check_attempts: int = 2,
    prompt_dir=None,
) -> Generator[LLMRequest, str, tuple[str | None, str]]:
    """Ask for a scenario until a reply holds a scene (:func:`_extract_scenario`)
    that ``check_model`` (``gen_model`` when None) accepts against ``original``
    (``jailbreak/scenario_check``, at :data:`CHECK_TEMPERATURE`).

    A reply without a scene is asked for again, up to 3 replies. A scene the
    check rejects is asked for again with the verdict as feedback, up to
    ``check_attempts`` scenes, so the scenario and its check cost 2 to
    ``4 * check_attempts`` calls. Returns ``(scene, "")``, or ``(None, info)``
    with the ``DISCARDED`` info when no attempt held a scene
    (:data:`MALFORMED_SCENARIO`) or every scene was rejected.
    """
    if check_attempts < 1:
        raise ValueError(f"check_attempts must be at least 1, got {check_attempts}.")

    def scene_of(reply: str) -> list[str]:
        scene = _extract_scenario(reply)
        return [scene] if scene else []

    check = PromptTemplate.load("jailbreak", "scenario_check", prompt_dir=prompt_dir)
    feedback = ""
    for _ in range(check_attempts):
        scenes, _attempts = yield from extracted(
            gen_model, partial(with_feedback, messages, feedback), scene_of)
        if not scenes:
            return None, MALFORMED_SCENARIO
        verdict = yield LLMRequest(
            check_model or gen_model,
            check(original_prompt=original, scenario=scenes[0]),
            temperature=CHECK_TEMPERATURE)
        if is_accepted(verdict):
            return scenes[0], ""
        feedback = verdict
    return None, f"{DISCARDED}; reason=scenario check failed; feedback={verdict}"


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
    check_model: str | None = None,
    check_attempts: int = 2,
    scenario: str | None = None,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Yield the scenario requests (:func:`_scenario_gen`), then the
    construction request for the technique ``key`` of the taxonomy; return
    ``(jailbreak, info)``.

    A given ``scenario`` skips the scenario and its check. A scenario that no
    attempt parses or the check rejects every time returns the prompt with the
    discard info (:data:`MALFORMED_SCENARIO`, ``scenario check failed``).
    """
    if scenario is None:
        scenario, discard = yield from _scenario_gen(
            gen_model, _scenario_messages(prompt, prompt_dir), original=prompt,
            check_model=check_model, check_attempts=check_attempts,
            prompt_dir=prompt_dir)
        if scenario is None:
            return prompt, discard
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
