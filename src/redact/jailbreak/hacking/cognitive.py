"""Cognitive/psychological hacking techniques.

Two-step generation: scenario -> jailbreak construction (two LLM rounds).

The public ``to_*`` functions are **technique generators** (see
``jailbreak/protocol.py``): they yield a scenario request, then a construction
request, so the batched engine advances both rounds across many samples at
once. The sync helpers ``get_situation`` / ``create_jailbreak`` are retained
for standalone/single-call use; both paths share the message builders and
scenario extractor below.

Technique definitions are loaded from configs/taxonomy/cognitive_techniques.json
so they can be edited without touching code.
"""

import re
from pathlib import Path

from redact.dataset.taxonomy import load_taxonomy
from redact.jailbreak.protocol import LLMRequest, TechniqueGen
from redact.llms.base import LLMBackend
from redact.llms.calls import generate_sample
from redact.llms.prompts import build_messages, load_prompt
from redact.llms.wrappers import RateLimiter

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
# Shared builders (used by both the sync helpers and the technique generators)
# ---------------------------------------------------------------------------

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
    prompt: str, category_index: int, scenario: str, prompt_dir=None
) -> list[dict]:
    category_name, category_description = HACKING_CATEGORIES[category_index]
    config = load_prompt("jailbreak", "jailbreak_construction", prompt_dir)
    return build_messages(
        config,
        jailbreak_type_name=category_name,
        jailbreak_type_description=category_description,
        prompt=prompt,
        scenario=scenario,
    )


# ---------------------------------------------------------------------------
# Sync helpers (single-call use; retained for tests and standalone callers)
# ---------------------------------------------------------------------------


def get_situation(
    prompt: str,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> str:
    """Generate a plausible scenario for the given prompt (single LLM call)."""
    situation = generate_sample(
        backend, model, _scenario_messages(prompt, prompt_dir), rate_limiter
    )
    return _extract_scenario(situation)


def create_jailbreak(
    prompt: str,
    category_index: int,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
    scenario: str | None = None,
) -> tuple[str, str, str]:
    """Two-step jailbreak (sync): scenario generation then construction.

    Returns (jailbreak_prompt, additional_info, scenario). The batched path
    uses :func:`cognitive_gen` instead.
    """
    if scenario is None:
        scenario = get_situation(prompt, backend, model, rate_limiter, prompt_dir)

    messages = _construction_messages(prompt, category_index, scenario, prompt_dir)
    jailbreak = generate_sample(backend, model, messages, rate_limiter)
    snake_name = CATEGORY_NAMES[category_index]
    return jailbreak, f"category={snake_name}", scenario


# ---------------------------------------------------------------------------
# Technique generator (batched path)
# ---------------------------------------------------------------------------


def cognitive_gen(
    prompt: str,
    category_index: int,
    *,
    gen_model: str | None = None,
    scenario: str | None = None,
    prompt_dir=None,
    **kwargs,
) -> TechniqueGen:
    """Yield scenario then construction requests; return ``(jailbreak, info)``.

    If ``scenario`` is provided (e.g. cached), the scenario round is skipped.
    """
    if scenario is None:
        raw = yield LLMRequest(gen_model, _scenario_messages(prompt, prompt_dir))
        scenario = _extract_scenario(raw)
    jailbreak = yield LLMRequest(
        gen_model, _construction_messages(prompt, category_index, scenario, prompt_dir)
    )
    return jailbreak, f"category={CATEGORY_NAMES[category_index]}"


# ---------------------------------------------------------------------------
# Per-technique public functions (technique generators)
# ---------------------------------------------------------------------------


def to_persona_roleplay(prompt: str, **kwargs) -> TechniqueGen:
    """Persona/Roleplay jailbreak."""
    return (yield from cognitive_gen(prompt, 0, **kwargs))


def to_hypothetical_framing(prompt: str, **kwargs) -> TechniqueGen:
    """Hypothetical Framing jailbreak."""
    return (yield from cognitive_gen(prompt, 1, **kwargs))


def to_authority_obedience(prompt: str, **kwargs) -> TechniqueGen:
    """Authority & Obedience jailbreak."""
    return (yield from cognitive_gen(prompt, 2, **kwargs))


def to_avi(prompt: str, **kwargs) -> TechniqueGen:
    """Anthropomorphic Vulnerability Inheritance (AVI) jailbreak."""
    return (yield from cognitive_gen(prompt, 3, **kwargs))


def to_deep_inception(prompt: str, **kwargs) -> TechniqueGen:
    """DeepInception jailbreak."""
    return (yield from cognitive_gen(prompt, 4, **kwargs))


def get_hacking_functions() -> list:
    """Return all hacking technique functions."""
    return [
        to_persona_roleplay,
        to_hypothetical_framing,
        to_authority_obedience,
        to_avi,
        to_deep_inception,
    ]
