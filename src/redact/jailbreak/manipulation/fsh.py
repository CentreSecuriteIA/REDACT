"""Few-shot (FSH) techniques: benign Q&A pairs, then the prompt as the last question.

``info`` is ``num_shots=...;sub_category=...``, plus ``selection=...`` when a
model chose the subcategory. The many-shot variants show ``MANY_PAIRS`` short
pairs (``pairs.py``); ``wanted=...`` follows ``num_shots`` when the cache held
fewer.
"""

import random
from collections.abc import Callable

from redact.jailbreak.technique import TechniqueGen

from .pairs import (
    run_random_manipulation,
    run_random_many,
    run_selected_manipulation,
    run_selected_many,
)


def _build_fsh_prompt(
    harmful_prompt: str,
    benign_pairs: list[dict],
    num_shots: int,
    rng: random.Random | None = None,
) -> tuple[str, dict]:
    """Assemble FSH prompt: N benign Q&A then harmful question; no extra info."""
    shots = (rng or random).sample(benign_pairs, min(num_shots, len(benign_pairs)))
    lines: list[str] = []
    for i, pair in enumerate(shots, 1):
        lines.append(f"Q{i}: {pair['prompt']}")
        lines.append(f"A{i}: {pair['answer']}\n")
    lines.append(f"Q{len(shots) + 1}: {harmful_prompt}")
    lines.append(f"A{len(shots) + 1}:")
    return "\n".join(lines), {}


def to_fsh_random_short(
    prompt: str, benign_data: dict, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """FSH with randomly chosen category, short answers. No LLM needed."""
    return run_random_manipulation(
        prompt, benign_data, "short", _build_fsh_prompt, rng)


def to_fsh_random_long(
    prompt: str, benign_data: dict, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """FSH with randomly chosen category, long answers. No LLM needed."""
    return run_random_manipulation(
        prompt, benign_data, "long", _build_fsh_prompt, rng)


def to_fsh_selected_short(prompt: str, **kwargs) -> TechniqueGen:
    """FSH with LLM-selected category, short answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, "short", _build_fsh_prompt, **kwargs))


def to_fsh_selected_long(prompt: str, **kwargs) -> TechniqueGen:
    """FSH with LLM-selected category, long answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, "long", _build_fsh_prompt, **kwargs))


def to_fsh_random_many(
    prompt: str, benign_data: dict, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """FSH with many short pairs from every subcategory. No LLM needed."""
    return run_random_many(prompt, benign_data, _build_fsh_prompt, rng)


def to_fsh_selected_many(prompt: str, **kwargs) -> TechniqueGen:
    """FSH with many short pairs, the LLM-selected category's first
    (technique generator)."""
    return (yield from run_selected_many(prompt, _build_fsh_prompt, **kwargs))


def get_fsh_functions() -> list[Callable]:
    """Return all FSH technique functions."""
    return [to_fsh_random_short, to_fsh_random_long,
            to_fsh_selected_short, to_fsh_selected_long]
