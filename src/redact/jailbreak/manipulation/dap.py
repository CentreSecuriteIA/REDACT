"""DAP techniques: the prompt as one question among benign Q&A pairs.

``info`` is ``num_shots=...;harmful_idx=...;sub_category=...``, plus
``selection=...`` when a model chose the subcategory. The many-shot variants
show ``MANY_PAIRS`` short pairs (``pairs.py``); ``wanted=...`` follows
``num_shots`` when the cache held fewer.
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


def _build_dap_prompt(
    harmful_prompt: str,
    benign_pairs: list[dict],
    num_shots: int,
    rng: random.Random | None = None,
) -> tuple[str, dict]:
    """Assemble DAP prompt with harmful question at random middle position.

    The question is never first, and last only where it cannot be otherwise:
    one shot puts it second (``harmful_idx`` 2), and no shot leaves it alone.

    Returns:
        (jailbreak_prompt, {"harmful_idx": harmful_idx}), harmful_idx 1-based.
    """
    rng = rng or random
    shots = rng.sample(benign_pairs, min(num_shots, len(benign_pairs)))
    if not shots:
        return f"Q1: {harmful_prompt}\nA1:\n", {"harmful_idx": 1}
    # Insert in the middle: never first (1), never last (len+1), which with one
    # shot leaves nowhere: the question follows it.
    harmful_idx = rng.randint(2, max(2, len(shots)))

    lines: list[str] = []
    idx = 1
    shot_iter = iter(shots)
    for pos in range(1, len(shots) + 2):  # +1 slot for the harmful prompt
        if pos == harmful_idx:
            lines.append(f"Q{idx}: {harmful_prompt}")
            lines.append(f"A{idx}:\n")
            idx += 1
        else:
            pair = next(shot_iter)
            lines.append(f"Q{idx}: {pair['prompt']}")
            lines.append(f"A{idx}: {pair['answer']}\n")
            idx += 1
    return "\n".join(lines), {"harmful_idx": harmful_idx}


def to_dap_random_short(
    prompt: str, benign_data: dict, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """DAP with randomly chosen category, short answers. No LLM needed."""
    return run_random_manipulation(
        prompt, benign_data, "short", _build_dap_prompt, rng)


def to_dap_random_long(
    prompt: str, benign_data: dict, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """DAP with randomly chosen category, long answers. No LLM needed."""
    return run_random_manipulation(
        prompt, benign_data, "long", _build_dap_prompt, rng)


def to_dap_selected_short(prompt: str, **kwargs) -> TechniqueGen:
    """DAP with LLM-selected category, short answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, "short", _build_dap_prompt, **kwargs))


def to_dap_selected_long(prompt: str, **kwargs) -> TechniqueGen:
    """DAP with LLM-selected category, long answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, "long", _build_dap_prompt, **kwargs))


def to_dap_random_many(
    prompt: str, benign_data: dict, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """DAP with many short pairs from every subcategory. No LLM needed."""
    return run_random_many(prompt, benign_data, _build_dap_prompt, rng)


def to_dap_selected_many(prompt: str, **kwargs) -> TechniqueGen:
    """DAP with many short pairs, the LLM-selected category's first
    (technique generator)."""
    return (yield from run_selected_many(prompt, _build_dap_prompt, **kwargs))


def get_dap_functions() -> list[Callable]:
    """Return all DAP technique functions."""
    return [to_dap_random_short, to_dap_random_long,
            to_dap_selected_short, to_dap_selected_long,
            to_dap_random_many, to_dap_selected_many]
