"""DAP (Distract and Persuade / Attention Shifting) techniques.

The harmful question is hidden among benign Q&A pairs at a random
middle position with an empty answer placeholder.

Ported from reference manipulation.py lines 356-468.
"""

import random

from redact.llms.base import LLMBackend
from redact.llms.wrappers import RateLimiter
from .fsh import select_best_subcategory


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_dap_prompt(
    harmful_prompt: str,
    benign_pairs: list[dict],
    num_shots: int,
) -> tuple[str, int]:
    """Assemble DAP prompt with harmful question at random middle position.

    Returns:
        (jailbreak_prompt, harmful_idx) where harmful_idx is 1-based.
    """
    shots = random.sample(benign_pairs, min(num_shots, len(benign_pairs)))
    # Insert in the middle: never first (1), never last (len+1)
    harmful_idx = random.randint(2, max(2, len(shots)))

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
    return "\n".join(lines), harmful_idx


# ---------------------------------------------------------------------------
# Technique functions
# ---------------------------------------------------------------------------


def to_dap_random_short(prompt: str, benign_data: dict) -> tuple[str, str]:
    """DAP with randomly chosen category, short answers. No LLM needed."""
    num_shots = random.randint(3, 7)
    subcat = random.choice(benign_data["all_subcategories"])
    pairs = benign_data["by_subcat_short"].get(subcat, benign_data["short"])
    jailbreak, harmful_idx = _build_dap_prompt(prompt, pairs, num_shots)
    return jailbreak, f"num_shots={num_shots};harmful_idx={harmful_idx};sub_category={subcat}"


def to_dap_random_long(prompt: str, benign_data: dict) -> tuple[str, str]:
    """DAP with randomly chosen category, long answers. No LLM needed."""
    num_shots = random.randint(3, 7)
    subcat = random.choice(benign_data["all_subcategories"])
    pairs = benign_data["by_subcat_long"].get(subcat, benign_data["long"])
    jailbreak, harmful_idx = _build_dap_prompt(prompt, pairs, num_shots)
    return jailbreak, f"num_shots={num_shots};harmful_idx={harmful_idx};sub_category={subcat}"


def to_dap_selected_short(
    prompt: str,
    benign_data: dict,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """DAP with LLM-selected category, short answers."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = select_best_subcategory(
        prompt, benign_data["all_subcategories"], backend, model, rate_limiter
    )
    pairs = benign_data["by_subcat_short"].get(subcat, benign_data["short"])
    jailbreak, harmful_idx = _build_dap_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, f"num_shots={num_shots};harmful_idx={harmful_idx};sub_category={subcat};selection={selection}"


def to_dap_selected_long(
    prompt: str,
    benign_data: dict,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """DAP with LLM-selected category, long answers."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = select_best_subcategory(
        prompt, benign_data["all_subcategories"], backend, model, rate_limiter
    )
    pairs = benign_data["by_subcat_long"].get(subcat, benign_data["long"])
    jailbreak, harmful_idx = _build_dap_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, f"num_shots={num_shots};harmful_idx={harmful_idx};sub_category={subcat};selection={selection}"


def get_dap_functions() -> list:
    """Return all DAP technique functions."""
    return [to_dap_random_short, to_dap_random_long,
            to_dap_selected_short, to_dap_selected_long]
