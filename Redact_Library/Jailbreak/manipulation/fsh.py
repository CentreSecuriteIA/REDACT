"""Few-Shot Hacking (FSH) manipulation techniques.

FSH places N benign Q&A pairs before the harmful question, priming
the model to continue the helpful pattern.

Ported from reference manipulation.py lines 340-426.
"""

import random
from pathlib import Path

from Redact_Library.LLMs.base import LLMBackend
from Redact_Library.LLMs.calls import generate_sample
from Redact_Library.LLMs.wrappers import RateLimiter
from Redact_Library.LLMs.prompts import load_prompt, build_messages


# ---------------------------------------------------------------------------
# LLM-based category selection (shared with dap.py)
# ---------------------------------------------------------------------------


def select_best_subcategory(
    harmful_prompt: str,
    all_subcategories: list[str],
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, bool]:
    """Use LLM to select the benign subcategory most similar to harmful prompt.

    Returns:
        (subcategory, was_fallback) where was_fallback is True if the
        LLM response couldn't be parsed and a random category was used.
    """
    config = load_prompt("Jailbreak", "category_selection", prompt_dir)
    subcategory_list = "\n".join(f"- {sc}" for sc in all_subcategories)
    messages = build_messages(
        config,
        harmful_prompt=harmful_prompt,
        subcategory_list=subcategory_list,
    )
    chosen = generate_sample(backend, model, messages, rate_limiter).strip()

    # Exact match
    if chosen in all_subcategories:
        return chosen, False
    # Fuzzy: check if the response contains a known subcategory
    for sc in all_subcategories:
        if sc.lower() in chosen.lower():
            return sc, False
    # Fallback: random
    return random.choice(all_subcategories), True


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _build_fsh_prompt(
    harmful_prompt: str,
    benign_pairs: list[dict],
    num_shots: int,
) -> str:
    """Assemble FSH prompt: N benign Q&A then harmful question."""
    shots = random.sample(benign_pairs, min(num_shots, len(benign_pairs)))
    lines: list[str] = []
    for i, pair in enumerate(shots, 1):
        lines.append(f"Q{i}: {pair['prompt']}")
        lines.append(f"A{i}: {pair['answer']}\n")
    lines.append(f"Q{len(shots) + 1}: {harmful_prompt}")
    lines.append(f"A{len(shots) + 1}:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Technique functions
# ---------------------------------------------------------------------------


def to_fsh_random_short(prompt: str, benign_data: dict) -> tuple[str, str]:
    """FSH with randomly chosen category, short answers. No LLM needed."""
    num_shots = random.randint(3, 7)
    subcat = random.choice(benign_data["all_subcategories"])
    pairs = benign_data["by_subcat_short"].get(subcat, benign_data["short"])
    jailbreak = _build_fsh_prompt(prompt, pairs, num_shots)
    return jailbreak, f"num_shots={num_shots};sub_category={subcat}"


def to_fsh_random_long(prompt: str, benign_data: dict) -> tuple[str, str]:
    """FSH with randomly chosen category, long answers. No LLM needed."""
    num_shots = random.randint(3, 7)
    subcat = random.choice(benign_data["all_subcategories"])
    pairs = benign_data["by_subcat_long"].get(subcat, benign_data["long"])
    jailbreak = _build_fsh_prompt(prompt, pairs, num_shots)
    return jailbreak, f"num_shots={num_shots};sub_category={subcat}"


def to_fsh_selected_short(
    prompt: str,
    benign_data: dict,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """FSH with LLM-selected category, short answers."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = select_best_subcategory(
        prompt, benign_data["all_subcategories"], backend, model, rate_limiter
    )
    pairs = benign_data["by_subcat_short"].get(subcat, benign_data["short"])
    jailbreak = _build_fsh_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, f"num_shots={num_shots};sub_category={subcat};selection={selection}"


def to_fsh_selected_long(
    prompt: str,
    benign_data: dict,
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> tuple[str, str]:
    """FSH with LLM-selected category, long answers."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = select_best_subcategory(
        prompt, benign_data["all_subcategories"], backend, model, rate_limiter
    )
    pairs = benign_data["by_subcat_long"].get(subcat, benign_data["long"])
    jailbreak = _build_fsh_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, f"num_shots={num_shots};sub_category={subcat};selection={selection}"


def get_fsh_functions() -> list:
    """Return all FSH technique functions."""
    return [to_fsh_random_short, to_fsh_random_long,
            to_fsh_selected_short, to_fsh_selected_long]
