"""Few-Shot Hacking (FSH) manipulation techniques.

FSH places N benign Q&A pairs before the harmful question, priming
the model to continue the helpful pattern.

Ported from reference manipulation.py lines 340-426.
"""

import random
from pathlib import Path

from redact.jailbreak.protocol import LLMRequest, TechniqueGen
from redact.llms.base import LLMBackend
from redact.llms.calls import generate_sample
from redact.llms.wrappers import RateLimiter
from redact.llms.prompts import load_prompt, build_messages


# ---------------------------------------------------------------------------
# LLM-based category selection (shared with dap.py)
# ---------------------------------------------------------------------------


def _build_subcategory_messages(
    harmful_prompt: str, all_subcategories: list[str], prompt_dir=None
) -> list[dict]:
    config = load_prompt("jailbreak", "category_selection", prompt_dir)
    subcategory_list = "\n".join(f"- {sc}" for sc in all_subcategories)
    return build_messages(
        config, harmful_prompt=harmful_prompt, subcategory_list=subcategory_list
    )


def _parse_subcategory(chosen: str, all_subcategories: list[str]) -> tuple[str, bool]:
    """Map an LLM reply to a known subcategory, falling back to random."""
    chosen = chosen.strip()
    if chosen in all_subcategories:
        return chosen, False
    for sc in all_subcategories:
        if sc.lower() in chosen.lower():
            return sc, False
    return random.choice(all_subcategories), True


def select_best_subcategory(
    harmful_prompt: str,
    all_subcategories: list[str],
    backend: LLMBackend,
    model: str,
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | Path | None = None,
) -> tuple[str, bool]:
    """Use LLM to select the benign subcategory most similar to harmful prompt (sync).

    Returns:
        (subcategory, was_fallback) where was_fallback is True if the
        LLM response couldn't be parsed and a random category was used.
    """
    messages = _build_subcategory_messages(harmful_prompt, all_subcategories, prompt_dir)
    chosen = generate_sample(backend, model, messages, rate_limiter)
    return _parse_subcategory(chosen, all_subcategories)


def _select_subcategory_gen(
    harmful_prompt: str,
    all_subcategories: list[str],
    *,
    gen_model: str | None = None,
    prompt_dir=None,
) -> TechniqueGen:
    """Yield one category-selection request; return ``(subcategory, was_fallback)``."""
    chosen = yield LLMRequest(
        gen_model, _build_subcategory_messages(harmful_prompt, all_subcategories, prompt_dir)
    )
    return _parse_subcategory(chosen, all_subcategories)


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
    prompt: str, benign_data: dict | None = None, *,
    gen_model: str | None = None, prompt_dir=None, **kwargs,
) -> TechniqueGen:
    """FSH with LLM-selected category, short answers (technique generator)."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = yield from _select_subcategory_gen(
        prompt, benign_data["all_subcategories"], gen_model=gen_model, prompt_dir=prompt_dir
    )
    pairs = benign_data["by_subcat_short"].get(subcat, benign_data["short"])
    jailbreak = _build_fsh_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, f"num_shots={num_shots};sub_category={subcat};selection={selection}"


def to_fsh_selected_long(
    prompt: str, benign_data: dict | None = None, *,
    gen_model: str | None = None, prompt_dir=None, **kwargs,
) -> TechniqueGen:
    """FSH with LLM-selected category, long answers (technique generator)."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = yield from _select_subcategory_gen(
        prompt, benign_data["all_subcategories"], gen_model=gen_model, prompt_dir=prompt_dir
    )
    pairs = benign_data["by_subcat_long"].get(subcat, benign_data["long"])
    jailbreak = _build_fsh_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, f"num_shots={num_shots};sub_category={subcat};selection={selection}"


def get_fsh_functions() -> list:
    """Return all FSH technique functions."""
    return [to_fsh_random_short, to_fsh_random_long,
            to_fsh_selected_short, to_fsh_selected_long]
