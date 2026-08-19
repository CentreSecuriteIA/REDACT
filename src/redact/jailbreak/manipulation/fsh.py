"""Few-Shot Hacking (FSH) manipulation techniques.

FSH places N benign Q&A pairs before the harmful question, priming
the model to continue the helpful pattern.

Ported from reference manipulation.py lines 340-426.
"""

import random
from collections.abc import Callable
from pathlib import Path

from redact.jailbreak.protocol import LLMRequest, TechniqueGen
from redact.llms.base import LLMBackend
from redact.llms.calls import generate_sample
from redact.llms.prompts import build_messages, load_prompt
from redact.llms.wrappers import RateLimiter

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
# Shared random/selected × short/long runner (also used by dap.py)
# ---------------------------------------------------------------------------
#
# FSH and DAP differ only in (a) how the prompt is assembled (a plain string
# vs. a string + a harmful_idx position) and (b) whether that extra field
# appears in additional_info. build_prompt normalizes both shapes to
# (jailbreak_text, extra_info_fields) so this runner covers every
# random/selected × short/long variant for both technique families.


def _pick_pairs(benign_data: dict, answer_type: str, subcat: str) -> list[dict]:
    """Benign Q&A pairs for one subcategory + answer-length tier ("short"/"long")."""
    return benign_data[f"by_subcat_{answer_type}"].get(subcat, benign_data[answer_type])


def _format_manipulation_info(
    num_shots: int, extra: dict, subcat: str, selection: str | None = None
) -> str:
    """additional_info string shared by every FSH/DAP random/selected variant."""
    parts = [f"num_shots={num_shots}"]
    parts.extend(f"{k}={v}" for k, v in extra.items())
    parts.append(f"sub_category={subcat}")
    if selection is not None:
        parts.append(f"selection={selection}")
    return ";".join(parts)


def run_random_manipulation(
    prompt: str,
    benign_data: dict,
    answer_type: str,
    build_prompt: Callable[[str, list[dict], int], tuple[str, dict]],
) -> tuple[str, str]:
    """Shared body for a manipulation technique's ``*_random_{short,long}`` variant. No LLM needed."""
    num_shots = random.randint(3, 7)
    subcat = random.choice(benign_data["all_subcategories"])
    pairs = _pick_pairs(benign_data, answer_type, subcat)
    jailbreak, extra = build_prompt(prompt, pairs, num_shots)
    return jailbreak, _format_manipulation_info(num_shots, extra, subcat)


def run_selected_manipulation(
    prompt: str,
    benign_data: dict,
    answer_type: str,
    build_prompt: Callable[[str, list[dict], int], tuple[str, dict]],
    *,
    gen_model: str | None = None,
    prompt_dir=None,
) -> TechniqueGen:
    """Shared body for a manipulation technique's ``*_selected_{short,long}`` variant."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = yield from _select_subcategory_gen(
        prompt, benign_data["all_subcategories"], gen_model=gen_model, prompt_dir=prompt_dir
    )
    pairs = _pick_pairs(benign_data, answer_type, subcat)
    jailbreak, extra = build_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, _format_manipulation_info(num_shots, extra, subcat, selection)


# ---------------------------------------------------------------------------
# Technique functions
# ---------------------------------------------------------------------------


def _fsh_build_prompt(harmful_prompt: str, pairs: list[dict], num_shots: int) -> tuple[str, dict]:
    """Adapt _build_fsh_prompt's (str) return to the shared runner's (str, extra) shape."""
    return _build_fsh_prompt(harmful_prompt, pairs, num_shots), {}


def to_fsh_random_short(prompt: str, benign_data: dict) -> tuple[str, str]:
    """FSH with randomly chosen category, short answers. No LLM needed."""
    return run_random_manipulation(prompt, benign_data, "short", _fsh_build_prompt)


def to_fsh_random_long(prompt: str, benign_data: dict) -> tuple[str, str]:
    """FSH with randomly chosen category, long answers. No LLM needed."""
    return run_random_manipulation(prompt, benign_data, "long", _fsh_build_prompt)


def to_fsh_selected_short(
    prompt: str, benign_data: dict | None = None, *,
    gen_model: str | None = None, prompt_dir=None, **kwargs,
) -> TechniqueGen:
    """FSH with LLM-selected category, short answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, benign_data, "short", _fsh_build_prompt,
        gen_model=gen_model, prompt_dir=prompt_dir,
    ))


def to_fsh_selected_long(
    prompt: str, benign_data: dict | None = None, *,
    gen_model: str | None = None, prompt_dir=None, **kwargs,
) -> TechniqueGen:
    """FSH with LLM-selected category, long answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, benign_data, "long", _fsh_build_prompt,
        gen_model=gen_model, prompt_dir=prompt_dir,
    ))


def get_fsh_functions() -> list:
    """Return all FSH technique functions."""
    return [to_fsh_random_short, to_fsh_random_long,
            to_fsh_selected_short, to_fsh_selected_long]
