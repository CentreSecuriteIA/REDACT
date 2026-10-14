"""The benign pairs both manipulation families build on: a subcategory, picked at
random or by a model, its pairs, and the info string.

A family supplies ``build_prompt(prompt, pairs, num_shots) -> (text, extra)``,
where ``extra`` holds its own info fields.
"""

import random
from collections.abc import Callable

from redact.llm_pipeline import LLMRequest
from redact.llms.prompting import build_messages, load_prompt

from ..technique import TechniqueGen


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


def _select_subcategory_gen(
    harmful_prompt: str,
    all_subcategories: list[str],
    *,
    gen_model: str | None = None,
    prompt_dir=None,
) -> TechniqueGen:
    """Yield one category-selection request; return ``(subcategory, was_fallback)``."""
    chosen = yield LLMRequest(gen_model, _build_subcategory_messages(
        harmful_prompt, all_subcategories, prompt_dir))
    return _parse_subcategory(chosen, all_subcategories)


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
    """The body of a ``*_random_{short,long}`` variant; no model call."""
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
    """The body of a ``*_selected_{short,long}`` variant, subcategory by a model."""
    num_shots = random.randint(3, 7)
    subcat, was_fallback = yield from _select_subcategory_gen(
        prompt, benign_data["all_subcategories"],
        gen_model=gen_model, prompt_dir=prompt_dir,
    )
    pairs = _pick_pairs(benign_data, answer_type, subcat)
    jailbreak, extra = build_prompt(prompt, pairs, num_shots)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, _format_manipulation_info(num_shots, extra, subcat, selection)
