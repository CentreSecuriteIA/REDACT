"""The benign pairs both manipulation families build on: a subcategory, picked at
random or by a model, its pairs, and the info string.

A family supplies ``build_prompt(prompt, pairs, num_shots, rng) -> (text,
extra)``, where ``extra`` holds its own info fields. The manipulation layer
applies last, so ``prompt`` is the finished question (obfuscated, its request
around it) and the pairs stay as written; a model picking the subcategory is
shown the clean input instead (``source_prompt``).

The many-shot variants (``*_many``) show :data:`MANY_PAIRS` short pairs. A
subcategory holds only eight of a kind in the cache, so they draw from a pool
across subcategories (:func:`_many_pairs`). A cache with no short pair raises
(:func:`_require_short_pairs`) rather than give a prompt with no shot.
"""

import random
from collections.abc import Callable

from redact.llm_pipeline import LLMRequest
from redact.llms.prompting import build_messages, load_prompt

from ..technique import TechniqueGen

BuildPrompt = Callable[[str, list[dict], int, random.Random], tuple[str, dict]]

#: The pairs a many-shot variant shows, all with short answers. A short pair
#: of the cache is about 150 characters as the prompt writes it (``Qn:`` and
#: ``An:`` lines), about 40 tokens at four characters a token; 32 of them are
#: about 1,200 tokens, 1,450 at the cache's 90th percentile and 2,800 were
#: every pair as long as its longest. The manipulation layer applies last, so
#: no later model step reads the pairs (a selected variant shows its model
#: the clean input alone), and the row stays well inside the generation
#: model's 10,000-token context.
MANY_PAIRS = 32


def _build_subcategory_messages(
    harmful_prompt: str, all_subcategories: list[str], prompt_dir=None
) -> list[dict]:
    config = load_prompt("jailbreak", "category_selection", prompt_dir)
    subcategory_list = "\n".join(f"- {sc}" for sc in all_subcategories)
    return build_messages(
        config, harmful_prompt=harmful_prompt, subcategory_list=subcategory_list
    )


def _parse_subcategory(
    chosen: str, all_subcategories: list[str], rng: random.Random | None = None,
) -> tuple[str, bool]:
    """Map an LLM reply to a known subcategory, falling back to random."""
    chosen = chosen.strip()
    if chosen in all_subcategories:
        return chosen, False
    for sc in all_subcategories:
        if sc.lower() in chosen.lower():
            return sc, False
    return (rng or random).choice(all_subcategories), True


def _select_subcategory_gen(
    harmful_prompt: str,
    all_subcategories: list[str],
    *,
    gen_model: str | None = None,
    prompt_dir=None,
    rng: random.Random | None = None,
) -> TechniqueGen:
    """Yield one category-selection request; return ``(subcategory, was_fallback)``."""
    chosen = yield LLMRequest(gen_model, _build_subcategory_messages(
        harmful_prompt, all_subcategories, prompt_dir))
    return _parse_subcategory(chosen, all_subcategories, rng)


def _pick_pairs(benign_data: dict, answer_type: str, subcat: str) -> list[dict]:
    """Benign Q&A pairs for one subcategory + answer-length tier ("short"/"long")."""
    return benign_data[f"by_subcat_{answer_type}"].get(subcat, benign_data[answer_type])


def _require_short_pairs(benign_data: dict) -> None:
    """Raise unless ``benign_data`` holds a short pair, which a many-shot
    variant shows; without one it would write the request alone, marked
    ``num_shots=0``, or fail on the empty subcategory list.

    Raises:
        ValueError: No short pair; the message says how to generate them again.
    """
    if not benign_data["short"]:
        raise ValueError(
            "The benign data holds no short pair, which a many-shot variant "
            "shows (every short unit was rejected, or the cache lacks them). "
            "Delete the benign cache, benign_samples.csv and the ledger beside "
            "it, to generate the pairs again.")


def _many_pairs(
    benign_data: dict, rng: random.Random, subcat: str | None = None,
) -> list[dict]:
    """Up to :data:`MANY_PAIRS` short pairs of the cache, each question once.

    ``subcat``'s pairs come first, then those of the other subcategories of
    its main category, then the rest, each group in an order drawn from
    ``rng``. Without ``subcat``, every pair is drawn alike. A cache of fewer
    pairs gives them all.
    """
    pairs = benign_data["short"]
    main = next((pair.get("main_category") for pair in pairs
                 if subcat is not None and pair.get("sub_category") == subcat), None)

    def closeness(pair: dict) -> int:
        if subcat is not None and pair.get("sub_category") == subcat:
            return 0
        if main is not None and pair.get("main_category") == main:
            return 1
        return 2

    chosen: list[dict] = []
    seen: set[str] = set()
    for pair in sorted(rng.sample(pairs, len(pairs)), key=closeness):
        if len(chosen) < MANY_PAIRS and pair["prompt"] not in seen:
            seen.add(pair["prompt"])
            chosen.append(pair)
    return chosen


def _format_manipulation_info(
    num_shots: int, extra: dict, subcat: str, selection: str | None = None,
    wanted: int | None = None,
) -> str:
    """additional_info string shared by every FSH/DAP variant; ``wanted=...``
    follows ``num_shots`` when a many-shot variant got fewer pairs than it
    wanted."""
    parts = [f"num_shots={num_shots}"]
    if wanted is not None and num_shots < wanted:
        parts.append(f"wanted={wanted}")
    parts.extend(f"{k}={v}" for k, v in extra.items())
    parts.append(f"sub_category={subcat}")
    if selection is not None:
        parts.append(f"selection={selection}")
    return ";".join(parts)


def run_random_manipulation(
    prompt: str,
    benign_data: dict,
    answer_type: str,
    build_prompt: BuildPrompt,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """The body of a ``*_random_{short,long}`` variant; no model call."""
    rng = rng or random
    num_shots = rng.randint(3, 7)
    subcat = rng.choice(benign_data["all_subcategories"])
    pairs = _pick_pairs(benign_data, answer_type, subcat)
    jailbreak, extra = build_prompt(prompt, pairs, num_shots, rng)
    return jailbreak, _format_manipulation_info(num_shots, extra, subcat)


def run_selected_manipulation(
    prompt: str,
    answer_type: str,
    build_prompt: BuildPrompt,
    *,
    benign_data: dict | None = None,
    gen_model: str | None = None,
    prompt_dir=None,
    rng: random.Random | None = None,
    source_prompt: str | None = None,
    **kwargs,
) -> TechniqueGen:
    """The body of a ``*_selected_{short,long}`` variant, subcategory by a
    model. The model is shown ``source_prompt``, the clean input the chain
    started from (:func:`~redact.jailbreak.chain.run_chain`), else
    ``prompt``; the pairs go around ``prompt``, the finished text."""
    rng = rng or random
    num_shots = rng.randint(3, 7)
    subcat, was_fallback = yield from _select_subcategory_gen(
        prompt if source_prompt is None else source_prompt,
        benign_data["all_subcategories"],
        gen_model=gen_model, prompt_dir=prompt_dir, rng=rng,
    )
    pairs = _pick_pairs(benign_data, answer_type, subcat)
    jailbreak, extra = build_prompt(prompt, pairs, num_shots, rng)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, _format_manipulation_info(num_shots, extra, subcat, selection)


def run_random_many(
    prompt: str,
    benign_data: dict,
    build_prompt: BuildPrompt,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """The body of a ``*_random_many`` variant: :data:`MANY_PAIRS` short pairs
    drawn from every subcategory (``sub_category=all``); no model call."""
    _require_short_pairs(benign_data)
    rng = rng or random
    pairs = _many_pairs(benign_data, rng)
    jailbreak, extra = build_prompt(prompt, pairs, len(pairs), rng)
    return jailbreak, _format_manipulation_info(
        len(pairs), extra, "all", wanted=MANY_PAIRS)


def run_selected_many(
    prompt: str,
    build_prompt: BuildPrompt,
    *,
    benign_data: dict | None = None,
    gen_model: str | None = None,
    prompt_dir=None,
    rng: random.Random | None = None,
    source_prompt: str | None = None,
    **kwargs,
) -> TechniqueGen:
    """The body of a ``*_selected_many`` variant: :data:`MANY_PAIRS` short
    pairs, those of the subcategory a model picks first, then its main
    category's, then the rest's. The model picks on ``source_prompt``, as
    :func:`run_selected_manipulation` does."""
    _require_short_pairs(benign_data)
    rng = rng or random
    subcat, was_fallback = yield from _select_subcategory_gen(
        prompt if source_prompt is None else source_prompt,
        benign_data["all_subcategories"],
        gen_model=gen_model, prompt_dir=prompt_dir, rng=rng,
    )
    pairs = _many_pairs(benign_data, rng, subcat)
    jailbreak, extra = build_prompt(prompt, pairs, len(pairs), rng)
    selection = "fallback_random" if was_fallback else "selected"
    return jailbreak, _format_manipulation_info(
        len(pairs), extra, subcat, selection, wanted=MANY_PAIRS)
