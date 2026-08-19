"""DAP (Distract and Persuade / Attention Shifting) techniques.

The harmful question is hidden among benign Q&A pairs at a random
middle position with an empty answer placeholder.

Ported from reference manipulation.py lines 356-468.
"""

import random

from redact.jailbreak.protocol import TechniqueGen

from .fsh import run_random_manipulation, run_selected_manipulation

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


def _dap_build_prompt(harmful_prompt: str, pairs: list[dict], num_shots: int) -> tuple[str, dict]:
    """Adapt _build_dap_prompt's (str, int) return to the shared runner's (str, extra) shape."""
    text, harmful_idx = _build_dap_prompt(harmful_prompt, pairs, num_shots)
    return text, {"harmful_idx": harmful_idx}


def to_dap_random_short(prompt: str, benign_data: dict) -> tuple[str, str]:
    """DAP with randomly chosen category, short answers. No LLM needed."""
    return run_random_manipulation(prompt, benign_data, "short", _dap_build_prompt)


def to_dap_random_long(prompt: str, benign_data: dict) -> tuple[str, str]:
    """DAP with randomly chosen category, long answers. No LLM needed."""
    return run_random_manipulation(prompt, benign_data, "long", _dap_build_prompt)


def to_dap_selected_short(
    prompt: str, benign_data: dict | None = None, *,
    gen_model: str | None = None, prompt_dir=None, **kwargs,
) -> TechniqueGen:
    """DAP with LLM-selected category, short answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, benign_data, "short", _dap_build_prompt,
        gen_model=gen_model, prompt_dir=prompt_dir,
    ))


def to_dap_selected_long(
    prompt: str, benign_data: dict | None = None, *,
    gen_model: str | None = None, prompt_dir=None, **kwargs,
) -> TechniqueGen:
    """DAP with LLM-selected category, long answers (technique generator)."""
    return (yield from run_selected_manipulation(
        prompt, benign_data, "long", _dap_build_prompt,
        gen_model=gen_model, prompt_dir=prompt_dir,
    ))


def get_dap_functions() -> list:
    """Return all DAP technique functions."""
    return [to_dap_random_short, to_dap_random_long,
            to_dap_selected_short, to_dap_selected_long]
