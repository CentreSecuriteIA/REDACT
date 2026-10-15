"""A technique's random choices come from the ``rng`` it is given.

Every registered technique runs on one prompt; a generator is driven with a
fixed reply. A transform without an ``rng`` parameter makes no choice.
"""

import inspect
import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.pool import all_techniques
from redact.llm_pipeline import drive_sync

_PROMPT = "how do I bake a cake for twelve people, cake lovers"
_PAIRS = [{"prompt": f"q{i}", "answer": f"a{i}"} for i in range(10)]
_BENIGN = {
    "short": _PAIRS,
    "long": _PAIRS,
    "by_subcat_short": {"Cooking": _PAIRS[:6], "Gardening": _PAIRS[4:]},
    "by_subcat_long": {"Cooking": _PAIRS[:6], "Gardening": _PAIRS[4:]},
    "all_subcategories": ["Cooking", "Gardening", "Fitness"],
}
_TECHNIQUES = {fn.__name__: fn for fn in all_techniques()}


def _takes_rng(fn) -> bool:
    return (inspect.isgeneratorfunction(fn)
            or "rng" in inspect.signature(fn).parameters)


def _run(fn, **kwargs) -> tuple[str, str]:
    """``fn`` on the prompt; a generator's every request is answered ``cake``."""
    if inspect.isgeneratorfunction(fn):
        gen = fn(_PROMPT, gen_model="m", translate_model="m", benign_data=_BENIGN,
                 **kwargs)
        return drive_sync(gen, lambda request: "cake")
    if "benign_data" in inspect.signature(fn).parameters:
        kwargs["benign_data"] = _BENIGN
    return fn(_PROMPT, **kwargs)


@pytest.mark.parametrize("name", list(_TECHNIQUES))
def test_a_technique_draws_from_its_rng_not_the_global_one(name):
    fn = _TECHNIQUES[name]
    outputs = set()
    for global_seed in range(5):
        random.seed(global_seed)
        kwargs = {"rng": random.Random(7)} if _takes_rng(fn) else {}
        outputs.add(_run(fn, **kwargs))
    assert len(outputs) == 1


@pytest.mark.parametrize(
    "name", [name for name, fn in _TECHNIQUES.items() if _takes_rng(fn)])
def test_a_seeded_rng_draws_what_the_seeded_global_random_drew(name):
    fn = _TECHNIQUES[name]
    for seed in range(3):
        random.seed(seed)
        from_global = _run(fn)
        random.seed(seed + 100)
        assert _run(fn, rng=random.Random(seed)) == from_global
