"""The registered techniques, against a snapshot of them.

``fixtures/technique_registry.json`` holds each layer's techniques in registry
order, with their spec tags, whether each is a generator and its parameters,
and the combinations ``sample_combination`` draws from the whole pool for five
seeds. A change that renames, reorders, retags or re-signs a technique fails
here. A change meant to alter them regenerates the fixture, from the
repository root, with ``py -m tests.jailbreak.test_registry``.
"""

import inspect
import json
import random
from pathlib import Path

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.hacking import get_all_hacking_functions
from redact.jailbreak.manipulation import get_all_manipulation_functions
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.requests import get_all_request_functions
from redact.jailbreak.sampling import sample_combination
from redact.jailbreak.technique import load_spec

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "technique_registry.json"

_TAGS = (
    "families", "complexity", "requires_llm", "requires_benign",
    "encode_weight", "max_tokens", "layer", "within_layer_order",
)
_SEEDS = (0, 1, 2, 3, 4)


def _layers() -> dict[str, list]:
    """Each layer's registered techniques, in the order the pool joins them."""
    return {
        "obfuscation": get_all_obfuscation_functions(),
        "hacking": get_all_hacking_functions(),
        "manipulation": get_all_manipulation_functions(),
        "requests": get_all_request_functions(),
    }


def _signature(fn) -> str:
    """The parameters of ``fn`` without their annotations."""
    sig = inspect.signature(fn)
    params = [p.replace(annotation=p.empty) for p in sig.parameters.values()]
    return str(sig.replace(parameters=params, return_annotation=sig.empty))


def snapshot() -> dict:
    """What the fixture holds, as the code gives it now."""
    layers = {
        layer: [
            {
                "name": fn.__name__,
                **{tag: getattr(fn, tag) for tag in _TAGS},
                "generator": inspect.isgeneratorfunction(fn),
                "signature": _signature(fn),
            }
            for fn in fns
        ]
        for layer, fns in _layers().items()
    }
    pool = [fn for fns in _layers().values() for fn in fns]
    combinations = [
        [t.__name__ for t in sample_combination(random.Random(seed), pool).techniques]
        for seed in _SEEDS
    ]
    return {"layers": layers, "combinations": combinations}


def _dumps(snap: dict) -> str:
    """``snap`` as JSON, one technique or combination per line."""
    def rows(items: list, indent: str) -> str:
        return ",\n".join(indent + json.dumps(item) for item in items)

    layers = ",\n".join(
        f"  {json.dumps(layer)}: [\n{rows(entries, '   ')}\n  ]"
        for layer, entries in snap["layers"].items()
    )
    combinations = rows(snap["combinations"], "  ")
    return (f'{{\n "layers": {{\n{layers}\n }},\n'
            f' "combinations": [\n{combinations}\n ]\n}}\n')


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def actual() -> dict:
    return snapshot()


@pytest.mark.parametrize(
    "layer", ["obfuscation", "hacking", "manipulation", "requests"])
def test_a_layer_registers_the_same_techniques_in_order(layer, expected, actual):
    assert actual["layers"][layer] == expected["layers"][layer]


def test_a_seed_draws_the_same_combination(expected, actual):
    assert actual["combinations"] == expected["combinations"]


def test_the_spec_and_the_registry_name_the_same_techniques():
    registered = [fn.__name__ for fns in _layers().values() for fn in fns]
    in_spec = {name for name in load_spec()["techniques"] if not name.startswith("_")}
    assert len(registered) == len(set(registered))
    assert set(registered) == in_spec


def test_every_generator_takes_the_prompt_and_keywords():
    generators = [fn for fns in _layers().values() for fn in fns
                  if inspect.isgeneratorfunction(fn)]
    assert generators
    assert {fn.__name__: _signature(fn) for fn in generators} == {
        fn.__name__: "(prompt, **kwargs)" for fn in generators}


if __name__ == "__main__":
    _FIXTURE.write_text(_dumps(snapshot()), encoding="utf-8")
