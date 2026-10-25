"""The catalog of every registered jailbreak technique, made from the code.

:func:`build` gathers, from the registry, the spec, the rules, the variant
pools, the augmentation kinds and the prompt tree:

- every registered technique (:func:`~redact.jailbreak.pool.all_techniques`),
  the retired ones marked: its spec tags, the variants of the template pool
  it draws from, the model prompt steps its code loads
  (:func:`prompt_steps`), the pools that hold it, and how it combines: the
  other active techniques :func:`~redact.jailbreak.rules.compatible` admits
  beside it, asked one pair at a time, and for each one it refuses, the
  rules that refuse the pair (:func:`refusal_rules`, which mirrors
  ``compatible``; a test holds the two together);
- each layer (cap, rate, order) and family, and a family by family matrix:
  ``always`` when every pair of their active members combines, ``sometimes``
  when some do, ``never`` when none does;
- the pools of the packaged kinds, of the variants a kind may ask for (the
  rewrite layer on, the translation switch flipped) and of each dataset
  type, with the rounds each plans;
- the prompt inventory: the template variants per technique, family, layer
  and in all; the model prompt files per pipeline and step; the input side
  (the taxonomies, the benign pairs, the generation axes, the answer modes,
  the entry types);
- combination examples: synthetic inputs planned through each packaged kind
  as the stage plans them (:data:`SIMULATED_UNITS` units, seed
  :data:`SIMULATED_SEED`, :data:`SIMULATED_CATEGORIES` categories times the
  kind's entry types), the units counted by their number of techniques and
  of layers, and the family pairs drawn most often.

:func:`dumps` writes it as JSON (``docs/catalog.json``) and :func:`markdown`
as ``docs/catalog.md``; every number in both comes from the code. The pools
are those of the packaged files: a ``config_dir`` is not read. A variant
pool's floor is the variant-floor fixture's
(``tests/fixtures/variant_floor.json``), read when the repository's tests are
there, else ``None``. ``tests/jailbreak/test_catalog.py`` fails until both
files are made again, from the repository root, with
``PYTHONPATH=src py -m redact.jailbreak.catalog`` (or ``scripts/catalog.py``).

The rules of a pair, by code (:data:`RULES`): ``S`` one technique per
family, ``L`` the layer cap (``layer_max_picks``), ``F`` a family's
``cross_incompatible_with``, ``T`` a technique's own, ``M`` the manipulation
block (``manipulation_blocked_obfuscation``). ``R`` is not a pair rule: the
families ``request_obfuscation_refuses`` keeps out of a combination whose
request obfuscation is a technique of the family it is keyed by.

The JSON's keys: ``headline`` (the totals); ``rules`` (the codes);
``layers``; ``sampling`` (the budget, the request obfuscation, the
manipulation block, the escalation); ``families`` (by name);
``family_matrix`` (``families`` in order, and ``cells[a][b]``, a list
named by ``cell``: status, allowed pairs, pairs, rule codes);
``techniques`` (by name, in registry order: the spec tags, ``uses`` (the
chain's keyword arguments its code reads) and ``models``, ``variant_pool``
and ``variants``, ``prompts`` (its steps under ``prompts/jailbreak/``),
``pools``, ``allowed_count`` and ``refused_count``, ``never_with`` and
``partly_with`` by family, and ``refused_with``: each active technique it
is refused with, and the codes of the rules that refuse the pair);
``pools`` (by name: kind, section, rounds, techniques); ``prompts``;
``inputs``; ``combinations`` (by pool); ``spec_notes`` (the spec's
comments on its rules).
"""

import argparse
import functools
import inspect
import json
import re
import sys
import tempfile
import types
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import pandas as pd

from redact import paths
from redact.config_overlay import DATASET_TYPES
from redact.types import ALL_ENTRY_TYPES

from . import rules
from .augmentation import (
    layers_off_by_default,
    natural_only_families,
    resolve,
    rounds_of,
)
from .plan import UNIT_HEADROOM, default_escalation_schedule, plan_run
from .pool import (
    all_techniques,
    technique_pool,
    variant_pools,
)
from .technique import (
    KINDS,
    MAX_COMPLEXITY,
    decode_hint_rate,
    heavy_encoding_weight,
    load_spec,
)

#: The units each packaged kind plans for the combination examples.
SIMULATED_UNITS = 3000
#: The seed of those plans.
SIMULATED_SEED = 42
#: The categories of the synthetic inputs, times each kind's entry types.
SIMULATED_CATEGORIES = 3
#: The family pairs listed per kind, the most frequent first.
TOP_FAMILY_PAIRS = 15
#: A refusal list this long or shorter is written out in the Markdown.
SHORT_LIST = 25
#: The command that writes the catalog again, from the repository root.
REGENERATE = "PYTHONPATH=src py -m redact.jailbreak.catalog"

#: The rules that may refuse a pair, by code, in the order they are named.
RULES = {
    "S": "one technique per family",
    "L": "layer cap",
    "F": "family rule",
    "T": "technique rule",
    "M": "manipulation block",
}
#: The rule of a combination whose request obfuscation is drawn.
REQUEST_RULE = "R"
_RULE_SOURCES = {
    "S": "rules.compatible",
    "L": "combination_spec.json: layer_max_picks",
    "F": "combination_spec.json: families.<family>.cross_incompatible_with",
    "T": "combination_spec.json: techniques.<technique>.cross_incompatible_with",
    "M": "combination_spec.json: manipulation_blocked_obfuscation",
    "R": "combination_spec.json: request_obfuscation_refuses",
}

#: The keyword arguments the chain gives a technique that takes them
#: (:func:`redact.jailbreak.chain.run_chain`), and what each is for.
_CHAIN_KWARGS = {
    "gen_model": "generation",
    "check_model": "check",
    "translate_model": "translation",
    "benign_data": "benign pairs",
    "prompt_dir": None,
    "source_prompt": None,
    "rng": None,
}
#: The names a code object uses when it reads a prompt file.
_LOADERS = {"load_prompt", "resolve_prompt", "load"}
#: How deep the walk reads nested constants, and the JSON its levels.
_NESTING = 3
_PACKAGE = __name__.rpartition(".")[0]

#: The pools of the packaged section: name, kind, the section given.
_KIND_POOLS = (
    ("adversarial", "adversarial", None),
    ("adversarial_rewrite", "adversarial", "rewrite"),
    ("adversarial_no_translation", "adversarial", {"translation": False}),
    ("natural", "natural", None),
    ("natural_translation", "natural", {"translation": True}),
)


# --- the rules of a pair --------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _rule_context() -> tuple:
    spec = load_spec()
    families = spec["families"]
    manipulation = {
        name
        for name, family in families.items()
        if family.get("layer") == "manipulation"
    }
    return (
        families,
        spec.get("layer_max_picks", {}),
        spec.get("manipulation_blocked_obfuscation", {}),
        manipulation,
    )


def refusal_rules(a: Callable, b: Callable) -> str:
    """The codes of the rules (:data:`RULES`) that refuse ``b`` beside ``a``,
    in that order; ``""`` when :func:`~redact.jailbreak.rules.compatible`
    admits the pair. It reads the spec as ``compatible`` does, and a test
    asserts the two agree on every pair of registered techniques."""
    families, caps, blocked, manipulation = _rule_context()

    def cross(names: set) -> set:
        return set().union(
            *(
                families.get(name, {}).get("cross_incompatible_with", [])
                for name in names
            )
        )

    fa, fb = rules._families(a), rules._families(b)
    codes = ""
    if fa & fb:
        codes += "S"
    layer = getattr(b, "layer", None)
    picked = 1 if getattr(a, "layer", None) == layer else 0
    if layer in caps and picked >= caps[layer]:
        codes += "L"
    if fb & cross(fa) or fa & cross(fb):
        codes += "F"
    if fb & rules._own_incompatible(a) or fa & rules._own_incompatible(b):
        codes += "T"
    if (fa & manipulation and rules._unfit_for_manipulation(b, blocked)) or (
        fb & manipulation and rules._unfit_for_manipulation(a, blocked)
    ):
        codes += "M"
    return codes


# --- the prompts a technique's code loads ---------------------------------------


def _prompt_files() -> dict[str, list[Path]]:
    """Every packaged prompt file by its step, ``<pipeline>/<folder>``."""
    root = paths.prompts_dir()
    steps: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.json")):
        steps[path.parent.relative_to(root).as_posix()].append(path)
    return dict(sorted(steps.items()))


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def _strings(value, depth: int = 0):
    """The strings of a code constant, nested tuples and frozensets
    included."""
    if isinstance(value, str):
        yield value
    elif depth < _NESTING and isinstance(value, tuple | frozenset):
        for item in value:
            yield from _strings(item, depth + 1)


def _filled(cell) -> list:
    try:
        return [cell.cell_contents]
    except ValueError:
        return []


def prompt_steps(fn: Callable, steps: set[str]) -> set[str]:
    """The prompt steps of ``steps`` (names as ``load_prompt`` takes them)
    that ``fn``'s code may load (:func:`code_reach`)."""
    return code_reach(fn, steps)[0]


def code_reach(fn: Callable, steps: set[str]) -> tuple[set[str], set[str]]:
    """What ``fn``'s code may use, read off its bytecode, not run: the
    prompt steps of ``steps`` it may load, and the keyword arguments of the
    chain (:data:`_CHAIN_KWARGS`) it may read.

    From ``fn``, every function of the jailbreak package its code (nested
    code included) names, through its globals, a module of the package it
    names, or its closure, is followed. In each code object that calls a
    prompt loader (:data:`_LOADERS`), a step is a string constant of the
    code, or a string its names or closure hold; in every code object, a
    keyword argument is a local name or a string constant (``kwargs.get``).
    So a step a technique loads through a helper of another module (the
    marking call of the sensitive words) is found, and one only another
    technique of its module loads is not.
    """
    found: set[str] = set()
    used: set[str] = set()
    seen: set[int] = set()
    stack = [fn]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, functools.partial):
            stack.append(obj.func)
            continue
        if hasattr(obj, "__wrapped__"):
            stack.append(obj.__wrapped__)
        code = getattr(obj, "__code__", None)
        if code is None:
            continue
        names = getattr(obj, "__globals__", {})
        cells = [value for cell in (obj.__closure__ or ()) for value in _filled(cell)]
        modules = [
            value
            for value in names.values()
            if isinstance(value, types.ModuleType)
            and value.__name__.startswith(_PACKAGE)
        ]
        for each in _code_objects(code):
            refs = [names[name] for name in each.co_names if name in names]
            refs += [
                getattr(module, name)
                for name in each.co_names
                for module in modules
                if hasattr(module, name)
            ]
            constants = {s for const in each.co_consts for s in _strings(const)}
            used.update((constants | set(each.co_varnames)) & set(_CHAIN_KWARGS))
            if _LOADERS & set(each.co_names):
                found.update(constants & steps)
                found.update(
                    value
                    for value in refs + cells
                    if isinstance(value, str) and value in steps
                )
            stack.extend(
                value
                for value in refs + cells
                if callable(value)
                and not isinstance(value, type)
                and (getattr(value, "__module__", "") or "").startswith(_PACKAGE)
            )
    return found, used


# --- the pools ------------------------------------------------------------------


def _resolved_pools() -> dict[str, dict]:
    """Each pool's kind, section, base dataset type, resolved kind (``None``
    when off) and techniques."""
    from redact.recipe import load_dataset_type  # the recipe imports the stages

    pools = {}
    for name, kind, given in _KIND_POOLS:
        if given == "rewrite":
            given = {"layers": {"allow": layers_off_by_default()}}
        section = {kind: given} if given is not None else None
        pools[name] = {
            "kind": kind,
            "section": given,
            "dataset_type": None,
            "resolved": resolve(section, kind),
        }
    for dataset_type in DATASET_TYPES:
        base = load_dataset_type(dataset_type)["augmentations"]
        for kind in KINDS:
            pools[f"{dataset_type}.{kind}"] = {
                "kind": kind,
                "section": base.get(kind, {}),
                "dataset_type": dataset_type,
                "resolved": resolve({}, kind, base=base),
            }
    for entry in pools.values():
        resolved = entry["resolved"]
        entry["techniques"] = (
            [fn.__name__ for fn in technique_pool(resolved)]
            if resolved is not None
            else []
        )
        entry["rounds"] = rounds_of(resolved) if resolved is not None else []
    return pools


# --- the combination examples ---------------------------------------------------


def _simulated_inputs(units: int, rounds: int, entry_types: list[str]) -> pd.DataFrame:
    count = units // rounds
    return pd.DataFrame(
        {
            "prompt": [f"simulated input {i} about a topic" for i in range(count)],
            "category": [f"category {i % SIMULATED_CATEGORIES}" for i in range(count)],
            "entry_type": [
                entry_types[(i // SIMULATED_CATEGORIES) % len(entry_types)]
                for i in range(count)
            ],
        }
    )


def _simulate(resolved: dict) -> dict:
    """What the kind ``resolved`` plans for synthetic inputs: the units by
    number of techniques and of layers, the units per layer and the family
    pairs drawn most often."""
    from redact.dataset import Manifest

    pool = technique_pool(resolved)
    rounds = rounds_of(resolved)
    entry_types = resolved["entry_types"] or [str(t.value) for t in ALL_ENTRY_TYPES]
    frame = _simulated_inputs(SIMULATED_UNITS, len(rounds), entry_types)
    with tempfile.TemporaryDirectory() as folder:
        manifest = Path(folder) / "simulated.manifest.jsonl"
        plan_run(
            frame,
            pool,
            manifest_path=manifest,
            seed=SIMULATED_SEED,
            iterations=len(rounds),
            settings_per_iteration=rounds,
            family_weights=resolved["family_weights"] or None,
            layer_rates=resolved.get("layer_rates") or None,
            verbose=False,
        )
        plan = Manifest(manifest).load()
    by_name = {fn.__name__: fn for fn in all_techniques()}
    sizes, touched, per_layer, pairs, drawn = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        set(),
    )
    for row in plan:
        names = row["combination"]
        drawn.update(names)
        layers = {by_name[name].layer for name in names}
        sizes[len(names)] += 1
        touched[len(layers)] += 1
        per_layer.update(layers)
        firsts = sorted({by_name[name].families[0] for name in names})
        pairs.update(combinations(firsts, 2))
    layer_order = load_spec()["layer_order"]
    top = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:TOP_FAMILY_PAIRS]
    return {
        "units": len(plan),
        "inputs": len(frame),
        "rounds": len(rounds),
        "groups": len({(row["category"], row["entry_type"]) for row in plan}),
        "entry_types": entry_types,
        "seed": SIMULATED_SEED,
        "pool_size": len(pool),
        "techniques_drawn": len(drawn),
        "never_drawn": [fn.__name__ for fn in pool if fn.__name__ not in drawn],
        "mean_techniques": round(sum(k * n for k, n in sizes.items()) / len(plan), 3),
        "by_size": {str(k): sizes[k] for k in sorted(sizes)},
        "by_layers_touched": {str(k): touched[k] for k in sorted(touched)},
        "units_per_layer": {layer: per_layer[layer] for layer in layer_order},
        "top_family_pairs": [[a, b, n] for (a, b), n in top],
    }


# --- the input side -------------------------------------------------------------


def _inputs() -> dict:
    from redact.content_moderation import axes, modes
    from redact.dataset.taxonomy import iter_categories, load_taxonomy

    from .manipulation import benign, pairs

    taxonomies = {}
    for name in ("harmful", benign.DEFAULT_TAXONOMY):
        categories = {
            category: len(info.get("subcategories") or [])
            for category, info in iter_categories(load_taxonomy(name))
        }
        taxonomies[name] = {
            "categories": len(categories),
            "subcategories": sum(categories.values()),
            "per_category": categories,
        }
    subcategories = benign.benign_categories(load_taxonomy(benign.DEFAULT_TAXONOMY))
    per_unit = (
        inspect.signature(benign.generate_benign_data).parameters["num_samples"].default
    )
    units = len(subcategories) * len(benign.ANSWER_TYPES)
    generation_axes = {}
    for stage in ("inputs", "outputs"):
        resolved = axes.resolve(stage, None) or {}
        generation_axes[stage] = {
            axis: {
                "dimension": entry.get("dimension"),
                "from": entry.get("from"),
                "values": {
                    value: spec.get("weight")
                    for value, spec in (entry.get("values") or {}).items()
                },
            }
            for axis, entry in resolved.items()
        }
    return {
        "entry_types": [str(t.value) for t in ALL_ENTRY_TYPES],
        "taxonomies": taxonomies,
        "benign_pairs": {
            "taxonomy": benign.DEFAULT_TAXONOMY,
            "subcategories": len(subcategories),
            "answer_types": list(benign.ANSWER_TYPES),
            "units": units,
            "pairs_per_unit": per_unit,
            "pairs": units * per_unit,
            "many_shot_pairs": pairs.MANY_PAIRS,
        },
        "axes": generation_axes,
        "answer_modes": {
            "modes": list(modes.MODES),
            "by_entry_type": {
                key: list(value) for key, value in modes.DEFAULT_MODES.items()
            },
        },
    }


# --- the catalog ----------------------------------------------------------------


def _floors() -> dict[str, int | None]:
    """Each variant pool's floor, from the variant-floor fixture when the
    repository's tests are there; empty otherwise."""
    fixture = (
        paths.package_dir().parents[1] / "tests" / "fixtures" / "variant_floor.json"
    )
    if not fixture.exists():
        return {}
    pools = json.loads(fixture.read_text(encoding="utf-8"))["pools"]
    return {name: entry.get("floor") for name, entry in pools.items()}


def _family_order(spec: dict) -> list[str]:
    """The spec's families by layer, then order within the layer, then name."""
    layers = spec["layer_order"]
    families = spec["families"]
    return sorted(
        families,
        key=lambda f: (
            layers.index(families[f]["layer"]),
            families[f].get("within_layer_order", 0),
            f,
        ),
    )


def _union(codes) -> str:
    """The rule codes of ``codes`` (strings of codes), each once, in
    :data:`RULES` order."""
    joined = "".join(codes)
    return "".join(code for code in RULES if code in joined)


def compatible_with(fn: Callable, partners: list[Callable]) -> list[Callable]:
    """The ``partners`` :func:`~redact.jailbreak.rules.compatible` admits
    beside ``fn`` alone, with no complexity budget."""
    return rules.compatible([fn], partners)


def _pairs(registered: list[Callable], active: list[Callable]) -> tuple[dict, dict]:
    """For each technique, the names of the active techniques
    :func:`compatible_with` admits beside it, and those it refuses with the
    codes of the rules that refuse them (:func:`refusal_rules`).

    Raises:
        RuntimeError: ``compatible`` refuses a pair :func:`refusal_rules`
            names no rule for: the two have drifted apart.
    """
    allowed: dict[str, set[str]] = {}
    refused: dict[str, dict[str, str]] = {}
    for fn in registered:
        partners = [other for other in active if other is not fn]
        allowed[fn.__name__] = {
            other.__name__ for other in compatible_with(fn, partners)
        }
        refused[fn.__name__] = {}
        for other in partners:
            if other.__name__ in allowed[fn.__name__]:
                continue
            codes = refusal_rules(fn, other)
            if not codes:
                raise RuntimeError(
                    f"rules.compatible refuses {other.__name__} beside {fn.__name__}, "
                    "and refusal_rules names no rule: bring it in line with "
                    "rules.compatible."
                )
            refused[fn.__name__][other.__name__] = codes
    return allowed, refused


def _against_families(
    fn: Callable, refused: dict[str, str], members: dict, family_order: list[str]
) -> tuple[dict, dict]:
    """The families of which ``fn`` is refused with every active member
    (by the rule codes) and those of which with some (how many of how
    many, by the codes); its own families are left out."""
    never, partly = {}, {}
    for family in family_order:
        others = [other for other in members.get(family, ()) if other is not fn]
        if family in fn.families or not others:
            continue
        codes = [
            refused[other.__name__] for other in others if other.__name__ in refused
        ]
        if len(codes) == len(others):
            never[family] = _union(codes)
        elif codes:
            partly[family] = {
                "refused": len(codes),
                "members": len(others),
                "rules": _union(codes),
            }
    return never, partly


def _matrix(
    family_order: list[str], members: dict, allowed: dict, refused: dict
) -> dict:
    """The family by family matrix of the families with an active member:
    each cell ``[status, allowed pairs, pairs, rule codes]``."""
    order = [family for family in family_order if members.get(family)]
    cells: dict[str, dict[str, list]] = {}
    for a in order:
        cells[a] = {}
        for b in order:
            if a == b:
                cells[a][b] = ["same", 0, 0, "S"]
                continue
            pairs = [
                (x.__name__, y.__name__)
                for x in members[a]
                for y in members[b]
                if x is not y
            ]
            ok = sum(y in allowed[x] for x, y in pairs)
            status = (
                "n/a"
                if not pairs
                else "always"
                if ok == len(pairs)
                else "never"
                if ok == 0
                else "sometimes"
            )
            cells[a][b] = [
                status,
                ok,
                len(pairs),
                _union(refused[x][y] for x, y in pairs if y in refused[x]),
            ]
    return {
        "families": order,
        "cell": ["status", "allowed", "pairs", "rules"],
        "cells": cells,
    }


def _technique(fn: Callable, context: dict) -> dict:
    """One technique's entry (the module docstring)."""
    name = fn.__name__
    ref = getattr(fn, "variant_pool", None)
    steps, uses = context["reach"][name]
    never, partly = _against_families(
        fn, context["refused"][name], context["members"], context["family_order"]
    )
    table = context["request_table"]
    candidate = name in context["candidates"]
    return {
        "layer": fn.layer,
        "families": list(fn.families),
        "first_family": fn.families[0],
        "kinds": list(fn.kinds),
        "complexity": fn.complexity,
        "requires_llm": fn.requires_llm,
        "requires_benign": fn.requires_benign,
        "encode_weight": fn.encode_weight,
        "needs_latin": fn.needs_latin,
        "max_tokens": fn.max_tokens,
        "check": fn.check,
        "calls": fn.calls,
        "applicability": hasattr(fn, "applicable"),
        "since": fn.since,
        "deprecated": fn.deprecated,
        "retired": fn.deprecated is not None,
        "within_layer_order": fn.within_layer_order,
        "cross_incompatible_with": list(fn.cross_incompatible_with),
        "uses": [key for key in _CHAIN_KWARGS if key in uses],
        "models": [
            model for key, model in _CHAIN_KWARGS.items() if key in uses and model
        ],
        "variant_pool": ref.name if ref else None,
        "variant_pool_optional": bool(ref and ref.optional),
        "variants": context["vpools"][ref.name]["variants"] if ref else 0,
        "prompts": steps,
        "prompt_files": sum(
            len(context["prompt_files"][context["steps"][step]]) for step in steps
        ),
        "pools": [pool for pool, held in context["pool_sets"].items() if name in held],
        "request_obfuscation_candidate": candidate,
        "as_request_obfuscation_keeps_out": (
            sorted(set().union(*(table.get(family, ()) for family in fn.families)))
            if candidate
            else []
        ),
        "kept_out_by_request_obfuscation_of": sorted(
            key for key, value in table.items() if set(fn.families) & set(value)
        ),
        "allowed_count": len(context["allowed"][name]),
        "refused_count": len(context["refused"][name]),
        "never_with": never,
        "partly_with": partly,
        "refused_with": dict(sorted(context["refused"][name].items())),
    }


def _family(family: str, context: dict, techniques: dict) -> dict:
    """One family's entry (the module docstring)."""
    spec = context["spec"]
    entry = spec["families"][family]
    firsts = [fn for fn in context["registered"] if fn.families[0] == family]
    current = [fn.__name__ for fn in firsts if not fn.deprecated]
    pools = [
        pool for pool, info in context["vpools"].items() if info["family"] == family
    ]
    floors = [
        context["floors"][pool]
        for pool in pools
        if context["floors"].get(pool) is not None
    ]
    calls = entry.get("calls")
    blocked = spec.get("manipulation_blocked_obfuscation", {}).get("families", [])
    return {
        "layer": entry["layer"],
        "dimension": entry.get("dimension"),
        "within_layer_order": entry.get("within_layer_order", 0),
        "kinds": list(entry.get("kinds", ["adversarial"])),
        "check": entry.get("check"),
        "calls": [calls, calls] if isinstance(calls, int) else calls,
        "description": entry.get("description"),
        "techniques": current,
        "retired": [fn.__name__ for fn in firsts if fn.deprecated],
        "also_in": [
            fn.__name__ for fn in context["active"] if family in fn.families[1:]
        ],
        "variant_pools": pools,
        "variants": sum(context["vpools"][pool]["variants"] for pool in pools),
        "floor": min(floors) if floors else None,
        "models": sorted(
            {model for name in current for model in techniques[name]["models"]}
        ),
        "prompts": sorted(
            {step for name in current for step in techniques[name]["prompts"]}
        ),
        "refuses": list(entry.get("cross_incompatible_with", [])),
        "refused_by": [
            other
            for other in context["family_order"]
            if other != family
            and family in spec["families"][other].get("cross_incompatible_with", [])
        ],
        "manipulation_blocked": family in blocked,
        "request_obfuscation_refuses": list(context["request_table"].get(family, [])),
        "natural_only_for_attacks": family in natural_only_families(),
    }


def _layers(spec: dict, family_order: list[str], families: dict) -> list[dict]:
    """Each layer, in the order they apply: cap, rate, families, counts."""
    caps, rates = spec.get("layer_max_picks", {}), spec.get("layer_rates", {})
    off = layers_off_by_default()
    layers = []
    for position, layer in enumerate(spec["layer_order"], start=1):
        names = [f for f in family_order if families[f]["layer"] == layer]
        layers.append(
            {
                "name": layer,
                "position": position,
                "cap": caps.get(layer),
                "rate": rates.get(layer),
                "off_by_default": layer in off,
                "families": names,
                "within_layer_order": {
                    f: families[f]["within_layer_order"] for f in names
                },
                "techniques": sum(len(families[f]["techniques"]) for f in names),
                "retired": sum(len(families[f]["retired"]) for f in names),
                "variants": sum(families[f]["variants"] for f in names),
            }
        )
    return layers


def _pipelines(prompt_files: dict[str, list[Path]]) -> dict:
    """The prompt files per pipeline and step, with their few-shot examples."""
    pipelines: dict[str, dict] = {}
    for step, files in prompt_files.items():
        pipeline, _, folder = step.partition("/")
        few_shot = sum(
            len(
                json.loads(path.read_text(encoding="utf-8")).get("few_shot_examples")
                or []
            )
            for path in files
        )
        entry = pipelines.setdefault(
            pipeline, {"files": 0, "few_shot_examples": 0, "steps": {}}
        )
        entry["files"] += len(files)
        entry["few_shot_examples"] += few_shot
        entry["steps"][folder or "."] = {
            "files": len(files),
            "few_shot_examples": few_shot,
        }
    return pipelines


def _sampling(spec: dict, adversarial: dict, context: dict) -> dict:
    """The budget, the request obfuscation, the manipulation block and the
    escalation schedule."""
    blocked = spec.get("manipulation_blocked_obfuscation", {})
    return {
        "max_technique_complexity": MAX_COMPLEXITY,
        "default_max_complexity": adversarial["max_complexity"],
        "default_max_obfuscations": adversarial["max_obfuscations"],
        "allow_request_obfuscation": adversarial["allow_request_obfuscation"],
        "group_budget_headroom": UNIT_HEADROOM,
        "heavy_encoding_weight": heavy_encoding_weight(),
        "decode_hint_rate": decode_hint_rate(),
        "layers_off_by_default": layers_off_by_default(),
        "request_obfuscation": {
            "rate": spec.get("layer_rates", {}).get("request_obfuscation"),
            "allowed_families": list(
                spec.get("request_obfuscation_allowed_families", [])
            ),
            "refuses": context["request_table"],
            "candidates": sorted(context["candidates"]),
        },
        "manipulation_blocked_obfuscation": {
            key: value for key, value in blocked.items() if not key.startswith("_")
        },
        "natural_only_families": natural_only_families(),
        "escalation": default_escalation_schedule(),
    }


def _spec_notes(spec: dict) -> dict[str, str]:
    """The spec's comments that give the reasons of its rules."""
    notes = {
        "spec": spec.get("_comment"),
        "layer_rates": spec.get("layer_rates", {}).get("_comment"),
        "manipulation_blocked_obfuscation": spec.get(
            "manipulation_blocked_obfuscation", {}
        ).get("_comment"),
        "request_obfuscation_refuses": spec.get("request_obfuscation_refuses", {}).get(
            "_comment"
        ),
        "techniques": spec.get("techniques", {}).get("_comment"),
    }
    return {key: value for key, value in notes.items() if value}


def build() -> dict:
    """The whole catalog, as JSON-ready data (the module docstring)."""
    from .manipulation import get_or_generate_benign_data

    spec = load_spec()
    registered = all_techniques()
    active = [fn for fn in registered if not fn.deprecated]
    members: dict[str, list[Callable]] = defaultdict(list)
    for fn in active:
        for family in fn.families:
            members[family].append(fn)
    prompt_files = _prompt_files()
    steps = {
        step.partition("/")[2]: step
        for step in prompt_files
        if step.startswith("jailbreak/")
    }
    allowed, refused = _pairs(registered, active)
    pools = _resolved_pools()
    context = {
        "spec": spec,
        "registered": registered,
        "active": active,
        "members": members,
        "family_order": _family_order(spec),
        "allowed": allowed,
        "refused": refused,
        "vpools": dict(sorted(variant_pools().items())),
        "floors": _floors(),
        "prompt_files": prompt_files,
        "steps": steps,
        "reach": {
            fn.__name__: tuple(sorted(found) for found in code_reach(fn, set(steps)))
            for fn in registered
        },
        "pool_sets": {name: set(entry["techniques"]) for name, entry in pools.items()},
        "request_table": {
            key: value
            for key, value in spec.get("request_obfuscation_refuses", {}).items()
            if not key.startswith("_")
        },
        "candidates": {
            fn.__name__ for fn in rules.request_obfuscation_candidates(registered)
        },
    }
    stage_steps = sorted(prompt_steps(get_or_generate_benign_data, set(steps)))
    techniques = {fn.__name__: _technique(fn, context) for fn in registered}
    families = {
        family: _family(family, context, techniques)
        for family in context["family_order"]
    }
    layers = _layers(spec, context["family_order"], families)
    vpools = context["vpools"]
    variants_total = sum(info["variants"] for info in vpools.values())
    on = {name: entry for name, entry in pools.items() if entry["resolved"] is not None}
    return {
        "_comment": "Generated by redact.jailbreak.catalog; do not edit.",
        "regenerate": REGENERATE,
        "spec_version": spec.get("version"),
        "headline": {
            "registered": len(registered),
            "active": len(active),
            "retired": len(registered) - len(active),
            "layers": len(spec["layer_order"]),
            "families": len({fn.families[0] for fn in active}),
            "spec_families": len(spec["families"]),
            "variant_pools": len(vpools),
            "template_variants": variants_total,
            "prompt_files": sum(len(files) for files in prompt_files.values()),
            "jailbreak_prompt_steps": len(steps),
            "compatible_pairs": sum(len(allowed[fn.__name__]) for fn in active) // 2,
            "pools": {name: len(entry["techniques"]) for name, entry in on.items()},
        },
        "rules": {
            code: {"rule": title, "source": _RULE_SOURCES[code]}
            for code, title in {**RULES, REQUEST_RULE: "request obfuscation"}.items()
        },
        "layers": layers,
        "sampling": _sampling(spec, pools["adversarial"]["resolved"], context),
        "families": families,
        "family_matrix": _matrix(context["family_order"], members, allowed, refused),
        "techniques": techniques,
        "pools": {
            name: {
                "kind": entry["kind"],
                "dataset_type": entry["dataset_type"],
                "section": entry["section"],
                "on": entry["resolved"] is not None,
                "size": len(entry["techniques"]),
                "rounds": entry["rounds"],
                "techniques": entry["techniques"],
            }
            for name, entry in pools.items()
        },
        "prompts": {
            "variant_pools": {
                name: {
                    **info,
                    "floor": context["floors"].get(name),
                    "layer": spec["families"][info["family"]]["layer"],
                }
                for name, info in vpools.items()
            },
            "variants_by_family": {
                f: entry["variants"] for f, entry in families.items()
            },
            "variants_by_layer": {layer["name"]: layer["variants"] for layer in layers},
            "variants_total": variants_total,
            "pipelines": _pipelines(prompt_files),
            "jailbreak_steps": {
                step: {
                    "techniques": [
                        n for n, t in techniques.items() if step in t["prompts"]
                    ],
                    "stage": step in stage_steps,
                }
                for step in steps
            },
            "stage_steps": stage_steps,
        },
        "inputs": _inputs(),
        "combinations": {
            name: _simulate(on[name]["resolved"])
            for name in ("adversarial", "natural", "eval.adversarial")
            if name in on
        },
        "spec_notes": _spec_notes(spec),
    }


# --- JSON -----------------------------------------------------------------------


def dumps(catalog: dict) -> str:
    """``catalog`` as JSON text: its first three levels one key a line, the
    rest of each value on its line."""
    return _dump(catalog, 0) + "\n"


def _dump(value, depth: int) -> str:
    if isinstance(value, dict) and value and depth < _NESTING:
        pad = " " * (depth + 1)
        items = [
            f"{pad}{json.dumps(key, ensure_ascii=False)}: {_dump(item, depth + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


# --- Markdown -------------------------------------------------------------------

#: The letter of a pool in the technique tables.
_POOL_LETTERS = {
    "adversarial": "A",
    "adversarial_rewrite": "R",
    "natural": "N",
    "natural_translation": "T",
    "eval.adversarial": "E",
}


def _row(*cells) -> str:
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


def _rule_line(columns: int) -> str:
    return "|" + "---|" * columns


def _code(name) -> str:
    return f"`{name}`"


def _names(names) -> str:
    return ", ".join(_code(name) for name in names)


def _anchor(heading: str) -> str:
    """The id Python-Markdown's ``toc`` gives ``heading``."""
    text = re.sub(r"[^\w\s-]", "", heading.replace("`", "")).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def _calls(entry: dict) -> str:
    calls = entry["calls"]
    if not calls:
        return ""
    low, high = calls
    text = str(low) if low == high else f"{low}-{high}"
    return f"{text} ({entry['check']})" if entry["check"] else text


def _tags(entry: dict) -> str:
    tags = []
    if entry["requires_benign"]:
        tags.append("benign")
    if entry["encode_weight"]:
        tags.append(entry["encode_weight"])
    if entry["needs_latin"]:
        tags.append("latin")
    if entry["applicability"]:
        tags.append("applies")
    if entry["max_tokens"]:
        tags.append(f"max {entry['max_tokens']}")
    return " ".join(tags)


def _pool_letters(entry: dict) -> str:
    held = set(entry["pools"])
    letters = []
    for pool, letter in _POOL_LETTERS.items():
        if pool not in held:
            continue
        if pool == "adversarial_rewrite" and "adversarial" in held:
            continue
        if pool == "natural_translation" and "natural" in held:
            continue
        letters.append(letter)
    return " ".join(letters)


def _never(entry: dict, catalog: dict) -> str:
    """The families ``entry`` never combines with, the rest of its layer
    (refused by the cap alone) in one item; those it combines with only in
    part, as ``some``; and what it keeps out as the request obfuscation."""
    families = catalog["families"]
    layer = entry["layer"]
    parts = []
    same_layer = [
        f
        for f in entry["never_with"]
        if families[f]["layer"] == layer and entry["never_with"][f] == "L"
    ]
    if same_layer:
        parts.append(f"rest of {_code(layer)} (L)")
    parts += [
        f"{_code(f)} ({codes})"
        for f, codes in entry["never_with"].items()
        if f not in same_layer
    ]
    parts += [
        f"some {_code(f)} ({part['refused']} of {part['members']}, {part['rules']})"
        for f, part in entry["partly_with"].items()
    ]
    if entry["as_request_obfuscation_keeps_out"]:
        parts.append(
            f"as request obfuscation: "
            f"{_names(entry['as_request_obfuscation_keeps_out'])} (R)"
        )
    return ", ".join(parts)


def _technique_name(name: str, entry: dict) -> str:
    if entry["retired"]:
        return f"{_code(name)} (retired {entry['deprecated']})"
    return _code(name)


def _section_headline(catalog: dict) -> list[str]:
    head = catalog["headline"]
    pools = catalog["pools"]
    lines = ["## Headline", "", _row("What", "Count"), _rule_line(2)]
    lines += [
        _row("Spec version", catalog["spec_version"]),
        _row("Techniques registered", head["registered"]),
        _row("Active techniques", head["active"]),
        _row("Retired techniques", head["retired"]),
        _row("Layers", head["layers"]),
        _row("Families with an active technique", head["families"]),
        _row("Families in the spec", head["spec_families"]),
        _row("Compatible pairs of active techniques", head["compatible_pairs"]),
        _row("Variant pools", head["variant_pools"]),
        _row("Template variants", head["template_variants"]),
        _row("Model prompt files", head["prompt_files"]),
        _row("Jailbreak prompt steps", head["jailbreak_prompt_steps"]),
    ]
    lines += [
        _row(f"Pool {_code(name)}", size)
        for name, size in head["pools"].items()
        if pools[name]["on"]
    ]
    return lines


def _section_layers(catalog: dict) -> list[str]:
    sampling = catalog["sampling"]
    lines = [
        "## Layers",
        "",
        "Applied in this order:",
        "",
        _row(
            "#",
            "Layer",
            "Cap",
            "Rate",
            "Off by default",
            "Families",
            "Active",
            "Retired",
            "Variants",
        ),
        _rule_line(9),
    ]
    for layer in catalog["layers"]:
        lines.append(
            _row(
                layer["position"],
                _code(layer["name"]),
                layer["cap"] if layer["cap"] is not None else "none",
                layer["rate"] if layer["rate"] is not None else "",
                "yes" if layer["off_by_default"] else "no",
                len(layer["families"]),
                layer["techniques"],
                layer["retired"],
                layer["variants"],
            )
        )
    lines += ["", "Order within a layer (`within_layer_order`):", ""]
    for layer in catalog["layers"]:
        order = sorted(
            layer["within_layer_order"].items(), key=lambda kv: (kv[1], kv[0])
        )
        lines.append(
            f"- {_code(layer['name'])}: "
            + ", ".join(f"{_code(f)} {n}" for f, n in order)
        )
    request = sampling["request_obfuscation"]
    blocked = sampling["manipulation_blocked_obfuscation"]
    lines += [
        "",
        "### Budget and sampling",
        "",
        _row("Setting", "Value"),
        _rule_line(2),
        _row(
            "Complexity of a technique", f"0 to {sampling['max_technique_complexity']}"
        ),
        _row("Default `max_complexity`", sampling["default_max_complexity"]),
        _row("Default `max_obfuscations`", sampling["default_max_obfuscations"]),
        _row("Group budget headroom", sampling["group_budget_headroom"]),
        _row("`heavy_encoding_weight`", sampling["heavy_encoding_weight"]),
        _row("`decode_hint_rate`", sampling["decode_hint_rate"]),
        _row("Layers off by default", _names(sampling["layers_off_by_default"])),
        _row("Natural-only for the attacks", _names(sampling["natural_only_families"])),
        _row(
            "Request obfuscation (phase 5)",
            "on" if sampling["allow_request_obfuscation"] else "off",
        ),
        _row("Request obfuscation rate", request["rate"]),
        _row("Request obfuscation families", _names(request["allowed_families"])),
        _row("Request obfuscation candidates", len(request["candidates"])),
    ]
    lines += [
        _row(f"A {_code(family)} request obfuscation keeps out", _names(kept))
        for family, kept in request["refuses"].items()
    ]
    lines += [
        _row("Manipulation block: families", _names(blocked.get("families", []))),
        _row("Manipulation block: `encode_weight`", blocked.get("encode_weight", "")),
        "",
        "### Escalation rounds",
        "",
        _row("Round", "Settings"),
        _rule_line(2),
    ]
    lines += [
        _row(i, _code(json.dumps(r, sort_keys=True)))
        for i, r in enumerate(sampling["escalation"], start=1)
    ]
    return lines


def _section_families(catalog: dict) -> list[str]:
    lines = [
        "## Families",
        "",
        "Active and retired count by first family.",
        "",
        _row(
            "Family",
            "Layer",
            "Order",
            "Dimension",
            "Kinds",
            "Active",
            "Retired",
            "Also in",
            "Variants",
            "Floor",
            "Models",
            "Calls",
            "Refuses",
            "Refused by",
            "Blocked by manipulation",
        ),
        _rule_line(15),
    ]
    for family, entry in catalog["families"].items():
        lines.append(
            _row(
                _code(family),
                entry["layer"],
                entry["within_layer_order"],
                entry["dimension"] or "",
                " ".join(k[:3] for k in entry["kinds"]),
                len(entry["techniques"]),
                len(entry["retired"]),
                len(entry["also_in"]),
                entry["variants"],
                entry["floor"] if entry["floor"] is not None else "",
                ", ".join(entry["models"]),
                _calls(entry) or (entry["check"] or ""),
                _names(entry["refuses"]),
                _names(entry["refused_by"]),
                "yes" if entry["manipulation_blocked"] else "",
            )
        )
    return lines


_SYMBOLS = {"always": "✓", "sometimes": "◐", "never": "✗", "same": "=", "n/a": "·"}


def _section_matrix(catalog: dict) -> list[str]:
    matrix = catalog["family_matrix"]
    order = matrix["families"]
    legend = ", ".join(f"{symbol} {status}" for status, symbol in _SYMBOLS.items())
    lines = [
        "## Family matrix",
        "",
        f"Every pair of active members: {legend}.",
        "",
        _row("#", "Family", *range(1, len(order) + 1)),
        _rule_line(len(order) + 2),
    ]
    for i, a in enumerate(order, start=1):
        lines.append(
            _row(i, _code(a), *(_SYMBOLS[matrix["cells"][a][b][0]] for b in order))
        )
    return lines


def _section_techniques(catalog: dict) -> list[str]:
    legend = ", ".join(
        f"{letter} {_code(pool)}" for pool, letter in _POOL_LETTERS.items()
    )
    rules_legend = ", ".join(f"{code} {title}" for code, title in RULES.items())
    lines = [
        "## Techniques",
        "",
        f"Pools: {legend}.",
        "",
        f"Rules: {rules_legend}, {REQUEST_RULE} request obfuscation.",
        "",
        "Combines with: active partners `rules.compatible` admits.",
        "Never with: whole families; `some`: in part.",
        "",
    ]
    by_layer: dict[str, list[str]] = defaultdict(list)
    for name, entry in catalog["techniques"].items():
        by_layer[entry["layer"]].append(name)
    family_rank = {f: i for i, f in enumerate(catalog["families"])}
    for layer in catalog["layers"]:
        names = sorted(
            by_layer.get(layer["name"], ()),
            key=lambda n: (family_rank[catalog["techniques"][n]["first_family"]], n),
        )
        lines += [
            f"### Layer {_code(layer['name'])}",
            "",
            _row(
                "Technique",
                "Families",
                "Kinds",
                "Cx",
                "Calls",
                "Tags",
                "Variants",
                "Prompts",
                "Pools",
                "Since",
                "Combines with",
                "Never with",
            ),
            _rule_line(12),
        ]
        short = []
        for name in names:
            entry = catalog["techniques"][name]
            variants = entry["variants"]
            lines.append(
                _row(
                    _technique_name(name, entry),
                    _names(entry["families"]),
                    " ".join(k[:3] for k in entry["kinds"]),
                    entry["complexity"],
                    _calls(entry),
                    _tags(entry),
                    f"{variants} opt"
                    if entry["variant_pool_optional"]
                    else variants or "",
                    len(entry["prompts"]) or "",
                    _pool_letters(entry),
                    entry["since"],
                    entry["allowed_count"],
                    _never(entry, catalog),
                )
            )
            if entry["refused_count"] <= SHORT_LIST:
                short.append(
                    f"- {_code(name)} ({entry['refused_count']}): "
                    + _names(entry["refused_with"])
                )
        if short:
            lines += ["", "Refused partners, where few:", "", *short]
        lines.append("")
    return lines[:-1]


def _section_pools(catalog: dict) -> list[str]:
    pools = catalog["pools"]
    lines = [
        "## Pools",
        "",
        _row("Pool", "Kind", "Dataset type", "Section", "On", "Size", "Families"),
        _rule_line(7),
    ]
    for name, entry in pools.items():
        lines.append(
            _row(
                _code(name),
                entry["kind"],
                entry["dataset_type"] or "",
                _code(f"{entry['dataset_type']}.json")
                if entry["dataset_type"]
                else _code(json.dumps(entry["section"], sort_keys=True))
                if entry["section"] is not None
                else "",
                "yes" if entry["on"] else "no",
                entry["size"],
                len(
                    {
                        catalog["techniques"][n]["first_family"]
                        for n in entry["techniques"]
                    }
                ),
            )
        )
    listed: dict[tuple, str] = {}
    for name, entry in pools.items():
        if not entry["on"]:
            continue
        lines += ["", f"### Pool {_code(name)}", ""]
        key = tuple(entry["techniques"])
        if key in listed:
            lines.append(f"Same techniques as {_code(listed[key])}.")
            continue
        listed[key] = name
        grouped: dict[str, list[str]] = defaultdict(list)
        for technique in entry["techniques"]:
            grouped[catalog["techniques"][technique]["first_family"]].append(technique)
        for family in catalog["families"]:
            if family in grouped:
                lines.append(
                    f"- {_code(family)} ({len(grouped[family])}): "
                    + _names(grouped[family])
                )
        lines += ["", "Rounds:", ""]
        lines += [f"- {_code(json.dumps(r, sort_keys=True))}" for r in entry["rounds"]]
    return lines


def _section_prompts(catalog: dict) -> list[str]:
    prompts = catalog["prompts"]
    lines = [
        "## Prompts",
        "",
        "### Template variants by layer",
        "",
        _row("Layer", "Variant pools", "Variants"),
        _rule_line(3),
    ]
    for layer in catalog["layers"]:
        count = sum(
            1 for p in prompts["variant_pools"].values() if p["layer"] == layer["name"]
        )
        lines.append(_row(_code(layer["name"]), count, layer["variants"]))
    lines.append(_row("all", len(prompts["variant_pools"]), prompts["variants_total"]))
    lines += [
        "",
        "### Variant pools",
        "",
        _row("Pool", "Family", "Techniques", "Variants", "Floor"),
        _rule_line(5),
    ]
    for name, pool in prompts["variant_pools"].items():
        drawn = pool["techniques"]
        lines.append(
            _row(
                _code(name),
                _code(pool["family"]),
                _names(drawn) if len(drawn) <= 3 else f"{len(drawn)} techniques",  # noqa: PLR2004
                pool["variants"],
                pool["floor"] if pool["floor"] is not None else "",
            )
        )
    lines += [
        "",
        "### Model prompt files",
        "",
        _row("Pipeline", "Steps", "Files", "Few-shot examples"),
        _rule_line(4),
    ]
    for pipeline, entry in prompts["pipelines"].items():
        lines.append(
            _row(
                _code(pipeline),
                len(entry["steps"]),
                entry["files"],
                entry["few_shot_examples"],
            )
        )
    lines.append(
        _row(
            "all",
            sum(len(e["steps"]) for e in prompts["pipelines"].values()),
            sum(e["files"] for e in prompts["pipelines"].values()),
            sum(e["few_shot_examples"] for e in prompts["pipelines"].values()),
        )
    )
    lines += [
        "",
        "### Jailbreak prompt steps",
        "",
        _row("Step", "Files", "Techniques", "Families", "Stage"),
        _rule_line(5),
    ]
    steps = prompts["pipelines"].get("jailbreak", {}).get("steps", {})
    for step, users in prompts["jailbreak_steps"].items():
        families = sorted(
            {catalog["techniques"][n]["first_family"] for n in users["techniques"]},
            key=list(catalog["families"]).index,
        )
        lines.append(
            _row(
                _code(step),
                steps.get(step, {}).get("files", ""),
                len(users["techniques"]),
                _names(families),
                "benign pairs" if users["stage"] else "",
            )
        )
    lines += [
        "",
        "### Prompt steps by technique",
        "",
        _row("Steps", "Techniques"),
        _rule_line(2),
    ]
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for name, entry in catalog["techniques"].items():
        if entry["prompts"]:
            grouped[tuple(entry["prompts"])].append(name)
    for steps_used, names in sorted(grouped.items()):
        lines.append(_row(_names(steps_used), f"{len(names)}: {_names(names)}"))
    lines += _section_inputs(catalog["inputs"])
    return lines


def _section_inputs(inputs: dict) -> list[str]:
    lines = [
        "",
        "### Input side",
        "",
        f"Entry types ({len(inputs['entry_types'])}): "
        + _names(inputs["entry_types"])
        + ".",
        "",
    ]
    for name, taxonomy in inputs["taxonomies"].items():
        lines += [
            f"Taxonomy {_code(name)}: {taxonomy['categories']} categories, "
            f"{taxonomy['subcategories']} subcategories.",
            "",
            _row("Category", "Subcategories"),
            _rule_line(2),
        ]
        lines += [
            _row(category, count)
            for category, count in taxonomy["per_category"].items()
        ]
        lines.append("")
    benign = inputs["benign_pairs"]
    lines += [
        "Benign pairs:",
        "",
        _row("What", "Count"),
        _rule_line(2),
        _row("Subcategories", benign["subcategories"]),
        _row("Answer types", _names(benign["answer_types"])),
        _row("Units", benign["units"]),
        _row("Pairs per unit", benign["pairs_per_unit"]),
        _row("Pairs in all", benign["pairs"]),
        _row("Many-shot pairs", benign["many_shot_pairs"]),
        "",
        "Generation axes:",
        "",
        _row("Stage", "Axis", "Dimension", "Values (weight)"),
        _rule_line(4),
    ]
    for stage, stage_axes in inputs["axes"].items():
        for axis, entry in stage_axes.items():
            values = (
                ", ".join(f"{_code(v)} ({w})" for v, w in entry["values"].items())
                if entry["values"]
                else f"from {entry['from']}"
            )
            lines.append(_row(stage, _code(axis), entry["dimension"] or "", values))
    modes = inputs["answer_modes"]
    lines += [
        "",
        f"Answer modes: {_names(modes['modes'])}.",
        "",
        _row("Entry type", "Modes"),
        _rule_line(2),
    ]
    lines += [_row(_code(t), _names(m)) for t, m in modes["by_entry_type"].items()]
    return lines


def _section_combinations(catalog: dict) -> list[str]:
    lines = ["## Combination examples", ""]
    for kind, sim in catalog["combinations"].items():
        lines += [
            f"### Kind {_code(kind)}",
            "",
            _row("Plan", "Value"),
            _rule_line(2),
            _row("Units", sim["units"]),
            _row("Synthetic inputs", sim["inputs"]),
            _row("Rounds", sim["rounds"]),
            _row("Groups", sim["groups"]),
            _row("Entry types", _names(sim["entry_types"])),
            _row("Seed", sim["seed"]),
            _row("Pool size", sim["pool_size"]),
            _row("Techniques drawn", sim["techniques_drawn"]),
            _row("Never drawn", _names(sim["never_drawn"])),
            _row("Mean techniques per unit", sim["mean_techniques"]),
            "",
            _row("Techniques", "Units"),
            _rule_line(2),
        ]
        lines += [_row(k, n) for k, n in sim["by_size"].items()]
        lines += ["", _row("Layers touched", "Units"), _rule_line(2)]
        lines += [_row(k, n) for k, n in sim["by_layers_touched"].items()]
        lines += ["", _row("Layer", "Units"), _rule_line(2)]
        lines += [_row(_code(layer), n) for layer, n in sim["units_per_layer"].items()]
        if sim["top_family_pairs"]:
            lines += ["", _row("Family pair", "Units"), _rule_line(2)]
            lines += [
                _row(f"{_code(a)} + {_code(b)}", n)
                for a, b, n in sim["top_family_pairs"]
            ]
        else:
            lines += ["", "No unit draws two families."]
        lines.append("")
    return lines[:-1]


def _section_rules(catalog: dict) -> list[str]:
    lines = ["## Rules", "", _row("Code", "Rule", "Source"), _rule_line(3)]
    lines += [
        _row(code, entry["rule"], _code(entry["source"]))
        for code, entry in catalog["rules"].items()
    ]
    for key, text in catalog["spec_notes"].items():
        lines += ["", f"### Spec notes: {_code(key)}", "", "> " + text]
    lines += ["", "### Family descriptions", ""]
    lines += [
        f"- {_code(family)}: {entry['description']}"
        for family, entry in catalog["families"].items()
        if entry["description"]
    ]
    return lines


_SECTIONS = (
    ("Headline", _section_headline),
    ("Layers", _section_layers),
    ("Families", _section_families),
    ("Family matrix", _section_matrix),
    ("Techniques", _section_techniques),
    ("Pools", _section_pools),
    ("Prompts", _section_prompts),
    ("Combination examples", _section_combinations),
    ("Rules", _section_rules),
)


def markdown(catalog: dict) -> str:
    """``catalog`` as the Markdown page ``docs/catalog.md``."""
    lines = [
        "# Technique catalog",
        "",
        "<!-- Generated: redact.jailbreak.catalog -->",
        "",
        "Generated from the code; do not edit.",
        f"Regenerate: `{REGENERATE}`.",
        "Data: [catalog.json](catalog.json).",
        "",
        "Contents:",
        "",
    ]
    lines += [f"- [{title}](#{_anchor(title)})" for title, _ in _SECTIONS]
    for _, section in _SECTIONS:
        lines += ["", *section(catalog)]
    return "\n".join(lines) + "\n"


# --- the command line -----------------------------------------------------------


def default_dir() -> Path:
    """The repository's ``docs`` folder."""
    return paths.package_dir().parents[1] / "docs"


def main(argv: list[str] | None = None) -> int:
    """Write ``catalog.json`` and ``catalog.md`` (``--check``: only say
    whether they are current); returns the exit status."""
    parser = argparse.ArgumentParser(description="Write the technique catalog.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_dir(),
        help="the folder of catalog.json and catalog.md",
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if a file is stale"
    )
    args = parser.parse_args(argv)
    catalog = build()
    files = {
        args.out_dir / "catalog.json": dumps(catalog),
        args.out_dir / "catalog.md": markdown(catalog),
    }
    stale = [
        path
        for path, text in files.items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    if args.check:
        for path in stale:
            print(f"stale: {path}")
        return 1 if stale else 0
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
