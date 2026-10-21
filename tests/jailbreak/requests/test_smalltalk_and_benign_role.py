"""Small talk and benign roles (requests/smalltalk.py, requests/benign_role.py):
an ordinary line before or after the prompt, the prompt kept as it is."""

import random
import re

import pytest

import redact.jailbreak.directives as D
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.requests.benign_role import get_benign_role_functions
from redact.jailbreak.requests.smalltalk import (
    get_smalltalk_functions,
    to_smalltalk_suffix,
)
from redact.jailbreak.technique import applies
from redact.types import ALL_ENTRY_TYPES

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, café.',
]

#: Words no variant holds: a greeting or a role gives no instruction and says
#: nothing about rules or replies. The neutral containers' starting list, and
#: the words of a role that sets limits aside.
NOT_NEUTRAL = (
    "ignore", "override", "bypass", "disregard", "obey", "unrestricted",
    "jailbreak", "system", "admin", "administrator", "instruction",
    "instructions", "policy", "policies", "safety", "rules", "answer",
    "respond", "reply", "must", "always", "never",
    "pretend", "character", "limit", "limits", "restriction", "restrictions",
    "filter", "unfiltered", "uncensored", "guidelines",
)

SMALLTALK = get_smalltalk_functions()
ROLES = get_benign_role_functions()
FUNCTIONS = SMALLTALK + ROLES

#: The variants each pool holds: its floor, 20 for small talk, 12 for a role.
_SIZES = {"to_smalltalk_prefix": 20, "to_smalltalk_suffix": 20,
          "to_benign_role_subject_helper": 12, "to_benign_role_tutor": 12,
          "to_benign_role_task_assistant": 12}

#: The key of each technique's info that names its directive.
_INFO_KEY = {fn.__name__: "smalltalk" for fn in SMALLTALK} | {
    fn.__name__: "role" for fn in ROLES}


def _pool(fn) -> list[dict]:
    ref = fn.variant_pool
    return read_pools(ref.file)[ref.directive]


def _variants(functions: list) -> list[tuple]:
    return [(fn, variant) for fn in functions for variant in _pool(fn)]


def _ids(pairs: list[tuple]) -> list[str]:
    return [f"{fn.__name__}-{variant['name']}" for fn, variant in pairs]


_ALL = _variants(FUNCTIONS)
#: A role stands before the prompt, as a small-talk prefix does.
_BEFORE = _variants([fn for fn in FUNCTIONS if fn is not to_smalltalk_suffix])
_AFTER = _variants([to_smalltalk_suffix])


def _written(monkeypatch, fn, variant: dict, text: str) -> tuple[str, str]:
    """What ``fn`` writes for ``text`` when it draws ``variant``."""
    monkeypatch.setattr(D, "draw_variant", lambda variants, rng=None: variant)
    return fn(text)


def test_the_pools_are_ones_their_techniques_can_draw():
    check_pools(fn.variant_pool for fn in FUNCTIONS)


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_each_pool_holds_its_floor_of_variants_each_named_once(fn):
    names = [variant["name"] for variant in _pool(fn)]
    assert len(names) == len(set(names)) == _SIZES[fn.__name__]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize(("fn", "variant"), _ALL, ids=_ids(_ALL))
def test_every_variant_holds_the_text_once_and_is_named_in_the_info(
        monkeypatch, fn, variant, text):
    written, info = _written(monkeypatch, fn, variant, text)
    assert variant["template"].count("{prompt}") == 1
    assert written.count(text) == 1
    assert info == (f"{_INFO_KEY[fn.__name__]}={fn.variant_pool.directive};"
                    f"variant={variant['name']}")


@pytest.mark.parametrize(("fn", "variant"), _BEFORE, ids=_ids(_BEFORE))
def test_a_prefix_or_a_role_puts_the_text_after_its_line(monkeypatch, fn, variant):
    written, _ = _written(monkeypatch, fn, variant, TEXTS[0])
    assert written.endswith(TEXTS[0])
    assert written[:-len(TEXTS[0])].strip()


@pytest.mark.parametrize(("fn", "variant"), _AFTER, ids=_ids(_AFTER))
def test_a_suffix_puts_the_text_before_its_line(monkeypatch, fn, variant):
    written, _ = _written(monkeypatch, fn, variant, TEXTS[0])
    assert written.startswith(TEXTS[0])
    assert written[len(TEXTS[0]):].strip()


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_a_seed_draws_its_variant_as_rng_choice_does(fn):
    for seed in range(20):
        ours, theirs = random.Random(seed), random.Random(seed)
        _, info = fn(TEXTS[0], rng=ours)
        assert info.endswith(f";variant={theirs.choice(_pool(fn))['name']}")
        assert ours.getstate() == theirs.getstate()


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_a_seed_gives_the_same_text(fn):
    assert fn(TEXTS[1], rng=random.Random(3)) == fn(TEXTS[1], rng=random.Random(3))


@pytest.mark.parametrize(("fn", "variant"), _ALL, ids=_ids(_ALL))
def test_no_variant_holds_a_word_of_instruction(fn, variant):
    words = set(re.findall(r"[a-z]+", variant["template"].lower()))
    assert sorted(words & set(NOT_NEUTRAL)) == []


@pytest.mark.parametrize("fn", ROLES, ids=lambda fn: fn.__name__)
def test_a_benign_role_applies_to_benign_and_dual_use_benign_inputs_only(fn):
    assert {entry_type.value for entry_type in ALL_ENTRY_TYPES
            if applies(fn, TEXTS[0], entry_type.value)} == {"benign", "dual_use_benign"}


@pytest.mark.parametrize("fn", SMALLTALK, ids=lambda fn: fn.__name__)
def test_small_talk_applies_to_every_input(fn):
    assert all(applies(fn, TEXTS[0], entry_type.value)
               for entry_type in ALL_ENTRY_TYPES)
