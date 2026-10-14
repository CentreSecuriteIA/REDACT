"""The spec's rules, over the registered techniques: they bind both ways."""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.hacking import get_all_hacking_functions
from redact.jailbreak.manipulation import get_all_manipulation_functions
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.requests import get_all_request_functions
from redact.jailbreak.rules import compatible, request_obfuscation_candidates
from redact.jailbreak.sampling import sample_exact_combination

POOL = (get_all_obfuscation_functions() + get_all_hacking_functions()
        + get_all_manipulation_functions() + get_all_request_functions())
BY_NAME = {fn.__name__: fn for fn in POOL}


def _joins(selected: str, candidate: str) -> bool:
    return BY_NAME[candidate] in compatible([BY_NAME[selected]], [BY_NAME[candidate]])


@pytest.mark.parametrize(("first", "then"), [
    ("to_fictional_world", "to_ascii_art"),
    ("to_ascii_art", "to_fictional_world"),
    ("to_fsh_random_short", "to_ascii_art"),
    ("to_ascii_art", "to_fsh_random_short"),
])
def test_a_cross_incompatibility_refuses_either_order(first, then):
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_french", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_french"),
    ("to_base64", "to_dap_random_long"),
    ("to_dap_random_long", "to_base64"),
])
def test_manipulation_blocking_refuses_either_order(first, then):
    assert not _joins(first, then)


@pytest.mark.parametrize(("first", "then"), [
    ("to_sensitive_words_encode_base64", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_sensitive_words_encode_base64"),
    ("to_rot13", "to_fsh_random_short"),
    ("to_fsh_random_short", "to_rot13"),
])
def test_manipulation_admits_light_and_word_level_encoding(first, then):
    assert _joins(first, then)


def test_every_pair_is_refused_or_admitted_both_ways():
    joins = {fn.__name__: {g.__name__ for g in compatible([fn], POOL)} for fn in POOL}
    one_way = [(a, b) for a in joins for b in joins[a] if a not in joins[b]]
    assert one_way == []


def test_an_exact_draw_never_holds_a_forbidden_pair():
    for seed in range(200):
        fn = sample_exact_combination(random.Random(seed), POOL, 3)
        names = [t.__name__ for t in fn.techniques]
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                assert _joins(name, other), (seed, names)


def test_request_obfuscation_is_light_encoding_or_translation():
    names = {fn.__name__ for fn in request_obfuscation_candidates(POOL)}
    assert {"to_rot13", "to_french"} <= names
    assert not names & {"to_base64", "to_ascii_art", "to_summarize"}
