"""The shared dealer: names dealt by count over weight, parts at a rate."""

import random

from redact.dataset.dealing import Dealer, stable_seed

_UNITS = 20
_HALF = _UNITS // 2


def _deal(dealer: Dealer, names: list[str], n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    dealt = []
    for _ in range(n):
        name = dealer.pick(rng, names)
        dealer.counts[name] += 1
        dealt.append(name)
    return dealt


def test_names_of_equal_weight_come_out_even():
    dealt = _deal(Dealer(), ["a", "b", "c"], 30)
    assert sorted(dealt.count(n) for n in "abc") == [10, 10, 10]


def test_a_name_gets_its_share_of_the_weights():
    dealt = _deal(Dealer({"a": 3, "b": 1}), ["a", "b"], 2 * _UNITS)
    assert dealt.count("a") == 3 * _HALF


def test_ties_are_broken_by_the_rng():
    firsts = {_deal(Dealer(), ["a", "b", "c"], 1, seed=s)[0] for s in range(20)}
    assert firsts == {"a", "b", "c"}


def test_pick_takes_only_the_names_given():
    dealer = Dealer()
    dealer.counts["a"] = 5
    assert dealer.pick(random.Random(0), ["a"]) == "a"


def test_a_part_at_rate_one_half_joins_half_the_units():
    dealer = Dealer(key=("group",))
    taken = 0
    for position in range(1, _UNITS + 1):
        dealer.position = position
        if dealer.include("layer", 0.5):
            dealer.included["layer"] += 1
            taken += 1
    assert taken == dealer.targets["layer"] == _HALF


def test_a_stable_seed_is_the_same_for_the_same_parts():
    # The first 16 hex digits of the SHA-256 of "a::1", in every process.
    assert stable_seed("a", 1) == int("c8af6862b04087ce", 16)
    assert stable_seed("a", 1) != stable_seed("a", 2)
