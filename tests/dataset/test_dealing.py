"""The shared dealer: names dealt by count over weight, a name's members by
count, parts at a rate, and the bank of what earlier units left unspent."""

import random

import pytest

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


def test_members_of_one_name_come_out_even_and_apart_from_the_names():
    dealer = Dealer()
    rng = random.Random(0)
    dealt = []
    for _ in range(30):
        member = dealer.pick_member(rng, ["x1", "x2", "x3"])
        dealer.members[member] += 1
        dealt.append(member)
    assert sorted(dealt.count(m) for m in ("x1", "x2", "x3")) == [10, 10, 10]
    assert dealer.counts == {}


def test_a_member_tie_is_broken_by_the_rng_among_the_least_dealt():
    dealer = Dealer()
    dealer.members.update({"x1": 2, "x2": 1, "x3": 1})
    names = ["x1", "x2", "x3"]
    picks = {dealer.pick_member(random.Random(s), names) for s in range(20)}
    assert picks == {"x2", "x3"}


def test_a_part_at_rate_one_half_joins_half_the_units():
    dealer = Dealer(key=("group",))
    taken = 0
    for position in range(1, _UNITS + 1):
        dealer.position = position
        if dealer.include("layer", 0.5):
            dealer.included["layer"] += 1
            taken += 1
    assert taken == dealer.targets["layer"] == _HALF


@pytest.mark.parametrize("size", [1000, 2000])
@pytest.mark.parametrize("seed", [1, 2, 42, 43])
def test_two_parts_of_one_rate_meet_on_about_rate_times_rate_of_the_units(
        seed, size):
    """Each part keeps its rate, and whether a unit takes one says nothing
    of the other: they meet on rate x rate of the units, within a fifth.
    With one offset per part they met on every unit that took one, or on
    almost none, for the whole group."""
    dealer = Dealer(key=(seed, "group"))
    both = 0
    for position in range(1, size + 1):
        dealer.position = position
        took = [dealer.include(part, 0.5) for part in ("a", "b")]
        dealer.included.update({part: t for part, t in zip("ab", took)})
        both += all(took)
    assert dealer.included == {"a": size // 2, "b": size // 2}
    assert abs(both - 0.25 * size) <= 0.2 * 0.25 * size


def test_every_prefix_of_a_group_is_within_one_pick_of_the_rate():
    dealer = Dealer(key=("group",))
    for position in range(1, 601):
        dealer.position = position
        dealer.included["layer"] += dealer.include("layer", 0.35)
        assert abs(dealer.included["layer"] - 0.35 * position) < 1


def test_the_bank_is_what_the_earlier_units_left_unspent():
    dealer = Dealer()
    dealer.position = 1
    assert dealer.bank(6) == 0                  # no unit before the first
    dealer.position, dealer.spent = 3, 8        # two units before, 12 allowed
    assert dealer.bank(6) == 4


def test_the_bank_never_goes_below_zero():
    dealer = Dealer()
    dealer.position, dealer.spent = 2, 9
    assert dealer.bank(6) == 0


def test_a_stable_seed_is_the_same_for_the_same_parts():
    # The first 16 hex digits of the SHA-256 of "a::1", in every process.
    assert stable_seed("a", 1) == int("c8af6862b04087ce", 16)
    assert stable_seed("a", 1) != stable_seed("a", 2)


# --- pick and include as they always dealt ------------------------------------
# The generation axes deal from the Dealer and are dealt again on every run of
# the inputs and outputs stages, so a change in how pick and include consume
# the rng or count would deal another spec to the units a resumed run has not
# generated yet. These pins were taken before the member counts joined it.
# The axes call pick only; include, which the jailbreak plan calls (a kept
# plan replays its names), took a jitter per unit in place of one offset per
# part, which moved the trace's include flags (14 of the 40; still 16 picks)
# and none of its names, nor the rng, nor the axes' specs.

def _value(weight, **not_with):
    return {"label": "x", "hint": "", "weight": weight, "not_with": not_with}


_AXES = {
    "size": {"label": "Size", "values": {
        "small": _value(3), "medium": _value(2), "large": _value(1, tone=["terse"])}},
    "tone": {"label": "Tone", "values": {
        "plain": _value(1), "warm": _value(1), "terse": _value(0.5)}},
    "who": {"label": "Who", "values": {f"w{i}": _value(1) for i in range(7)}},
}
_PINNED_SPECS = "30e81995929826dfb2e1882b877655d465a1cd786e7ce8a947b1c98642e34958"
_PINNED_TRACE = ("c0b1a1b0b0a0c1b0a0c1b0a1b0b0c1b1c0a0b0b1a1c0b0a0b1c0a1b0b1c0b1"
                 "b0a0b1c0c0b1a0a1b0")


def test_the_axes_deal_the_pinned_specs():
    import hashlib
    import json

    from redact.content_moderation import axes as A

    units = [((category, entry_type), (f"entry {i}", it))
             for category in ("A", "B") for entry_type in ("harmful", "benign")
             for i in range(6) for it in range(3)]
    specs = A.deal_specs(_AXES, units, seed=7)
    assert specs[:3] == [{"size": "small", "tone": "terse", "who": "w1"},
                         {"size": "medium", "tone": "warm", "who": "w4"},
                         {"size": "large", "tone": "plain", "who": "w0"}]
    digest = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
    assert digest == _PINNED_SPECS


def test_pick_and_include_deal_the_pinned_trace():
    dealer = Dealer({"b": 2}, key=("g", 1))
    rng = random.Random(5)
    trace = []
    for position in range(1, 41):
        dealer.position = position
        name = dealer.pick(rng, ["a", "b", "c"] if position % 3 else ["a", "c"])
        dealer.counts[name] += 1
        took = dealer.include("layer", 0.4)
        dealer.included["layer"] += took
        trace.append(f"{name}{int(took)}")
    assert "".join(trace) == _PINNED_TRACE
    assert rng.random() == 0.9665643123171954       # the rng consumed as before
