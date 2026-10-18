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


# --- pick and include as they always dealt ------------------------------------
# The generation axes deal from the Dealer and are dealt again on every run of
# the inputs and outputs stages, so a change in how pick and include consume
# the rng or count would deal another spec to the units a resumed run has not
# generated yet. These pins were taken before the member counts joined it.

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
_PINNED_TRACE = ("c0b1a0b0b1a0c1b0a0c1b0a1b0b0c1b0c1a0b0b1a0c1b0a0b1c0a1b0b0c1b0b1"
                 "a0b0c1c0b1a0a0b1")


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
