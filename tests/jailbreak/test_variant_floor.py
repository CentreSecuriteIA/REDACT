"""The variant floor: how many variants each template pool may draw, ratcheted.

``fixtures/variant_floor.json`` records, for every variant pool a technique
draws from (``pool.variant_pools``, named ``<file>:<directive>``), its
family, its count of variants on (weight above 0) and its floor, the fewest
it should have, from :data:`FLOORS` by family. Its ``known_short`` list
names the pools below their floor that are known to be: the gaps still to
close.

The tests fail when:

- a pool has fewer variants than the fixture records (a variant removed or
  turned off); a drop is the owner's call;
- the fixture does not record every pool as the code counts it (a new pool,
  or a pool that grew): regenerate it to ratchet the count up;
- a pool's family has no floor in :data:`FLOORS`: add one;
- a pool is below its floor and not in ``known_short``;
- a pool in ``known_short`` has reached its floor, or no longer exists:
  take it out of the list (closing a gap shrinks the list).

Regenerate the fixture from the repository root with
``PYTHONPATH=src py -m tests.jailbreak.test_variant_floor``: it writes every
pool's family, count and floor from the code, and keeps ``known_short`` as
the file has it (a new pool below its floor is added to the list by hand,
with the owner's agreement); with no fixture yet, it lists every pool below
its floor.
"""

import json
from pathlib import Path

import pytest

from redact.jailbreak.pool import variant_pools

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "variant_floor.json"

#: The fewest variants a pool should have, by its family: 8 for a request or
#: framing directive and for a container, 12 for a benign role, 20 for small
#: talk. ``neutral_container``, ``benign_role`` and the small-talk families
#: have no pool yet.
FLOORS = {
    "framing": 8,
    "answer_format": 8,
    "answer_language": 8,
    "asking": 8,
    "continuation": 8,
    "distractor_prefix": 8,
    "distractor_suffix": 8,
    "impersonation": 8,
    "indirect": 8,
    "temporal": 8,
    "structural": 8,
    "neutral_container": 8,
    "benign_role": 12,
    "smalltalk_prefix": 20,
    "smalltalk_suffix": 20,
}


def snapshot() -> dict[str, dict]:
    """Each pool's family, count of variants on and floor, as the code
    gives them now; a family without a floor has floor ``None``."""
    return {name: {"family": pool["family"], "variants": pool["variants"],
                   "floor": FLOORS.get(pool["family"])}
            for name, pool in sorted(variant_pools().items())}


def _dumps(pools: dict[str, dict], known_short: list[str]) -> str:
    """The fixture as JSON, one pool or known-short name per line."""
    rows = ",\n".join(f"  {json.dumps(name)}: {json.dumps(entry)}"
                      for name, entry in pools.items())
    short = ",\n".join(f"  {json.dumps(name)}" for name in sorted(known_short))
    return (
        '{\n "_comment": "Regenerate with PYTHONPATH=src py -m '
        'tests.jailbreak.test_variant_floor; known_short is kept by hand.",\n'
        f' "pools": {{\n{rows}\n }},\n "known_short": [\n{short}\n ]\n}}\n')


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict[str, dict]:
    return snapshot()


def test_no_pool_has_fewer_variants_than_recorded(recorded, current):
    dropped = sorted(
        f"{name}: {current.get(name, {}).get('variants', 0)} of {entry['variants']}"
        for name, entry in recorded["pools"].items()
        if current.get(name, {}).get("variants", 0) < entry["variants"])
    assert dropped == [], (
        "these pools lost variants (removed, or weight 0); if that is meant, "
        "regenerate the fixture")


def test_the_fixture_records_every_pool_as_counted(recorded, current):
    assert current == recorded["pools"], (
        "regenerate with PYTHONPATH=src py -m tests.jailbreak.test_variant_floor")


def test_every_pool_s_family_has_a_floor(current):
    unfloored = sorted({entry["family"] for entry in current.values()
                        if entry["floor"] is None})
    assert unfloored == [], "add the families to FLOORS in this module"


def test_a_pool_below_its_floor_is_known_short(recorded, current):
    short = sorted(name for name, entry in current.items()
                   if entry["floor"] is not None and entry["variants"] < entry["floor"])
    unknown = sorted(set(short) - set(recorded["known_short"]))
    assert unknown == [], (
        "these pools are below their floor: give them more variants, or add "
        "them to known_short with the owner's agreement")


def test_a_known_short_pool_is_still_short(recorded, current):
    closed = sorted(name for name in recorded["known_short"]
                    if name not in current
                    or current[name]["variants"] >= current[name]["floor"])
    assert closed == [], "take these pools out of known_short"


if __name__ == "__main__":
    existing = (json.loads(_FIXTURE.read_text(encoding="utf-8"))
                if _FIXTURE.exists() else None)
    pools = snapshot()
    known_short = (existing["known_short"] if existing is not None
                   else [name for name, entry in pools.items()
                         if entry["floor"] and entry["variants"] < entry["floor"]])
    _FIXTURE.write_text(_dumps(pools, known_short), encoding="utf-8")
