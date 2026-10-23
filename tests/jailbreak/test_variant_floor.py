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
its floor. It refuses to record a pool with fewer variants than the fixture
has, or one the code no longer has: pass ``--allow-drop`` with the owner's
agreement.
"""

import argparse
import json
import re
from pathlib import Path

import pytest

from redact.jailbreak.pool import variant_pools

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "variant_floor.json"

#: The fewest variants a pool should have, by its family: 8 for a request or
#: framing directive, a container and a fake turn, 12 for a benign role, 20
#: for small talk, 6 for the persona templates' mechanics, 8 for the payload
#: split's reassembly and the word art's decoding and the wording's key and
#: puzzles (two pools), 4 for the decode hints of an encoding (each encoding
#: has a pool of its own, named ``<file>:<technique>``). A persona template's
#: frame is not counted: it is its technique's own text, one per persona by
#: design, not a pool the technique draws from, so the mechanics are the
#: family's one pool here.
FLOORS = {
    "framing": 8,
    "persona_template": 6,
    "encode": 4,
    "fake_turns": 8,
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
    "payload_split": 8,
    "word_art": 8,
    "wording": 8,
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


def lost(recorded: dict[str, dict], current: dict[str, dict]) -> list[str]:
    """The pools that have fewer variants now than ``recorded`` has for them,
    or are gone, as ``<name>: <now> of <recorded>``."""
    return sorted(
        f"{name}: {current.get(name, {}).get('variants', 0)} of {entry['variants']}"
        for name, entry in recorded.items()
        if current.get(name, {}).get("variants", 0) < entry["variants"])


def regenerate(fixture: Path = _FIXTURE, *, allow_drop: bool = False) -> None:
    """Write ``fixture`` from the code, keeping its ``known_short``.

    Raises:
        SystemExit: A pool has fewer variants than the fixture records, or is
            gone, and ``allow_drop`` is false; nothing is written.
    """
    existing = (json.loads(fixture.read_text(encoding="utf-8"))
                if fixture.exists() else None)
    pools = snapshot()
    if existing is not None and not allow_drop and (
            dropped := lost(existing["pools"], pools)):
        raise SystemExit(
            f"refusing to record fewer variants than {fixture.name} has: "
            f"{'; '.join(dropped)}. A drop is the owner's call; run with "
            "--allow-drop to record it.")
    known_short = (existing["known_short"] if existing is not None
                   else [name for name, entry in pools.items()
                         if entry["floor"] and entry["variants"] < entry["floor"]])
    fixture.write_text(_dumps(pools, known_short), encoding="utf-8")


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict[str, dict]:
    return snapshot()


def test_no_pool_has_fewer_variants_than_recorded(recorded, current):
    dropped = lost(recorded["pools"], current)
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


# --- the regeneration -----------------------------------------------------------

def _copy_of_the_fixture(tmp_path, change=None) -> Path:
    """A copy of the fixture in ``tmp_path``, after ``change(data)``."""
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    if change:
        change(data)
    copy = tmp_path / "variant_floor.json"
    copy.write_text(json.dumps(data), encoding="utf-8")
    return copy


def test_regenerating_the_current_fixture_writes_it_byte_for_byte(tmp_path):
    copy = _copy_of_the_fixture(tmp_path)
    regenerate(copy)
    assert copy.read_bytes() == _FIXTURE.read_bytes()


def test_a_pool_with_fewer_variants_is_refused_and_nothing_is_written(tmp_path):
    pool, count = next(iter(snapshot().items()))
    count = count["variants"]

    def one_more_recorded(data):
        data["pools"][pool]["variants"] += 1

    copy = _copy_of_the_fixture(tmp_path, one_more_recorded)
    before = copy.read_bytes()
    with pytest.raises(SystemExit, match=rf"{re.escape(pool)}: {count} of {count + 1}"):
        regenerate(copy)
    assert copy.read_bytes() == before


def test_a_pool_the_code_no_longer_has_is_refused(tmp_path):
    def a_pool_since_removed(data):
        data["pools"]["requests/answer_templates.json:answer_as_haiku"] = {
            "family": "answer_format", "variants": 8, "floor": 8}

    copy = _copy_of_the_fixture(tmp_path, a_pool_since_removed)
    with pytest.raises(SystemExit, match="answer_as_haiku: 0 of 8"):
        regenerate(copy)


def test_allow_drop_records_the_lower_count(tmp_path):
    pool = next(iter(snapshot()))

    def one_more_recorded(data):
        data["pools"][pool]["variants"] += 1

    copy = _copy_of_the_fixture(tmp_path, one_more_recorded)
    regenerate(copy, allow_drop=True)
    assert copy.read_bytes() == _FIXTURE.read_bytes()


def test_a_pool_that_gained_variants_is_recorded_without_a_flag(tmp_path):
    pool = next(iter(snapshot()))

    def fewer_recorded(data):
        data["pools"][pool]["variants"] -= 1

    copy = _copy_of_the_fixture(tmp_path, fewer_recorded)
    regenerate(copy)
    assert copy.read_bytes() == _FIXTURE.read_bytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate the floor fixture.")
    parser.add_argument("--allow-drop", action="store_true",
                        help="record a pool that has fewer variants than before")
    regenerate(allow_drop=parser.parse_args().allow_drop)
