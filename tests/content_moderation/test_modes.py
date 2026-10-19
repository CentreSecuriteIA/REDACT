"""The answer modes of the outputs stage: the mapping from entry type to
modes, the label of a response, and the deal of one mode per unit."""

import random
from collections import Counter

import pytest

from redact.content_moderation import modes as M
from redact.types import ALL_ENTRY_TYPES, EntryType

_SEED = 42
_BOTH = ["harmful", "helpful"]


def _units(n: int, group=("Cyber", "dual_use_harmful"), k: int = 1) -> list[M.Unit]:
    """``n`` inputs of ``group``, ``k`` units each."""
    return [(group, (f"in-{i}", it)) for i in range(n) for it in range(k)]


def _dealt(units, modes=None) -> list[str]:
    return M.deal_modes(modes or M.resolve_modes(), units, seed=_SEED)


def _per_input(units, dealt) -> dict[str, Counter]:
    per: dict[str, Counter] = {}
    for (_, (entry, _)), mode in zip(units, dealt, strict=True):
        per.setdefault(entry, Counter())[mode] += 1
    return per


class TestTheMapping:
    def test_the_default_is_the_owners_table_for_every_entry_type(self):
        assert M.resolve_modes() == {
            "harmful": ["harmful"],
            "dual_use_harmful": _BOTH,
            "dual_use_benign": _BOTH,
            "benign": ["helpful"],
        }
        assert list(M.resolve_modes()) == [t.value for t in ALL_ENTRY_TYPES]

    def test_the_default_is_a_copy(self):
        M.resolve_modes()["harmful"].append("helpful")
        assert M.resolve_modes()["harmful"] == ["harmful"]
        assert M.DEFAULT_MODES["harmful"] == ["harmful"]

    def test_an_entry_type_given_replaces_its_modes_and_keeps_the_others(self):
        resolved = M.resolve_modes({"benign": ["harmful", "helpful"]})
        assert resolved["benign"] == _BOTH
        assert resolved["harmful"] == ["harmful"]
        assert resolved["dual_use_benign"] == _BOTH

    def test_a_comment_key_is_ignored(self):
        assert M.resolve_modes({"_note": "anything", "harmful": ["helpful"]}) == {
            **M.resolve_modes(), "harmful": ["helpful"]}

    @pytest.mark.parametrize(("given", "message"), [
        (["harmful"], "must be an object"),
        ("harmful", "must be an object"),
        ({"general_benign": ["helpful"]}, "names no entry type 'general_benign'"),
        ({"benign": []}, "non-empty list of distinct modes"),
        ({"benign": "helpful"}, "non-empty list of distinct modes"),
        ({"benign": ["safe"]}, "non-empty list of distinct modes"),
        ({"benign": ["helpful", "helpful"]}, "non-empty list of distinct modes"),
        ({"benign": [{"helpful": 1}]}, "non-empty list of distinct modes"),
    ])
    def test_a_wrong_mapping_raises_naming_the_key(self, given, message):
        with pytest.raises(ValueError, match=message) as raised:
            M.resolve_modes(given)
        assert "outputs.modes" in str(raised.value)


class TestTheLabel:
    @pytest.mark.parametrize(("entry_type", "mode", "label"), [
        # The four the default mapping deals.
        ("harmful", "harmful", "harmful_harmful"),
        ("dual_use_harmful", "harmful", "dual_use_harmful_harmful"),
        ("dual_use_harmful", "helpful", "dual_use_harmful_benign"),
        ("dual_use_benign", "harmful", "dual_use_benign_harmful"),
        ("dual_use_benign", "helpful", "dual_use_benign_benign"),
        ("benign", "helpful", "benign_benign"),
        # The two a custom mapping can give besides.
        ("harmful", "helpful", "harmful_benign"),
        ("benign", "harmful", "benign_harmful"),
    ])
    def test_the_label_is_the_entry_type_then_the_modes_output_label(
            self, entry_type, mode, label):
        assert M.response_label(entry_type, mode) == label
        assert label.rsplit("_", 1) == [entry_type, M.OUTPUT_LABEL[mode]]

    def test_an_entry_type_member_gives_its_value(self):
        assert M.response_label(EntryType.DUAL_USE_BENIGN, "helpful") == (
            "dual_use_benign_benign")

    def test_another_mode_raises_naming_the_modes(self):
        with pytest.raises(ValueError, match=r"\['harmful', 'helpful'\]"):
            M.response_label("benign", "benign")


class TestTheDeal:
    @pytest.mark.parametrize(("entry_type", "mode"), [
        ("harmful", "harmful"), ("benign", "helpful")])
    def test_one_allowed_mode_gives_every_unit_that_mode(self, entry_type, mode):
        units = _units(5, group=("Cyber", entry_type), k=3)
        assert set(_dealt(units)) == {mode}

    @pytest.mark.parametrize("entry_type", ["dual_use_harmful", "dual_use_benign"])
    def test_at_two_per_input_a_dual_use_input_gets_one_of_each(self, entry_type):
        units = _units(9, group=("Cyber", entry_type), k=2)
        dealt = _dealt(units)
        assert all(counts == Counter(_BOTH) for counts in
                   _per_input(units, dealt).values())
        assert Counter(dealt) == Counter({"harmful": 9, "helpful": 9})

    def test_at_one_per_input_a_group_of_six_gives_three_and_three(self):
        assert Counter(_dealt(_units(6))) == Counter({"harmful": 3, "helpful": 3})

    def test_at_three_per_input_each_input_gets_two_and_one(self):
        units = _units(7, k=3)
        dealt = _dealt(units)
        assert all(sorted(counts.values()) == [1, 2]
                   for counts in _per_input(units, dealt).values())
        totals = Counter(dealt)
        assert abs(totals["harmful"] - totals["helpful"]) <= 1

    def test_the_deal_is_the_same_whatever_the_order_of_the_units(self):
        units = _units(8, k=2) + _units(5, group=("Fraud", "dual_use_benign"), k=3)
        shuffled = list(units)
        random.Random(3).shuffle(shuffled)
        by_unit = dict(zip(units, _dealt(units), strict=True))
        assert dict(zip(shuffled, _dealt(shuffled), strict=True)) == by_unit

    def test_at_two_per_input_an_inputs_modes_stay_when_others_join_its_group(self):
        few = _units(4, k=2)
        many = _units(4, k=2) + [(("Cyber", "dual_use_harmful"), (f"new-{i}", it))
                                 for i in range(7) for it in range(2)]
        before = dict(zip(few, _dealt(few), strict=True))
        after = dict(zip(many, _dealt(many), strict=True))
        assert {unit: after[unit] for unit in few} == before

    def test_two_rows_with_one_input_id_in_one_group_are_dealt_without_an_error(self):
        units = _units(1, k=2) * 2
        dealt = _dealt(units)
        assert len(dealt) == len(units)
        assert Counter(dealt) == Counter({"harmful": 2, "helpful": 2})

    def test_groups_are_dealt_apart_from_the_modes_their_entry_type_allows(self):
        units = (_units(4, group=("Cyber", "harmful"), k=2)
                 + _units(4, group=("Cyber", "benign"), k=2)
                 + _units(4, group=("Cyber", "dual_use_benign"), k=2))
        by_group: dict[str, Counter] = {}
        for (group, _), mode in zip(units, _dealt(units), strict=True):
            by_group.setdefault(group[1], Counter())[mode] += 1
        assert by_group == {"harmful": Counter(harmful=8),
                            "benign": Counter(helpful=8),
                            "dual_use_benign": Counter(harmful=4, helpful=4)}

    def test_a_custom_mapping_deals_its_modes(self):
        modes = M.resolve_modes({"harmful": ["helpful", "harmful"]})
        units = _units(4, group=("Cyber", "harmful"), k=2)
        assert Counter(_dealt(units, modes)) == Counter(harmful=4, helpful=4)
