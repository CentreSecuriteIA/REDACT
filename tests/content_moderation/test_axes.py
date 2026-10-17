"""The generation axes: the packaged table, its overlay, the deal of one spec
per unit, the rendered spec, the columns and the coverage tables."""

import csv
from collections import Counter

import pytest

from redact.content_moderation import axes as A
from redact.dataset import Ledger, Manifest

_ENTRIES = 40       # entries of one group in a deal
_K = 3              # units (siblings) per entry
_SEED = 42
_GIVEN = 0.5        # a weight a params file gives
_LENGTHS = ("fragment", "one_liner", "short", "long", "very_long")
# How far a value's share of a group of 54 may be from its weight: the
# writer's 14 roles at .05 each take 3 of 54 rather than 2.7.
_POINTS = 0.08
_EXPLICIT_AT_MOST = 2   # ceil(.55 * 3)


def _units(n: int, group=("Cyber", "harmful"), k: int = _K) -> list[A.Unit]:
    """``n`` entries of ``group``, ``k`` units each, an entry's units in a row."""
    return [(group, (f"entry {i}", it)) for i in range(n) for it in range(k)]


def _spec_counts(specs: list[dict], axis: str) -> Counter:
    return Counter(spec[axis] for spec in specs)


class TestThePackagedTable:
    @pytest.mark.parametrize("stage", A.STAGES)
    def test_each_stage_has_axes_with_a_value_to_deal(self, stage):
        axes = A.resolve(stage)
        dealt = [name for name, entry in axes.items() if "values" in entry]
        assert dealt
        for name in dealt:
            values = axes[name]["values"].values()
            assert any(value["weight"] > 0 for value in values), name
            assert all(value["label"] and value["hint"] for value in values), name

    def test_the_input_axes_are_those_of_the_plan(self):
        assert list(A.resolve("inputs")) == [
            "length", "register", "writer", "framing", "directness", "focus",
            "domain"]

    def test_the_writer_is_anonymous_or_one_of_the_roles(self):
        writer = A.resolve("inputs")["writer"]["values"]
        assert writer["anonymous"]["weight"] == pytest.approx(0.30)
        roles = [v for name, v in writer.items() if name != "anonymous"]
        assert sum(v["weight"] for v in roles) == pytest.approx(0.70)

    def test_the_response_axes_are_length_format_and_register(self):
        assert list(A.resolve("outputs")) == ["length", "format", "register"]


    def test_focus_deals_one_subject_only(self):
        values = A.resolve("inputs")["focus"]["values"]
        assert [name for name, v in values.items() if v["weight"] > 0] == ["single"]


class TestTheOverlay:
    def test_none_and_true_are_the_packaged_axes(self):
        packaged = A.resolve("inputs")
        assert A.resolve("inputs", None) == A.resolve("inputs", True) == packaged

    def test_false_is_no_axes(self):
        assert A.resolve("inputs", False) is None

    def test_a_weight_given_replaces_the_packaged_one(self):
        axes = A.resolve("inputs", {"length": {"very_long": _GIVEN}})
        assert axes["length"]["values"]["very_long"]["weight"] == _GIVEN
        assert axes["length"]["values"]["fragment"]["weight"] == pytest.approx(0.15)

    def test_an_axis_given_as_null_is_left_out(self):
        assert "writer" not in A.resolve("inputs", {"writer": None})

    def test_comment_keys_are_skipped(self):
        assert A.resolve("inputs", {"_note": "x", "length": {"_why": "y"}}) == \
            A.resolve("inputs")

    def test_no_axis_left_to_deal_is_no_axes(self):
        given = dict.fromkeys(("length", "format", "register"))
        assert A.resolve("outputs", given) is None

    @pytest.mark.parametrize(("given", "message"), [
        ({"lenght": {}}, "no axis 'lenght'"),
        ({"length": {"tiny": 1}}, "no value 'tiny'"),
        ({"length": {"short": -1}}, "a number of at least 0"),
        ({"length": {"short": True}}, "a number of at least 0"),
        ({"length": {"short": "1"}}, "a number of at least 0"),
        ({"length": dict.fromkeys(_LENGTHS, 0)}, "every weight"),
        ({"length": ["short"]}, "an object of value: weight"),
        ({"domain": {"x": 1}}, "takes no weights"),
        (["length"], "an object of axis"),
        ("on", "an object of axis"),
    ])
    def test_a_wrong_name_or_weight_raises(self, given, message):
        with pytest.raises(ValueError, match=message):
            A.resolve("inputs", given)

    def test_an_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="no generation axes"):
            A.resolve("jailbreaks")


class TestTheDeal:
    def test_each_value_gets_its_share_of_a_group(self):
        axes = A.resolve("inputs", {"length": {"fragment": 1, "one_liner": 1,
                                                "short": 2, "long": 0, "very_long": 0}})
        counts = _spec_counts(A.deal_specs(axes, _units(_ENTRIES), seed=_SEED),
                              "length")
        total = _ENTRIES * _K
        assert counts == {"fragment": total // 4, "one_liner": total // 4,
                          "short": total // 2}

    def test_a_group_of_54_lands_within_a_few_points_of_the_weights(self):
        axes = A.resolve("inputs")
        specs = A.deal_specs(axes, _units(18), seed=_SEED)
        for axis, entry in axes.items():
            counts = _spec_counts(specs, axis) if "values" in entry else {}
            weight = sum(v["weight"] for v in entry.get("values", {}).values())
            for value, v in entry.get("values", {}).items():
                share = counts.get(value, 0) / len(specs)
                assert share == pytest.approx(v["weight"] / weight, abs=_POINTS)

    def test_siblings_differ_on_an_axis_none_of_whose_values_has_over_a_kth(self):
        """At k = 3: length and writer (no value above .30); register only
        when its .35 value is brought down."""
        axes = A.resolve("inputs", {"register": dict.fromkeys(
            ("casual", "neutral", "formal", "technical", "emotional"), 1)})
        specs = A.deal_specs(axes, _units(_ENTRIES), seed=_SEED)
        for i in range(0, len(specs), _K):
            siblings = specs[i:i + _K]
            for axis in ("length", "writer", "register"):
                assert len({spec[axis] for spec in siblings}) == _K

    def test_an_entry_takes_a_value_at_most_its_share_of_k_rounded_up(self):
        """Directness at k = 3: explicit (.55) twice at most, the others once."""
        specs = A.deal_specs(A.resolve("inputs"), _units(_ENTRIES), seed=_SEED)
        for i in range(0, len(specs), _K):
            counts = _spec_counts(specs[i:i + _K], "directness")
            assert counts["explicit"] <= _EXPLICIT_AT_MOST
            assert counts["implicit"] <= 1 and counts["euphemistic"] <= 1

    def test_the_deal_is_the_same_whatever_the_order_of_the_units(self):
        axes = A.resolve("inputs")
        units = _units(_ENTRIES)
        dealt = dict(zip(units, map(str, A.deal_specs(axes, units, seed=_SEED))))
        backwards = list(reversed(units))
        again = A.deal_specs(axes, backwards, seed=_SEED)
        assert dict(zip(backwards, map(str, again))) == dealt

    def test_groups_are_dealt_apart(self):
        axes = A.resolve("inputs")
        one = A.deal_specs(axes, _units(_ENTRIES), seed=_SEED)
        mixed = _units(_ENTRIES) + _units(_ENTRIES, ("Fraud", "benign"))
        assert A.deal_specs(axes, mixed, seed=_SEED)[:len(one)] == one

    def test_another_seed_deals_another_plan(self):
        axes = A.resolve("inputs")
        assert A.deal_specs(axes, _units(_ENTRIES), seed=1) != \
            A.deal_specs(axes, _units(_ENTRIES), seed=2)

    def test_a_value_is_never_dealt_with_one_it_names_in_not_with(self):
        axes = A.resolve("inputs", {"focus": {"mixed": 0.25}})
        for spec in A.deal_specs(axes, _units(4 * _ENTRIES), seed=_SEED):
            if spec["focus"] == "mixed":
                assert spec["length"] not in ("fragment", "one_liner")
            if spec["framing"] == "advice":
                assert spec["length"] != "fragment"

    def test_a_value_of_weight_zero_is_never_dealt(self):
        axes = A.resolve("outputs", {"format": {"code": 0}})
        specs = A.deal_specs(axes, _units(_ENTRIES, k=1), seed=_SEED)
        assert "code" not in _spec_counts(specs, "format")

    def test_a_spec_names_every_dealt_axis_only(self):
        axes = A.resolve("inputs")
        (spec,) = A.deal_specs(axes, _units(1, k=1), seed=_SEED)
        assert list(spec) == ["length", "register", "writer", "framing",
                              "directness", "focus"]

    def test_no_axes_deal_empty_specs(self):
        assert A.deal_specs(None, _units(2, k=2), seed=_SEED) == [{}] * 4


class TestRendering:
    _AXES = {
        "length": {"label": "Length", "values": {
            "short": {"label": "short", "hint": "2 to 4 sentences.", "weight": 1}}},
        "tone": {"label": "Tone", "values": {
            "dry": {"label": "dry", "hint": "", "weight": 1}}},
        "domain": {"label": "Domain", "from": "subcategory"},
    }

    def test_one_spec_is_a_line_per_axis(self):
        text = A.render_spec(self._AXES, {"length": "short", "tone": "dry"},
                             {"subcategory": "Phishing"})
        assert text == ("Length: short — 2 to 4 sentences.\nTone: dry\n"
                        "Domain: Phishing")

    def test_an_empty_field_has_no_line(self):
        text = A.render_spec(self._AXES, {"length": "short", "tone": "dry"},
                             {"subcategory": ""})
        assert text == "Length: short — 2 to 4 sentences.\nTone: dry"

    def test_no_axes_render_nothing(self):
        assert A.render_spec(None, {}, {}) == ""

    def test_columns_are_the_dealt_axes_with_the_stage_prefix(self):
        spec = {"length": "short", "tone": "dry"}
        assert A.spec_columns(self._AXES, spec, prefix="response_") == {
            "response_length": "short", "response_tone": "dry"}
        assert A.spec_columns(self._AXES, {"length": "short"}) == {
            "length": "short", "tone": ""}
        assert A.spec_columns(None, {}) == {}


class TestCoverage:
    def test_every_value_of_every_group_has_a_row(self, tmp_path):
        axes = {"length": {"label": "Length", "values": {
            "short": {"label": "s", "hint": "", "weight": 1},
            "long": {"label": "l", "hint": "", "weight": 1}}},
            "domain": {"label": "Domain", "from": "subcategory"}}
        cyber, fraud = ("Cyber", "harmful"), ("Fraud", "benign")
        planned = [(cyber, {"length": "short"}), (cyber, {"length": "short"}),
                   (fraud, {"length": "long"})]
        accepted = [(cyber, {"length": "short"})]
        path = A.write_coverage(tmp_path / "x.coverage.csv", axes, planned, accepted)

        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == [
            {"axis": "length", "value": "short", "category": "Cyber",
             "entry_type": "harmful", "planned": "2", "accepted": "1"},
            {"axis": "length", "value": "long", "category": "Cyber",
             "entry_type": "harmful", "planned": "0", "accepted": "0"},
            {"axis": "length", "value": "short", "category": "Fraud",
             "entry_type": "benign", "planned": "0", "accepted": "0"},
            {"axis": "length", "value": "long", "category": "Fraud",
             "entry_type": "benign", "planned": "1", "accepted": "0"},
        ]

    def test_a_value_the_table_no_longer_has_keeps_its_counts(self, tmp_path):
        axes = {"length": {"label": "Length", "values": {
            "short": {"label": "s", "hint": "", "weight": 1}}}}
        group = ("Cyber", "harmful")
        path = A.write_coverage(tmp_path / "x.coverage.csv", axes, [],
                                [(group, {"length": "gone"})])
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[-1] == {"axis": "length", "value": "gone", "category": "Cyber",
                            "entry_type": "harmful", "planned": "0", "accepted": "1"}


class TestTheStageCoverage:
    _AXES = {"length": {"label": "Length", "values": {
        "short": {"label": "s", "hint": "", "weight": 1},
        "long": {"label": "l", "hint": "", "weight": 1}}}}

    def _sidecars(self, tmp_path) -> tuple[Manifest, Ledger]:
        manifest = Manifest(tmp_path / "x.manifest.jsonl")
        ledger = Ledger(tmp_path / "x.state.jsonl",
                        key_fields=("entry", "iteration"), casters={"iteration": int})
        manifest.write(
            {"entry": entry, "iteration": it, "category": "Cyber",
             "entry_type": "harmful", "spec": {"length": length}, "status": "planned"}
            for entry, it, length in (("a", 0, "short"), ("a", 1, "long"),
                                      ("b", 0, "short"), ("b", 1, "long")))
        ledger.record([], saved={})
        ledger.record([
            {"entry": "a", "iteration": 0, "ids": ["1"], "accepted": True},
            {"entry": "a", "iteration": 1, "ids": ["2"], "accepted": False},
            {"entry": "b", "iteration": 0, "status": "rejected"},
            {"entry": "b", "iteration": "1", "ids": ["4"], "accepted": True},
        ], saved={"f": [1, "x"]})
        return manifest, ledger

    def test_it_joins_the_planned_specs_to_the_acks_by_key(self, tmp_path):
        manifest, ledger = self._sidecars(tmp_path)
        path = A.write_stage_coverage(self._AXES, manifest, ledger)

        assert path == tmp_path / "x.coverage.csv"
        with path.open(encoding="utf-8", newline="") as fh:
            rows = {row["value"]: (row["planned"], row["accepted"])
                    for row in csv.DictReader(fh)}
        assert rows == {"short": ("2", "1"), "long": ("2", "1")}

    def test_without_axes_it_removes_the_table(self, tmp_path):
        manifest, ledger = self._sidecars(tmp_path)
        path = A.write_stage_coverage(self._AXES, manifest, ledger)
        assert A.write_stage_coverage(None, manifest, ledger) is None
        assert not path.exists()
